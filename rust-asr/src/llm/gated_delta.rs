//! Gated delta rule recurrence (Gated DeltaNet), the state update behind
//! Qwen3.5's linear-attention layers.
//!
//! Prompt prefill runs it as a custom Metal kernel — the same kernel source
//! mlx-lm uses — because an op-by-op loop over thousands of tokens × 24 layers
//! is far too slow. `step_ops` is the plain-ops reference for one timestep,
//! used for single-token decode steps and to check the kernel.

use anyhow::{anyhow, Result};
use mlx_rs::ops::{expand_dims, multiply, subtract, sum_axis};
use mlx_rs::{Array, Dtype};
use std::ffi::CString;

/// One SIMD group (32 threads) per value row; each thread owns Dk/32 state
/// elements. Shapes: q, k [B, T, Hk, Dk]; v, y [B, T, Hv, Dv]; g, beta
/// [B, T, Hv]; state [B, Hv, Dv, Dk].
const KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    // g, beta: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      // Increment data pointers to next time step
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"#;

fn check(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(anyhow!("MLX {what} failed"))
    }
}

/// The compiled-on-first-use Metal kernel for the recurrence.
pub struct GatedDeltaKernel {
    kernel: mlx_sys::mlx_fast_metal_kernel,
}

// The handle is an immutable kernel description; MLX kernels may be applied
// from any thread (callers serialize GPU work through the engine mutex).
unsafe impl Send for GatedDeltaKernel {}
unsafe impl Sync for GatedDeltaKernel {}

impl GatedDeltaKernel {
    pub fn new() -> Result<Self> {
        let cstr = |s: &str| CString::new(s).expect("no interior NUL");
        unsafe {
            let inputs = mlx_sys::mlx_vector_string_new();
            for name in ["q", "k", "v", "g", "beta", "state_in", "T"] {
                check(
                    mlx_sys::mlx_vector_string_append_value(inputs, cstr(name).as_ptr()),
                    "kernel input name",
                )?;
            }
            let outputs = mlx_sys::mlx_vector_string_new();
            for name in ["y", "state_out"] {
                check(
                    mlx_sys::mlx_vector_string_append_value(outputs, cstr(name).as_ptr()),
                    "kernel output name",
                )?;
            }
            let kernel = mlx_sys::mlx_fast_metal_kernel_new(
                cstr("gated_delta_step").as_ptr(),
                inputs,
                outputs,
                cstr(KERNEL_SOURCE).as_ptr(),
                cstr("").as_ptr(),
                true,
                false,
            );
            mlx_sys::mlx_vector_string_free(inputs);
            mlx_sys::mlx_vector_string_free(outputs);
            Ok(Self { kernel })
        }
    }

    /// Run the recurrence over all T steps. Returns (y, new state).
    /// Requires Dk to be a multiple of 32.
    pub fn apply(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> Result<(Array, Array)> {
        let (b, t, hk, dk) = (k.shape()[0], k.shape()[1], k.shape()[2], k.shape()[3]);
        let (hv, dv) = (v.shape()[2], v.shape()[3]);
        if dk % 32 != 0 {
            return Err(anyhow!("gated delta kernel needs Dk % 32 == 0, got {dk}"));
        }
        let steps = Array::from_int(t);
        let y_shape = [b, t, hv, dv];
        let cstr = |s: &str| CString::new(s).expect("no interior NUL");

        unsafe {
            let config = mlx_sys::mlx_fast_metal_kernel_config_new();
            let inputs = mlx_sys::mlx_vector_array_new();
            let mut outputs = mlx_sys::mlx_vector_array_new();
            let stream = mlx_sys::mlx_default_gpu_stream_new();

            let result = (|| -> Result<(Array, Array)> {
                check(
                    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config,
                        y_shape.as_ptr(),
                        y_shape.len(),
                        q.dtype() as mlx_sys::mlx_dtype,
                    ),
                    "kernel output y",
                )?;
                check(
                    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config,
                        state.shape().as_ptr(),
                        state.shape().len(),
                        state.dtype() as mlx_sys::mlx_dtype,
                    ),
                    "kernel output state",
                )?;
                check(
                    mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, dv, b * hv),
                    "kernel grid",
                )?;
                check(
                    mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1),
                    "kernel thread group",
                )?;
                for (name, dtype) in [("InT", q.dtype()), ("StT", state.dtype())] {
                    check(
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config,
                            cstr(name).as_ptr(),
                            dtype as mlx_sys::mlx_dtype,
                        ),
                        "kernel template dtype",
                    )?;
                }
                for (name, value) in [("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)] {
                    check(
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config,
                            cstr(name).as_ptr(),
                            value,
                        ),
                        "kernel template int",
                    )?;
                }
                for a in [q, k, v, g, beta, state, &steps] {
                    check(
                        mlx_sys::mlx_vector_array_append_value(inputs, a.as_ptr()),
                        "kernel input",
                    )?;
                }
                check(
                    mlx_sys::mlx_fast_metal_kernel_apply(
                        &mut outputs,
                        self.kernel,
                        inputs,
                        config,
                        stream,
                    ),
                    "gated delta kernel",
                )?;
                let mut y = mlx_sys::mlx_array_new();
                let mut new_state = mlx_sys::mlx_array_new();
                check(mlx_sys::mlx_vector_array_get(&mut y, outputs, 0), "kernel y")?;
                check(
                    mlx_sys::mlx_vector_array_get(&mut new_state, outputs, 1),
                    "kernel state",
                )?;
                Ok((Array::from_ptr(y), Array::from_ptr(new_state)))
            })();

            mlx_sys::mlx_stream_free(stream);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            result
        }
    }
}

impl Drop for GatedDeltaKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.kernel) };
    }
}

/// One timestep with plain ops. q, k: [B, Hv, Dk] (already repeated to the
/// value heads); v: [B, Hv, Dv]; g, beta: [B, Hv]; state: [B, Hv, Dv, Dk].
/// Returns (y [B, Hv, Dv], new state).
pub fn step_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    let decay = expand_dims(&expand_dims(g, -1)?, -1)?; // [B, H, 1, 1]
    let k_row = expand_dims(k, -2)?; // [B, H, 1, Dk]
    let state = multiply(state, &decay)?;
    let kv_mem = sum_axis(&multiply(&state, &k_row)?, -1, false)?; // [B, H, Dv]
    let delta = multiply(&subtract(v, &kv_mem)?, &expand_dims(beta, -1)?)?;
    let state = &state + &multiply(&k_row, &expand_dims(&delta, -1)?)?;
    let y = sum_axis(&multiply(&state, &expand_dims(q, -2)?)?, -1, false)?;
    Ok((y.as_dtype(q.dtype())?, state.as_dtype(Dtype::Float32)?))
}
