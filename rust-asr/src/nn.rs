//! Stateless neural-net helpers (functional, f32). Weights are passed in
//! explicitly so the encoder/decoder modules own loading + layout fixups.

use anyhow::Result;
use mlx_rs::ops::{mean_axis, rsqrt, sqrt, square, sum_axis};
use mlx_rs::Array;

/// y = x @ w^T (+ b). w is stored [out, in] (PyTorch/MLX Linear layout).
pub fn linear(x: &Array, w: &Array, b: Option<&Array>) -> Result<Array> {
    let y = x.matmul(&w.t())?;
    Ok(match b {
        Some(b) => &y + b,
        None => y,
    })
}

/// LayerNorm over the last axis.
pub fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f32) -> Result<Array> {
    let last = (x.ndim() - 1) as i32;
    let mean = mean_axis(x, last, true)?;
    let centered = x - &mean;
    let var = mean_axis(&square(&centered)?, last, true)?;
    let normed = &centered * &rsqrt(&(&var + Array::from_f32(eps)))?;
    Ok(&(&normed * w) + b)
}

/// BatchNorm in eval mode using running statistics, applied over the last axis
/// (channels-last activations [..., C]).
pub fn batch_norm_eval(
    x: &Array,
    w: &Array,
    b: &Array,
    running_mean: &Array,
    running_var: &Array,
    eps: f32,
) -> Result<Array> {
    let inv = &rsqrt(&(running_var + Array::from_f32(eps)))?;
    let normed = &(x - running_mean) * inv;
    Ok(&(&normed * w) + b)
}

/// Population/sample stats helper kept for parity experiments (unused in the
/// hot path but handy when diffing intermediate tensors).
#[allow(dead_code)]
pub fn std_axis(x: &Array, axis: i32, ddof: i32, keepdims: bool) -> Result<Array> {
    let n = x.shape()[axis as usize] - ddof;
    let mean = mean_axis(x, axis, true)?;
    let centered = x - &mean;
    let var = &sum_axis(&square(&centered)?, axis, keepdims)? / Array::from_f32(n as f32);
    Ok(sqrt(&var)?)
}
