//! Kimi Delta Attention (KDA) prediction network of Haas-ja-v1.0, recurrent
//! form only.
//!
//! Haas replaces the 2-layer LSTM of NeMo's TDT prediction network with
//! pre-norm residual KDA blocks (`x = x + kda(norm(x))`, then a final
//! LayerNorm). A transducer feeds the prediction network one label at a time,
//! so only the step-by-step recurrence is needed; the math follows
//! flash-linear-attention 0.5.2's `KimiDeltaAttention` with its pure-PyTorch
//! references (`fla/ops/kda/naive.py::naive_recurrent_kda`,
//! `fla/ops/kda/gate.py::naive_kda_gate`):
//!
//! - q, k, v = silu(causal depthwise conv4(proj(x))); q, k L2-normalized per head
//! - g = -exp(A_log) * softplus(f_proj(x) + dt_bias)   (per head *and* channel)
//! - beta = sigmoid(b_proj(x))
//! - S = S * exp(g);  S += beta * k ⊗ (v - kᵀS);  o = (q / √K)ᵀ S
//! - out = o_proj(rms_norm(o) * sigmoid(g_proj(x)))
//!
//! The state is tiny (5 × 64 × 64 per layer plus three 4-tap conv buffers) and
//! kept in f32, like the rest of this f32 port.

use anyhow::Result;
use mlx_rs::nn::{sigmoid, silu, softplus};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::ops::{concatenate, exp, expand_dims, mean_axis, rsqrt, square, sum_axis, zeros};
use mlx_rs::Array;

use crate::nn::{layer_norm, linear};
use crate::weights::Weights;

pub const HIDDEN: i32 = 640;
const N_LAYERS: usize = 2;
const N_HEADS: i32 = 5;
const HEAD_DIM: i32 = 64;
const KEY_DIM: i32 = N_HEADS * HEAD_DIM; // 320 (value_dim is the same: expand_v = 1)
const CONV: i32 = 4;
const NORM_EPS: f32 = 1e-5;
const L2_EPS: f32 = 1e-6;

struct KdaLayer {
    norm_w: Array,
    norm_b: Array,
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    q_conv: Array, // [320, 4]
    k_conv: Array,
    v_conv: Array,
    f_first: Array, // [64, 640]
    f_second: Array, // [320, 64]
    b_proj: Array, // [5, 640]
    neg_a: Array, // -exp(A_log), [5, 1]
    dt_bias: Array, // [5, 64]
    g_first: Array, // [64, 640]
    g_second: Array, // [320, 64]
    g_second_b: Array,
    o_norm: Array, // [64]
    o_proj: Array, // [640, 320]
}

/// Per-layer decoding state.
#[derive(Clone)]
pub struct LayerState {
    recurrent: Array, // [1, H, K, V]
    conv_q: Array,    // [1, 320, 4], oldest input first
    conv_k: Array,
    conv_v: Array,
}

pub type State = Vec<LayerState>;

pub struct PredictionNetwork {
    layers: Vec<KdaLayer>,
    final_w: Array,
    final_b: Array,
}

impl PredictionNetwork {
    pub fn load(w: &Weights) -> Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let g = |s: &str| w.f32(&format!("prediction.layers.{i}.{s}"));
            let a_log = g("A_log")?.reshape(&[N_HEADS, 1])?;
            layers.push(KdaLayer {
                norm_w: w.f32(&format!("prediction.norms.{i}.weight"))?,
                norm_b: w.f32(&format!("prediction.norms.{i}.bias"))?,
                q_proj: g("q_proj.weight")?,
                k_proj: g("k_proj.weight")?,
                v_proj: g("v_proj.weight")?,
                q_conv: g("q_conv_weight")?,
                k_conv: g("k_conv_weight")?,
                v_conv: g("v_conv_weight")?,
                f_first: g("f_proj.first.weight")?,
                f_second: g("f_proj.second.weight")?,
                b_proj: g("b_proj.weight")?,
                neg_a: -exp(&a_log)?,
                dt_bias: g("dt_bias")?.reshape(&[N_HEADS, HEAD_DIM])?,
                g_first: g("g_proj.first.weight")?,
                g_second: g("g_proj.second.weight")?,
                g_second_b: g("g_proj.second.bias")?,
                o_norm: g("o_norm_weight")?,
                o_proj: g("o_proj.weight")?,
            });
        }
        Ok(Self {
            layers,
            final_w: w.f32("prediction.final_norm.weight")?,
            final_b: w.f32("prediction.final_norm.bias")?,
        })
    }

    pub fn empty_state(&self) -> Result<State> {
        let conv = zeros::<f32>(&[1, KEY_DIM, CONV])?;
        let layer = LayerState {
            recurrent: zeros::<f32>(&[1, N_HEADS, HEAD_DIM, HEAD_DIM])?,
            conv_q: conv.clone(),
            conv_k: conv.clone(),
            conv_v: conv,
        };
        Ok(vec![layer; N_LAYERS])
    }

    /// One label step. y: embedded label [1, 640] -> (output [1, 640], new state).
    pub fn step(&self, y: &Array, state: &State) -> Result<(Array, State)> {
        let mut x = y.clone();
        let mut next = Vec::with_capacity(N_LAYERS);
        for (layer, st) in self.layers.iter().zip(state) {
            let h = layer_norm(&x, &layer.norm_w, &layer.norm_b, NORM_EPS)?;
            let (out, new_st) = kda_step(layer, &h, st)?;
            x = &x + &out;
            next.push(new_st);
        }
        Ok((layer_norm(&x, &self.final_w, &self.final_b, NORM_EPS)?, next))
    }
}

/// Causal depthwise conv over the last `CONV` inputs, then SiLU.
/// u: [1, D]; state: [1, D, CONV] -> (out [1, D], new state).
fn short_conv(u: &Array, state: &Array, weight: &Array) -> Result<(Array, Array)> {
    let window = concatenate(&[state.index((.., .., 1..)), expand_dims(u, -1)?], -1)?;
    let out = sum_axis(&(&window * weight), -1, false)?;
    Ok((silu(&out)?, window))
}

/// x / sqrt(sum(x²) + eps) over the last axis (fla's L2Norm).
fn l2norm(x: &Array) -> Result<Array> {
    let ss = sum_axis(&square(x)?, -1, true)?;
    Ok(x * &rsqrt(&(&ss + Array::from_f32(L2_EPS)))?)
}

fn kda_step(l: &KdaLayer, x: &Array, st: &LayerState) -> Result<(Array, LayerState)> {
    let heads = |a: &Array| a.reshape(&[1, N_HEADS, HEAD_DIM]);

    let (q, conv_q) = short_conv(&linear(x, &l.q_proj, None)?, &st.conv_q, &l.q_conv)?;
    let (k, conv_k) = short_conv(&linear(x, &l.k_proj, None)?, &st.conv_k, &l.k_conv)?;
    let (v, conv_v) = short_conv(&linear(x, &l.v_proj, None)?, &st.conv_v, &l.v_conv)?;
    let q = &l2norm(&heads(&q)?)? * Array::from_f32((HEAD_DIM as f32).powf(-0.5));
    let k = l2norm(&heads(&k)?)?;
    let v = heads(&v)?;

    // Log-space forget gate, per (head, key channel): [1, H, K].
    let f = heads(&linear(&linear(x, &l.f_first, None)?, &l.f_second, None)?)?;
    let g = &l.neg_a * &softplus(&(&f + &l.dt_bias))?;
    let beta = sigmoid(&linear(x, &l.b_proj, None)?)?; // [1, H]

    // Delta rule on S [1, H, K, V].
    let s = &st.recurrent * &expand_dims(&exp(&g)?, -1)?;
    let k_col = expand_dims(&k, -1)?; // [1, H, K, 1]
    let kv_mem = sum_axis(&(&k_col * &s), 2, false)?; // [1, H, V]
    let delta = &v - &kv_mem;
    let bk = &k_col * &beta.index((.., .., NewAxis, NewAxis));
    let s = &s + &(&bk * &expand_dims(&delta, 2)?);
    let o = sum_axis(&(&expand_dims(&q, -1)? * &s), 2, false)?; // [1, H, V]

    // Sigmoid-gated RMSNorm over each head's values, then the output projection.
    let gate = heads(&linear(
        &linear(x, &l.g_first, None)?,
        &l.g_second,
        Some(&l.g_second_b),
    )?)?;
    let ms = mean_axis(&square(&o)?, -1, true)?;
    let o = &(&(&o * &rsqrt(&(&ms + Array::from_f32(NORM_EPS)))?) * &l.o_norm) * &sigmoid(&gate)?;
    let out = linear(&o.reshape(&[1, KEY_DIM])?, &l.o_proj, None)?;

    Ok((
        out,
        LayerState {
            recurrent: s,
            conv_q,
            conv_k,
            conv_v,
        },
    ))
}
