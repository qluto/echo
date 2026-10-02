//! Qwen3.5 text decoder (4-bit quantized) on Apple MLX, for ASR
//! post-processing. Port of mlx-lm's `qwen3_5.py` (text path only — the
//! checkpoint's vision tower is not loaded).
//!
//! Unlike Qwen3's plain transformer, three of every four layers replace
//! softmax attention with a Gated DeltaNet ("linear attention"): a causal
//! depthwise conv over the q/k/v projections followed by a gated delta-rule
//! state update, with constant-size recurrent state instead of a KV cache.
//! Every fourth layer is GQA attention with per-head q/k RMSNorm, RoPE on the
//! first quarter of each head's dims, and a sigmoid output gate. Greedy
//! decoding.

use anyhow::{anyhow, Result};
use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::{sigmoid, silu, softplus};
use mlx_rs::ops::indexing::{argmax_axis, take_axis, IndexOp};
use mlx_rs::ops::{
    concatenate_axis, conv1d, dequantize, exp, expand_dims, multiply, ones, quantized_matmul,
    reshape, zeros_dtype,
};
use mlx_rs::{Array, Dtype};

use super::gated_delta::{step_ops, GatedDeltaKernel};
use super::qwen3::{causal_mask, repeat_kv, Kv, QLinear, BITS, GROUP_SIZE};
use crate::weights::Weights;

const PREFIX: &str = "language_model.model";

#[derive(Clone)]
pub struct Config {
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    /// Leading dims of each attention head that RoPE rotates.
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// Every n-th layer is full attention; the rest are Gated DeltaNet.
    pub full_attention_interval: usize,
    pub linear_k_heads: i32,
    pub linear_v_heads: i32,
    pub linear_k_dim: i32,
    pub linear_v_dim: i32,
    pub conv_kernel: i32,
    /// Generation stops at any of these.
    pub stop_token_ids: Vec<i32>,
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Array,
    k_norm: Array,
}

struct GatedDeltaNet {
    in_proj_qkv: QLinear,
    in_proj_z: QLinear,
    in_proj_b: QLinear,
    in_proj_a: QLinear,
    /// Depthwise conv weight [conv_dim, kernel, 1].
    conv_w: Array,
    /// exp(A_log), f32 [Hv].
    a: Array,
    dt_bias: Array,
    norm: Array,
    out_proj: QLinear,
}

enum Mixer {
    Attention(Attention),
    Linear(GatedDeltaNet),
}

struct Layer {
    input_ln: Array,
    mixer: Mixer,
    post_ln: Array,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

/// Per-layer recurrent state.
enum Cache {
    Attention(Option<Kv>),
    Linear {
        /// Last `conv_kernel - 1` qkv rows, [1, K-1, conv_dim].
        conv: Option<Array>,
        /// Delta-rule state, f32 [1, Hv, Dv, Dk].
        state: Option<Array>,
    },
}

pub struct Qwen3_5 {
    cfg: Config,
    embed_w: Array,
    embed_scales: Array,
    embed_biases: Array,
    layers: Vec<Layer>,
    norm: Array,
    kernel: GatedDeltaKernel,
}

impl Qwen3_5 {
    pub fn load(w: &Weights, cfg: Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("{PREFIX}.layers.{i}");
            let mixer = if (i + 1) % cfg.full_attention_interval == 0 {
                let a = format!("{p}.self_attn");
                Mixer::Attention(Attention {
                    q_proj: QLinear::load(w, &format!("{a}.q_proj"))?,
                    k_proj: QLinear::load(w, &format!("{a}.k_proj"))?,
                    v_proj: QLinear::load(w, &format!("{a}.v_proj"))?,
                    o_proj: QLinear::load(w, &format!("{a}.o_proj"))?,
                    q_norm: w.raw(&format!("{a}.q_norm.weight"))?.clone(),
                    k_norm: w.raw(&format!("{a}.k_norm.weight"))?.clone(),
                })
            } else {
                let a = format!("{p}.linear_attn");
                Mixer::Linear(GatedDeltaNet {
                    in_proj_qkv: QLinear::load(w, &format!("{a}.in_proj_qkv"))?,
                    in_proj_z: QLinear::load(w, &format!("{a}.in_proj_z"))?,
                    in_proj_b: QLinear::load(w, &format!("{a}.in_proj_b"))?,
                    in_proj_a: QLinear::load(w, &format!("{a}.in_proj_a"))?,
                    conv_w: w.raw(&format!("{a}.conv1d.weight"))?.clone(),
                    a: exp(w.f32(&format!("{a}.A_log"))?)?,
                    dt_bias: w.raw(&format!("{a}.dt_bias"))?.clone(),
                    norm: w.raw(&format!("{a}.norm.weight"))?.clone(),
                    out_proj: QLinear::load(w, &format!("{a}.out_proj"))?,
                })
            };
            layers.push(Layer {
                input_ln: w.raw(&format!("{p}.input_layernorm.weight"))?.clone(),
                mixer,
                post_ln: w.raw(&format!("{p}.post_attention_layernorm.weight"))?.clone(),
                gate: QLinear::load(w, &format!("{p}.mlp.gate_proj"))?,
                up: QLinear::load(w, &format!("{p}.mlp.up_proj"))?,
                down: QLinear::load(w, &format!("{p}.mlp.down_proj"))?,
            });
        }
        Ok(Self {
            embed_w: w.raw(&format!("{PREFIX}.embed_tokens.weight"))?.clone(),
            embed_scales: w.raw(&format!("{PREFIX}.embed_tokens.scales"))?.clone(),
            embed_biases: w.raw(&format!("{PREFIX}.embed_tokens.biases"))?.clone(),
            layers,
            norm: w.raw(&format!("{PREFIX}.norm.weight"))?.clone(),
            kernel: GatedDeltaKernel::new()?,
            cfg,
        })
    }

    /// Embedding lookup for token ids [T] -> [1, T, hidden] (gather + dequantize).
    fn embed(&self, ids: &Array) -> Result<Array> {
        let qw = take_axis(&self.embed_w, ids, 0)?;
        let qs = take_axis(&self.embed_scales, ids, 0)?;
        let qb = take_axis(&self.embed_biases, ids, 0)?;
        let e = dequantize(&qw, &qs, &qb, GROUP_SIZE, BITS)?;
        Ok(expand_dims(&e, 0)?)
    }

    /// Tied lm_head: logits for the last position only. h: [1, T, hidden].
    fn lm_head_last(&self, h: &Array) -> Result<Array> {
        let t = h.shape()[1];
        let last = h.index((.., t - 1, ..));
        Ok(quantized_matmul(
            &last,
            &self.embed_w,
            &self.embed_scales,
            &self.embed_biases,
            true,
            GROUP_SIZE,
            BITS,
        )?)
    }

    fn new_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Attention(_) => Cache::Attention(None),
                Mixer::Linear(_) => Cache::Linear {
                    conv: None,
                    state: None,
                },
            })
            .collect()
    }

    fn forward(&self, ids: &Array, caches: &mut [Cache], offset: i32) -> Result<Array> {
        let mut h = self.embed(ids)?;
        let t = h.shape()[1];
        let mask = if t > 1 { Some(causal_mask(t)?) } else { None };
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let hn = rms_norm(&h, &layer.input_ln, self.cfg.rms_eps)?;
            let r = match (&layer.mixer, cache) {
                (Mixer::Attention(a), Cache::Attention(kv)) => {
                    self.attention(a, &hn, kv, offset, mask.as_ref())?
                }
                (Mixer::Linear(g), Cache::Linear { conv, state }) => {
                    self.gated_delta_net(g, &hn, conv, state)?
                }
                _ => return Err(anyhow!("cache does not match layer type")),
            };
            h = &h + &r;
            let hn = rms_norm(&h, &layer.post_ln, self.cfg.rms_eps)?;
            let mlp = layer
                .down
                .forward(&(&silu(&layer.gate.forward(&hn)?)? * &layer.up.forward(&hn)?))?;
            h = &h + &mlp;
        }
        Ok(rms_norm(&h, &self.norm, self.cfg.rms_eps)?)
    }

    fn attention(
        &self,
        a: &Attention,
        x: &Array,
        cache: &mut Option<Kv>,
        offset: i32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, t) = (x.shape()[0], x.shape()[1]);

        // q_proj yields queries and the output gate, interleaved per head.
        let qg = reshape(&a.q_proj.forward(x)?, &[b, t, cfg.n_heads, 2 * cfg.head_dim])?;
        let q = qg.index((.., .., .., ..cfg.head_dim));
        let gate = reshape(
            &qg.index((.., .., .., cfg.head_dim..)),
            &[b, t, cfg.n_heads * cfg.head_dim],
        )?;
        let k = reshape(&a.k_proj.forward(x)?, &[b, t, cfg.n_kv_heads, cfg.head_dim])?;
        let v = reshape(&a.v_proj.forward(x)?, &[b, t, cfg.n_kv_heads, cfg.head_dim])?;

        let q = rms_norm(&q, &a.q_norm, cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = rms_norm(&k, &a.k_norm, cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
        let mut v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Partial RoPE: only the first `rope_dims` features rotate.
        let q = rope(&q, cfg.rope_dims, false, Some(cfg.rope_theta), 1.0, offset, None)?;
        let mut k = rope(&k, cfg.rope_dims, false, Some(cfg.rope_theta), 1.0, offset, None)?;

        if let Some(prev) = cache.as_ref() {
            k = concatenate_axis(&[prev.k.clone(), k], 2)?;
            v = concatenate_axis(&[prev.v.clone(), v], 2)?;
        }
        *cache = Some(Kv {
            k: k.clone(),
            v: v.clone(),
        });

        let rep = cfg.n_heads / cfg.n_kv_heads;
        let k = repeat_kv(&k, rep)?;
        let v = repeat_kv(&v, rep)?;

        let scale = (cfg.head_dim as f32).powf(-0.5);
        let mask = mask.map(ScaledDotProductAttentionMask::Array);
        let out = scaled_dot_product_attention(&q, &k, &v, scale, mask)?;
        let out = reshape(
            &out.transpose_axes(&[0, 2, 1, 3])?,
            &[b, t, cfg.n_heads * cfg.head_dim],
        )?;
        a.o_proj.forward(&multiply(&out, &sigmoid(&gate)?)?)
    }

    fn gated_delta_net(
        &self,
        g: &GatedDeltaNet,
        x: &Array,
        conv_cache: &mut Option<Array>,
        state_cache: &mut Option<Array>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let (hk, hv, dk, dv) = (
            cfg.linear_k_heads,
            cfg.linear_v_heads,
            cfg.linear_k_dim,
            cfg.linear_v_dim,
        );
        let key_dim = hk * dk;
        let conv_dim = 2 * key_dim + hv * dv;

        let qkv = g.in_proj_qkv.forward(x)?;
        let z = reshape(&g.in_proj_z.forward(x)?, &[b, t, hv, dv])?;
        let beta = sigmoid(&g.in_proj_b.forward(x)?)?;
        let a = g.in_proj_a.forward(x)?;

        // Causal depthwise conv over [previous K-1 rows, new rows].
        let keep = cfg.conv_kernel - 1;
        let conv_state = match conv_cache.take() {
            Some(s) => s,
            None => zeros_dtype(&[b, keep, conv_dim], qkv.dtype())?,
        };
        let conv_input = concatenate_axis(&[conv_state, qkv], 1)?;
        *conv_cache = Some(conv_input.index((.., t.., ..)));
        let conv_out = silu(&conv1d(&conv_input, &g.conv_w, 1, 0, 1, conv_dim)?)?;

        let q = reshape(&conv_out.index((.., .., ..key_dim)), &[b, t, hk, dk])?;
        let k = reshape(&conv_out.index((.., .., key_dim..2 * key_dim)), &[b, t, hk, dk])?;
        let v = reshape(&conv_out.index((.., .., 2 * key_dim..)), &[b, t, hv, dv])?;

        // L2-normalize q/k, folding the Dk^-0.5 readout scale into q.
        // rms_norm adds eps to mean(x²) where the reference l2norm adds it to
        // sum(x²), hence eps / Dk.
        let inv_scale = (dk as f32).powf(-0.5);
        let unit = ones::<f32>(&[dk])?.as_dtype(q.dtype())?;
        let l2_eps = 1e-6 * inv_scale * inv_scale;
        let q = multiply(
            &rms_norm(&q, &unit, l2_eps)?,
            &Array::from_f32(inv_scale * inv_scale).as_dtype(q.dtype())?,
        )?;
        let k = multiply(
            &rms_norm(&k, &unit, l2_eps)?,
            &Array::from_f32(inv_scale).as_dtype(k.dtype())?,
        )?;

        // Decay gate: exp(-exp(A_log) * softplus(a + dt_bias)), f32 [B, T, Hv].
        let decay = exp(&(-&multiply(&g.a, &softplus(&(&a + &g.dt_bias))?)?))?;

        let state = match state_cache.take() {
            Some(s) => s,
            None => zeros_dtype(&[b, hv, dv, dk], Dtype::Float32)?,
        };
        let (out, state) = if t == 1 {
            let rep = hv / hk;
            let step = |x: &Array| x.index((.., 0, ..));
            let q1 = repeat_heads(&q.index((.., 0, .., ..)), rep)?;
            let k1 = repeat_heads(&k.index((.., 0, .., ..)), rep)?;
            let (y, state) = step_ops(
                &q1,
                &k1,
                &v.index((.., 0, .., ..)),
                &step(&decay),
                &step(&beta),
                &state,
            )?;
            (expand_dims(&y, 1)?, state)
        } else {
            self.kernel.apply(&q, &k, &v, &decay, &beta, &state)?
        };
        *state_cache = Some(state);

        // Gated RMSNorm: norm(out) * silu(z), in f32.
        let normed = rms_norm(&out, &g.norm, cfg.rms_eps)?.as_dtype(Dtype::Float32)?;
        let gated = multiply(&silu(&z.as_dtype(Dtype::Float32)?)?, &normed)?.as_dtype(out.dtype())?;
        g.out_proj.forward(&reshape(&gated, &[b, t, hv * dv])?)
    }

    /// Greedy generation. Returns generated token ids (excluding the prompt),
    /// stopping at a stop token or max_tokens.
    pub fn generate(&self, prompt_ids: &[i32], max_tokens: usize) -> Result<Vec<i32>> {
        let mut caches = self.new_caches();
        let prompt = Array::from_slice(prompt_ids, &[prompt_ids.len() as i32]);
        let h = self.forward(&prompt, &mut caches, 0)?;
        let mut next = argmax_axis(&self.lm_head_last(&h)?, 1, false)?.item::<u32>() as i32;

        let mut out = Vec::new();
        let mut pos = prompt_ids.len() as i32;
        for _ in 0..max_tokens {
            if self.cfg.stop_token_ids.contains(&next) {
                break;
            }
            out.push(next);
            let tok = Array::from_slice(&[next], &[1]);
            let h = self.forward(&tok, &mut caches, pos)?;
            next = argmax_axis(&self.lm_head_last(&h)?, 1, false)?.item::<u32>() as i32;
            pos += 1;
        }
        Ok(out)
    }
}

/// [B, Hk, D] -> [B, Hk * rep, D], each key head repeated for its value heads.
fn repeat_heads(x: &Array, rep: i32) -> Result<Array> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let x = mlx_rs::ops::broadcast_to(&reshape(x, &[b, h, 1, d])?, &[b, h, rep, d])?;
    Ok(reshape(&x, &[b, h * rep, d])?)
}
