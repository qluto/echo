//! Gemma 4 text decoder (4-bit quantized) on Apple MLX, for ASR
//! post-processing. Port of mlx-lm's `gemma4_text.py` for the dense
//! "E" models (E2B/E4B) — text path only; the checkpoint's vision and audio
//! towers are never loaded.
//!
//! Differences from a plain Llama/Qwen3 decoder:
//! - Layers alternate sliding-window attention (window 512, RoPE θ=1e4,
//!   head_dim 256) with full attention (head_dim 512, "proportional" RoPE
//!   that rotates only the first quarter of each head).
//! - The last `num_kv_shared_layers` layers compute no K/V of their own; they
//!   reuse those of the last earlier layer of the same attention type.
//! - Per-layer embeddings (PLE): every layer gets its own 256-dim token
//!   embedding, gated into the residual stream after the MLP.
//! - q/k/v are RMS-normalized (v without a scale), attention scale is 1, the
//!   MLP is GeGLU, and each layer output is multiplied by a learned scalar.
//! - Final logits are soft-capped (tanh). Greedy decoding.

use anyhow::{anyhow, Result};
use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{take_axis, IndexOp};
use mlx_rs::ops::{
    arange, concatenate, dequantize, divide, expand_dims, full, le, logical_and, lt,
    multiply, power, quantized_matmul, reshape, subtract, tanh,
};
use mlx_rs::{Array, Dtype};

use super::decode;
use super::kv_cache::KvCache;
use super::qwen3::{QLinear, BITS, GROUP_SIZE};
use crate::weights::Weights;

const PREFIX: &str = "language_model.model";

#[derive(Clone)]
pub struct Config {
    pub hidden_size: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    /// head_dim of sliding-window layers.
    pub head_dim: i32,
    /// head_dim of full-attention layers.
    pub global_head_dim: i32,
    pub rms_eps: f32,
    pub sliding_window: i32,
    /// Per layer: true for full attention, false for sliding window.
    pub full_attention: Vec<bool>,
    pub n_kv_shared_layers: usize,
    /// Per-layer-embedding width (hidden_size_per_layer_input).
    pub ple_dim: i32,
    pub sliding_rope_theta: f32,
    pub full_rope_theta: f32,
    pub full_partial_rotary: f32,
    pub final_logit_softcap: Option<f32>,
    /// Generation stops at any of these.
    pub stop_token_ids: Vec<i32>,
}

impl Config {
    fn n_layers(&self) -> usize {
        self.full_attention.len()
    }

    /// Index of the first layer that reuses an earlier layer's K/V.
    fn first_shared(&self) -> usize {
        self.n_layers() - self.n_kv_shared_layers
    }
}

/// A quantized embedding table (gathered by row, then dequantized).
struct QEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
}

impl QEmbedding {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.raw(&format!("{prefix}.weight"))?.clone(),
            scales: w.raw(&format!("{prefix}.scales"))?.clone(),
            biases: w.raw(&format!("{prefix}.biases"))?.clone(),
        })
    }

    /// ids [T] -> [T, dim].
    fn lookup(&self, ids: &Array) -> Result<Array> {
        let qw = take_axis(&self.weight, ids, 0)?;
        let qs = take_axis(&self.scales, ids, 0)?;
        let qb = take_axis(&self.biases, ids, 0)?;
        Ok(dequantize(&qw, &qs, &qb, GROUP_SIZE, BITS)?)
    }

    /// Tied output projection: x [.., hidden] -> logits [.., vocab].
    fn as_linear(&self, x: &Array) -> Result<Array> {
        Ok(quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            GROUP_SIZE,
            BITS,
        )?)
    }
}

struct KvProj {
    k_proj: QLinear,
    v_proj: QLinear,
    k_norm: Array,
}

struct Attention {
    q_proj: QLinear,
    /// None on KV-shared layers.
    kv: Option<KvProj>,
    o_proj: QLinear,
    q_norm: Array,
}

struct Layer {
    full: bool,
    head_dim: i32,
    /// Layer whose K/V this layer attends over (itself unless KV-shared).
    kv_source: usize,
    input_ln: Array,
    attn: Attention,
    post_attn_ln: Array,
    pre_ff_ln: Array,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    post_ff_ln: Array,
    ple_gate: QLinear,
    ple_proj: QLinear,
    ple_norm: Array,
    scalar: Array,
}

pub struct Gemma4 {
    cfg: Config,
    embed: QEmbedding,
    ple_embed: QEmbedding,
    ple_model_proj: QLinear,
    ple_proj_norm: Array,
    layers: Vec<Layer>,
    norm: Array,
    /// RoPE frequencies of full-attention layers (proportional, partial).
    full_rope_freqs: Array,
    scalars: Scalars,
}

/// A bf16 scalar, so multiplying a bf16 tensor by it stays in bf16 (like a
/// Python float in mlx). Evaluated up front: built per call, the cast would be
/// one more GPU op every time.
fn scalar(v: f32) -> Result<Array> {
    let a = Array::from_f32(v).as_dtype(Dtype::Bfloat16)?;
    a.eval()?;
    Ok(a)
}

/// The scalar constants of the forward pass, as bf16.
struct Scalars {
    embed: Array,
    ple_embed: Array,
    ple_proj: Array,
    ple_combine: Array,
    softcap: Option<Array>,
    gelu: Gelu,
}

/// Constants of the tanh GELU approximation.
struct Gelu {
    half: Array,
    one: Array,
    three: Array,
    cubic: Array,
    sqrt_2_over_pi: Array,
}

impl Gelu {
    /// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`, op for op like
    /// mlx's `nn.gelu_approx` and staying in bf16 (mlx-rs's
    /// `gelu_approximate` uses f32 constants, which promotes its output — and
    /// with it the whole residual stream — to f32).
    fn apply(&self, x: &Array) -> Result<Array> {
        let inner = x + &multiply(&self.cubic, &power(x, &self.three)?)?;
        let t = tanh(&multiply(&self.sqrt_2_over_pi, &inner)?)?;
        Ok(multiply(&multiply(&self.half, x)?, &(&self.one + &t))?)
    }
}

/// How queries are masked over the keys they attend to.
#[derive(Clone)]
enum Mask {
    /// A single query: every returned key is visible.
    None,
    /// Plain causal (lower-right aligned), which SDPA runs fused.
    Causal,
    /// Causal and limited to a sliding window.
    Window(Array),
}

impl Mask {
    /// The mask for `t` queries ending at position `end` over the `k_len` keys
    /// ending there, on a layer with an optional sliding `window`.
    fn new(t: i32, end: i32, k_len: i32, window: Option<i32>) -> Result<Self> {
        Ok(match window {
            _ if t == 1 => Mask::None,
            // No query is further than k_len - 1 from a key.
            Some(w) if k_len > w => {
                let q = expand_dims(&arange::<_, i32>(end - t, end, None)?, 1)?;
                let k = expand_dims(&arange::<_, i32>(end - k_len, end, None)?, 0)?;
                let in_window = lt(&subtract(&q, &k)?, &Array::from_int(w))?;
                Mask::Window(logical_and(&le(&k, &q)?, &in_window)?)
            }
            _ => Mask::Causal,
        })
    }

    fn sdpa(&self) -> Option<ScaledDotProductAttentionMask<'_>> {
        match self {
            Mask::None => None,
            Mask::Causal => Some(ScaledDotProductAttentionMask::Causal),
            Mask::Window(m) => Some(ScaledDotProductAttentionMask::Array(m)),
        }
    }
}

impl Gemma4 {
    pub fn load(w: &Weights, cfg: Config) -> Result<Self> {
        let n = cfg.n_layers();
        let first_shared = cfg.first_shared();
        // KV-shared layers read the last non-shared layer of their own type.
        let last_of = |full: bool| {
            (0..first_shared)
                .rev()
                .find(|&i| cfg.full_attention[i] == full)
                .ok_or_else(|| anyhow!("no non-shared layer of this attention type"))
        };

        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("{PREFIX}.layers.{i}");
            let a = format!("{p}.self_attn");
            let full = cfg.full_attention[i];
            let shared = i >= first_shared;
            let kv = if shared {
                None
            } else {
                Some(KvProj {
                    k_proj: QLinear::load(w, &format!("{a}.k_proj"))?,
                    v_proj: QLinear::load(w, &format!("{a}.v_proj"))?,
                    k_norm: w.raw(&format!("{a}.k_norm.weight"))?.clone(),
                })
            };
            layers.push(Layer {
                full,
                head_dim: if full { cfg.global_head_dim } else { cfg.head_dim },
                kv_source: if shared { last_of(full)? } else { i },
                input_ln: w.raw(&format!("{p}.input_layernorm.weight"))?.clone(),
                attn: Attention {
                    q_proj: QLinear::load(w, &format!("{a}.q_proj"))?,
                    kv,
                    o_proj: QLinear::load(w, &format!("{a}.o_proj"))?,
                    q_norm: w.raw(&format!("{a}.q_norm.weight"))?.clone(),
                },
                post_attn_ln: w.raw(&format!("{p}.post_attention_layernorm.weight"))?.clone(),
                pre_ff_ln: w.raw(&format!("{p}.pre_feedforward_layernorm.weight"))?.clone(),
                gate: QLinear::load(w, &format!("{p}.mlp.gate_proj"))?,
                up: QLinear::load(w, &format!("{p}.mlp.up_proj"))?,
                down: QLinear::load(w, &format!("{p}.mlp.down_proj"))?,
                post_ff_ln: w.raw(&format!("{p}.post_feedforward_layernorm.weight"))?.clone(),
                ple_gate: QLinear::load(w, &format!("{p}.per_layer_input_gate"))?,
                ple_proj: QLinear::load(w, &format!("{p}.per_layer_projection"))?,
                ple_norm: w.raw(&format!("{p}.post_per_layer_input_norm.weight"))?.clone(),
                scalar: w.raw(&format!("{p}.layer_scalar"))?.clone(),
            });
        }

        // Proportional RoPE: periods base^(2i/head_dim) for the rotated pairs,
        // infinite (no rotation) for the rest.
        let d = cfg.global_head_dim;
        let rotated = (d as f32 * cfg.full_partial_rotary) as i32;
        let periods: Vec<f32> = (0..rotated / 2)
            .map(|i| cfg.full_rope_theta.powf((2 * i) as f32 / d as f32))
            .collect();
        let full_rope_freqs = concatenate(
            &[
                Array::from_slice(&periods, &[rotated / 2]),
                full::<f32>(&[(d - rotated) / 2], &Array::from_f32(f32::INFINITY))?,
            ],
            0,
        )?;

        Ok(Self {
            embed: QEmbedding::load(w, &format!("{PREFIX}.embed_tokens"))?,
            ple_embed: QEmbedding::load(w, &format!("{PREFIX}.embed_tokens_per_layer"))?,
            ple_model_proj: QLinear::load(w, &format!("{PREFIX}.per_layer_model_projection"))?,
            ple_proj_norm: w.raw(&format!("{PREFIX}.per_layer_projection_norm.weight"))?.clone(),
            layers,
            norm: w.raw(&format!("{PREFIX}.norm.weight"))?.clone(),
            full_rope_freqs,
            scalars: Scalars {
                embed: scalar((cfg.hidden_size as f32).sqrt())?,
                ple_embed: scalar((cfg.ple_dim as f32).sqrt())?,
                ple_proj: scalar((cfg.hidden_size as f32).powf(-0.5))?,
                ple_combine: scalar(0.5f32.sqrt())?,
                softcap: cfg.final_logit_softcap.map(scalar).transpose()?,
                gelu: Gelu {
                    half: scalar(0.5)?,
                    one: scalar(1.0)?,
                    three: scalar(3.0)?,
                    cubic: scalar(0.044715)?,
                    sqrt_2_over_pi: scalar((2.0 / std::f32::consts::PI).sqrt())?,
                },
            },
            cfg,
        })
    }

    fn rope(&self, x: &Array, layer: &Layer, offset: i32) -> Result<Array> {
        Ok(if layer.full {
            rope(x, layer.head_dim, false, None, 1.0, offset, Some(&self.full_rope_freqs))?
        } else {
            rope(x, layer.head_dim, false, Some(self.cfg.sliding_rope_theta), 1.0, offset, None)?
        })
    }

    /// Per-layer inputs for token ids [T] and the scaled embeddings h
    /// [1, T, hidden] -> [1, T, n_layers, ple_dim].
    fn per_layer_inputs(&self, ids: &Array, h: &Array) -> Result<Array> {
        let cfg = &self.cfg;
        let t = h.shape()[1];
        let shape = [1, t, cfg.n_layers() as i32, cfg.ple_dim];
        let from_tokens = multiply(
            &reshape(&self.ple_embed.lookup(ids)?, &shape)?,
            &self.scalars.ple_embed,
        )?;
        let projected = multiply(
            &self.ple_model_proj.forward(h)?,
            &self.scalars.ple_proj,
        )?;
        let projected = rms_norm(&reshape(&projected, &shape)?, Some(&self.ple_proj_norm), cfg.rms_eps)?;
        Ok(multiply(&(&projected + &from_tokens), &self.scalars.ple_combine)?)
    }

    /// Runs token ids [T] at positions offset.. and returns the final hidden
    /// states [1, T, hidden]. `caches` holds one entry per non-shared layer.
    fn forward(&self, ids: &Array, caches: &mut [KvCache], offset: i32) -> Result<Array> {
        let cfg = &self.cfg;
        let emb = expand_dims(&self.embed.lookup(ids)?, 0)?;
        let mut h = multiply(&emb, &self.scalars.embed)?;
        let ple = self.per_layer_inputs(ids, &h)?;
        let t = h.shape()[1];

        // K/V (and mask) each layer attended over, for the KV-shared layers.
        let mut attended: Vec<Option<(Array, Array, Mask)>> = vec![None; self.layers.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let x = rms_norm(&h, Some(&layer.input_ln), cfg.rms_eps)?;
            let (b, hd) = (1, layer.head_dim);

            let q = reshape(&layer.attn.q_proj.forward(&x)?, &[b, t, cfg.n_heads, hd])?;
            let q = rms_norm(&q, Some(&layer.attn.q_norm), cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let q = self.rope(&q, layer, offset)?;

            let (k, v, mask) = match &layer.attn.kv {
                Some(kv) => {
                    let k = reshape(&kv.k_proj.forward(&x)?, &[b, t, cfg.n_kv_heads, hd])?;
                    let v = reshape(&kv.v_proj.forward(&x)?, &[b, t, cfg.n_kv_heads, hd])?;
                    let k = rms_norm(&k, Some(&kv.k_norm), cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
                    let k = self.rope(&k, layer, offset)?;
                    let v = rms_norm(&v, None, cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
                    let (k, v) = caches[i].update(&k, &v)?;
                    let window = (!layer.full).then_some(cfg.sliding_window);
                    let mask = Mask::new(t, offset + t, k.shape()[2], window)?;
                    attended[i] = Some((k.clone(), v.clone(), mask.clone()));
                    (k, v, mask)
                }
                None => attended[layer.kv_source]
                    .clone()
                    .ok_or_else(|| anyhow!("KV source layer {} not run", layer.kv_source))?,
            };

            let out = scaled_dot_product_attention(&q, &k, &v, 1.0, mask.sdpa(), None)?;
            let out = reshape(&out.transpose_axes(&[0, 2, 1, 3])?, &[b, t, cfg.n_heads * hd])?;
            let out = rms_norm(&layer.attn.o_proj.forward(&out)?, Some(&layer.post_attn_ln), cfg.rms_eps)?;
            h = &h + &out;

            let x = rms_norm(&h, Some(&layer.pre_ff_ln), cfg.rms_eps)?;
            let mlp = layer.down.forward(&multiply(
                &self.scalars.gelu.apply(&layer.gate.forward(&x)?)?,
                &layer.up.forward(&x)?,
            )?)?;
            h = &h + &rms_norm(&mlp, Some(&layer.post_ff_ln), cfg.rms_eps)?;

            // Per-layer input, gated into the residual stream.
            let g = multiply(
                &self.scalars.gelu.apply(&layer.ple_gate.forward(&h)?)?,
                &ple.index((.., .., i as i32, ..)),
            )?;
            let g = rms_norm(&layer.ple_proj.forward(&g)?, Some(&layer.ple_norm), cfg.rms_eps)?;
            h = multiply(&(&h + &g), &layer.scalar)?;
        }
        Ok(rms_norm(&h, Some(&self.norm), cfg.rms_eps)?)
    }

    /// Soft-capped logits for the last position of h [1, T, hidden].
    fn logits_last(&self, h: &Array) -> Result<Array> {
        let t = h.shape()[1];
        let logits = self.embed.as_linear(&h.index((.., t - 1, ..)))?;
        Ok(match &self.scalars.softcap {
            Some(cap) => multiply(&tanh(&divide(&logits, cap)?)?, cap)?,
            None => logits,
        })
    }

    /// Greedy generation. Returns generated token ids (excluding the prompt),
    /// stopping at a stop token or max_tokens.
    pub fn generate(&self, prompt_ids: &[i32], max_tokens: usize) -> Result<Vec<i32>> {
        let mut caches: Vec<KvCache> = {
            let _gpu = crate::gpu::lock();
            self.layers[..self.cfg.first_shared()]
                .iter()
                .map(|l| KvCache::new((!l.full).then_some(self.cfg.sliding_window)))
                .collect()
        };
        decode::greedy(prompt_ids, max_tokens, &self.cfg.stop_token_ids, |ids, offset| {
            self.logits_last(&self.forward(ids, &mut caches, offset)?)
        })
    }
}
