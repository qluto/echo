//! Qwen3 causal LLM (4-bit quantized) on Apple MLX, for ASR post-processing.
//!
//! Standard Qwen3 dense decoder: quantized embedding (tied lm_head), 28 layers
//! of RMSNorm + GQA attention (per-head q/k RMSNorm, RoPE, KV cache) + SwiGLU
//! MLP, all linears 4-bit (group 64) via `quantized_matmul`. Greedy decoding.

use anyhow::Result;
use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::silu;
use mlx_rs::ops::indexing::{argmax_axis, take_axis, IndexOp};
use mlx_rs::ops::{
    broadcast_to, concatenate_axis, dequantize, expand_dims, quantized_matmul, reshape,
};
use mlx_rs::{Array, Dtype};

use crate::weights::Weights;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

#[derive(Clone)]
pub struct Config {
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub eos_token_id: i32,
}

/// A 4-bit quantized linear: y = x @ dequant(weight)^T.
struct QLinear {
    weight: Array,
    scales: Array,
    biases: Array,
}

impl QLinear {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        // Keep native BF16 scales/biases so inference matches mlx-lm bit-for-bit.
        Ok(Self {
            weight: w.raw(&format!("{prefix}.weight"))?.clone(),
            scales: w.raw(&format!("{prefix}.scales"))?.clone(),
            biases: w.raw(&format!("{prefix}.biases"))?.clone(),
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
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

struct Layer {
    input_ln: Array,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Array,
    k_norm: Array,
    post_ln: Array,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

pub struct Qwen3 {
    cfg: Config,
    embed_w: Array, // quantized [vocab, hidden/8]
    embed_scales: Array,
    embed_biases: Array,
    layers: Vec<Layer>,
    norm: Array,
}

/// Per-layer KV cache.
struct Kv {
    k: Array,
    v: Array,
}

fn rmsnorm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    Ok(rms_norm(x, weight, eps)?)
}

impl Qwen3 {
    pub fn load(w: &Weights, cfg: Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_ln: w.raw(&format!("{p}.input_layernorm.weight"))?.clone(),
                q_proj: QLinear::load(w, &format!("{p}.self_attn.q_proj"))?,
                k_proj: QLinear::load(w, &format!("{p}.self_attn.k_proj"))?,
                v_proj: QLinear::load(w, &format!("{p}.self_attn.v_proj"))?,
                o_proj: QLinear::load(w, &format!("{p}.self_attn.o_proj"))?,
                q_norm: w.raw(&format!("{p}.self_attn.q_norm.weight"))?.clone(),
                k_norm: w.raw(&format!("{p}.self_attn.k_norm.weight"))?.clone(),
                post_ln: w.raw(&format!("{p}.post_attention_layernorm.weight"))?.clone(),
                gate: QLinear::load(w, &format!("{p}.mlp.gate_proj"))?,
                up: QLinear::load(w, &format!("{p}.mlp.up_proj"))?,
                down: QLinear::load(w, &format!("{p}.mlp.down_proj"))?,
            });
        }
        Ok(Self {
            cfg,
            embed_w: w.raw("model.embed_tokens.weight")?.clone(),
            embed_scales: w.raw("model.embed_tokens.scales")?.clone(),
            embed_biases: w.raw("model.embed_tokens.biases")?.clone(),
            layers,
            norm: w.raw("model.norm.weight")?.clone(),
        })
    }

    /// Embedding lookup for token ids [T] -> [1, T, hidden] (gather + dequantize).
    fn embed(&self, ids: &Array) -> Result<Array> {
        let qw = take_axis(&self.embed_w, ids, 0)?;
        let qs = take_axis(&self.embed_scales, ids, 0)?;
        let qb = take_axis(&self.embed_biases, ids, 0)?;
        let e = dequantize(&qw, &qs, &qb, GROUP_SIZE, BITS)?; // [T, hidden]
        Ok(expand_dims(&e, 0)?)
    }

    /// Tied lm_head: logits for the last position only. h: [1, T, hidden].
    fn lm_head_last(&self, h: &Array) -> Result<Array> {
        let t = h.shape()[1];
        let last = h.index((.., t - 1, ..)); // [1, hidden]
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

    fn forward(&self, ids: &Array, caches: &mut [Option<Kv>], offset: i32) -> Result<Array> {
        let mut h = self.embed(ids)?;
        let t = h.shape()[1];
        let mask = if t > 1 { Some(causal_mask(t)?) } else { None };
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = self.layer_forward(layer, &h, cache, offset, mask.as_ref())?;
        }
        rmsnorm(&h, &self.norm, self.cfg.rms_eps)
    }

    fn layer_forward(
        &self,
        l: &Layer,
        h: &Array,
        cache: &mut Option<Kv>,
        offset: i32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, t) = (h.shape()[0], h.shape()[1]);
        let residual = h.clone();
        let hn = rms_norm(h, &l.input_ln, cfg.rms_eps)?;

        // projections
        let q = reshape(&l.q_proj.forward(&hn)?, &[b, t, cfg.n_heads, cfg.head_dim])?;
        let k = reshape(&l.k_proj.forward(&hn)?, &[b, t, cfg.n_kv_heads, cfg.head_dim])?;
        let v = reshape(&l.v_proj.forward(&hn)?, &[b, t, cfg.n_kv_heads, cfg.head_dim])?;

        // per-head RMSNorm on q,k (over head_dim), then to [b, heads, t, hd]
        let q = rmsnorm(&q, &l.q_norm, cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = rmsnorm(&k, &l.k_norm, cfg.rms_eps)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // RoPE
        let q = rope(&q, cfg.head_dim, false, Some(cfg.rope_theta), 1.0, offset, None)?;
        let mut k = rope(&k, cfg.head_dim, false, Some(cfg.rope_theta), 1.0, offset, None)?;
        let mut v = v;

        // KV cache
        if let Some(prev) = cache.as_ref() {
            k = concatenate_axis(&[prev.k.clone(), k], 2)?;
            v = concatenate_axis(&[prev.v.clone(), v], 2)?;
        }
        *cache = Some(Kv {
            k: k.clone(),
            v: v.clone(),
        });

        // GQA: repeat kv heads to match q heads
        let rep = cfg.n_heads / cfg.n_kv_heads;
        let k = repeat_kv(&k, rep)?;
        let v = repeat_kv(&v, rep)?;

        let scale = (cfg.head_dim as f32).powf(-0.5);
        let mask = mask.map(ScaledDotProductAttentionMask::Array);
        let attn = scaled_dot_product_attention(&q, &k, &v, scale, mask)?;
        let attn = reshape(
            &attn.transpose_axes(&[0, 2, 1, 3])?,
            &[b, t, cfg.n_heads * cfg.head_dim],
        )?;
        let h = &residual + &l.o_proj.forward(&attn)?;

        // MLP (SwiGLU)
        let residual = h.clone();
        let hn = rmsnorm(&h, &l.post_ln, cfg.rms_eps)?;
        let mlp = l.down.forward(&(&silu(&l.gate.forward(&hn)?)? * &l.up.forward(&hn)?))?;
        Ok(&residual + &mlp)
    }

    /// Greedy generation. prompt_ids: token ids. Returns generated token ids
    /// (excluding the prompt), stopping at EOS or max_tokens.
    pub fn generate(&self, prompt_ids: &[i32], max_tokens: usize) -> Result<Vec<i32>> {
        let mut caches: Vec<Option<Kv>> = (0..self.cfg.n_layers).map(|_| None).collect();
        let prompt = Array::from_slice(prompt_ids, &[prompt_ids.len() as i32]);
        let h = self.forward(&prompt, &mut caches, 0)?;
        let mut next = argmax_axis(&self.lm_head_last(&h)?, 1, false)?.item::<u32>() as i32;

        let mut out = Vec::new();
        let mut pos = prompt_ids.len() as i32;
        for _ in 0..max_tokens {
            if next == self.cfg.eos_token_id {
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

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}

fn repeat_kv(x: &Array, rep: i32) -> Result<Array> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape().to_vec(); // [b, n_kv, t, hd]
    let (b, n_kv, t, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = reshape(x, &[b, n_kv, 1, t, hd])?;
    let x = broadcast_to(&x, &[b, n_kv, rep, t, hd])?;
    Ok(reshape(&x, &[b, n_kv * rep, t, hd])?)
}

fn causal_mask(s: i32) -> Result<Array> {
    let n = s as usize;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if j > i {
                data[i * n + j] = -1e9;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, s, s]).as_dtype(Dtype::Bfloat16)?)
}
