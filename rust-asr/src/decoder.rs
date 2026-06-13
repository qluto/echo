//! Transformer AED decoder + greedy generation, ported from
//! `mlx_audio.stt.models.cohere_asr` (TransformerDecoder* + _generate_batch_tokens).
//!
//! Single utterance, batch 1, greedy (beam_size 1). Pre-LN layers with a
//! self-attention KV cache and a cross-attention cache computed once from the
//! encoder output.

use anyhow::Result;
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::relu;
use mlx_rs::ops::indexing::{argmax_axis, take_axis, IndexOp};
use mlx_rs::ops::{concatenate_axis, reshape};
use mlx_rs::Array;

use crate::nn::{layer_norm, linear};
use crate::tokenizer::EOS_ID;
use crate::weights::Weights;

const H: i32 = 1024; // hidden_size
const N_HEADS: i32 = 8;
const HD: i32 = H / N_HEADS; // 128
const N_LAYERS: usize = 8;
const LN_EPS: f32 = 1e-5;

struct AttnW {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    o_b: Array,
}

struct DecLayer {
    ln1_w: Array,
    ln1_b: Array,
    self_attn: AttnW,
    ln2_w: Array,
    ln2_b: Array,
    cross_attn: AttnW,
    ln3_w: Array,
    ln3_b: Array,
    ff_in_w: Array,
    ff_in_b: Array,
    ff_out_w: Array,
    ff_out_b: Array,
}

struct Kv {
    k: Array,
    v: Array,
}

#[derive(Default)]
struct LayerCache {
    self_kv: Option<Kv>,
    cross_kv: Option<Kv>,
}

pub struct Decoder {
    tok_emb: Array,  // [16384, 1024]
    pos_enc: Array,  // [1024, 1024]
    emb_ln_w: Array,
    emb_ln_b: Array,
    layers: Vec<DecLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
    head_w: Array, // [16384, 1024]
    head_b: Array,
}

fn attn_w(w: &Weights, prefix: &str) -> Result<AttnW> {
    let g = |s: &str| w.f32(&format!("{prefix}.{s}"));
    Ok(AttnW {
        q_w: g("query_net.weight")?,
        q_b: g("query_net.bias")?,
        k_w: g("key_net.weight")?,
        k_b: g("key_net.bias")?,
        v_w: g("value_net.weight")?,
        v_b: g("value_net.bias")?,
        o_w: g("out_projection.weight")?,
        o_b: g("out_projection.bias")?,
    })
}

impl Decoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let lp = format!("transf_decoder._decoder.layers.{i}");
            let g = |s: &str| w.f32(&format!("{lp}.{s}"));
            layers.push(DecLayer {
                ln1_w: g("layer_norm_1.weight")?,
                ln1_b: g("layer_norm_1.bias")?,
                self_attn: attn_w(w, &format!("{lp}.first_sub_layer"))?,
                ln2_w: g("layer_norm_2.weight")?,
                ln2_b: g("layer_norm_2.bias")?,
                cross_attn: attn_w(w, &format!("{lp}.second_sub_layer"))?,
                ln3_w: g("layer_norm_3.weight")?,
                ln3_b: g("layer_norm_3.bias")?,
                ff_in_w: g("third_sub_layer.dense_in.weight")?,
                ff_in_b: g("third_sub_layer.dense_in.bias")?,
                ff_out_w: g("third_sub_layer.dense_out.weight")?,
                ff_out_b: g("third_sub_layer.dense_out.bias")?,
            });
        }
        Ok(Self {
            tok_emb: w.f32("transf_decoder._embedding.token_embedding.weight")?,
            pos_enc: w.f32("transf_decoder._embedding.position_embedding.pos_enc")?,
            emb_ln_w: w.f32("transf_decoder._embedding.layer_norm.weight")?,
            emb_ln_b: w.f32("transf_decoder._embedding.layer_norm.bias")?,
            layers,
            final_ln_w: w.f32("transf_decoder._decoder.final_layer_norm.weight")?,
            final_ln_b: w.f32("transf_decoder._decoder.final_layer_norm.bias")?,
            head_w: w.f32("log_softmax.mlp.layer0.weight")?,
            head_b: w.f32("log_softmax.mlp.layer0.bias")?,
        })
    }

    /// Greedy decode given encoder hidden states [1, T, 1024]. Returns token ids.
    pub fn generate(&self, enc: &Array, prompt: &[i32], max_tokens: usize) -> Result<Vec<u32>> {
        let mut caches: Vec<LayerCache> = (0..N_LAYERS).map(|_| LayerCache::default()).collect();
        let mut generated: Vec<u32> = Vec::new();

        // First pass: full prompt.
        let prompt_arr = Array::from_slice(prompt, &[1, prompt.len() as i32]);
        let hidden = self.forward(&prompt_arr, enc, &mut caches, 0)?;
        let mut next = self.argmax_last(&hidden)?;
        if next != EOS_ID {
            generated.push(next as u32);
        }

        let mut pos = prompt.len() as i32;
        for _ in 0..max_tokens.saturating_sub(1) {
            if next == EOS_ID {
                break;
            }
            let ids = Array::from_slice(&[next], &[1, 1]);
            let hidden = self.forward(&ids, enc, &mut caches, pos)?;
            next = self.argmax_last(&hidden)?;
            if next != EOS_ID {
                generated.push(next as u32);
            }
            pos += 1;
        }
        Ok(generated)
    }

    fn argmax_last(&self, hidden: &Array) -> Result<i32> {
        // hidden [1, S, 1024] -> last token logits -> argmax
        let s = hidden.shape()[1];
        let last = hidden.index((.., s - 1, ..)); // [1, 1024]
        let logits = linear(&last, &self.head_w, Some(&self.head_b))?; // [1, 16384]
        let am = argmax_axis(&logits, 1, false)?; // [1]
        am.eval()?;
        Ok(am.as_slice::<u32>()[0] as i32)
    }

    fn forward(
        &self,
        ids: &Array,
        enc: &Array,
        caches: &mut [LayerCache],
        start_pos: i32,
    ) -> Result<Array> {
        let s = ids.shape()[1];

        // --- embedding ---
        let tok = take_axis(&self.tok_emb, ids, 0)?; // [1, S, 1024]
        let positions = Array::from_slice(
            &(start_pos..start_pos + s).collect::<Vec<i32>>(),
            &[s],
        );
        let pos = take_axis(&self.pos_enc, &positions, 0)?; // [S, 1024]
        let mut h = layer_norm(&(&tok + &pos), &self.emb_ln_w, &self.emb_ln_b, LN_EPS)?;

        // causal self-attention mask (only needed when S > 1, i.e. the prompt pass)
        let self_mask = if s > 1 {
            Some(causal_mask(s)?)
        } else {
            None
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            // self-attention
            let residual = h.clone();
            let hn = layer_norm(&h, &layer.ln1_w, &layer.ln1_b, LN_EPS)?;
            let self_out = self_attention(&layer.self_attn, &hn, self_mask.as_ref(), &mut cache.self_kv)?;
            h = &residual + &self_out;

            // cross-attention
            let residual = h.clone();
            let hn = layer_norm(&h, &layer.ln2_w, &layer.ln2_b, LN_EPS)?;
            let cross_out = cross_attention(&layer.cross_attn, &hn, enc, &mut cache.cross_kv)?;
            h = &residual + &cross_out;

            // feed-forward (relu)
            let residual = h.clone();
            let hn = layer_norm(&h, &layer.ln3_w, &layer.ln3_b, LN_EPS)?;
            let ff = linear(&hn, &layer.ff_in_w, Some(&layer.ff_in_b))?;
            let ff = relu(&ff)?;
            let ff = linear(&ff, &layer.ff_out_w, Some(&layer.ff_out_b))?;
            h = &residual + &ff;
        }

        layer_norm(&h, &self.final_ln_w, &self.final_ln_b, LN_EPS)
    }
}

fn split_heads(x: &Array) -> Result<Array> {
    // [1, S, 1024] -> [1, 8, S, 128]
    let sh = x.shape().to_vec();
    let r = reshape(x, &[sh[0], sh[1], N_HEADS, HD])?;
    Ok(r.transpose_axes(&[0, 2, 1, 3])?)
}

fn merge_heads(x: &Array, s: i32) -> Result<Array> {
    // [1, 8, S, 128] -> [1, S, 1024]
    let m = x.transpose_axes(&[0, 2, 1, 3])?;
    Ok(reshape(&m, &[1, s, H])?)
}

fn self_attention(
    a: &AttnW,
    hn: &Array,
    mask: Option<&Array>,
    cache: &mut Option<Kv>,
) -> Result<Array> {
    let s = hn.shape()[1];
    let scale = (HD as f32).powf(-0.5);
    let q = split_heads(&linear(hn, &a.q_w, Some(&a.q_b))?)?;
    let mut k = split_heads(&linear(hn, &a.k_w, Some(&a.k_b))?)?;
    let mut v = split_heads(&linear(hn, &a.v_w, Some(&a.v_b))?)?;
    if let Some(prev) = cache.as_ref() {
        k = concatenate_axis(&[prev.k.clone(), k], 2)?;
        v = concatenate_axis(&[prev.v.clone(), v], 2)?;
    }
    *cache = Some(Kv {
        k: k.clone(),
        v: v.clone(),
    });
    let mask = mask.map(ScaledDotProductAttentionMask::Array);
    let out = scaled_dot_product_attention(&q, &k, &v, scale, mask)?;
    linear(&merge_heads(&out, s)?, &a.o_w, Some(&a.o_b))
}

fn cross_attention(a: &AttnW, hn: &Array, enc: &Array, cache: &mut Option<Kv>) -> Result<Array> {
    let s = hn.shape()[1];
    let scale = (HD as f32).powf(-0.5);
    let q = split_heads(&linear(hn, &a.q_w, Some(&a.q_b))?)?;
    let (k, v) = match cache.as_ref() {
        Some(kv) => (kv.k.clone(), kv.v.clone()),
        None => {
            let k = split_heads(&linear(enc, &a.k_w, Some(&a.k_b))?)?;
            let v = split_heads(&linear(enc, &a.v_w, Some(&a.v_b))?)?;
            *cache = Some(Kv {
                k: k.clone(),
                v: v.clone(),
            });
            (k, v)
        }
    };
    let no_mask: Option<ScaledDotProductAttentionMask> = None;
    let out = scaled_dot_product_attention(&q, &k, &v, scale, no_mask)?;
    linear(&merge_heads(&out, s)?, &a.o_w, Some(&a.o_b))
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
    Ok(Array::from_slice(&data, &[1, 1, s, s]))
}
