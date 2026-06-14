//! FastConformer encoder ported from
//! `mlx_audio.stt.models.cohere_asr.cohere_asr` (ConformerEncoder).
//!
//! Single-utterance inference only: batch == 1 and the whole sequence is
//! valid, so all padding masks are identity and are omitted. That keeps the
//! port to the math that actually matters.

use anyhow::Result;
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::{glu, relu, silu};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{conv1d, conv2d, cos, expand_dims, pad, reshape, sin, stack_axis};
use mlx_rs::Array;

use crate::nn::{batch_norm_eval, layer_norm, linear};
use crate::weights::Weights;

const D_MODEL: i32 = 1280;
const N_HEADS: i32 = 8;
const HEAD_DIM: i32 = D_MODEL / N_HEADS; // 160
const N_LAYERS: usize = 48;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

struct Attn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    pos_w: Array, // linear_pos, no bias
    out_w: Array,
    out_b: Array,
    bias_u: Array, // [8,160]
    bias_v: Array,
}

struct ConvMod {
    pw1_w: Array,
    pw1_b: Array,
    dw_w: Array,
    dw_b: Array,
    bn_w: Array,
    bn_b: Array,
    bn_rm: Array,
    bn_rv: Array,
    pw2_w: Array,
    pw2_b: Array,
}

struct Ff {
    l1_w: Array,
    l1_b: Array,
    l2_w: Array,
    l2_b: Array,
}

struct Layer {
    n_ff1_w: Array,
    n_ff1_b: Array,
    ff1: Ff,
    n_att_w: Array,
    n_att_b: Array,
    attn: Attn,
    n_conv_w: Array,
    n_conv_b: Array,
    conv: ConvMod,
    n_ff2_w: Array,
    n_ff2_b: Array,
    ff2: Ff,
    n_out_w: Array,
    n_out_b: Array,
}

pub struct Encoder {
    // ConvSubsampling
    sub_w: Vec<(Array, Array)>, // (weight already in MLX layout, bias) for conv idx 0,2,3,5,6
    sub_groups: Vec<i32>,
    sub_stride: Vec<i32>,
    sub_pad: Vec<i32>,
    out_w: Array,
    out_b: Array,
    layers: Vec<Layer>,
    proj_w: Array,
    proj_b: Array,
}

fn conv2d_w(w: &Weights, key: &str) -> Result<Array> {
    // checkpoint NCHW [O, I/g, H, W] -> MLX [O, H, W, I/g]
    Ok(w.f32(key)?.transpose_axes(&[0, 2, 3, 1])?)
}
fn conv1d_w(w: &Weights, key: &str) -> Result<Array> {
    // checkpoint [O, I/g, k] -> MLX [O, k, I/g]
    Ok(w.f32(key)?.transpose_axes(&[0, 2, 1])?)
}

impl Encoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let p = "encoder.pre_encode.conv";
        // conv indices that carry weights: 0,2,3,5,6 (1,4,7 are ReLU)
        let mut sub_w = Vec::new();
        for idx in [0usize, 2, 3, 5, 6] {
            let wt = conv2d_w(w, &format!("{p}.{idx}.weight"))?;
            let b = w.f32(&format!("{p}.{idx}.bias"))?;
            sub_w.push((wt, b));
        }
        // groups/stride/pad per the five conv layers (0,2,3,5,6)
        let sub_groups = vec![1, 256, 1, 256, 1];
        let sub_stride = vec![2, 2, 1, 2, 1];
        let sub_pad = vec![1, 1, 0, 1, 0];

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let lp = format!("encoder.layers.{i}");
            let g = |s: &str| w.f32(&format!("{lp}.{s}"));
            let layer = Layer {
                n_ff1_w: g("norm_feed_forward1.weight")?,
                n_ff1_b: g("norm_feed_forward1.bias")?,
                ff1: Ff {
                    l1_w: g("feed_forward1.linear1.weight")?,
                    l1_b: g("feed_forward1.linear1.bias")?,
                    l2_w: g("feed_forward1.linear2.weight")?,
                    l2_b: g("feed_forward1.linear2.bias")?,
                },
                n_att_w: g("norm_self_att.weight")?,
                n_att_b: g("norm_self_att.bias")?,
                attn: Attn {
                    q_w: g("self_attn.linear_q.weight")?,
                    q_b: g("self_attn.linear_q.bias")?,
                    k_w: g("self_attn.linear_k.weight")?,
                    k_b: g("self_attn.linear_k.bias")?,
                    v_w: g("self_attn.linear_v.weight")?,
                    v_b: g("self_attn.linear_v.bias")?,
                    pos_w: g("self_attn.linear_pos.weight")?,
                    out_w: g("self_attn.linear_out.weight")?,
                    out_b: g("self_attn.linear_out.bias")?,
                    bias_u: g("self_attn.pos_bias_u")?,
                    bias_v: g("self_attn.pos_bias_v")?,
                },
                n_conv_w: g("norm_conv.weight")?,
                n_conv_b: g("norm_conv.bias")?,
                conv: ConvMod {
                    pw1_w: conv1d_w(w, &format!("{lp}.conv.pointwise_conv1.weight"))?,
                    pw1_b: g("conv.pointwise_conv1.bias")?,
                    dw_w: conv1d_w(w, &format!("{lp}.conv.depthwise_conv.weight"))?,
                    dw_b: g("conv.depthwise_conv.bias")?,
                    bn_w: g("conv.batch_norm.weight")?,
                    bn_b: g("conv.batch_norm.bias")?,
                    bn_rm: g("conv.batch_norm.running_mean")?,
                    bn_rv: g("conv.batch_norm.running_var")?,
                    pw2_w: conv1d_w(w, &format!("{lp}.conv.pointwise_conv2.weight"))?,
                    pw2_b: g("conv.pointwise_conv2.bias")?,
                },
                n_ff2_w: g("norm_feed_forward2.weight")?,
                n_ff2_b: g("norm_feed_forward2.bias")?,
                ff2: Ff {
                    l1_w: g("feed_forward2.linear1.weight")?,
                    l1_b: g("feed_forward2.linear1.bias")?,
                    l2_w: g("feed_forward2.linear2.weight")?,
                    l2_b: g("feed_forward2.linear2.bias")?,
                },
                n_out_w: g("norm_out.weight")?,
                n_out_b: g("norm_out.bias")?,
            };
            layers.push(layer);
        }

        Ok(Self {
            sub_w,
            sub_groups,
            sub_stride,
            sub_pad,
            out_w: w.f32("encoder.pre_encode.out.weight")?,
            out_b: w.f32("encoder.pre_encode.out.bias")?,
            layers,
            proj_w: w.f32("encoder_decoder_proj.weight")?,
            proj_b: w.f32("encoder_decoder_proj.bias")?,
        })
    }

    /// mel [1, T, 128] -> encoder hidden states [1, T', 1024]
    pub fn forward(&self, mel: &Array, seq_len: i32) -> Result<Array> {
        let mut x = self.subsample(mel, seq_len)?; // [1, T', 1280]
        let pos_emb = rel_pos_encoding(x.shape()[1], D_MODEL)?; // [1, 2T'-1, 1280]
        for layer in &self.layers {
            x = conformer_layer(layer, &x, &pos_emb)?;
        }
        // encoder_decoder_proj 1280 -> 1024
        linear(&x, &self.proj_w, Some(&self.proj_b))
    }

    /// Debug: return (subsample_out, pos_emb, layer0_out) for stage-by-stage diffing.
    pub fn debug_stages(&self, mel: &Array, seq_len: i32) -> Result<(Array, Array, Array)> {
        let sub = self.subsample(mel, seq_len)?;
        let pos_emb = rel_pos_encoding(sub.shape()[1], D_MODEL)?;
        let l0 = conformer_layer(&self.layers[0], &sub, &pos_emb)?;
        Ok((sub, pos_emb, l0))
    }

    fn subsample(&self, mel: &Array, seq_len: i32) -> Result<Array> {
        // ConvSubsampling masks invalid time frames before every conv/relu slot
        // (and once more at the end), updating the valid length after each
        // stride-2 conv. Omitting this leaks padding into boundary frames.
        // [1,T,128] -> [1,1,T,128] -> [1,T,128,1]
        let mut x = expand_dims(mel, 1)?.transpose_axes(&[0, 2, 3, 1])?;
        let mut len = seq_len;

        // (weight_index_into_sub_w | None for ReLU). Slots: conv0, relu, conv2,
        // conv3, relu, conv5, conv6, relu. Stride convs (0,2,5) update length.
        let slots: [(Option<usize>, bool); 8] = [
            (Some(0), true),  // conv0, stride
            (None, false),    // relu
            (Some(1), true),  // conv2 (depthwise), stride
            (Some(2), false), // conv3 (pointwise)
            (None, false),    // relu
            (Some(3), true),  // conv5 (depthwise), stride
            (Some(4), false), // conv6 (pointwise)
            (None, false),    // relu
        ];

        for (wi, is_stride) in slots {
            x = mask_time(&x, len)?;
            match wi {
                Some(i) => {
                    let (wt, b) = &self.sub_w[i];
                    let s = self.sub_stride[i];
                    let pd = self.sub_pad[i];
                    let g = self.sub_groups[i];
                    x = conv2d(&x, wt, Some((s, s)), Some((pd, pd)), Some((1, 1)), Some(g))?;
                    x = &x + b;
                }
                None => x = relu(&x)?,
            }
            if is_stride {
                len = ((len - 1) / 2) + 1;
            }
        }
        x = mask_time(&x, len)?;

        // x [1, T', 16, 256] -> [1, T', 256, 16] -> [1, T', 4096]
        let sh = x.shape().to_vec();
        x = x.transpose_axes(&[0, 1, 3, 2])?;
        x = reshape(&x, &[sh[0], sh[1], sh[2] * sh[3]])?;
        linear(&x, &self.out_w, Some(&self.out_b))
    }
}

/// Zero out time frames at index >= len. x is [1, T, W, C]; mask broadcasts.
fn mask_time(x: &Array, len: i32) -> Result<Array> {
    let t = x.shape()[1];
    let m: Vec<f32> = (0..t).map(|i| if i < len { 1.0 } else { 0.0 }).collect();
    let mask = Array::from_slice(&m, &[1, t, 1, 1]);
    Ok(x * &mask)
}

/// Relative positional encoding buffer slice: [1, 2T-1, d_model].
fn rel_pos_encoding(t: i32, d: i32) -> Result<Array> {
    let len = (2 * t - 1) as usize;
    let half = (d / 2) as usize;
    // positions: T-1, T-2, ..., 0, ..., -(T-1)
    let pos: Vec<f32> = (0..len).map(|i| (t - 1) as f32 - i as f32).collect();
    let pos_col = Array::from_slice(&pos, &[len as i32, 1]);
    // div_term[j] = exp(2j * -(ln 10000)/d)
    let ln10000 = (10000.0f32).ln();
    let div: Vec<f32> = (0..half)
        .map(|j| ((2 * j) as f32 * -(ln10000 / d as f32)).exp())
        .collect();
    let div_row = Array::from_slice(&div, &[1, half as i32]);
    let angles = &pos_col * &div_row; // [len, half]
    let s = sin(&angles)?;
    let c = cos(&angles)?;
    // interleave: stack on new last axis -> [len, half, 2] -> [len, d]
    let stacked = stack_axis(&[s, c], -1)?;
    let pe = reshape(&stacked, &[len as i32, d])?;
    Ok(expand_dims(&pe, 0)?)
}

fn rel_shift(x: &Array) -> Result<Array> {
    // x [B,H,q,p] -> pad last dim left by 1 -> reshape -> drop first -> reshape back
    let sh = x.shape().to_vec();
    let (b, h, q, p) = (sh[0], sh[1], sh[2], sh[3]);
    let x = pad(x, &[(0, 0), (0, 0), (0, 0), (1, 0)][..], Array::from_f32(0.0), None)?;
    let x = reshape(&x, &[b, h, p + 1, q])?;
    let x = x.index((.., .., 1.., ..)); // drop first along axis 2
    Ok(reshape(&x, &[b, h, q, p])?)
}

fn self_attention(a: &Attn, x: &Array, pos_emb: &Array) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, q_len) = (sh[0], sh[1]);
    let scale = (HEAD_DIM as f32).powf(-0.5);

    let p = linear(pos_emb, &a.pos_w, None)?; // [1, 2T-1, 1280]
    let pos_len = p.shape()[1];

    let q = reshape(&linear(x, &a.q_w, Some(&a.q_b))?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let k = reshape(&linear(x, &a.k_w, Some(&a.k_b))?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let v = reshape(&linear(x, &a.v_w, Some(&a.v_b))?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let p = reshape(&p, &[1, pos_len, N_HEADS, HEAD_DIM])?;

    // q + bias_u/v : bias [8,160] broadcasts over [b,q,8,160]
    let q_u = (&q + &a.bias_u).transpose_axes(&[0, 2, 1, 3])?; // [b,8,q,160]
    let q_v = (&q + &a.bias_v).transpose_axes(&[0, 2, 1, 3])?;
    let k = k.transpose_axes(&[0, 2, 1, 3])?; // [b,8,q,160]
    let v = v.transpose_axes(&[0, 2, 1, 3])?;
    let p = p.transpose_axes(&[0, 2, 1, 3])?; // [1,8,2T-1,160]

    // matrix_bd = rel_shift(q_v @ p^T) then crop to k length, scaled
    let bd = q_v.matmul(&p.transpose_axes(&[0, 1, 3, 2])?)?; // [b,8,q,2T-1]
    let bd = rel_shift(&bd)?;
    let bd = bd.index((.., .., .., 0..k.shape()[2])); // [b,8,q,q]
    let bd = &bd * Array::from_f32(scale);

    let out = scaled_dot_product_attention(
        &q_u,
        &k,
        &v,
        scale,
        Some(ScaledDotProductAttentionMask::Array(&bd)),
    )?;
    let out = reshape(&out.transpose_axes(&[0, 2, 1, 3])?, &[b, q_len, D_MODEL])?;
    linear(&out, &a.out_w, Some(&a.out_b))
}

fn conv_module(c: &ConvMod, x: &Array) -> Result<Array> {
    // x [1,T,1280] (NLC)
    let x = conv1d(x, &c.pw1_w, Some(1), Some(0), Some(1), Some(1))?; // [1,T,2560]
    let x = &x + &c.pw1_b;
    let x = glu(&x, 2)?; // -> [1,T,1280]
    let x = conv1d(&x, &c.dw_w, Some(1), Some(4), Some(1), Some(D_MODEL))?; // depthwise k9 p4
    let x = &x + &c.dw_b;
    let x = batch_norm_eval(&x, &c.bn_w, &c.bn_b, &c.bn_rm, &c.bn_rv, BN_EPS)?;
    let x = silu(&x)?;
    let x = conv1d(&x, &c.pw2_w, Some(1), Some(0), Some(1), Some(1))?;
    Ok(&x + &c.pw2_b)
}

fn feed_forward(f: &Ff, x: &Array) -> Result<Array> {
    let h = linear(x, &f.l1_w, Some(&f.l1_b))?;
    let h = silu(&h)?;
    linear(&h, &f.l2_w, Some(&f.l2_b))
}

fn conformer_layer(l: &Layer, x: &Array, pos_emb: &Array) -> Result<Array> {
    let h = layer_norm(x, &l.n_ff1_w, &l.n_ff1_b, LN_EPS)?;
    let x = x + &(&feed_forward(&l.ff1, &h)? * Array::from_f32(0.5));

    let h = layer_norm(&x, &l.n_att_w, &l.n_att_b, LN_EPS)?;
    let x = &x + &self_attention(&l.attn, &h, pos_emb)?;

    let h = layer_norm(&x, &l.n_conv_w, &l.n_conv_b, LN_EPS)?;
    let x = &x + &conv_module(&l.conv, &h)?;

    let h = layer_norm(&x, &l.n_ff2_w, &l.n_ff2_b, LN_EPS)?;
    let x = &x + &(&feed_forward(&l.ff2, &h)? * Array::from_f32(0.5));

    layer_norm(&x, &l.n_out_w, &l.n_out_b, LN_EPS)
}
