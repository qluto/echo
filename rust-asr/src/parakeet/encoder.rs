//! Parakeet FastConformer encoder (NeMo), ported from
//! `mlx_audio.stt.models.parakeet.conformer`. Structurally identical to the
//! Cohere encoder, with three differences: conv weights are already in MLX
//! layout (no transpose), there is no inter-conv length masking (single
//! utterance), and rel-pos encoding scales the input by sqrt(d_model)
//! (xscaling=True). d_model 1024, 24 layers, 8 heads, 80-mel input.

use anyhow::Result;
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::{glu, relu, silu};
use mlx_rs::ops::{conv1d, conv2d, cos, expand_dims, pad, reshape, sin, stack_axis};
use mlx_rs::Array;

use crate::nn::{batch_norm_eval, layer_norm, linear};
use crate::weights::Weights;

const D_MODEL: i32 = 1024;
const N_HEADS: i32 = 8;
const HEAD_DIM: i32 = D_MODEL / N_HEADS; // 128
const FF_DIM: i32 = D_MODEL * 4; // ff_expansion_factor 4
const N_LAYERS: usize = 24;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;
const XSCALE: f32 = 32.0; // sqrt(1024)

struct Attn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    pos_w: Array,
    out_w: Array,
    out_b: Array,
    bias_u: Array,
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
    sub_w: Vec<(Array, Array)>,
    sub_groups: Vec<i32>,
    sub_stride: Vec<i32>,
    sub_pad: Vec<i32>,
    out_w: Array,
    out_b: Array,
    layers: Vec<Layer>,
}

impl Encoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let p = "encoder.pre_encode.conv";
        let mut sub_w = Vec::new();
        for idx in [0usize, 2, 3, 5, 6] {
            // conv weights are already in MLX layout in this checkpoint.
            let wt = w.f32(&format!("{p}.{idx}.weight"))?;
            let b = w.f32(&format!("{p}.{idx}.bias"))?;
            sub_w.push((wt, b));
        }
        let sub_groups = vec![1, 256, 1, 256, 1];
        let sub_stride = vec![2, 2, 1, 2, 1];
        let sub_pad = vec![1, 1, 0, 1, 0];

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let lp = format!("encoder.layers.{i}");
            let g = |s: &str| w.f32(&format!("{lp}.{s}"));
            layers.push(Layer {
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
                    pw1_w: g("conv.pointwise_conv1.weight")?,
                    pw1_b: g("conv.pointwise_conv1.bias")?,
                    dw_w: g("conv.depthwise_conv.weight")?,
                    dw_b: g("conv.depthwise_conv.bias")?,
                    bn_w: g("conv.batch_norm.weight")?,
                    bn_b: g("conv.batch_norm.bias")?,
                    bn_rm: g("conv.batch_norm.running_mean")?,
                    bn_rv: g("conv.batch_norm.running_var")?,
                    pw2_w: g("conv.pointwise_conv2.weight")?,
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
            });
        }

        Ok(Self {
            sub_w,
            sub_groups,
            sub_stride,
            sub_pad,
            out_w: w.f32("encoder.pre_encode.out.weight")?,
            out_b: w.f32("encoder.pre_encode.out.bias")?,
            layers,
        })
    }

    /// mel [1, T, 80] -> encoder hidden states [1, T', 1024]
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        let mut x = self.subsample(mel)?;
        let pos_emb = rel_pos_encoding(x.shape()[1], D_MODEL)?;
        // xscaling: the input (not pos_emb) is scaled by sqrt(d_model)
        x = &x * Array::from_f32(XSCALE);
        for layer in &self.layers {
            x = conformer_layer(layer, &x, &pos_emb)?;
        }
        Ok(x)
    }

    fn subsample(&self, mel: &Array) -> Result<Array> {
        // [1,T,80] -> [1,1,T,80] -> [1,T,80,1]
        let mut x = expand_dims(mel, 1)?.transpose_axes(&[0, 2, 3, 1])?;
        let relu_after = [true, false, true, false, true];
        for (i, (wt, b)) in self.sub_w.iter().enumerate() {
            let s = self.sub_stride[i];
            let pd = self.sub_pad[i];
            let g = self.sub_groups[i];
            x = conv2d(&x, wt, Some((s, s)), Some((pd, pd)), Some((1, 1)), Some(g))?;
            x = &x + b;
            if relu_after[i] {
                x = relu(&x)?;
            }
        }
        // [1, T', 10, 256] -> [1, T', 256, 10] -> [1, T', 2560]
        let sh = x.shape().to_vec();
        x = x.transpose_axes(&[0, 1, 3, 2])?;
        x = reshape(&x, &[sh[0], sh[1], sh[2] * sh[3]])?;
        linear(&x, &self.out_w, Some(&self.out_b))
    }
}

fn rel_pos_encoding(t: i32, d: i32) -> Result<Array> {
    let len = (2 * t - 1) as usize;
    let half = (d / 2) as usize;
    let pos: Vec<f32> = (0..len).map(|i| (t - 1) as f32 - i as f32).collect();
    let pos_col = Array::from_slice(&pos, &[len as i32, 1]);
    let ln10000 = (10000.0f32).ln();
    let div: Vec<f32> = (0..half)
        .map(|j| ((2 * j) as f32 * -(ln10000 / d as f32)).exp())
        .collect();
    let div_row = Array::from_slice(&div, &[1, half as i32]);
    let angles = &pos_col * &div_row;
    let s = sin(&angles)?;
    let c = cos(&angles)?;
    let stacked = stack_axis(&[s, c], -1)?;
    let pe = reshape(&stacked, &[len as i32, d])?;
    Ok(expand_dims(&pe, 0)?)
}

fn rel_shift(x: &Array) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, q, p) = (sh[0], sh[1], sh[2], sh[3]);
    let x = pad(x, &[(0, 0), (0, 0), (0, 0), (1, 0)][..], Array::from_f32(0.0), None)?;
    let x = reshape(&x, &[b, h, p + 1, q])?;
    let x = x.index((.., .., 1.., ..));
    Ok(reshape(&x, &[b, h, q, p])?)
}

use mlx_rs::ops::indexing::IndexOp;

fn self_attention(a: &Attn, x: &Array, pos_emb: &Array) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, q_len) = (sh[0], sh[1]);
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let p = linear(pos_emb, &a.pos_w, None)?;
    let pos_len = p.shape()[1];
    let q = reshape(&linear(x, &a.q_w, Some(&a.q_b))?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let k = reshape(&linear(x, &a.k_w, Some(&a.k_b))?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let v = reshape(&linear(x, &a.v_w, Some(&a.v_b))?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let p = reshape(&p, &[1, pos_len, N_HEADS, HEAD_DIM])?;
    let q_u = (&q + &a.bias_u).transpose_axes(&[0, 2, 1, 3])?;
    let q_v = (&q + &a.bias_v).transpose_axes(&[0, 2, 1, 3])?;
    let k = k.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.transpose_axes(&[0, 2, 1, 3])?;
    let p = p.transpose_axes(&[0, 2, 1, 3])?;
    let bd = q_v.matmul(&p.transpose_axes(&[0, 1, 3, 2])?)?;
    let bd = rel_shift(&bd)?;
    let bd = bd.index((.., .., .., 0..k.shape()[2]));
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
    let x = conv1d(x, &c.pw1_w, Some(1), Some(0), Some(1), Some(1))?;
    let x = &x + &c.pw1_b;
    let x = glu(&x, 2)?;
    let x = conv1d(&x, &c.dw_w, Some(1), Some(4), Some(1), Some(D_MODEL))?;
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
