//! Parakeet FastConformer encoder (NeMo), ported from
//! `mlx_audio.stt.models.parakeet.conformer`. Structurally identical to the
//! Cohere encoder, with three differences: conv weights are already in MLX
//! layout (no transpose), there is no inter-conv length masking (single
//! utterance), and rel-pos encoding scales the input by sqrt(d_model)
//! (xscaling=True). d_model 1024, 24 layers, 8 heads, 80-mel input.
//!
//! Haas-ja (`crate::haas`) uses this encoder too: 128-mel input, no biases in
//! the conformer layers (`use_bias: false`), and subsampling convs keyed
//! `pre_encode.convs.0..4`. Both layouts are detected from the weights.

use anyhow::Result;
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::{glu, relu, silu};
use mlx_rs::ops::{conv1d, conv2d, cos, expand_dims, pad, reshape, sin, stack};
use mlx_rs::Array;

use crate::nn::{batch_norm_eval, layer_norm, linear};
use crate::weights::Weights;

const D_MODEL: i32 = 1024;
const N_HEADS: i32 = 8;
const HEAD_DIM: i32 = D_MODEL / N_HEADS; // 128
const N_LAYERS: usize = 24;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;
const XSCALE: f32 = 32.0; // sqrt(1024)

struct Attn {
    q_w: Array,
    q_b: Option<Array>,
    k_w: Array,
    k_b: Option<Array>,
    v_w: Array,
    v_b: Option<Array>,
    pos_w: Array,
    out_w: Array,
    out_b: Option<Array>,
    bias_u: Array,
    bias_v: Array,
}

struct ConvMod {
    pw1_w: Array,
    pw1_b: Option<Array>,
    dw_w: Array,
    dw_b: Option<Array>,
    bn_w: Array,
    bn_b: Array,
    bn_rm: Array,
    bn_rv: Array,
    pw2_w: Array,
    pw2_b: Option<Array>,
}

struct Ff {
    l1_w: Array,
    l1_b: Option<Array>,
    l2_w: Array,
    l2_b: Option<Array>,
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
        // The five subsampling convs: NeMo's Sequential indices (ReLUs in
        // between) in Parakeet-ja, renumbered 0..4 in the Haas-ja conversion.
        // Conv weights are already in MLX layout in both checkpoints.
        let names: Vec<String> = if w.contains("encoder.pre_encode.conv.0.weight") {
            [0, 2, 3, 5, 6].iter().map(|i| format!("encoder.pre_encode.conv.{i}")).collect()
        } else {
            (0..5).map(|i| format!("encoder.pre_encode.convs.{i}")).collect()
        };
        let mut sub_w = Vec::new();
        for p in &names {
            let wt = w.f32(&format!("{p}.weight"))?;
            let b = w.f32(&format!("{p}.bias"))?;
            sub_w.push((wt, b));
        }
        let sub_groups = vec![1, 256, 1, 256, 1];
        let sub_stride = vec![2, 2, 1, 2, 1];
        let sub_pad = vec![1, 1, 0, 1, 0];

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let lp = format!("encoder.layers.{i}");
            let g = |s: &str| w.f32(&format!("{lp}.{s}"));
            let opt = |s: &str| optional(w, &format!("{lp}.{s}"));
            layers.push(Layer {
                n_ff1_w: g("norm_feed_forward1.weight")?,
                n_ff1_b: g("norm_feed_forward1.bias")?,
                ff1: Ff {
                    l1_w: g("feed_forward1.linear1.weight")?,
                    l1_b: opt("feed_forward1.linear1.bias")?,
                    l2_w: g("feed_forward1.linear2.weight")?,
                    l2_b: opt("feed_forward1.linear2.bias")?,
                },
                n_att_w: g("norm_self_att.weight")?,
                n_att_b: g("norm_self_att.bias")?,
                attn: Attn {
                    q_w: g("self_attn.linear_q.weight")?,
                    q_b: opt("self_attn.linear_q.bias")?,
                    k_w: g("self_attn.linear_k.weight")?,
                    k_b: opt("self_attn.linear_k.bias")?,
                    v_w: g("self_attn.linear_v.weight")?,
                    v_b: opt("self_attn.linear_v.bias")?,
                    pos_w: g("self_attn.linear_pos.weight")?,
                    out_w: g("self_attn.linear_out.weight")?,
                    out_b: opt("self_attn.linear_out.bias")?,
                    bias_u: g("self_attn.pos_bias_u")?,
                    bias_v: g("self_attn.pos_bias_v")?,
                },
                n_conv_w: g("norm_conv.weight")?,
                n_conv_b: g("norm_conv.bias")?,
                conv: ConvMod {
                    pw1_w: g("conv.pointwise_conv1.weight")?,
                    pw1_b: opt("conv.pointwise_conv1.bias")?,
                    dw_w: g("conv.depthwise_conv.weight")?,
                    dw_b: opt("conv.depthwise_conv.bias")?,
                    bn_w: g("conv.batch_norm.weight")?,
                    bn_b: g("conv.batch_norm.bias")?,
                    bn_rm: g("conv.batch_norm.running_mean")?,
                    bn_rv: g("conv.batch_norm.running_var")?,
                    pw2_w: g("conv.pointwise_conv2.weight")?,
                    pw2_b: opt("conv.pointwise_conv2.bias")?,
                },
                n_ff2_w: g("norm_feed_forward2.weight")?,
                n_ff2_b: g("norm_feed_forward2.bias")?,
                ff2: Ff {
                    l1_w: g("feed_forward2.linear1.weight")?,
                    l1_b: opt("feed_forward2.linear1.bias")?,
                    l2_w: g("feed_forward2.linear2.weight")?,
                    l2_b: opt("feed_forward2.linear2.bias")?,
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
        self.forward_valid(mel, None)
    }

    /// As [`forward`](Self::forward), for features padded past `valid_frames`
    /// mel frames (NeMo's `pad_to`): the encoder frames computed from the
    /// padding are dropped after subsampling. NeMo masks them in attention and
    /// convolution; for a single utterance, dropping them is the same.
    pub fn forward_valid(&self, mel: &Array, valid_frames: Option<i32>) -> Result<Array> {
        let mut x = self.subsample(mel)?;
        if let Some(n) = valid_frames {
            // NeMo calc_length for three stride-2 convs (kernel 3, pad 1).
            let t = (0..3).fold(n, |n, _| (n - 1) / 2 + 1);
            x = x.index((.., 0..t, ..));
        }
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
    let stacked = stack(&[s, c], -1)?;
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
    let q = reshape(&linear(x, &a.q_w, a.q_b.as_ref())?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let k = reshape(&linear(x, &a.k_w, a.k_b.as_ref())?, &[b, q_len, N_HEADS, HEAD_DIM])?;
    let v = reshape(&linear(x, &a.v_w, a.v_b.as_ref())?, &[b, q_len, N_HEADS, HEAD_DIM])?;
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
        None,
    )?;
    let out = reshape(&out.transpose_axes(&[0, 2, 1, 3])?, &[b, q_len, D_MODEL])?;
    linear(&out, &a.out_w, a.out_b.as_ref())
}

fn conv_module(c: &ConvMod, x: &Array) -> Result<Array> {
    let x = conv1d(x, &c.pw1_w, Some(1), Some(0), Some(1), Some(1))?;
    let x = add_bias(x, c.pw1_b.as_ref());
    let x = glu(&x, 2)?;
    let x = conv1d(&x, &c.dw_w, Some(1), Some(4), Some(1), Some(D_MODEL))?;
    let x = add_bias(x, c.dw_b.as_ref());
    let x = batch_norm_eval(&x, &c.bn_w, &c.bn_b, &c.bn_rm, &c.bn_rv, BN_EPS)?;
    let x = silu(&x)?;
    let x = conv1d(&x, &c.pw2_w, Some(1), Some(0), Some(1), Some(1))?;
    Ok(add_bias(x, c.pw2_b.as_ref()))
}

fn add_bias(x: Array, b: Option<&Array>) -> Array {
    match b {
        Some(b) => &x + b,
        None => x,
    }
}

/// A bias that only some checkpoints carry (Haas-ja is trained with
/// `use_bias: false`; Parakeet-ja has them).
fn optional(w: &Weights, key: &str) -> Result<Option<Array>> {
    if w.contains(key) {
        Ok(Some(w.f32(key)?))
    } else {
        Ok(None)
    }
}

fn feed_forward(f: &Ff, x: &Array) -> Result<Array> {
    let h = linear(x, &f.l1_w, f.l1_b.as_ref())?;
    let h = silu(&h)?;
    linear(&h, &f.l2_w, f.l2_b.as_ref())
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
