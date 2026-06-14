//! Parakeet TDT (Token-and-Duration Transducer) decoder, ported from
//! `mlx_audio.stt.models.parakeet` (PredictNetwork + JointNetwork + TDT decode).
//!
//! Prediction network = embedding + 2-layer LSTM (hand-rolled cell, gate order
//! i,f,g,o to match MLX). Joint network combines an encoder frame with the
//! prediction state and emits token logits + duration logits. Greedy TDT loop:
//! advance the LSTM state only on non-blank emissions, skip `duration` frames
//! each step.

use anyhow::Result;
use mlx_rs::nn::sigmoid;
use mlx_rs::ops::indexing::{argmax, IndexOp};
use mlx_rs::ops::{split, tanh};
use mlx_rs::Array;

use crate::nn::linear;
use crate::weights::Weights;

const PRED_HIDDEN: i32 = 640;
const N_LSTM_LAYERS: usize = 2;
const VOCAB: i32 = 3072;
const BLANK_ID: i32 = VOCAB; // 3072
const N_DURATIONS: usize = 5;
const DURATIONS: [i32; N_DURATIONS] = [0, 1, 2, 3, 4];
const MAX_SYMBOLS: i32 = 10;

struct LstmLayer {
    wx: Array, // [4H, in]
    wh: Array, // [4H, H]
    bias: Array, // [4H]
}

pub struct Decoder {
    embed: Array, // [VOCAB+1, 640]
    lstm: Vec<LstmLayer>,
    joint_enc_w: Array,
    joint_enc_b: Array,
    joint_pred_w: Array,
    joint_pred_b: Array,
    joint_out_w: Array,
    joint_out_b: Array,
    vocab: Vec<String>,
}

impl Decoder {
    pub fn load(w: &Weights, vocab: Vec<String>) -> Result<Self> {
        let mut lstm = Vec::new();
        for i in 0..N_LSTM_LAYERS {
            let lp = format!("decoder.prediction.dec_rnn.lstm.{i}");
            lstm.push(LstmLayer {
                wx: w.f32(&format!("{lp}.Wx"))?,
                wh: w.f32(&format!("{lp}.Wh"))?,
                bias: w.f32(&format!("{lp}.bias"))?,
            });
        }
        Ok(Self {
            embed: w.f32("decoder.prediction.embed.weight")?,
            lstm,
            joint_enc_w: w.f32("joint.enc.weight")?,
            joint_enc_b: w.f32("joint.enc.bias")?,
            joint_pred_w: w.f32("joint.pred.weight")?,
            joint_pred_b: w.f32("joint.pred.bias")?,
            joint_out_w: w.f32("joint.joint_net.2.weight")?,
            joint_out_b: w.f32("joint.joint_net.2.bias")?,
            vocab,
        })
    }

    /// Greedy TDT decode. encoder: [1, T, 1024]. Returns decoded text.
    pub fn decode(&self, encoder: &Array) -> Result<String> {
        let max_length = encoder.shape()[1];

        // LSTM state per layer: h, c each [1, 640]
        let mut h: Vec<Array> = (0..N_LSTM_LAYERS)
            .map(|_| Array::from_slice(&vec![0.0f32; PRED_HIDDEN as usize], &[1, PRED_HIDDEN]))
            .collect();
        let mut c = h.clone();

        let mut last_token = BLANK_ID;
        let mut hypothesis: Vec<i32> = Vec::new();
        let mut time = 0i32;
        let mut new_symbols = 0i32;

        while time < max_length {
            let feature = encoder.index((.., time, ..)); // [1, 1024]

            // embed(last_token); blank -> zeros
            let embedded = if last_token == BLANK_ID {
                Array::from_slice(&vec![0.0f32; PRED_HIDDEN as usize], &[1, PRED_HIDDEN])
            } else {
                self.embed.index((last_token, ..)).reshape(&[1, PRED_HIDDEN])?
            };

            // run 2-layer LSTM from current (h, c)
            let mut next_h = Vec::with_capacity(N_LSTM_LAYERS);
            let mut next_c = Vec::with_capacity(N_LSTM_LAYERS);
            let mut x = embedded;
            for (li, layer) in self.lstm.iter().enumerate() {
                let (nh, nc) = lstm_step(layer, &x, &h[li], &c[li])?;
                x = nh.clone();
                next_h.push(nh);
                next_c.push(nc);
            }
            let dec_out = x; // [1, 640]

            // joint(feature, dec_out) -> logits [3078]
            let logits = self.joint(&feature, &dec_out)?;
            let token_logits = logits.index(0..(BLANK_ID + 1)); // [3073]
            let dur_logits = logits.index((BLANK_ID + 1)..); // [5]
            let pred_token = argmax(&token_logits, false)?.item::<u32>() as i32;
            let decision = argmax(&dur_logits, false)?.item::<u32>() as usize;
            let duration = DURATIONS[decision];

            if pred_token != BLANK_ID {
                last_token = pred_token;
                h = next_h;
                c = next_c;
                if !self.is_special(pred_token) {
                    hypothesis.push(pred_token);
                }
            }

            time += duration;
            new_symbols += 1;
            if duration != 0 {
                new_symbols = 0;
            } else if MAX_SYMBOLS <= new_symbols {
                time += 1;
                new_symbols = 0;
            }
        }

        Ok(self.decode_tokens(&hypothesis))
    }

    fn joint(&self, feature: &Array, pred: &Array) -> Result<Array> {
        let enc = linear(feature, &self.joint_enc_w, Some(&self.joint_enc_b))?; // [1,640]
        let pred = linear(pred, &self.joint_pred_w, Some(&self.joint_pred_b))?; // [1,640]
        let h = mlx_rs::nn::relu(&(&enc + &pred))?;
        let out = linear(&h, &self.joint_out_w, Some(&self.joint_out_b))?; // [1,3078]
        Ok(out.reshape(&[VOCAB + 1 + N_DURATIONS as i32])?)
    }

    fn is_special(&self, token: i32) -> bool {
        let t = token as usize;
        if t >= self.vocab.len() {
            return true;
        }
        let p = &self.vocab[t];
        (p.starts_with("<|") && p.ends_with("|>")) || p == "<unk>" || p == "<pad>"
    }

    fn decode_tokens(&self, tokens: &[i32]) -> String {
        let mut s = String::new();
        for &t in tokens {
            let i = t as usize;
            if i >= self.vocab.len() {
                continue;
            }
            let p = &self.vocab[i];
            if (p.starts_with("<|") && p.ends_with("|>")) || p == "<unk>" || p == "<pad>" {
                continue;
            }
            s.push_str(&p.replace('▁', " "));
        }
        s.trim().to_string()
    }
}

/// One LSTM step. gates = x @ Wx^T + bias + h @ Wh^T; split i,f,g,o.
fn lstm_step(layer: &LstmLayer, x: &Array, h: &Array, c: &Array) -> Result<(Array, Array)> {
    let gates = &(&(&x.matmul(&layer.wx.t())? + &layer.bias) + &h.matmul(&layer.wh.t())?);
    let parts = split(gates, 4, -1)?; // [i, f, g, o] each [1, H]
    let i = sigmoid(&parts[0])?;
    let f = sigmoid(&parts[1])?;
    let g = tanh(&parts[2])?;
    let o = sigmoid(&parts[3])?;
    let c2 = &(&f * c) + &(&i * &g);
    let h2 = &o * &tanh(&c2)?;
    Ok((h2, c2))
}
