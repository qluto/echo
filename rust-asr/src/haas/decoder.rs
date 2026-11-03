//! Haas-ja TDT joint network + greedy decoding.
//!
//! The loop follows NeMo's `greedy_batch` TDT label-looping decoder for one
//! utterance (as in matsuurajun/haas-mlx `model.py`), which differs from the
//! Parakeet-ja loop in `parakeet/decoder.rs`:
//! - a blank always advances time by at least one frame;
//! - the max-symbols guard counts tokens emitted on the same frame and only
//!   forces a step when time has not moved.

use anyhow::Result;
use mlx_rs::nn::relu;
use mlx_rs::ops::indexing::{argmax, IndexOp};
use mlx_rs::Array;

use super::kda::{PredictionNetwork, State, HIDDEN};
use crate::nn::linear;
use crate::weights::Weights;

const VOCAB: i32 = 3072;
const BLANK_ID: i32 = VOCAB;
const DURATIONS: [i32; 5] = [0, 1, 2, 3, 4];
const MAX_SYMBOLS: i32 = 10;

pub struct Decoder {
    embed: Array, // [VOCAB + 1, 640]
    prediction: PredictionNetwork,
    joint_enc_w: Array,
    joint_enc_b: Array,
    joint_pred_w: Array,
    joint_pred_b: Array,
    joint_out_w: Array,
    joint_out_b: Array,
}

impl Decoder {
    pub fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            embed: w.f32("embed.weight")?,
            prediction: PredictionNetwork::load(w)?,
            joint_enc_w: w.f32("joint_enc.weight")?,
            joint_enc_b: w.f32("joint_enc.bias")?,
            joint_pred_w: w.f32("joint_pred.weight")?,
            joint_pred_b: w.f32("joint_pred.bias")?,
            joint_out_w: w.f32("joint_out.weight")?,
            joint_out_b: w.f32("joint_out.bias")?,
        })
    }

    /// Prediction-network step for `token`, projected into the joint space.
    fn predict(&self, token: i32, state: &State) -> Result<(Array, State)> {
        let y = self.embed.index((token, ..)).reshape(&[1, HIDDEN])?;
        let (g, next) = self.prediction.step(&y, state)?;
        Ok((linear(&g, &self.joint_pred_w, Some(&self.joint_pred_b))?, next))
    }

    /// Greedy TDT decode. encoder: [1, T, 1024] -> token ids (blank excluded).
    pub fn decode(&self, encoder: &Array) -> Result<Vec<i32>> {
        let total = encoder.shape()[1];
        let enc_proj = linear(encoder, &self.joint_enc_w, Some(&self.joint_enc_b))?; // [1, T, 640]
        enc_proj.eval()?;

        // NeMo's start-of-sequence input is the blank embedding (blank_as_pad).
        let mut state = self.prediction.empty_state()?;
        let (mut pred, mut next_state) = self.predict(BLANK_ID, &state)?;

        let mut tokens = Vec::new();
        let mut t = 0i32;
        let mut last_t = -1i32;
        let mut on_last_t = 0i32;
        while t < total {
            let h = relu(&(&enc_proj.index((.., t, ..)) + &pred))?;
            let logits = linear(&h, &self.joint_out_w, Some(&self.joint_out_b))?
                .reshape(&[VOCAB + 1 + DURATIONS.len() as i32])?;
            let token = argmax(&logits.index(0..(BLANK_ID + 1)), false)?.item_cast::<u32>() as i32;
            let skip = DURATIONS
                [argmax(&logits.index((BLANK_ID + 1)..), false)?.item_cast::<u32>() as usize];

            if token == BLANK_ID {
                t += skip.max(1);
                continue;
            }
            tokens.push(token);
            on_last_t = if t == last_t { on_last_t + 1 } else { 1 };
            last_t = t;
            state = next_state;
            (pred, next_state) = self.predict(token, &state)?;
            t += skip;
            if on_last_t >= MAX_SYMBOLS && t == last_t {
                t += 1;
            }
        }
        Ok(tokens)
    }
}
