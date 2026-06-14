//! Cohere ASR audio frontend ported from `mlx_audio.stt.models.cohere_asr.audio`.
//!
//! waveform -> preemphasis -> centered STFT (n_fft=512, hann win 400 padded to
//! 512, hop 160) -> power spectrum -> mel filterbank (loaded from checkpoint)
//! -> log -> per-feature normalize -> trailing-frame mask -> [1, T, 128].
//!
//! Dither is intentionally omitted: it is N(0, 1e-5) noise that never changes
//! the decoded tokens, and skipping it makes the output deterministic for
//! parity checks against the Python dump (which we also run with dither off).

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    concatenate_axis, expand_dims, log as mlx_log, mean_axis, pad, reshape, sqrt, square,
    sum_axis,
};
use mlx_rs::{Array, Dtype};

use crate::weights::Weights;

const N_FFT: i32 = 512;
const HOP: i32 = 160;
const WIN_LEN: i32 = 400;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const NORM_EPS: f32 = 1e-5;

/// Returns (features [1, T, 128] f32, sequence_length).
pub fn extract_features(w: &Weights, waveform: &Array) -> Result<(Array, i32)> {
    let l = waveform.shape()[0];

    // --- preemphasis: out = [x0, x[1:] - 0.97*x[:-1]] ---
    let x0 = waveform.index(0..1);
    let cur = waveform.index(1..l);
    let prev = waveform.index(0..(l - 1));
    let body = &cur - &(&prev * Array::from_f32(PREEMPH));
    let x = concatenate_axis(&[x0, body], 0)?;

    // --- window: checkpoint hann(400) padded symmetrically to 512 ---
    let win = w.f32("preprocessor.featurizer.window")?; // [400]
    debug_assert_eq!(win.shape(), &[WIN_LEN]);
    let total_pad = N_FFT - WIN_LEN; // 112
    let left = total_pad / 2; // 56
    let right = total_pad - left; // 56
    let window = pad(&win, (left, right), Array::from_f32(0.0), None)?; // [512]

    // --- center pad signal by n_fft/2 (constant 0) ---
    let xc = pad(&x, (N_FFT / 2, N_FFT / 2), Array::from_f32(0.0), None)?;
    let p = xc.shape()[0];
    let num_frames = 1 + (p - N_FFT) / HOP;

    // --- frame via as_strided then windowed rfft ---
    let frames = xc.as_strided(&[num_frames, N_FFT][..], &[HOP as i64, 1i64][..], 0)?;
    let windowed = &frames * &window; // broadcast [512] over rows
    let spec = mlx_rs::fft::rfft(&windowed, N_FFT, 1)?; // complex [T, 257]
    let power = square(mlx_rs::ops::abs(&spec)?)?.as_dtype(Dtype::Float32)?; // [T, 257]

    // --- mel filterbank from checkpoint: [1,128,257] -> [128,257] ---
    let fb = w.f32("preprocessor.featurizer.fb")?;
    let fb = reshape(&fb, &[128, 257])?;
    let mel = fb.matmul(&power.t())?; // [128, T]
    let mel = mlx_log(&(&mel + Array::from_f32(LOG_GUARD)))?;

    // --- per-feature normalize over the first seq_len frames ---
    let mut seq_len = l / HOP;
    if seq_len > num_frames {
        seq_len = num_frames;
    }
    let valid = mel.index((.., 0..seq_len)); // [128, seq_len]
    let mean = mean_axis(&valid, 1, true)?; // [128,1]
    let centered = &valid - &mean;
    let var = &sum_axis(&square(&centered)?, 1, true)? / Array::from_f32((seq_len - 1) as f32);
    let std = sqrt(&var)?;
    let mel = &(&mel - &mean) / &(&std + Array::from_f32(NORM_EPS));

    // --- zero out trailing (>= seq_len) frames (host-built mask) ---
    let mask_vals: Vec<f32> = (0..num_frames)
        .map(|i| if i < seq_len { 1.0 } else { 0.0 })
        .collect();
    let mask = Array::from_slice(&mask_vals, &[1, num_frames]); // broadcast over 128 rows
    let mel = &mel * &mask;

    // --- [128, T] -> [T, 128] -> [1, T, 128] ---
    let mel = mel.t();
    let feats = expand_dims(&mel, 0)?;
    Ok((feats, seq_len))
}
