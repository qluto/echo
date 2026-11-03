//! Haas-ja 128-mel log-mel frontend, matching NeMo's `FilterbankFeatures` with
//! the released model's config (as ported in matsuurajun/haas-mlx `audio.py`).
//!
//! Unlike the Parakeet-ja frontend, the mel filterbank and the Hann window are
//! the checkpoint's own buffers (`fb` [128, 257], `window` [400]), and the
//! per-feature normalization runs over the *valid* frames only (`N / 160`,
//! one fewer than the centered STFT yields). The features are then zero-padded
//! to a multiple of 16 frames (NeMo's `pad_to`).

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    concatenate, expand_dims, log as mlx_log, mean_axis, pad, sqrt, square, sum_axis,
};
use mlx_rs::{Array, Dtype};

pub const HOP: i32 = 160;
const N_FFT: i32 = 512;
const WIN_LEN: i32 = 400;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const NORM_EPS: f32 = 1e-5;
const PAD_TO: i32 = 16;

/// The checkpoint's 400-point window centered in `N_FFT` zeros, as torch.stft does.
pub fn padded_window(window: &Array) -> Result<Array> {
    let left = (N_FFT - WIN_LEN) / 2;
    let right = N_FFT - WIN_LEN - left;
    Ok(pad(window, (left, right), Array::from_f32(0.0), None)?)
}

/// waveform [L] f32 (L >= HOP) -> (features [1, T, 128], valid frames).
/// `fb`: [128, 257] f32; `window`: [512] f32 from [`padded_window`].
pub fn extract_features(waveform: &Array, fb: &Array, window: &Array) -> Result<(Array, i32)> {
    let l = waveform.shape()[0];
    let n_valid = l / HOP;

    // Pre-emphasis: y[0] = x[0]; y[n] = x[n] - 0.97 * x[n-1].
    let x0 = waveform.index(0..1);
    let body = &waveform.index(1..l) - &(&waveform.index(0..(l - 1)) * Array::from_f32(PREEMPH));
    let x = concatenate(&[x0, body], 0)?;

    // Centered STFT with zero padding, power spectrum.
    let xc = pad(&x, (N_FFT / 2, N_FFT / 2), Array::from_f32(0.0), None)?;
    let p = xc.shape()[0];
    let num_frames = 1 + (p - N_FFT) / HOP;
    let frames = xc.as_strided(&[num_frames, N_FFT][..], &[HOP as i64, 1i64][..], 0)?;
    let spec = mlx_rs::fft::rfft(&(&frames * window), N_FFT, 1)?; // [T, 257]
    let power = square(mlx_rs::ops::abs(&spec)?)?.as_dtype(Dtype::Float32)?;

    // [T, 257] @ [257, 128] -> [T, 128], log, keep the valid frames.
    let mel = power.matmul(&fb.t())?;
    let mel = mlx_log(&(&mel + Array::from_f32(LOG_GUARD)))?;
    let mel = mel.index((0..n_valid, ..));

    // Per-feature normalization over time (Bessel-corrected std, + 1e-5).
    let mean = mean_axis(&mel, 0, true)?;
    let centered = &mel - &mean;
    let std = if n_valid > 1 {
        let var = &sum_axis(&square(&centered)?, 0, true)? / Array::from_f32((n_valid - 1) as f32);
        sqrt(&var)?
    } else {
        mlx_rs::ops::zeros_like(&mean)?
    };
    let mel = &centered / &(&std + Array::from_f32(NORM_EPS));

    let padded = (num_frames + PAD_TO - 1) / PAD_TO * PAD_TO;
    let mel = pad(
        &mel,
        &[(0, padded - n_valid), (0, 0)][..],
        Array::from_f32(0.0),
        None,
    )?;
    Ok((expand_dims(&mel, 0)?, n_valid))
}
