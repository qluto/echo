//! Parakeet 80-mel log-mel frontend, ported from
//! `mlx_audio.stt.models.parakeet.audio.log_mel_spectrogram`.
//!
//! Differs from the Cohere frontend: 80 mel bins, the slaney mel filterbank is
//! *computed* here (not loaded from the checkpoint), and per-feature normalize
//! runs over the full spectrogram (single utterance, no length masking).

use anyhow::Result;
use mlx_rs::ops::{
    concatenate_axis, expand_dims, log as mlx_log, mean_axis, pad, sqrt, square, sum_axis,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

const N_FFT: i32 = 512;
const HOP: i32 = 160;
const WIN_LEN: i32 = 400; // 0.025 * 16000
const FEATURES: usize = 80;
const SAMPLE_RATE: f32 = 16000.0;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const NORM_EPS: f32 = 1e-5;
// Parakeet's NeMo preprocessor config has preemph=None → no pre-emphasis.
const PREEMPH: f32 = 0.97;

/// Symmetric Hann window (periodic=false), matching mlx_audio.hanning.
fn hann(n: i32) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / denom).cos()))
        .collect()
}

// ---- slaney mel scale (librosa-compatible) ----
fn hz_to_mel(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    } else {
        f_sp * m
    }
}

/// Slaney-normalized mel filterbank, shape [n_mels, n_fft/2+1].
pub fn mel_filterbank() -> Array {
    let n_freqs = (N_FFT / 2 + 1) as usize; // 257
    let fmax = SAMPLE_RATE / 2.0;
    let fftfreqs: Vec<f32> = (0..n_freqs)
        .map(|k| k as f32 * fmax / (n_freqs as f32 - 1.0))
        .collect();

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);
    let n_points = FEATURES + 2;
    let hz_points: Vec<f32> = (0..n_points)
        .map(|i| {
            let m = mel_min + (mel_max - mel_min) * i as f32 / (n_points as f32 - 1.0);
            mel_to_hz(m)
        })
        .collect();

    let mut fb = vec![0.0f32; FEATURES * n_freqs];
    for m in 0..FEATURES {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];
        let enorm = 2.0 / (f_right - f_left);
        for (k, &freq) in fftfreqs.iter().enumerate() {
            let left = (freq - f_left) / (f_center - f_left);
            let right = (f_right - freq) / (f_right - f_center);
            let tri = left.min(right).max(0.0);
            fb[m * n_freqs + k] = tri * enorm;
        }
    }
    Array::from_slice(&fb, &[FEATURES as i32, n_freqs as i32])
}

/// waveform [L] f32 -> log-mel features [1, T, 80].
pub fn extract_features(waveform: &Array, fb: &Array) -> Result<Array> {
    let l = waveform.shape()[0];

    // optional pre-emphasis (disabled for parakeet)
    let x = if PREEMPH > 0.0 {
        let x0 = waveform.index(0..1);
        let cur = waveform.index(1..l);
        let prev = waveform.index(0..(l - 1));
        let body = &cur - &(&prev * Array::from_f32(PREEMPH));
        concatenate_axis(&[x0, body], 0)?
    } else {
        waveform.clone()
    };

    // hann(400) center-padded to n_fft=512
    let win = hann(WIN_LEN);
    let win = Array::from_slice(&win, &[WIN_LEN]);
    let total_pad = N_FFT - WIN_LEN;
    let left = total_pad / 2;
    let right = total_pad - left;
    let window = pad(&win, (left, right), Array::from_f32(0.0), None)?;

    // centered STFT (constant pad)
    let xc = pad(&x, (N_FFT / 2, N_FFT / 2), Array::from_f32(0.0), None)?;
    let p = xc.shape()[0];
    let num_frames = 1 + (p - N_FFT) / HOP;
    let frames = xc.as_strided(&[num_frames, N_FFT][..], &[HOP as i64, 1i64][..], 0)?;
    let windowed = &frames * &window;
    let spec = mlx_rs::fft::rfft(&windowed, N_FFT, 1)?; // [T, 257]
    let power = square(mlx_rs::ops::abs(&spec)?)?.as_dtype(Dtype::Float32)?;

    // mel: fb [80,257] @ power.T [257,T] -> [80, T]
    let mel = fb.matmul(&power.t())?;
    let mel = mlx_log(&(&mel + Array::from_f32(LOG_GUARD)))?;

    // per-feature normalize over all frames (axis=1)
    let mean = mean_axis(&mel, 1, true)?;
    let n = (num_frames - 1).max(1);
    let var = &sum_axis(&square(&(&mel - &mean))?, 1, true)? / Array::from_f32(n as f32);
    let std = sqrt(&var)?;
    let mel = &(&mel - &mean) / &(&std + Array::from_f32(NORM_EPS));

    // [80, T] -> [T, 80] -> [1, T, 80]
    let mel = mel.t();
    Ok(expand_dims(&mel, 0)?)
}
