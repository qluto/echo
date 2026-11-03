//! Full-Rust Haas-ja-v1.0 (Third Intelligence) on Apple MLX.
//!
//! Japanese ASR on the Parakeet-TDT architecture: the same FastConformer
//! encoder as Parakeet-ja (128-mel input, no biases), and a TDT decoder whose
//! prediction network is Kimi Delta Attention instead of an LSTM. Apache-2.0,
//! non-gated. Weights come from the unofficial MLX conversion
//! `mrjnzzz/Haas-ja-v1.0-mlx` (the checkpoint's bf16 tensors, rearranged for
//! MLX), pinned to a commit; they run in f32 here like the other encoders.
//!
//! The model was trained on utterances of up to 30 s and its attention spans
//! the whole input, so longer audio is cut at quiet points first (as the
//! official `transcribe_longform` does).

pub mod audio;
pub mod decoder;
pub mod kda;
pub mod spm;

use anyhow::{anyhow, Result};
use std::path::Path;

use mlx_rs::Array;

use crate::parakeet::encoder::Encoder;
use crate::weights::Weights;
use crate::Transcription;
use decoder::Decoder;
use spm::SentencePiece;

pub const HAAS_MODEL_ID: &str = "mrjnzzz/Haas-ja-v1.0-mlx";
/// Pinned commit of [`HAAS_MODEL_ID`]: an unofficial personal conversion, so
/// later pushes must not change what Echo downloads.
pub const HAAS_REVISION: &str = "15ef4f11618c6a1ee50e0eafaae13beded8ecdf9";

const SAMPLE_RATE: usize = 16_000;
const CHUNK_SECONDS: usize = 30;
const SEARCH_SECONDS: usize = 2;
const QUIET_WINDOW: usize = SAMPLE_RATE / 10;

pub fn is_haas_model(name: &str) -> bool {
    name.to_lowercase().contains("haas-ja")
}

pub struct HaasEngine {
    encoder: Encoder,
    decoder: Decoder,
    tokenizer: SentencePiece,
    fb: Array,
    window: Array,
}

impl HaasEngine {
    /// Download (if needed) the pinned model into `hub_cache_dir` and load it.
    pub fn load(hub_cache_dir: &Path) -> Result<Self> {
        use hf_hub::api::sync::ApiBuilder;
        use hf_hub::{Repo, RepoType};
        let api = ApiBuilder::new()
            .with_cache_dir(hub_cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|e| anyhow!("hf-hub init: {e}"))?;
        let repo = api.repo(Repo::with_revision(
            HAAS_MODEL_ID.to_string(),
            RepoType::Model,
            HAAS_REVISION.to_string(),
        ));
        // Both are LFS files, which hf-hub 0.3 downloads fine (see hf.rs).
        let tokenizer_path = repo
            .get("tokenizer.model")
            .map_err(|e| anyhow!("fetch tokenizer.model: {e}"))?;
        let safetensors = repo
            .get("model.safetensors")
            .map_err(|e| anyhow!("fetch model.safetensors: {e}"))?;
        let tokenizer = SentencePiece::load(&tokenizer_path)?;

        let _gpu = crate::gpu::lock();
        let weights = Weights::load(safetensors.to_str().ok_or_else(|| anyhow!("bad path"))?)?;
        let engine = Self {
            encoder: Encoder::load(&weights)?,
            decoder: Decoder::load(&weights)?,
            fb: weights.f32("fb")?,
            window: audio::padded_window(&weights.f32("window")?)?,
            tokenizer,
        };
        log::info!(
            "Haas-ja model loaded ({} vocab pieces)",
            engine.tokenizer.len()
        );
        Ok(engine)
    }

    /// Transcribe mono 16 kHz f32 samples. Haas-ja is Japanese-only; the
    /// language argument is ignored.
    pub fn transcribe_samples(&self, samples: &[f32], _language: &str) -> Result<Transcription> {
        let mut text = String::new();
        for (start, end) in split_at_quiet_points(samples) {
            // Each chunk's text is trimmed and joined with nothing between, as
            // the official long-form transcription does.
            text.push_str(&self.transcribe_chunk(&samples[start..end])?);
        }
        Ok(Transcription {
            text,
            language: "ja".to_string(),
        })
    }

    /// One utterance of up to ~30 s. The GPU lock is held per chunk, so other
    /// work can run between the chunks of a long recording.
    fn transcribe_chunk(&self, samples: &[f32]) -> Result<String> {
        if samples.len() < audio::HOP as usize {
            return Ok(String::new());
        }
        let _gpu = crate::gpu::lock();
        let waveform = Array::from_slice(samples, &[samples.len() as i32]);
        let (mel, valid) = audio::extract_features(&waveform, &self.fb, &self.window)?;
        let enc = self.encoder.forward_valid(&mel, Some(valid))?;
        let tokens = self.decoder.decode(&enc)?;
        Ok(self.tokenizer.decode(&tokens))
    }

    pub fn transcribe_wav(&self, wav_path: &str, language: &str) -> Result<Transcription> {
        let samples = crate::read_wav_16k_mono(wav_path)?;
        self.transcribe_samples(&samples, language)
    }

    pub fn warmup(&self) -> Result<()> {
        self.transcribe_samples(&vec![0.0f32; SAMPLE_RATE / 2], "ja")?;
        Ok(())
    }
}

/// `(start, end)` sample ranges covering `audio` without gaps, each at most
/// ~32 s: a cut lands at the centre of the quietest 0.1 s within 2 s of every
/// 30 s mark, and never earlier than 15 s after the previous cut (port of
/// `haas_asr.longform` / haas-mlx `split_at_quiet_points`).
fn split_at_quiet_points(audio: &[f32]) -> Vec<(usize, usize)> {
    let step = CHUNK_SECONDS * SAMPLE_RATE;
    let search = SEARCH_SECONDS * SAMPLE_RATE;
    let window = QUIET_WINDOW;
    let mut bounds = Vec::new();
    let mut start = 0usize;
    while audio.len() - start > step {
        let target = start + step;
        let low = (start + step / 2).max(target - search);
        let high = (audio.len() - window).min(target + search);
        let cut = if high > low {
            low + quietest_window(&audio[low..high + window], window) + window / 2
        } else {
            target
        };
        bounds.push((start, cut));
        start = cut;
    }
    if start < audio.len() {
        bounds.push((start, audio.len()));
    }
    bounds
}

/// Offset of the `window`-sample span of `audio` with the least energy.
fn quietest_window(audio: &[f32], window: usize) -> usize {
    let mut energy = Vec::with_capacity(audio.len() + 1);
    energy.push(0.0f64);
    for &s in audio {
        energy.push(energy.last().unwrap() + f64::from(s) * f64::from(s));
    }
    (0..=audio.len() - window)
        .min_by(|&a, &b| {
            let ea = energy[a + window] - energy[a];
            let eb = energy[b + window] - energy[b];
            ea.total_cmp(&eb)
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_is_one_chunk() {
        assert_eq!(split_at_quiet_points(&vec![0.1; 1000]), vec![(0, 1000)]);
        assert!(split_at_quiet_points(&[]).is_empty());
    }

    #[test]
    fn long_audio_is_cut_at_the_quiet_point() {
        let n = 70 * SAMPLE_RATE;
        let mut audio = vec![0.5f32; n];
        // A silent 0.2 s just after the 29 s mark, inside the ±2 s search span.
        let quiet = 29 * SAMPLE_RATE + SAMPLE_RATE / 2;
        audio[quiet..quiet + SAMPLE_RATE / 5].fill(0.0);
        let bounds = split_at_quiet_points(&audio);
        assert_eq!(bounds.first().unwrap().0, 0);
        assert_eq!(bounds.last().unwrap().1, n);
        for w in bounds.windows(2) {
            assert_eq!(w[0].1, w[1].0, "chunks must be contiguous");
        }
        let cut = bounds[0].1;
        assert!(cut >= quiet && cut <= quiet + SAMPLE_RATE / 5, "cut {cut}");
        assert!(bounds.iter().all(|(s, e)| e - s <= 32 * SAMPLE_RATE));
    }
}
