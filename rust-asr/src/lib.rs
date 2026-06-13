//! Full-Rust Cohere Transcribe engine (cohere-transcribe-03-2026) on Apple MLX.
//!
//! Exposes [`CohereEngine`], a self-contained in-process ASR engine that
//! replaces the Python/MLX sidecar for the Cohere model: it locates (and, if
//! needed, downloads) the gated checkpoint in the shared HuggingFace hub cache,
//! then runs the FastConformer encoder + Transformer AED decoder on Metal.

pub mod audio;
pub mod decoder;
pub mod encoder;
pub mod nn;
pub mod tokenizer;
pub mod weights;

use anyhow::{anyhow, Result};
use std::path::Path;

use decoder::Decoder;
use encoder::Encoder;
use mlx_rs::Array;
use tokenizer::Tok;
use weights::Weights;

pub const COHERE_MODEL_ID: &str = "CohereLabs/cohere-transcribe-03-2026";

/// Languages Cohere Transcribe supports (ISO codes). Used to validate input.
pub const SUPPORTED_LANGUAGES: [&str; 14] = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
];

/// Returns true if the given model id should be served by this in-process engine.
pub fn is_cohere_model(model_name: &str) -> bool {
    model_name.to_lowercase().contains("cohere-transcribe")
}

/// A single transcription result.
pub struct Transcription {
    pub text: String,
    pub language: String,
}

pub struct CohereEngine {
    weights: Weights, // retained for the audio frontend (mel filterbank / window)
    encoder: Encoder,
    decoder: Decoder,
    tokenizer: Tok,
}

impl CohereEngine {
    /// Ensure the checkpoint is present in `hub_cache_dir` (the HF hub cache root
    /// that contains `models--*` dirs), downloading the gated model with
    /// `hf_token` if necessary, then load it.
    pub fn load(hub_cache_dir: &Path, hf_token: Option<&str>) -> Result<Self> {
        let (safetensors, tokenizer_json) = ensure_model_files(hub_cache_dir, hf_token)?;

        let weights = Weights::load(safetensors.to_str().ok_or_else(|| anyhow!("bad path"))?)?;
        let encoder = Encoder::load(&weights)?;
        let decoder = Decoder::load(&weights)?;
        let tokenizer = Tok::load(tokenizer_json.to_str().ok_or_else(|| anyhow!("bad path"))?)?;

        Ok(Self {
            weights,
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Transcribe mono 16 kHz f32 samples in [-1, 1].
    pub fn transcribe_samples(&self, samples: &[f32], language: &str) -> Result<Transcription> {
        let lang = normalize_language(language);
        if samples.is_empty() {
            return Ok(Transcription {
                text: String::new(),
                language: lang,
            });
        }
        let waveform = Array::from_slice(samples, &[samples.len() as i32]);
        let (mel, seq_len) = audio::extract_features(&self.weights, &waveform)?;
        let enc = self.encoder.forward(&mel, seq_len)?;
        let prompt = self.tokenizer.build_prompt(&lang, true)?;
        let ids = self.decoder.generate(&enc, &prompt, 256)?;
        let text = self.tokenizer.decode(&ids)?;
        Ok(Transcription {
            text: text.trim().to_string(),
            language: lang,
        })
    }

    /// Transcribe a 16 kHz mono WAV file.
    pub fn transcribe_wav(&self, wav_path: &str, language: &str) -> Result<Transcription> {
        let samples = read_wav_16k_mono(wav_path)?;
        self.transcribe_samples(&samples, language)
    }

    /// Run a tiny dummy inference to trigger Metal kernel compilation so the
    /// first real transcription is fast.
    pub fn warmup(&self) -> Result<()> {
        let samples = vec![0.0f32; 16000 / 2]; // 0.5 s of silence
        let _ = self.transcribe_samples(&samples, "en")?;
        Ok(())
    }
}

/// Cohere requires a concrete supported language; "auto"/unknown -> "en"
/// (matching the Python reference, which falls back to the model default).
fn normalize_language(language: &str) -> String {
    let l = language.trim().to_lowercase();
    if SUPPORTED_LANGUAGES.contains(&l.as_str()) {
        l
    } else {
        "en".to_string()
    }
}

/// Locate (downloading if needed) model.safetensors + tokenizer.json in the
/// shared HF hub cache. Returns their on-disk paths.
fn ensure_model_files(
    hub_cache_dir: &Path,
    hf_token: Option<&str>,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    use hf_hub::api::sync::ApiBuilder;

    let mut builder = ApiBuilder::new()
        .with_cache_dir(hub_cache_dir.to_path_buf())
        .with_progress(false);
    if let Some(tok) = hf_token.filter(|t| !t.is_empty()) {
        builder = builder.with_token(Some(tok.to_string()));
    }
    let api = builder
        .build()
        .map_err(|e| anyhow!("hf-hub init failed: {e}"))?;
    let repo = api.model(COHERE_MODEL_ID.to_string());

    let safetensors = repo
        .get("model.safetensors")
        .map_err(|e| anyhow!("fetch model.safetensors ({COHERE_MODEL_ID}): {e}"))?;
    let tokenizer_json = repo
        .get("tokenizer.json")
        .map_err(|e| anyhow!("fetch tokenizer.json ({COHERE_MODEL_ID}): {e}"))?;
    Ok((safetensors, tokenizer_json))
}

/// Read a WAV as mono 16 kHz f32 in [-1, 1] (Echo always feeds 16 kHz mono).
pub fn read_wav_16k_mono(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).map_err(|e| anyhow!("open wav: {e}"))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| anyhow!("read samples: {e}"))?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow!("read samples: {e}"))?,
    };
    let mono: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        raw.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()
    } else {
        raw
    };
    Ok(mono)
}
