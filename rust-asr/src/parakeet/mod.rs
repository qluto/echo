//! Full-Rust NVIDIA Parakeet TDT (parakeet-tdt_ctc-0.6b-ja) on Apple MLX.
//!
//! Japanese-specialized ASR: FastConformer encoder + TDT transducer decoder.
//! Non-gated (CC-BY-4.0). Weights are f32 in the mlx-community conversion.

pub mod audio;
pub mod decoder;
pub mod encoder;

use anyhow::{anyhow, Result};
use std::path::Path;

use decoder::Decoder;
use encoder::Encoder;
use mlx_rs::Array;

use crate::weights::Weights;
use crate::Transcription;

pub const PARAKEET_MODEL_ID: &str = "mlx-community/parakeet-tdt_ctc-0.6b-ja";

pub fn is_parakeet_model(name: &str) -> bool {
    name.to_lowercase().contains("parakeet")
}

pub struct ParakeetEngine {
    encoder: Encoder,
    decoder: Decoder,
    mel_fb: Array,
}

impl ParakeetEngine {
    /// Download (if needed) the model into `hub_cache_dir` and load it.
    pub fn load(hub_cache_dir: &Path) -> Result<Self> {
        use hf_hub::api::sync::ApiBuilder;
        let api = ApiBuilder::new()
            .with_cache_dir(hub_cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|e| anyhow!("hf-hub init: {e}"))?;
        let repo = api.model(PARAKEET_MODEL_ID.to_string());
        // config.json is a small non-LFS file: HF serves it via a relative
        // resolve-cache redirect that hf-hub 0.3.2 can't follow, so fetch it
        // directly. The LFS weights below still use hf-hub (absolute redirect).
        let config_path =
            crate::hf::fetch_small_file(hub_cache_dir, PARAKEET_MODEL_ID, "config.json", None)
                .map_err(|e| anyhow!("fetch config.json: {e}"))?;
        let safetensors = repo
            .get("model.safetensors")
            .map_err(|e| anyhow!("fetch model.safetensors: {e}"))?;

        let vocab = load_vocabulary(&config_path)?;
        let weights = Weights::load(safetensors.to_str().ok_or_else(|| anyhow!("bad path"))?)?;
        let encoder = Encoder::load(&weights)?;
        let decoder = Decoder::load(&weights, vocab)?;
        let mel_fb = audio::mel_filterbank();
        log::info!("Parakeet-ja model loaded ({} vocab pieces)", decoder_vocab_len(&decoder));
        Ok(Self {
            encoder,
            decoder,
            mel_fb,
        })
    }

    /// Transcribe mono 16 kHz f32 samples. Parakeet-ja is Japanese-only; the
    /// language argument is ignored.
    pub fn transcribe_samples(&self, samples: &[f32], _language: &str) -> Result<Transcription> {
        if samples.is_empty() {
            return Ok(Transcription {
                text: String::new(),
                language: "ja".to_string(),
            });
        }
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        log::info!(
            "Parakeet transcribe: {} samples (~{:.2}s @16k), rms={:.5}",
            samples.len(),
            samples.len() as f32 / 16000.0,
            rms
        );
        let waveform = Array::from_slice(samples, &[samples.len() as i32]);
        let mel = audio::extract_features(&waveform, &self.mel_fb)?;
        let enc = self.encoder.forward(&mel)?;
        let text = self.decoder.decode(&enc)?;
        Ok(Transcription {
            text,
            language: "ja".to_string(),
        })
    }

    pub fn transcribe_wav(&self, wav_path: &str, language: &str) -> Result<Transcription> {
        let samples = crate::read_wav_16k_mono(wav_path)?;
        self.transcribe_samples(&samples, language)
    }

    pub fn warmup(&self) -> Result<()> {
        let _ = self.transcribe_samples(&vec![0.0f32; 16000 / 2], "ja")?;
        Ok(())
    }
}

fn decoder_vocab_len(_d: &Decoder) -> usize {
    3072
}

/// Parse the 3072-piece vocabulary from config.json's joint.vocabulary.
fn load_vocabulary(config_path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(config_path)?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let arr = v
        .get("joint")
        .and_then(|j| j.get("vocabulary"))
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow!("joint.vocabulary missing in config.json"))?;
    Ok(arr
        .iter()
        .map(|s| s.as_str().unwrap_or("").to_string())
        .collect())
}
