//! Whisper engine backed by whisper.cpp (via the `whisper-rs` crate) on Metal.
//!
//! This is the non-gated default ASR path. whisper.cpp handles the full
//! pipeline — log-mel, encoder/decoder, the multilingual tokenizer, language
//! detection and decoding — so we only download a ggml model and feed it
//! 16 kHz mono f32 samples (produced by `read_wav_16k_mono`, which resamples).

use anyhow::{anyhow, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::Transcription;

/// Whisper checkpoints live in ggerganov/whisper.cpp as ggml files.
const GGML_REPO: &str = "ggerganov/whisper.cpp";

pub fn is_whisper_model(name: &str) -> bool {
    name.to_lowercase().contains("whisper")
}

/// Map an mlx-community / openai whisper model id to a ggml file name.
fn ggml_file_for(model_id: &str) -> &'static str {
    let m = model_id.to_lowercase();
    if m.contains("large-v3-turbo") || m.contains("turbo") {
        "ggml-large-v3-turbo.bin"
    } else if m.contains("large-v3") || m.contains("large") {
        "ggml-large-v3.bin"
    } else if m.contains("medium") {
        "ggml-medium.bin"
    } else if m.contains("small") {
        "ggml-small.bin"
    } else if m.contains("base") {
        "ggml-base.bin"
    } else if m.contains("tiny") {
        "ggml-tiny.bin"
    } else {
        "ggml-large-v3-turbo.bin"
    }
}

pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    /// Download (if needed) the ggml file for `model_id` into the shared hub
    /// cache and load it on the GPU (Metal).
    pub fn load(hub_cache_dir: &Path, model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::ApiBuilder;
        let file = ggml_file_for(model_id);
        let api = ApiBuilder::new()
            .with_cache_dir(hub_cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|e| anyhow!("hf-hub init: {e}"))?;
        let path = api
            .model(GGML_REPO.to_string())
            .get(file)
            .map_err(|e| anyhow!("fetch {file} from {GGML_REPO}: {e}"))?;

        let mut params = WhisperContextParameters::default();
        params.use_gpu(true);
        let ctx = WhisperContext::new_with_params(
            path.to_str().ok_or_else(|| anyhow!("bad model path"))?,
            params,
        )
        .map_err(|e| anyhow!("whisper context: {e}"))?;
        log::info!("Whisper model loaded: {file}");
        Ok(Self { ctx })
    }

    /// Transcribe mono 16 kHz f32 samples in [-1, 1].
    pub fn transcribe_samples(&self, samples: &[f32], language: &str) -> Result<Transcription> {
        let lang = language.trim().to_lowercase();
        if samples.is_empty() {
            return Ok(Transcription {
                text: String::new(),
                language: lang,
            });
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow!("whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let lang_opt = if lang.is_empty() || lang == "auto" {
            None // let whisper detect
        } else {
            Some(lang.as_str())
        };
        params.set_language(lang_opt);
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        params.set_n_threads(threads);

        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        log::info!(
            "Whisper transcribe: {} samples (~{:.2}s @16k), rms={:.5}, lang={}",
            samples.len(),
            samples.len() as f32 / 16000.0,
            rms,
            lang_opt.unwrap_or("auto")
        );

        state
            .full(params, samples)
            .map_err(|e| anyhow!("whisper full: {e}"))?;

        let n = state
            .full_n_segments()
            .map_err(|e| anyhow!("n_segments: {e}"))?;
        let mut text = String::new();
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                text.push_str(&seg);
            }
        }

        Ok(Transcription {
            text: text.trim().to_string(),
            language: if lang.is_empty() { "auto".to_string() } else { lang },
        })
    }

    pub fn transcribe_wav(&self, wav_path: &str, language: &str) -> Result<Transcription> {
        let samples = crate::read_wav_16k_mono(wav_path)?;
        self.transcribe_samples(&samples, language)
    }

    /// Warm up with a short silent buffer to compile Metal kernels.
    pub fn warmup(&self) -> Result<()> {
        let _ = self.transcribe_samples(&vec![0.0f32; 16000], "en")?;
        Ok(())
    }
}
