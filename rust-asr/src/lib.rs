//! Full-Rust Cohere Transcribe engine (cohere-transcribe-03-2026) on Apple MLX.
//!
//! Exposes [`CohereEngine`], a self-contained in-process ASR engine that
//! replaces the Python/MLX sidecar for the Cohere model: it locates (and, if
//! needed, downloads) the gated checkpoint in the shared HuggingFace hub cache,
//! then runs the FastConformer encoder + Transformer AED decoder on Metal.

pub mod audio;
pub mod decoder;
pub mod encoder;
pub mod hf;
pub mod llm;
pub mod nn;
pub mod parakeet;
pub mod tokenizer;
pub mod weights;
pub mod whisper;

pub use llm::PostProcessor;
pub use parakeet::{is_parakeet_model, ParakeetEngine};
pub use whisper::{is_whisper_model, WhisperEngine};

use anyhow::{anyhow, Result};
use std::path::Path;

use decoder::Decoder;
use encoder::Encoder;
use mlx_rs::Array;
use tokenizer::Tok;
use weights::Weights;

pub const COHERE_MODEL_ID: &str = "CohereLabs/cohere-transcribe-03-2026";

/// (active, cache) MLX GPU memory in bytes — for diagnostics.
pub fn mlx_memory() -> (usize, usize) {
    unsafe {
        let (mut active, mut cache) = (0usize, 0usize);
        mlx_sys::mlx_get_active_memory(&mut active as *mut usize);
        mlx_sys::mlx_get_cache_memory(&mut cache as *mut usize);
        (active, cache)
    }
}

/// Release MLX's cached (unused) GPU buffers back to the OS.
///
/// MLX keeps freed buffers in an allocator pool for reuse, so simply dropping a
/// loaded engine's arrays does not lower OS-visible memory. Call this *after*
/// dropping an engine (when switching models) to actually free that memory.
/// Only affects the MLX cache; in-use arrays and other backends are untouched.
pub fn release_unused_memory() {
    unsafe {
        let mut cache: usize = 0;
        mlx_sys::mlx_get_cache_memory(&mut cache as *mut usize);
        mlx_sys::mlx_clear_cache();
        if cache > 1024 * 1024 {
            log::info!(
                "Released ~{} MB of cached MLX GPU memory",
                cache / (1024 * 1024)
            );
        }
    }
}

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
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        log::info!(
            "Cohere transcribe: {} samples (~{:.2}s @16k), rms={:.5}, lang={}",
            samples.len(),
            samples.len() as f32 / 16000.0,
            rms,
            lang
        );
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
    // tokenizer.json is a small non-LFS file served via HF's relative
    // resolve-cache redirect (hf-hub 0.3.2 can't follow it); fetch it directly,
    // passing the token through for this gated repo.
    let tokenizer_json =
        hf::fetch_small_file(hub_cache_dir, COHERE_MODEL_ID, "tokenizer.json", hf_token)
            .map_err(|e| anyhow!("fetch tokenizer.json ({COHERE_MODEL_ID}): {e}"))?;
    Ok((safetensors, tokenizer_json))
}

/// Read a WAV as mono 16 kHz f32 in [-1, 1]. Hotkey recordings are written at
/// the input device's native rate (often 48 kHz) and channel count, so we
/// downmix to mono and resample to 16 kHz — mirroring the Python engine's
/// `load_audio(path, sr=16000)`.
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
    log::info!(
        "read wav: {} Hz, {} ch, {} frames -> resample to 16k",
        spec.sample_rate,
        spec.channels,
        mono.len()
    );
    resample_to_16k(&mono, spec.sample_rate)
}

/// High-quality sinc resample of a mono signal to 16 kHz (no-op at 16 kHz).
fn resample_to_16k(input: &[f32], from_rate: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;
    if from_rate == 16000 || input.is_empty() {
        return Ok(input.to_vec());
    }
    let params = rubato::SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: rubato::SincInterpolationType::Cubic,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let chunk = 1024usize;
    let mut resampler =
        rubato::SincFixedIn::<f32>::new(16000.0 / from_rate as f64, 1.0, params, chunk, 1)
            .map_err(|e| anyhow!("resampler init: {e}"))?;

    let need = resampler.input_frames_next();
    let mut out: Vec<f32> = Vec::with_capacity(input.len() * 16000 / from_rate as usize + chunk);
    let mut pos = 0usize;
    while pos < input.len() {
        let take = need.min(input.len() - pos);
        let mut buf = input[pos..pos + take].to_vec();
        if buf.len() < need {
            buf.resize(need, 0.0); // pad the final chunk with silence
        }
        let res = resampler
            .process(&[buf], None)
            .map_err(|e| anyhow!("resample: {e}"))?;
        out.extend_from_slice(&res[0]);
        pos += take;
    }
    // Drop output beyond the expected duration (trailing padding artifacts).
    let expected = (input.len() as f64 * 16000.0 / from_rate as f64).round() as usize;
    out.truncate(expected.min(out.len()));
    Ok(out)
}
