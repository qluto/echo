//! ASR engine — fully in-process (no Python sidecar).
//!
//! All speech recognition (Whisper via whisper.cpp, Parakeet-JA and Cohere via
//! Apple MLX) and LLM post-processing (Qwen3 via MLX) run natively in Rust
//! through the `rust-asr` crate. `ASREngine` owns the loaded engines and
//! dispatches by the active model id. The dispatch is internal so all call
//! sites (hotkey, commands, always-on pipeline) are unchanged.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};

use rust_asr::{CohereEngine, ParakeetEngine, PostProcessor, WhisperEngine};

use crate::{TranscriptionResult, TranscriptionSegment};

/// ASR models the picker offers; all are served in-process.
const AVAILABLE_MODELS: [&str; 8] = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3",
    "mlx-community/whisper-medium",
    "mlx-community/whisper-small",
    "mlx-community/whisper-base",
    "mlx-community/whisper-tiny",
    "mlx-community/parakeet-tdt_ctc-0.6b-ja",
    "CohereLabs/cohere-transcribe-03-2026",
];

/// Post-processing LLMs (Qwen3) served by the in-process engine.
const POSTPROCESS_MODELS: [&str; 3] = [
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
];
const DEFAULT_POSTPROCESS_MODEL: &str = "mlx-community/Qwen3-4B-4bit";

// ===== Public status types (consumed by Tauri commands / the frontend) =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub model_name: String,
    pub loaded: bool,
    pub loading: bool,
    pub error: Option<String>,
    pub available_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResult {
    pub success: bool,
    pub warmup_time_ms: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheStatus {
    pub cached: bool,
    pub model_name: String,
}

/// In-process ASR + post-processing engine manager.
pub struct ASREngine {
    cohere: Option<CohereEngine>,
    cohere_loaded: bool,
    whisper: Option<WhisperEngine>,
    whisper_loaded: bool,
    parakeet: Option<ParakeetEngine>,
    parakeet_loaded: bool,
    postproc: Option<PostProcessor>,
    postproc_loaded: bool,
    postproc_model: String,
    /// Currently selected ASR model id (drives dispatch).
    active_model: String,
    /// HF hub cache root (`<appcache>/huggingface/hub`), captured at start().
    hub_dir: Option<PathBuf>,
    /// HF token for gated downloads (Cohere).
    hf_token: Option<String>,
}

impl ASREngine {
    pub fn new() -> Self {
        Self {
            cohere: None,
            cohere_loaded: false,
            whisper: None,
            whisper_loaded: false,
            parakeet: None,
            parakeet_loaded: false,
            postproc: None,
            postproc_loaded: false,
            postproc_model: DEFAULT_POSTPROCESS_MODEL.to_string(),
            active_model: String::new(),
            hub_dir: None,
            hf_token: None,
        }
    }

    /// Resolve the HF cache dir + token. Cheap and synchronous — there is no
    /// sidecar process to spawn. Idempotent.
    pub fn start(&mut self, app: &AppHandle, hf_token: Option<&str>) -> Result<()> {
        let cache_dir = app
            .path()
            .resolve("huggingface", BaseDirectory::AppCache)
            .map_err(|e| anyhow!("Failed to resolve cache directory: {}", e))?;
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            log::warn!("Failed to create cache directory {:?}: {}", cache_dir, e);
        }
        log::info!("ASR engine ready (in-process). Cache: {:?}", cache_dir);
        self.hub_dir = Some(cache_dir.join("hub"));
        self.hf_token = hf_token.filter(|t| !t.is_empty()).map(|t| t.to_string());
        Ok(())
    }

    /// Free all loaded engines and release their GPU memory.
    pub fn stop(&mut self) -> Result<()> {
        self.cohere = None;
        self.cohere_loaded = false;
        self.whisper = None;
        self.whisper_loaded = false;
        self.parakeet = None;
        self.parakeet_loaded = false;
        self.postproc = None;
        self.postproc_loaded = false;
        rust_asr::release_unused_memory();
        Ok(())
    }

    /// "Running" once start() has captured the cache dir.
    pub fn ping(&mut self) -> Result<bool> {
        Ok(self.hub_dir.is_some())
    }

    fn hub(&self) -> Result<PathBuf> {
        self.hub_dir
            .clone()
            .ok_or_else(|| anyhow!("Engine not started; call start_asr_engine first"))
    }

    fn status(&self, loaded: bool) -> ModelStatus {
        ModelStatus {
            model_name: self.active_model.clone(),
            loaded,
            loading: false,
            error: None,
            available_models: AVAILABLE_MODELS.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn active_loaded(&self) -> bool {
        if rust_asr::is_whisper_model(&self.active_model) {
            self.whisper_loaded
        } else if rust_asr::is_parakeet_model(&self.active_model) {
            self.parakeet_loaded
        } else if rust_asr::is_cohere_model(&self.active_model) {
            self.cohere_loaded
        } else {
            false
        }
    }

    /// HF hub cache repo dir name for a model id (for local cache checks).
    fn cache_repo(name: &str) -> Option<&'static str> {
        if rust_asr::is_cohere_model(name) {
            Some("models--CohereLabs--cohere-transcribe-03-2026")
        } else if rust_asr::is_whisper_model(name) {
            Some("models--ggerganov--whisper.cpp")
        } else if rust_asr::is_parakeet_model(name) {
            Some("models--mlx-community--parakeet-tdt_ctc-0.6b-ja")
        } else {
            None
        }
    }

    // ===== ASR model operations =====

    pub fn get_model_status(&mut self) -> Result<ModelStatus> {
        Ok(self.status(self.active_loaded()))
    }

    pub fn is_model_cached(&mut self, model_name: Option<&str>) -> Result<ModelCacheStatus> {
        let target = model_name.unwrap_or(&self.active_model).to_string();
        let cached = match (Self::cache_repo(&target), self.hub_dir.as_ref()) {
            (Some(repo), Some(hub)) => hub.join(repo).join("snapshots").exists(),
            _ => false,
        };
        Ok(ModelCacheStatus {
            cached,
            model_name: target,
        })
    }

    /// Select the active ASR model. Frees the previously-loaded engine.
    pub fn set_model(&mut self, model_name: &str) -> Result<ModelStatus> {
        if !Self::is_supported(model_name) {
            return Err(anyhow!(
                "Model '{}' is not supported by the in-process engine. \
                 Choose Whisper, Parakeet (JA) or Cohere.",
                model_name
            ));
        }
        self.active_model = model_name.to_string();
        // Drop any loaded ASR engine; the selected one loads on next load_model.
        self.cohere = None;
        self.cohere_loaded = false;
        self.whisper = None;
        self.whisper_loaded = false;
        self.parakeet = None;
        self.parakeet_loaded = false;
        // Return the freed model's GPU buffers to the OS (MLX caches them).
        rust_asr::release_unused_memory();
        Ok(self.status(false))
    }

    fn is_supported(name: &str) -> bool {
        rust_asr::is_whisper_model(name)
            || rust_asr::is_parakeet_model(name)
            || rust_asr::is_cohere_model(name)
    }

    /// Load the active ASR model into its in-process engine.
    pub fn load_model(&mut self) -> Result<ModelStatus> {
        if self.active_loaded() {
            return Ok(self.status(true));
        }
        let hub = self.hub()?;
        let model = self.active_model.clone();

        if rust_asr::is_whisper_model(&model) {
            log::info!("Loading Whisper (whisper.cpp / Metal): {model}");
            let eng = WhisperEngine::load(&hub, &model)
                .map_err(|e| anyhow!("Failed to load Whisper: {e}"))?;
            self.whisper = Some(eng);
            self.whisper_loaded = true;
        } else if rust_asr::is_parakeet_model(&model) {
            log::info!("Loading Parakeet (full Rust / MLX): {model}");
            let eng =
                ParakeetEngine::load(&hub).map_err(|e| anyhow!("Failed to load Parakeet: {e}"))?;
            self.parakeet = Some(eng);
            self.parakeet_loaded = true;
        } else if rust_asr::is_cohere_model(&model) {
            log::info!("Loading Cohere (full Rust / MLX): {model}");
            let eng = CohereEngine::load(&hub, self.hf_token.as_deref())
                .map_err(|e| anyhow!("Failed to load Cohere: {e}"))?;
            self.cohere = Some(eng);
            self.cohere_loaded = true;
        } else {
            return Err(anyhow!("Unsupported model: {model}"));
        }
        Ok(self.status(true))
    }

    /// VAD for the always-on pipeline runs natively in `vad.rs`; these are
    /// no-ops kept for command compatibility.
    pub fn load_vad(&mut self) -> Result<()> {
        Ok(())
    }
    pub fn warmup_vad(&mut self) -> Result<WarmupResult> {
        Ok(WarmupResult {
            success: true,
            warmup_time_ms: Some(0.0),
            error: None,
        })
    }

    /// Warm up the active ASR engine (compile Metal kernels).
    pub fn warmup_model(&mut self) -> Result<WarmupResult> {
        let start = std::time::Instant::now();
        let result: Result<()> = if let Some(e) = self.whisper.as_ref() {
            e.warmup().map_err(Into::into)
        } else if let Some(e) = self.parakeet.as_ref() {
            e.warmup().map_err(Into::into)
        } else if let Some(e) = self.cohere.as_ref() {
            e.warmup().map_err(Into::into)
        } else {
            return Ok(WarmupResult {
                success: false,
                warmup_time_ms: None,
                error: Some("No ASR model loaded".to_string()),
            });
        };
        Ok(WarmupResult {
            success: result.is_ok(),
            warmup_time_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
            error: result.err().map(|e| e.to_string()),
        })
    }

    /// Update the HF token (for gated Cohere downloads).
    pub fn set_hf_token(&mut self, token: Option<&str>) -> Result<()> {
        self.hf_token = token.filter(|t| !t.is_empty()).map(|t| t.to_string());
        Ok(())
    }

    /// Transcribe an audio file with the active in-process engine.
    pub fn transcribe(
        &mut self,
        audio_path: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        if !self.active_loaded() {
            self.load_model()?;
        }
        let lang = language.unwrap_or("auto");
        let out = if let Some(e) = self.whisper.as_ref() {
            e.transcribe_wav(audio_path, lang)
        } else if let Some(e) = self.parakeet.as_ref() {
            e.transcribe_wav(audio_path, lang)
        } else if let Some(e) = self.cohere.as_ref() {
            e.transcribe_wav(audio_path, lang)
        } else {
            return Err(anyhow!("No ASR model loaded"));
        }
        .map_err(|e| anyhow!("Transcription failed: {e}"))?;

        let no_speech = if out.text.trim().is_empty() {
            Some(true)
        } else {
            None
        };
        log::info!("Transcription complete: {} chars", out.text.len());
        Ok(TranscriptionResult {
            success: true,
            text: out.text,
            segments: Vec::<TranscriptionSegment>::new(),
            language: out.language,
            no_speech,
        })
    }

    // ===== Post-processing (in-process Qwen3) =====

    fn postproc_status(&self, loaded: bool, error: Option<String>) -> crate::PostProcessModelStatus {
        crate::PostProcessModelStatus {
            model_name: self.postproc_model.clone(),
            loaded,
            loading: false,
            error,
            available_models: POSTPROCESS_MODELS.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn load_postprocess_model(&mut self) -> Result<crate::PostProcessModelStatus> {
        if self.postproc_loaded && self.postproc.is_some() {
            return Ok(self.postproc_status(true, None));
        }
        let hub = self.hub()?;
        log::info!("Loading post-processor (Qwen3): {}", self.postproc_model);
        match PostProcessor::load(&hub, &self.postproc_model) {
            Ok(pp) => {
                self.postproc = Some(pp);
                self.postproc_loaded = true;
                log::info!("Post-processor loaded");
                Ok(self.postproc_status(true, None))
            }
            Err(e) => {
                let msg = e.to_string();
                log::error!("Failed to load post-processor: {}", msg);
                Ok(self.postproc_status(false, Some(msg)))
            }
        }
    }

    pub fn unload_postprocess_model(&mut self) -> Result<()> {
        self.postproc = None;
        self.postproc_loaded = false;
        rust_asr::release_unused_memory();
        Ok(())
    }

    pub fn is_postprocess_model_cached(&mut self) -> Result<crate::PostProcessModelStatus> {
        let repo = format!("models--{}", self.postproc_model.replace('/', "--"));
        let cached = self
            .hub_dir
            .as_ref()
            .map(|hub| hub.join(&repo).join("snapshots").exists())
            .unwrap_or(false);
        Ok(self.postproc_status(cached, None))
    }

    pub fn set_postprocess_model(
        &mut self,
        model_name: &str,
    ) -> Result<crate::PostProcessModelStatus> {
        if !POSTPROCESS_MODELS.contains(&model_name) {
            return Err(anyhow!("Unknown post-process model: {}", model_name));
        }
        self.postproc_model = model_name.to_string();
        self.postproc = None;
        self.postproc_loaded = false;
        rust_asr::release_unused_memory();
        Ok(self.postproc_status(false, None))
    }

    pub fn get_postprocess_status(&mut self) -> Result<crate::PostProcessModelStatus> {
        Ok(self.postproc_status(self.postproc_loaded, None))
    }

    pub fn postprocess_text(
        &mut self,
        text: &str,
        app_name: Option<&str>,
        app_bundle_id: Option<&str>,
        dictionary: Option<&HashMap<String, String>>,
        custom_prompt: Option<&str>,
    ) -> Result<crate::PostProcessResult> {
        if text.trim().is_empty() {
            return Ok(crate::PostProcessResult {
                success: true,
                processed_text: String::new(),
                processing_time_ms: Some(0.0),
                error: None,
            });
        }
        if !self.postproc_loaded || self.postproc.is_none() {
            self.load_postprocess_model()?;
        }
        let pp = self
            .postproc
            .as_ref()
            .ok_or_else(|| anyhow!("Post-processor not loaded"))?;
        let start = std::time::Instant::now();
        match pp.process(text, app_name, app_bundle_id, dictionary, custom_prompt) {
            Ok(out) => Ok(crate::PostProcessResult {
                success: true,
                processed_text: out,
                processing_time_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
                error: None,
            }),
            Err(e) => Ok(crate::PostProcessResult {
                success: false,
                processed_text: text.to_string(),
                processing_time_ms: None,
                error: Some(e.to_string()),
            }),
        }
    }

    pub fn summarize_transcriptions(
        &mut self,
        texts: &[crate::SummarizeEntry],
        language_hint: Option<&str>,
        custom_prompt: Option<&str>,
    ) -> Result<crate::SummarizeResult> {
        if texts.is_empty() {
            return Ok(crate::SummarizeResult {
                success: true,
                summary: String::new(),
                processing_time_ms: Some(0.0),
                error: None,
                entry_count: 0,
            });
        }
        if !self.postproc_loaded || self.postproc.is_none() {
            self.load_postprocess_model()?;
        }
        let pp = self
            .postproc
            .as_ref()
            .ok_or_else(|| anyhow!("Post-processor not loaded"))?;
        let entries: Vec<(String, String)> = texts
            .iter()
            .map(|e| (e.created_at.clone(), e.text.clone()))
            .collect();
        let start = std::time::Instant::now();
        match pp.summarize(&entries, language_hint, custom_prompt) {
            Ok(summary) => Ok(crate::SummarizeResult {
                success: true,
                summary,
                processing_time_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
                error: None,
                entry_count: texts.len(),
            }),
            Err(e) => Ok(crate::SummarizeResult {
                success: false,
                summary: String::new(),
                processing_time_ms: None,
                error: Some(e.to_string()),
                entry_count: texts.len(),
            }),
        }
    }
}

impl Default for ASREngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod in_process_tests {
    use super::*;
    use std::path::PathBuf;

    fn hub() -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        Some(PathBuf::from(home).join("Library/Caches/io.qluto.echo/huggingface/hub"))
    }

    const WAV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/resources/start_v1.wav");

    fn engine_with_hub() -> Option<ASREngine> {
        let hub = hub()?;
        let mut e = ASREngine::new();
        e.hub_dir = Some(hub);
        Some(e)
    }

    #[test]
    fn cohere_routes_in_process() {
        let Some(mut engine) = engine_with_hub() else { return };
        if !engine
            .hub_dir
            .as_ref()
            .unwrap()
            .join("models--CohereLabs--cohere-transcribe-03-2026")
            .exists()
        {
            eprintln!("skipping: cohere not cached");
            return;
        }
        engine.set_model("CohereLabs/cohere-transcribe-03-2026").expect("set");
        assert!(engine.load_model().expect("load").loaded);
        let r = engine.transcribe(WAV, Some("en")).expect("transcribe");
        assert!(r.success);
    }

    #[test]
    fn whisper_routes_in_process() {
        let Some(mut engine) = engine_with_hub() else { return };
        engine.set_model("mlx-community/whisper-tiny").expect("set");
        let s = match engine.load_model() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping whisper-tiny load: {e}");
                return;
            }
        };
        assert!(s.loaded);
        assert!(engine.transcribe(WAV, Some("en")).expect("transcribe").success);
    }

    #[test]
    fn parakeet_routes_in_process() {
        let Some(mut engine) = engine_with_hub() else { return };
        if !engine
            .hub_dir
            .as_ref()
            .unwrap()
            .join("models--mlx-community--parakeet-tdt_ctc-0.6b-ja")
            .exists()
        {
            eprintln!("skipping: parakeet not cached");
            return;
        }
        engine
            .set_model("mlx-community/parakeet-tdt_ctc-0.6b-ja")
            .expect("set");
        assert!(engine.load_model().expect("load").loaded);
        assert!(engine.transcribe(WAV, Some("ja")).expect("transcribe").success);
    }

    #[test]
    fn postprocess_in_process() {
        let Some(mut engine) = engine_with_hub() else { return };
        if !engine
            .hub_dir
            .as_ref()
            .unwrap()
            .join("models--mlx-community--Qwen3-1.7B-4bit")
            .exists()
        {
            eprintln!("skipping: Qwen3-1.7B-4bit not cached");
            return;
        }
        engine
            .set_postprocess_model("mlx-community/Qwen3-1.7B-4bit")
            .expect("set postproc");
        assert!(engine.load_postprocess_model().expect("load").loaded);
        let r = engine
            .postprocess_text("えーと、3時に、いや4時に会議があります。", None, None, None, None)
            .expect("postprocess");
        assert!(r.success && !r.processed_text.trim().is_empty());
    }
}
