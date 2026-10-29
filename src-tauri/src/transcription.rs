//! ASR engine — fully in-process (no Python sidecar).
//!
//! All speech recognition (Whisper via whisper.cpp, Parakeet-JA and Cohere via
//! Apple MLX) and LLMs (post-processing, memo minutes, meeting notes; via MLX) run natively in Rust
//! through the `rust-asr` crate. `ASREngine` owns the loaded engines and
//! dispatches by the active model id. The dispatch is internal so all call
//! sites (hotkey, commands, always-on pipeline) are unchanged.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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

/// Post-processing LLMs served by the in-process engine. Gemma 4 E4B is the
/// default: on Japanese cleanup it applies self-corrections that Qwen3.5-4B
/// misses, and it is faster. The Qwen models stay selectable for installs
/// that already have them downloaded.
const POSTPROCESS_MODELS: [&str; 6] = [
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/Qwen3.5-4B-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
];
const DEFAULT_POSTPROCESS_MODEL: &str = "mlx-community/gemma-4-e4b-it-4bit";
/// LLMs for voice-memo minutes and meeting notes, chosen separately from the
/// dictation post-processor: these favor quality over speed. Qwen3.6-27B
/// (Qwen3.5 architecture, ~16 GB) got 22/23 on the meeting-notes evaluation
/// against 17 for Gemma 4 E4B, which left a real meeting's notes empty; it
/// takes ~10–30 s per revision, too slow for dictation.
const NOTES_ONLY_MODELS: [&str; 1] = ["mlx-community/Qwen3.6-27B-4bit"];

fn notes_models() -> impl Iterator<Item = &'static str> {
    NOTES_ONLY_MODELS.into_iter().chain(POSTPROCESS_MODELS)
}

/// Qwen3.6-27B where it fits next to ASR, a meeting app and the OS; Gemma 4
/// E4B on smaller Macs.
fn default_notes_model() -> &'static str {
    const MIN_BYTES_FOR_27B: u64 = 32 * 1024 * 1024 * 1024;
    if physical_memory().is_some_and(|m| m >= MIN_BYTES_FOR_27B) {
        NOTES_ONLY_MODELS[0]
    } else {
        DEFAULT_POSTPROCESS_MODEL
    }
}

fn physical_memory() -> Option<u64> {
    extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> std::ffi::c_int;
    }
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let ok = unsafe {
        sysctlbyname(c"hw.memsize".as_ptr(), &mut bytes as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0)
    } == 0;
    ok.then_some(bytes)
}

/// Cap for MLX's freed-buffer cache (see `ASREngine::start`).
const MLX_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;

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
    /// Loaded LLMs by model id: the post-processor's and the notes model's
    /// (one instance when they are the same). Shared so a generation runs
    /// without the engine lock — ASR keeps going meanwhile (rust-asr only
    /// holds its GPU lock per decode step).
    llms: HashMap<String, Arc<PostProcessor>>,
    postproc_model: String,
    notes_model: String,
    /// Currently selected ASR model id (drives dispatch).
    active_model: String,
    /// HF hub cache root (`<appcache>/huggingface/hub`), captured at start().
    hub_dir: Option<PathBuf>,
    /// HF token for gated downloads (Cohere).
    hf_token: Option<String>,
    /// Silero VAD, lazily created, to gate out silent recordings before ASR
    /// (otherwise a "pressed hotkey, said nothing" clip hallucinates).
    vad: Option<crate::vad::VadProcessor>,
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
            llms: HashMap::new(),
            postproc_model: DEFAULT_POSTPROCESS_MODEL.to_string(),
            notes_model: default_notes_model().to_string(),
            active_model: String::new(),
            hub_dir: None,
            hf_token: None,
            vad: None,
        }
    }

    /// Engine pointed at an existing HF hub cache, without a Tauri app.
    #[cfg(test)]
    pub(crate) fn with_hub(hub_dir: PathBuf) -> Self {
        Self {
            hub_dir: Some(hub_dir),
            ..Self::new()
        }
    }

    /// Currently selected ASR model id. Empty until set_model is called.
    pub fn active_model_name(&self) -> &str {
        &self.active_model
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
        // Cap MLX's freed-buffer cache. Without a limit (default ≈ device
        // memory), continuous transcription grows process memory unboundedly:
        // every variable-length segment leaves differently-sized buffers in
        // the cache. 512 MB keeps steady-state buffer reuse while returning
        // the excess to the OS.
        let prev = rust_asr::set_cache_limit(MLX_CACHE_LIMIT_BYTES);
        log::info!(
            "MLX cache limit set to {} MB (was {} MB)",
            MLX_CACHE_LIMIT_BYTES / (1024 * 1024),
            prev / (1024 * 1024)
        );
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
        self.llms.clear();
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

    /// Pre-create the Silero VAD used to gate silent recordings (called from
    /// load_asr_model so the first transcription isn't slowed by VAD init).
    pub fn load_vad(&mut self) -> Result<()> {
        if self.vad.is_none() {
            match crate::vad::VadProcessor::new(0.5) {
                Ok(v) => self.vad = Some(v),
                Err(e) => log::warn!("VAD init failed: {e}"),
            }
        }
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
        // Read once at 16 kHz mono (also what the engines need).
        let samples =
            rust_asr::read_wav_16k_mono(audio_path).map_err(|e| anyhow!("read audio: {e}"))?;
        self.transcribe_samples(&samples, language, true)
    }

    /// Transcribe 16 kHz mono f32 samples with the active in-process engine.
    ///
    /// `vad_gate`: run Silero VAD first and skip ASR when there is no speech
    /// (so silence doesn't hallucinate). In-progress partial decodes pass
    /// `false`: a draft of silence is harmless and the extra pass costs time.
    pub fn transcribe_samples(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        vad_gate: bool,
    ) -> Result<TranscriptionResult> {
        if !self.active_loaded() {
            self.load_model()?;
        }

        if vad_gate && !self.has_speech(samples) {
            log::info!("No speech detected (VAD); skipping transcription");
            return Ok(TranscriptionResult {
                success: true,
                text: String::new(),
                segments: Vec::new(),
                language: language.unwrap_or("auto").to_string(),
                no_speech: Some(true),
                raw_text: None,
            });
        }

        let lang = language.unwrap_or("auto");
        let out = if let Some(e) = self.whisper.as_ref() {
            e.transcribe_samples(samples, lang)
        } else if let Some(e) = self.parakeet.as_ref() {
            e.transcribe_samples(samples, lang)
        } else if let Some(e) = self.cohere.as_ref() {
            e.transcribe_samples(samples, lang)
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
            raw_text: None,
        })
    }

    /// Whether the active, already-loaded ASR model is suitable for cheap
    /// in-progress (partial) decoding. whisper.cpp is too slow per call and
    /// hallucinates on truncated audio; the MLX encoder models are fine.
    pub fn supports_partial(&self) -> bool {
        self.whisper.is_none() && (self.parakeet.is_some() || self.cohere.is_some())
    }

    /// Run Silero VAD over the whole 16 kHz clip; speech if enough frames clear
    /// the threshold. Always-on segments (already speech) pass; silent hotkey
    /// recordings are rejected.
    fn has_speech(&mut self, samples: &[f32]) -> bool {
        use crate::vad::{VadEvent, VadProcessor, VAD_FRAME_SIZE};
        const MIN_SPEECH_FRAMES: usize = 3; // ~96 ms of speech

        if self.vad.is_none() {
            match VadProcessor::new(0.5) {
                Ok(v) => self.vad = Some(v),
                Err(e) => {
                    log::warn!("VAD init failed ({e}); skipping speech gate");
                    return true; // fail open — better to transcribe than to drop speech
                }
            }
        }
        let vad = self.vad.as_mut().unwrap();
        vad.reset();

        let mut speech_frames = 0usize;
        let mut max_prob = 0.0f32;
        for frame in samples.chunks_exact(VAD_FRAME_SIZE) {
            if let VadEvent::Speech { probability } = vad.process_frame(frame) {
                speech_frames += 1;
                if probability > max_prob {
                    max_prob = probability;
                }
                if speech_frames >= MIN_SPEECH_FRAMES {
                    return true;
                }
            }
        }
        log::debug!(
            "VAD: {} speech frames (max prob {:.2}) — below gate",
            speech_frames,
            max_prob
        );
        false
    }

    // ===== LLMs: dictation post-processing, memo minutes / meeting notes =====

    fn postproc_status(&self, loaded: bool, error: Option<String>) -> crate::PostProcessModelStatus {
        crate::PostProcessModelStatus {
            model_name: self.postproc_model.clone(),
            loaded,
            loading: false,
            error,
            available_models: POSTPROCESS_MODELS.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Drop loaded LLMs that are no longer selected for anything.
    fn prune_llms(&mut self) {
        let before = self.llms.len();
        let (pp, notes) = (&self.postproc_model, &self.notes_model);
        self.llms.retain(|id, _| id == pp || id == notes);
        if self.llms.len() < before {
            rust_asr::release_unused_memory();
        }
    }

    fn is_cached(&self, model: &str) -> bool {
        let repo = format!("models--{}", model.replace('/', "--"));
        self.hub_dir
            .as_ref()
            .is_some_and(|hub| hub.join(&repo).join("snapshots").exists())
    }

    /// The LLM `pick` selects, loaded (and downloaded) if needed without
    /// holding the engine lock, so ASR is never stalled by an LLM.
    fn shared_llm(engine: &Mutex<Self>, pick: fn(&Self) -> &str) -> Result<Arc<PostProcessor>> {
        let (model, hub) = {
            let e = engine.lock().map_err(|e| anyhow!("{e}"))?;
            let model = pick(&e).to_string();
            if let Some(llm) = e.llms.get(&model) {
                return Ok(Arc::clone(llm));
            }
            (model, e.hub()?)
        };
        log::info!("Loading LLM: {model}");
        let llm = Arc::new(PostProcessor::load(&hub, &model).map_err(|e| anyhow!("Failed to load {model}: {e}"))?);
        let mut e = engine.lock().map_err(|e| anyhow!("{e}"))?;
        // Another thread may have loaded it meanwhile: keep a single copy.
        let llm = Arc::clone(e.llms.entry(model).or_insert(llm));
        // If the selection changed during the load, this call still gets its model.
        e.prune_llms();
        Ok(llm)
    }

    /// The dictation post-processor.
    pub fn postprocessor(engine: &Mutex<Self>) -> Result<Arc<PostProcessor>> {
        Self::shared_llm(engine, |e| &e.postproc_model)
    }

    /// The LLM for voice-memo minutes and meeting notes. Generate with it after
    /// the engine lock is released.
    pub fn notes_llm(engine: &Mutex<Self>) -> Result<Arc<PostProcessor>> {
        Self::shared_llm(engine, |e| &e.notes_model)
    }

    /// Free the notes model after a memo or meeting, unless dictation uses it too.
    pub fn release_notes_model(&mut self) {
        if self.notes_model != self.postproc_model && self.llms.remove(&self.notes_model).is_some() {
            rust_asr::release_unused_memory();
        }
    }

    pub fn load_postprocess_model(&mut self) -> Result<crate::PostProcessModelStatus> {
        if self.llms.contains_key(&self.postproc_model) {
            return Ok(self.postproc_status(true, None));
        }
        let hub = self.hub()?;
        log::info!("Loading post-processor: {}", self.postproc_model);
        match PostProcessor::load(&hub, &self.postproc_model) {
            Ok(pp) => {
                self.llms.insert(self.postproc_model.clone(), Arc::new(pp));
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
        if self.llms.remove(&self.postproc_model).is_some() {
            rust_asr::release_unused_memory();
        }
        Ok(())
    }

    pub fn is_postprocess_model_cached(&mut self) -> Result<crate::PostProcessModelStatus> {
        Ok(self.postproc_status(self.is_cached(&self.postproc_model), None))
    }

    pub fn set_postprocess_model(
        &mut self,
        model_name: &str,
    ) -> Result<crate::PostProcessModelStatus> {
        if !POSTPROCESS_MODELS.contains(&model_name) {
            return Err(anyhow!("Unknown post-process model: {}", model_name));
        }
        self.postproc_model = model_name.to_string();
        self.prune_llms();
        Ok(self.postproc_status(self.llms.contains_key(model_name), None))
    }

    pub fn get_postprocess_status(&mut self) -> Result<crate::PostProcessModelStatus> {
        Ok(self.postproc_status(self.llms.contains_key(&self.postproc_model), None))
    }

    /// Selected notes model; `loaded` reports whether it is downloaded.
    pub fn notes_model_status(&self) -> crate::PostProcessModelStatus {
        crate::PostProcessModelStatus {
            model_name: self.notes_model.clone(),
            loaded: self.is_cached(&self.notes_model),
            loading: false,
            error: None,
            available_models: notes_models().map(String::from).collect(),
        }
    }

    pub fn set_notes_model(&mut self, model_name: &str) -> Result<crate::PostProcessModelStatus> {
        if !notes_models().any(|m| m == model_name) {
            return Err(anyhow!("Unknown notes model: {}", model_name));
        }
        self.notes_model = model_name.to_string();
        self.prune_llms();
        Ok(self.notes_model_status())
    }

    /// Clean up dictated text with the post-processor. The generation runs
    /// without the engine lock.
    pub fn postprocess_text(
        engine: &Mutex<Self>,
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
        let pp = Self::postprocessor(engine)?;
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

    /// Summarize history entries with the post-processor, without the engine lock.
    pub fn summarize_transcriptions(
        engine: &Mutex<Self>,
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
        let pp = Self::postprocessor(engine)?;
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
    fn vad_gates_silence_but_passes_speech() {
        let mut e = ASREngine::new();
        // 1s of pure silence -> not speech
        assert!(!e.has_speech(&vec![0.0f32; 16000]));
        // a real speech clip (synthesized earlier this session) -> speech
        if let Ok(samples) = rust_asr::read_wav_16k_mono("/tmp/ja.wav") {
            assert!(e.has_speech(&samples), "ja.wav should be detected as speech");
        } else {
            eprintln!("skipping speech half: /tmp/ja.wav not present");
        }
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
        let engine = Mutex::new(engine);
        let r = ASREngine::postprocess_text(&engine, "えーと、3時に、いや4時に会議があります。", None, None, None, None)
            .expect("postprocess");
        assert!(r.success && !r.processed_text.trim().is_empty());
    }

    /// Transcription through the engine mutex keeps going while the notes LLM
    /// generates on another thread (as during a meeting round):
    /// `cargo test --release transcription_continues_during_notes_llm -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn transcription_continues_during_notes_llm() {
        use std::time::{Duration, Instant};
        let Some(mut engine) = engine_with_hub() else { return };
        engine.set_model("mlx-community/parakeet-tdt_ctc-0.6b-ja").expect("set");
        engine.load_model().expect("load");
        let samples = rust_asr::read_wav_16k_mono(WAV).expect("wav");
        engine.transcribe_samples(&samples, Some("ja"), false).expect("warm up");
        let engine = Arc::new(Mutex::new(engine));
        ASREngine::notes_llm(&engine).expect("notes llm"); // load before timing

        let llm = {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                let t = Instant::now();
                ASREngine::notes_llm(&engine)
                    .and_then(|llm| llm.chat("Write a 30-item checklist.", "Japanese, one line each.", 300))
                    .expect("chat");
                t.elapsed()
            })
        };
        std::thread::sleep(Duration::from_millis(500));
        let mut worst = Duration::ZERO;
        let mut chunks = 0;
        while !llm.is_finished() {
            let t = Instant::now();
            engine.lock().unwrap().transcribe_samples(&samples, Some("ja"), false).expect("transcribe");
            worst = worst.max(t.elapsed());
            chunks += 1;
            std::thread::sleep(Duration::from_secs(2));
        }
        let llm_time = llm.join().unwrap();
        println!("{chunks} chunks during a {llm_time:?} generation; slowest chunk {worst:?}");
        assert!(chunks >= 2);
        assert!(worst < Duration::from_secs(1), "transcription waited on the LLM: {worst:?}");
    }

    /// Downloads the default post-processor (Gemma 4 E4B, ~5 GB) on first run:
    /// `cargo test default_postprocess -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn default_postprocess() {
        let Some(mut engine) = engine_with_hub() else { return };
        assert_eq!(engine.postproc_model, DEFAULT_POSTPROCESS_MODEL);
        let t = std::time::Instant::now();
        assert!(engine.load_postprocess_model().expect("load").loaded);
        println!("loaded in {:?}", t.elapsed());
        let engine = Mutex::new(engine);
        for text in [
            "えーと、3時に、いや4時に会議があります。",
            "あの、資料はですね、まあ明日までに送ります。",
            "um so I think we should uh ship it on Monday, no wait, Tuesday",
        ] {
            let r = ASREngine::postprocess_text(&engine, text, None, None, None, None).expect("postprocess");
            println!("in:  {text}\nout: {} ({:.0} ms)", r.processed_text, r.processing_time_ms.unwrap_or(0.0));
            assert!(r.success && !r.processed_text.trim().is_empty());
        }
    }
}
