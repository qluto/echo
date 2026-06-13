//! ASR Engine module for communicating with the Python sidecar
//!
//! This module manages the ASR (Automatic Speech Recognition) engine
//! which runs as a separate Python process using MLX-Audio.

use anyhow::{anyhow, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};

use rust_asr::{CohereEngine, ParakeetEngine, PostProcessor, WhisperEngine};

/// Post-processing LLMs supported by the in-process Qwen3 engine.
const POSTPROCESS_MODELS: [&str; 3] = [
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
];
const DEFAULT_POSTPROCESS_MODEL: &str = "mlx-community/Qwen3-4B-4bit";

use crate::{TranscriptionResult, TranscriptionSegment};

// ===== Protocol types =====

/// Request sent to the ASR engine
#[derive(Debug, Serialize)]
struct ASRRequest {
    command: String,
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hf_token: Option<String>,
}

impl ASRRequest {
    fn new(command: &str, id: u64) -> Self {
        Self {
            command: command.to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: None,
            hf_token: None,
        }
    }
}

/// Generic JSON-RPC response wrapper
#[derive(Debug, Deserialize)]
struct EngineResponse<T> {
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<T>,
    error: Option<String>,
}

impl<T> EngineResponse<T> {
    fn into_result(self, context: &str) -> Result<T> {
        if let Some(error) = self.error {
            return Err(anyhow!("{}: {}", context, error));
        }
        self.result.ok_or_else(|| anyhow!("No result in response"))
    }
}

/// Status response (used during startup)
#[derive(Debug, Deserialize)]
struct StatusResponse {
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ASRResultInner {
    success: bool,
    text: String,
    segments: Vec<SegmentInner>,
    language: String,
    error: Option<String>,
    no_speech: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SegmentInner {
    start: f64,
    end: f64,
    text: String,
}

/// Model status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub model_name: String,
    pub loaded: bool,
    pub loading: bool,
    pub error: Option<String>,
    pub available_models: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelOperationResult {
    success: Option<bool>,
    model_name: Option<String>,
    loaded: Option<bool>,
    loading: Option<bool>,
    error: Option<String>,
    available_models: Option<Vec<String>>,
    already_loaded: Option<bool>,
}

/// Warmup result response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResult {
    pub success: bool,
    pub warmup_time_ms: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WarmupResponseResult {
    success: Option<bool>,
    warmup_time_ms: Option<f64>,
    error: Option<String>,
}

/// Model cache check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheStatus {
    pub cached: bool,
    pub model_name: String,
}

#[derive(Debug, Deserialize)]
struct CacheCheckResult {
    cached: Option<bool>,
    model_name: Option<String>,
}

// ===== Process management =====

/// Process handle with I/O streams
struct ASRProcess {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl ASRProcess {
    /// Send a JSON-RPC request and deserialize the response
    fn send_command<T: DeserializeOwned>(&mut self, request: &impl Serialize) -> Result<T> {
        let json = serde_json::to_string(request)?;
        writeln!(self.stdin, "{}", json)?;
        self.stdin.flush()?;
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .map_err(|e| anyhow!("Failed to read response: {}", e))?;
        serde_json::from_str(&line).map_err(|e| anyhow!("Failed to parse response: {}", e))
    }
}

/// ASR Engine manager.
///
/// Routes ASR work for the Cohere model to an in-process full-Rust engine
/// ([`CohereEngine`]) and everything else (other models, VAD, post-processing,
/// summarization) to the Python sidecar. The dispatch is internal so all call
/// sites (hotkey, commands, always-on pipeline) are unchanged.
pub struct ASREngine {
    process: Option<ASRProcess>,
    request_id: AtomicU64,
    /// In-process Cohere engine, loaded on demand when the Cohere model is active.
    cohere: Option<CohereEngine>,
    cohere_loaded: bool,
    /// In-process Whisper engine (whisper.cpp), the non-gated default path.
    whisper: Option<WhisperEngine>,
    whisper_loaded: bool,
    /// In-process Parakeet engine (full-Rust MLX), Japanese-specialized.
    parakeet: Option<ParakeetEngine>,
    parakeet_loaded: bool,
    /// In-process post-processing LLM (Qwen3, full-Rust MLX).
    postproc: Option<PostProcessor>,
    postproc_loaded: bool,
    postproc_model: String,
    /// Currently selected ASR model id (drives the in-process vs sidecar dispatch).
    active_model: String,
    /// HF hub cache root (`<appcache>/huggingface/hub`), captured at start().
    cohere_hub_dir: Option<PathBuf>,
    /// HF token for gated download, captured at start() / set_hf_token().
    hf_token: Option<String>,
}

impl ASREngine {
    pub fn new() -> Self {
        Self {
            process: None,
            request_id: AtomicU64::new(1),
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
            cohere_hub_dir: None,
            hf_token: None,
        }
    }

    fn get_process(&mut self) -> Result<&mut ASRProcess> {
        self.process
            .as_mut()
            .ok_or_else(|| anyhow!("Engine not running"))
    }

    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the target triple for the current platform
    fn get_target_triple() -> &'static str {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            "aarch64-apple-darwin"
        }
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        {
            "x86_64-apple-darwin"
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            "x86_64-unknown-linux-gnu"
        }
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            "x86_64-pc-windows-msvc"
        }
    }

    /// Start the ASR engine.
    ///
    /// The Python sidecar is **optional**: the in-process Cohere engine needs
    /// only the HF cache dir + token (captured here up front). If the sidecar
    /// binary is missing/empty or fails to come up, we log a warning and return
    /// Ok with no process — Cohere still works fully in-process; other models,
    /// VAD and post-processing report errors only when actually used.
    pub fn start(&mut self, app: &AppHandle, hf_token: Option<&str>) -> Result<()> {
        if self.process.is_some() {
            log::info!("ASR engine already running");
            return Ok(());
        }

        // Resolve the app cache dir and capture what the in-process Cohere engine
        // needs FIRST, so it works regardless of the sidecar's fate.
        let cache_dir = app
            .path()
            .resolve("huggingface", BaseDirectory::AppCache)
            .map_err(|e| anyhow!("Failed to resolve cache directory: {}", e))?;
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            log::warn!("Failed to create cache directory {:?}: {}", cache_dir, e);
        }
        log::info!("Using model cache directory: {:?}", cache_dir);
        // hf-hub stores `models--*` under `<cache>/hub`.
        self.cohere_hub_dir = Some(cache_dir.join("hub"));
        self.hf_token = hf_token.filter(|t| !t.is_empty()).map(|t| t.to_string());

        // Locate a usable (existing, non-empty) sidecar binary.
        let binary_name_with_suffix = format!("mlx-asr-engine-{}", Self::get_target_triple());
        let binary_name_no_suffix = "mlx-asr-engine";
        let candidates = [
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                .map(|p| p.join(binary_name_no_suffix)),
            std::env::current_dir()
                .ok()
                .map(|p| p.join("binaries").join(&binary_name_with_suffix)),
            std::env::current_dir()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                .map(|p| p.join("src-tauri").join("binaries").join(&binary_name_with_suffix)),
        ];
        let non_empty = |p: &std::path::Path| {
            std::fs::metadata(p).map(|m| m.is_file() && m.len() > 0).unwrap_or(false)
        };
        let program = match candidates.into_iter().flatten().find(|p| non_empty(p)) {
            Some(p) => {
                log::info!("Using ASR sidecar binary: {:?}", p);
                p.to_string_lossy().to_string()
            }
            None => {
                log::warn!(
                    "Python ASR sidecar binary not found or empty. Cohere runs in-process \
                     (full Rust); other models, VAD and post-processing are unavailable until \
                     you build it via 'cd python-engine && ./build.sh'."
                );
                return Ok(());
            }
        };

        log::info!("Starting ASR engine: {} [\"daemon\"]", program);
        let mut cmd = Command::new(&program);
        cmd.arg("daemon")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("HF_HOME", &cache_dir)
            .env("TORCH_HOME", &cache_dir);
        if let Some(token) = hf_token.filter(|t| !t.is_empty()) {
            cmd.env("HF_TOKEN", token);
            log::info!("HF_TOKEN configured for ASR engine (length={})", token.len());
        }

        // Any sidecar failure below is non-fatal: log and continue without it.
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("Failed to start ASR sidecar ({}); continuing without it", e);
                return Ok(());
            }
        };
        let (stdin, stdout) = match (child.stdin.take(), child.stdout.take()) {
            (Some(i), Some(o)) => (i, o),
            _ => {
                log::warn!("Failed to get ASR sidecar I/O handles; continuing without it");
                child.kill().ok();
                return Ok(());
            }
        };
        let stdin_writer = BufWriter::new(stdin);
        let mut stdout_reader = BufReader::new(stdout);

        log::info!("Waiting for ASR engine ready signal...");
        let mut line = String::new();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(60);
        let ready = loop {
            if start.elapsed() > timeout {
                log::warn!("Timeout waiting for ASR sidecar; continuing without it");
                child.kill().ok();
                break false;
            }
            line.clear();
            match stdout_reader.read_line(&mut line) {
                Ok(0) => {
                    log::warn!("ASR sidecar closed stdout; continuing without it");
                    child.kill().ok();
                    break false;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    log::debug!("ASR engine output: {}", trimmed);
                    if let Ok(status) = serde_json::from_str::<StatusResponse>(trimmed) {
                        if status.status.as_deref() == Some("ready") {
                            log::info!("ASR engine ready");
                            break true;
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                Err(e) => {
                    log::warn!("Error reading from ASR sidecar ({}); continuing without it", e);
                    child.kill().ok();
                    break false;
                }
            }
        };

        if ready {
            self.process = Some(ASRProcess {
                child,
                stdin: stdin_writer,
                stdout: stdout_reader,
            });
        }
        Ok(())
    }

    /// Stop the ASR engine
    pub fn stop(&mut self) -> Result<()> {
        if let Some(mut process) = self.process.take() {
            let request = ASRRequest::new("quit", 0);
            if let Ok(json) = serde_json::to_string(&request) {
                let _ = writeln!(process.stdin, "{}", json);
                let _ = process.stdin.flush();
            }

            std::thread::sleep(Duration::from_millis(500));
            let _ = process.child.kill();
            let _ = process.child.wait();
            log::info!("ASR engine stopped");
        }
        Ok(())
    }

    // ===== ASR model operations =====

    /// Get model status from the engine
    pub fn get_model_status(&mut self) -> Result<ModelStatus> {
        // In-process models (Cohere, Whisper) may be active without a sidecar.
        if Self::is_in_process(&self.active_model) && self.process.is_none() {
            return Ok(self.in_process_status(self.in_process_loaded()));
        }
        let process = match self.process.as_mut() {
            Some(p) => p,
            None => {
                return Ok(ModelStatus {
                    model_name: "unknown".to_string(),
                    loaded: false,
                    loading: false,
                    error: Some("Engine not running".to_string()),
                    available_models: vec![],
                });
            }
        };

        let request = ASRRequest::new("get_status", self.request_id.fetch_add(1, Ordering::SeqCst));
        let response: EngineResponse<ModelOperationResult> = process.send_command(&request)?;
        let result = response.into_result("Failed to get status")?;

        let available_models = result.available_models.unwrap_or_default();

        // When an in-process model is active, report its state but keep the
        // sidecar's available_models so the model picker still works.
        if Self::is_in_process(&self.active_model) {
            return Ok(ModelStatus {
                model_name: self.active_model.clone(),
                loaded: self.in_process_loaded(),
                loading: false,
                error: None,
                available_models,
            });
        }

        Ok(ModelStatus {
            model_name: result.model_name.unwrap_or_else(|| "unknown".to_string()),
            loaded: result.loaded.unwrap_or(false),
            loading: result.loading.unwrap_or(false),
            error: result.error,
            available_models,
        })
    }

    /// Check if a model is cached locally
    pub fn is_model_cached(&mut self, model_name: Option<&str>) -> Result<ModelCacheStatus> {
        let target = model_name.unwrap_or(&self.active_model);

        // In-process models: check the shared HF hub cache directly (no Python).
        if let Some(repo_dir) = Self::in_process_cache_repo(target) {
            let cached = self
                .cohere_hub_dir
                .as_ref()
                .map(|hub| hub.join(repo_dir).join("snapshots").exists())
                .unwrap_or(false);
            return Ok(ModelCacheStatus {
                cached,
                model_name: target.to_string(),
            });
        }

        let mut request = ASRRequest::new("is_model_cached", self.next_id());
        request.model_name = model_name.map(|s| s.to_string());

        let response: EngineResponse<CacheCheckResult> =
            self.get_process()?.send_command(&request)?;
        let result = response.into_result("Failed to check cache")?;

        Ok(ModelCacheStatus {
            cached: result.cached.unwrap_or(false),
            model_name: result
                .model_name
                .unwrap_or_else(|| "unknown".to_string()),
        })
    }

    /// Load the model
    pub fn load_model(&mut self) -> Result<ModelStatus> {
        // Cohere → load the in-process full-Rust engine (no Python, fast start).
        if rust_asr::is_cohere_model(&self.active_model) {
            if self.cohere_loaded && self.cohere.is_some() {
                return Ok(self.in_process_status(true));
            }
            let hub = self
                .cohere_hub_dir
                .clone()
                .ok_or_else(|| anyhow!("Cohere cache dir not set; start the engine first"))?;
            let token = self.hf_token.clone();
            log::info!("Loading Cohere model in-process (full Rust / MLX)...");
            let engine = CohereEngine::load(&hub, token.as_deref())
                .map_err(|e| anyhow!("Failed to load Cohere engine: {}", e))?;
            self.cohere = Some(engine);
            self.cohere_loaded = true;
            log::info!("Cohere in-process engine loaded");
            return Ok(self.in_process_status(true));
        }

        // Whisper → load the in-process whisper.cpp engine (downloads the ggml
        // model on first use). Non-gated default path.
        if rust_asr::is_whisper_model(&self.active_model) {
            if self.whisper_loaded && self.whisper.is_some() {
                return Ok(self.in_process_status(true));
            }
            let hub = self
                .cohere_hub_dir
                .clone()
                .ok_or_else(|| anyhow!("Cache dir not set; start the engine first"))?;
            log::info!("Loading Whisper model in-process (whisper.cpp / Metal)...");
            let engine = WhisperEngine::load(&hub, &self.active_model)
                .map_err(|e| anyhow!("Failed to load Whisper engine: {}", e))?;
            self.whisper = Some(engine);
            self.whisper_loaded = true;
            log::info!("Whisper in-process engine loaded");
            return Ok(self.in_process_status(true));
        }

        // Parakeet → load the in-process full-Rust MLX engine (JA-specialized).
        if rust_asr::is_parakeet_model(&self.active_model) {
            if self.parakeet_loaded && self.parakeet.is_some() {
                return Ok(self.in_process_status(true));
            }
            let hub = self
                .cohere_hub_dir
                .clone()
                .ok_or_else(|| anyhow!("Cache dir not set; start the engine first"))?;
            log::info!("Loading Parakeet model in-process (full Rust / MLX)...");
            let engine = ParakeetEngine::load(&hub)
                .map_err(|e| anyhow!("Failed to load Parakeet engine: {}", e))?;
            self.parakeet = Some(engine);
            self.parakeet_loaded = true;
            log::info!("Parakeet in-process engine loaded");
            return Ok(self.in_process_status(true));
        }

        // Any other model needs the Python sidecar. Fail fast with a clear
        // message if it isn't running (rather than hanging the load screen).
        if self.process.is_none() {
            return Err(anyhow!(
                "Model '{}' is not supported by the in-process engine. Choose Whisper \
                 or Cohere, or build the Python sidecar (cd python-engine && ./build.sh).",
                self.active_model
            ));
        }

        let request = ASRRequest::new("load_model", self.next_id());
        let response: EngineResponse<ModelOperationResult> =
            self.get_process()?.send_command(&request)?;
        let result = response.into_result("Failed to load model")?;

        if result.success == Some(false) {
            return Err(anyhow!(
                "Failed to load model: {}",
                result.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        Ok(ModelStatus {
            model_name: result
                .model_name
                .unwrap_or_else(|| "unknown".to_string()),
            loaded: true,
            loading: false,
            error: None,
            available_models: vec![],
        })
    }

    /// Load the VAD model
    pub fn load_vad(&mut self) -> Result<()> {
        let request = ASRRequest::new("load_vad", self.next_id());
        let response: EngineResponse<ModelOperationResult> =
            self.get_process()?.send_command(&request)?;
        let result = response.into_result("Failed to load VAD")?;

        if result.success == Some(false) {
            return Err(anyhow!(
                "Failed to load VAD: {}",
                result.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        log::info!("VAD model loaded successfully");
        Ok(())
    }

    /// Warm up the ASR model by running inference on dummy audio
    pub fn warmup_model(&mut self) -> Result<WarmupResult> {
        // Cohere → warm up the in-process engine (compiles Metal kernels).
        if rust_asr::is_cohere_model(&self.active_model) {
            if let Some(engine) = self.cohere.as_ref() {
                let start = std::time::Instant::now();
                let result = engine.warmup();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(WarmupResult {
                    success: result.is_ok(),
                    warmup_time_ms: Some(ms),
                    error: result.err().map(|e| e.to_string()),
                });
            }
            return Ok(WarmupResult {
                success: false,
                warmup_time_ms: None,
                error: Some("Cohere engine not loaded".to_string()),
            });
        }

        // Whisper → warm up the in-process engine.
        if rust_asr::is_whisper_model(&self.active_model) {
            if let Some(engine) = self.whisper.as_ref() {
                let start = std::time::Instant::now();
                let result = engine.warmup();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(WarmupResult {
                    success: result.is_ok(),
                    warmup_time_ms: Some(ms),
                    error: result.err().map(|e| e.to_string()),
                });
            }
            return Ok(WarmupResult {
                success: false,
                warmup_time_ms: None,
                error: Some("Whisper engine not loaded".to_string()),
            });
        }

        // Parakeet → warm up the in-process engine.
        if rust_asr::is_parakeet_model(&self.active_model) {
            if let Some(engine) = self.parakeet.as_ref() {
                let start = std::time::Instant::now();
                let result = engine.warmup();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(WarmupResult {
                    success: result.is_ok(),
                    warmup_time_ms: Some(ms),
                    error: result.err().map(|e| e.to_string()),
                });
            }
            return Ok(WarmupResult {
                success: false,
                warmup_time_ms: None,
                error: Some("Parakeet engine not loaded".to_string()),
            });
        }
        // Non-in-process models without a sidecar: nothing to warm up.
        if self.process.is_none() {
            return Ok(WarmupResult { success: false, warmup_time_ms: None, error: None });
        }

        let request = ASRRequest::new("warmup_model", self.next_id());
        let response: EngineResponse<WarmupResponseResult> =
            self.get_process()?.send_command(&request)?;
        let result = response.into_result("Failed to warmup model")?;

        if result.success == Some(false) {
            log::warn!("Model warmup failed: {:?}", result.error);
        } else {
            log::info!("Model warmup complete in {:?}ms", result.warmup_time_ms);
        }

        Ok(WarmupResult {
            success: result.success.unwrap_or(false),
            warmup_time_ms: result.warmup_time_ms,
            error: result.error,
        })
    }

    /// Warm up the VAD model
    pub fn warmup_vad(&mut self) -> Result<WarmupResult> {
        let request = ASRRequest::new("warmup_vad", self.next_id());
        let response: EngineResponse<WarmupResponseResult> =
            self.get_process()?.send_command(&request)?;
        let result = response.into_result("Failed to warmup VAD")?;

        if result.success == Some(false) {
            log::warn!("VAD warmup failed: {:?}", result.error);
        } else {
            log::info!("VAD warmup complete in {:?}ms", result.warmup_time_ms);
        }

        Ok(WarmupResult {
            success: result.success.unwrap_or(false),
            warmup_time_ms: result.warmup_time_ms,
            error: result.error,
        })
    }

    /// Update the HuggingFace token in the running sidecar.
    /// Pass None to unset. Takes effect for subsequent model downloads.
    pub fn set_hf_token(&mut self, token: Option<&str>) -> Result<()> {
        // Keep the token for the in-process Cohere engine's gated download too.
        self.hf_token = token.filter(|t| !t.is_empty()).map(|t| t.to_string());
        // If the sidecar is not yet running there is nothing to update —
        // the token will be picked up via env var at spawn time.
        if self.process.is_none() {
            return Ok(());
        }
        let mut request = ASRRequest::new("set_hf_token", self.next_id());
        request.hf_token = Some(token.unwrap_or("").to_string());
        let response: EngineResponse<ModelOperationResult> =
            self.get_process()?.send_command(&request)?;
        let _ = response.into_result("Failed to set HF token")?;
        Ok(())
    }

    /// Status for the in-process Cohere engine.
    fn in_process_status(&self, loaded: bool) -> ModelStatus {
        ModelStatus {
            model_name: self.active_model.clone(),
            loaded,
            loading: false,
            error: None,
            available_models: vec![],
        }
    }

    /// Whether the given model id is served by an in-process Rust engine.
    fn is_in_process(name: &str) -> bool {
        rust_asr::is_cohere_model(name)
            || rust_asr::is_whisper_model(name)
            || rust_asr::is_parakeet_model(name)
    }

    /// Loaded state of the currently-active in-process engine.
    fn in_process_loaded(&self) -> bool {
        if rust_asr::is_whisper_model(&self.active_model) {
            self.whisper_loaded
        } else if rust_asr::is_parakeet_model(&self.active_model) {
            self.parakeet_loaded
        } else {
            self.cohere_loaded
        }
    }

    /// HF hub cache repo dir name for an in-process model (for cache checks).
    fn in_process_cache_repo(name: &str) -> Option<&'static str> {
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

    /// Drop all in-process engines and reset their loaded flags.
    fn drop_in_process_engines(&mut self) {
        self.cohere = None;
        self.cohere_loaded = false;
        self.whisper = None;
        self.whisper_loaded = false;
        self.parakeet = None;
        self.parakeet_loaded = false;
    }

    /// Set the model (requires reload). In-process models (Cohere, Whisper,
    /// Parakeet) are tracked locally with no Python involvement; others forward
    /// to the sidecar.
    pub fn set_model(&mut self, model_name: &str) -> Result<ModelStatus> {
        self.active_model = model_name.to_string();
        // Free any previously-loaded in-process engine; the selected one loads
        // fresh on the next load_model.
        self.drop_in_process_engines();
        if Self::is_in_process(model_name) {
            return Ok(self.in_process_status(false));
        }

        if self.process.is_none() {
            return Err(anyhow!(
                "Model '{}' is not supported in-process. Choose Whisper or Cohere.",
                model_name
            ));
        }

        let mut request = ASRRequest::new("set_model", self.next_id());
        request.model_name = Some(model_name.to_string());

        let response: EngineResponse<ModelOperationResult> =
            self.get_process()?.send_command(&request)?;
        let result = response.into_result("Failed to set model")?;

        if result.success == Some(false) {
            return Err(anyhow!(
                "Failed to set model: {}",
                result.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        Ok(ModelStatus {
            model_name: result
                .model_name
                .unwrap_or_else(|| model_name.to_string()),
            loaded: false,
            loading: false,
            error: None,
            available_models: vec![],
        })
    }

    /// Check if the engine is running
    pub fn ping(&mut self) -> Result<bool> {
        let process = match self.process.as_mut() {
            Some(p) => p,
            None => return Ok(false),
        };

        // Check if process is still alive
        match process.child.try_wait() {
            Ok(Some(_)) => {
                self.process = None;
                return Ok(false);
            }
            Ok(None) => {}
            Err(_) => return Ok(false),
        }

        let request = ASRRequest::new("ping", self.request_id.fetch_add(1, Ordering::SeqCst));
        let response: EngineResponse<ASRResultInner> = process.send_command(&request)?;

        Ok(response.error.is_none() && response.result.is_some())
    }

    /// Transcribe an audio file
    pub fn transcribe(
        &mut self,
        audio_path: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        // Cohere → transcribe in-process with the full-Rust engine.
        if rust_asr::is_cohere_model(&self.active_model) {
            if !self.cohere_loaded || self.cohere.is_none() {
                self.load_model()?;
            }
            let engine = self
                .cohere
                .as_ref()
                .ok_or_else(|| anyhow!("Cohere engine not loaded"))?;
            let lang = language.unwrap_or("auto");
            let out = engine
                .transcribe_wav(audio_path, lang)
                .map_err(|e| anyhow!("Cohere transcription failed: {}", e))?;
            let no_speech = if out.text.trim().is_empty() {
                Some(true)
            } else {
                None
            };
            log::info!(
                "Cohere (in-process) transcription complete: {} chars",
                out.text.len()
            );
            return Ok(TranscriptionResult {
                success: true,
                text: out.text,
                segments: vec![],
                language: out.language,
                no_speech,
            });
        }

        // Whisper → transcribe in-process with whisper.cpp.
        if rust_asr::is_whisper_model(&self.active_model) {
            if !self.whisper_loaded || self.whisper.is_none() {
                self.load_model()?;
            }
            let engine = self
                .whisper
                .as_ref()
                .ok_or_else(|| anyhow!("Whisper engine not loaded"))?;
            let out = engine
                .transcribe_wav(audio_path, language.unwrap_or("auto"))
                .map_err(|e| anyhow!("Whisper transcription failed: {}", e))?;
            let no_speech = if out.text.trim().is_empty() {
                Some(true)
            } else {
                None
            };
            log::info!(
                "Whisper (in-process) transcription complete: {} chars",
                out.text.len()
            );
            return Ok(TranscriptionResult {
                success: true,
                text: out.text,
                segments: vec![],
                language: out.language,
                no_speech,
            });
        }

        // Parakeet → transcribe in-process (full-Rust MLX, Japanese).
        if rust_asr::is_parakeet_model(&self.active_model) {
            if !self.parakeet_loaded || self.parakeet.is_none() {
                self.load_model()?;
            }
            let engine = self
                .parakeet
                .as_ref()
                .ok_or_else(|| anyhow!("Parakeet engine not loaded"))?;
            let out = engine
                .transcribe_wav(audio_path, language.unwrap_or("ja"))
                .map_err(|e| anyhow!("Parakeet transcription failed: {}", e))?;
            let no_speech = if out.text.trim().is_empty() {
                Some(true)
            } else {
                None
            };
            log::info!(
                "Parakeet (in-process) transcription complete: {} chars",
                out.text.len()
            );
            return Ok(TranscriptionResult {
                success: true,
                text: out.text,
                segments: vec![],
                language: out.language,
                no_speech,
            });
        }

        let process = self.get_process().map_err(|_| {
            anyhow!("ASR engine not running. Call start_asr_engine first.")
        })?;

        // Check if process is still alive
        match process.child.try_wait() {
            Ok(Some(status)) => {
                self.process = None;
                return Err(anyhow!("ASR engine exited unexpectedly: {:?}", status));
            }
            Ok(None) => {}
            Err(e) => {
                self.process = None;
                return Err(anyhow!("Failed to check ASR engine status: {}", e));
            }
        }

        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = ASRRequest {
            command: "transcribe".to_string(),
            id,
            audio_path: Some(audio_path.to_string()),
            language: language.map(|s| s.to_string()),
            model_name: None,
            hf_token: None,
        };

        log::info!(
            "Sending transcribe request for: {} with language: {:?}",
            audio_path,
            language
        );

        // Need to re-borrow process after the try_wait check above
        let process = self.process.as_mut().unwrap();
        let response: EngineResponse<ASRResultInner> = process.send_command(&request)?;
        let result = response.into_result("ASR engine error")?;

        if !result.success {
            if let Some(error) = result.error {
                return Err(anyhow!("Transcription failed: {}", error));
            }
            return Err(anyhow!("Transcription failed: unknown error"));
        }

        let segments: Vec<TranscriptionSegment> = result
            .segments
            .into_iter()
            .map(|seg| TranscriptionSegment {
                start: seg.start,
                end: seg.end,
                text: seg.text,
            })
            .collect();

        log::info!(
            "Transcription complete: {} chars, {} segments",
            result.text.len(),
            segments.len()
        );

        Ok(TranscriptionResult {
            success: true,
            text: result.text,
            segments,
            language: result.language,
            no_speech: result.no_speech,
        })
    }

    // ===== Post-processing methods (in-process Qwen3, full Rust) =====

    fn postproc_status(&self, loaded: bool, error: Option<String>) -> crate::PostProcessModelStatus {
        crate::PostProcessModelStatus {
            model_name: self.postproc_model.clone(),
            loaded,
            loading: false,
            error,
            available_models: POSTPROCESS_MODELS.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Load the post-processing LLM (Qwen3) in-process.
    pub fn load_postprocess_model(&mut self) -> Result<crate::PostProcessModelStatus> {
        if self.postproc_loaded && self.postproc.is_some() {
            return Ok(self.postproc_status(true, None));
        }
        let hub = self
            .cohere_hub_dir
            .clone()
            .ok_or_else(|| anyhow!("Cache dir not set; start the engine first"))?;
        log::info!(
            "Loading post-processor (Qwen3) in-process: {}",
            self.postproc_model
        );
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

    /// Unload the post-processing model to free memory.
    pub fn unload_postprocess_model(&mut self) -> Result<()> {
        self.postproc = None;
        self.postproc_loaded = false;
        Ok(())
    }

    /// Check if the post-processing model is present in the shared HF cache.
    pub fn is_postprocess_model_cached(&mut self) -> Result<crate::PostProcessModelStatus> {
        let repo = format!("models--{}", self.postproc_model.replace('/', "--"));
        let cached = self
            .cohere_hub_dir
            .as_ref()
            .map(|hub| hub.join(&repo).join("snapshots").exists())
            .unwrap_or(false);
        Ok(crate::PostProcessModelStatus {
            model_name: self.postproc_model.clone(),
            loaded: cached,
            loading: false,
            error: None,
            available_models: POSTPROCESS_MODELS.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Set the post-processing model (requires reload).
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
        Ok(self.postproc_status(false, None))
    }

    /// Get post-processor status.
    pub fn get_postprocess_status(&mut self) -> Result<crate::PostProcessModelStatus> {
        Ok(self.postproc_status(self.postproc_loaded, None))
    }

    /// Post-process transcribed text with the in-process LLM.
    pub fn postprocess_text(
        &mut self,
        text: &str,
        app_name: Option<&str>,
        app_bundle_id: Option<&str>,
        dictionary: Option<&std::collections::HashMap<String, String>>,
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
                processed_text: text.to_string(), // fallback to raw text
                processing_time_ms: None,
                error: Some(e.to_string()),
            }),
        }
    }

    /// Summarize a list of transcription entries with the in-process LLM.
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
mod cohere_inprocess_tests {
    use super::*;

    /// End-to-end check of the in-process Cohere routing without the Python
    /// sidecar: set the private fields directly, load the Rust engine, and
    /// transcribe a bundled WAV. Skips if the gated model isn't cached locally.
    #[test]
    fn cohere_routes_in_process() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let hub = PathBuf::from(&home)
            .join("Library/Caches/io.qluto.echo/huggingface/hub");
        let cached = hub.join("models--CohereLabs--cohere-transcribe-03-2026");
        if !cached.exists() {
            eprintln!("skipping: Cohere model not cached at {cached:?}");
            return;
        }

        // Mirror the app startup flow with NO Python sidecar (process == None):
        // set_model -> is_model_cached -> get_model_status -> load -> warmup ->
        // transcribe must all work in-process for Cohere.
        let mut engine = ASREngine::new();
        engine.cohere_hub_dir = Some(hub); // normally captured in start()

        let set = engine.set_model(rust_asr::COHERE_MODEL_ID).expect("set_model");
        assert!(rust_asr::is_cohere_model(&set.model_name));

        let cache = engine.is_model_cached(None).expect("is_model_cached");
        assert!(cache.cached, "Cohere model should be reported cached");

        let before = engine.get_model_status().expect("status");
        assert!(!before.loaded);

        let status = engine.load_model().expect("load cohere");
        assert!(status.loaded, "cohere engine should report loaded");

        let after = engine.get_model_status().expect("status");
        assert!(after.loaded, "status should reflect loaded engine");

        let warm = engine.warmup_model().expect("warmup");
        assert!(warm.success, "warmup should succeed: {:?}", warm.error);

        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/resources/start_v1.wav");
        let result = engine.transcribe(wav, Some("en")).expect("transcribe");
        assert!(result.success);
        // Routed in-process: segments are empty (the Rust engine returns text only).
        eprintln!("in-process cohere text: {:?}", result.text);
    }

    /// Whisper routes through the in-process whisper.cpp engine with no sidecar.
    /// Uses whisper-tiny (small, fast). Skips if the ggml model can't be fetched.
    #[test]
    fn whisper_routes_in_process() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let hub = PathBuf::from(&home).join("Library/Caches/io.qluto.echo/huggingface/hub");

        let mut engine = ASREngine::new();
        engine.cohere_hub_dir = Some(hub);

        let set = engine
            .set_model("mlx-community/whisper-tiny")
            .expect("set_model");
        assert!(rust_asr::is_whisper_model(&set.model_name));

        // load (downloads ggml-tiny.bin if missing — small)
        let status = match engine.load_model() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: could not load whisper-tiny ({e})");
                return;
            }
        };
        assert!(status.loaded);

        let after = engine.get_model_status().expect("status");
        assert!(after.loaded);

        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/resources/start_v1.wav");
        let result = engine.transcribe(wav, Some("en")).expect("transcribe");
        assert!(result.success);
        eprintln!("in-process whisper text: {:?}", result.text);
    }

    /// Parakeet (JA) routes through the in-process full-Rust MLX engine.
    /// Skips if the model isn't cached locally.
    #[test]
    fn parakeet_routes_in_process() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let hub = PathBuf::from(&home).join("Library/Caches/io.qluto.echo/huggingface/hub");
        if !hub
            .join("models--mlx-community--parakeet-tdt_ctc-0.6b-ja")
            .exists()
        {
            eprintln!("skipping: parakeet model not cached");
            return;
        }

        let mut engine = ASREngine::new();
        engine.cohere_hub_dir = Some(hub);

        let set = engine
            .set_model("mlx-community/parakeet-tdt_ctc-0.6b-ja")
            .expect("set_model");
        assert!(rust_asr::is_parakeet_model(&set.model_name));

        let status = engine.load_model().expect("load parakeet");
        assert!(status.loaded);
        let after = engine.get_model_status().expect("status");
        assert!(after.loaded);

        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/resources/start_v1.wav");
        let result = engine.transcribe(wav, Some("ja")).expect("transcribe");
        assert!(result.success);
        eprintln!("in-process parakeet text: {:?}", result.text);
    }

    /// Post-processing runs in-process via the Qwen3 LLM, no Python sidecar.
    /// Uses the small 1.7B model; skips if it isn't cached.
    #[test]
    fn postprocess_in_process() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let hub = PathBuf::from(&home).join("Library/Caches/io.qluto.echo/huggingface/hub");
        if !hub.join("models--mlx-community--Qwen3-1.7B-4bit").exists() {
            eprintln!("skipping: Qwen3-1.7B-4bit not cached");
            return;
        }

        let mut engine = ASREngine::new();
        engine.cohere_hub_dir = Some(hub);
        engine
            .set_postprocess_model("mlx-community/Qwen3-1.7B-4bit")
            .expect("set postproc model");

        let status = engine.load_postprocess_model().expect("load postproc");
        assert!(status.loaded, "postproc error: {:?}", status.error);

        let r = engine
            .postprocess_text("えーと、3時に、いや4時に会議があります。", None, None, None, None)
            .expect("postprocess");
        assert!(r.success);
        assert!(!r.processed_text.trim().is_empty());
        eprintln!("postprocessed: {:?}", r.processed_text);
    }
}

impl Drop for ASREngine {
    fn drop(&mut self) {
        self.stop().ok();
    }
}
