//! ASR Engine module for communicating with the Python sidecar
//!
//! This module manages the ASR (Automatic Speech Recognition) engine
//! which runs as a separate Python process using MLX-Audio.

use anyhow::{anyhow, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};

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
}

impl ASRRequest {
    fn new(command: &str, id: u64) -> Self {
        Self {
            command: command.to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: None,
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

/// ASR Engine manager
pub struct ASREngine {
    process: Option<ASRProcess>,
    request_id: AtomicU64,
}

impl ASREngine {
    pub fn new() -> Self {
        Self {
            process: None,
            request_id: AtomicU64::new(1),
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

    /// Start the ASR engine sidecar
    pub fn start(&mut self, app: &AppHandle) -> Result<()> {
        if self.process.is_some() {
            log::info!("ASR engine already running");
            return Ok(());
        }

        // Binary name with platform suffix (for development builds)
        let binary_name_with_suffix = format!("mlx-asr-engine-{}", Self::get_target_triple());
        // Binary name without suffix (Tauri strips the suffix when bundling externalBin)
        let binary_name_no_suffix = "mlx-asr-engine";

        // Priority 1: Check for bundled binary in app's MacOS directory (production mode)
        // Tauri 2.x places externalBin in Contents/MacOS/ with the platform suffix stripped
        let bundled_binary = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .map(|p| p.join(binary_name_no_suffix));

        // Priority 2: Check for development binary in src-tauri/binaries
        let dev_binary_paths = [
            // When running from src-tauri directory
            std::env::current_dir()
                .ok()
                .map(|p| p.join("binaries").join(&binary_name_with_suffix)),
            // When running from project root
            std::env::current_dir()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                .map(|p| p.join("src-tauri").join("binaries").join(&binary_name_with_suffix)),
        ];

        let program = if let Some(ref bundled) = bundled_binary {
            if bundled.exists() {
                log::info!("Using bundled ASR binary: {:?}", bundled);
                bundled.to_string_lossy().to_string()
            } else if let Some(dev_binary) = dev_binary_paths
                .into_iter()
                .flatten()
                .find(|p| p.exists())
            {
                log::info!("Using development ASR binary: {:?}", dev_binary);
                dev_binary.to_string_lossy().to_string()
            } else {
                return Err(anyhow!(
                    "ASR engine binary not found. Expected '{}' in {:?} or '{}' in src-tauri/binaries/. \
                    Run 'cd python-engine && ./build.sh' to build the binary.",
                    binary_name_no_suffix,
                    bundled,
                    binary_name_with_suffix
                ));
            }
        } else if let Some(dev_binary) = dev_binary_paths
            .into_iter()
            .flatten()
            .find(|p| p.exists())
        {
            log::info!("Using development ASR binary: {:?}", dev_binary);
            dev_binary.to_string_lossy().to_string()
        } else {
            return Err(anyhow!(
                "ASR engine binary not found. Run 'cd python-engine && ./build.sh' to build the binary."
            ));
        };

        let args = vec!["daemon".to_string()];

        log::info!("Starting ASR engine: {} {:?}", program, args);

        // Get app cache directory for model storage
        let cache_dir = app
            .path()
            .resolve("huggingface", BaseDirectory::AppCache)
            .map_err(|e| anyhow!("Failed to resolve cache directory: {}", e))?;

        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            log::warn!("Failed to create cache directory {:?}: {}", cache_dir, e);
        }

        log::info!("Using model cache directory: {:?}", cache_dir);

        // Build command with cache environment variables
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("HF_HOME", &cache_dir)
            .env("TORCH_HOME", &cache_dir);

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to start ASR engine: {}", e))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdin handle"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdout handle"))?;

        let stdin_writer = BufWriter::new(stdin);
        let mut stdout_reader = BufReader::new(stdout);

        // Wait for ready signal with timeout
        log::info!("Waiting for ASR engine ready signal...");
        let mut line = String::new();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(60);

        loop {
            if start.elapsed() > timeout {
                child.kill().ok();
                return Err(anyhow!("Timeout waiting for ASR engine to start"));
            }

            line.clear();
            match stdout_reader.read_line(&mut line) {
                Ok(0) => {
                    child.kill().ok();
                    return Err(anyhow!("ASR engine closed stdout unexpectedly"));
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    log::debug!("ASR engine output: {}", trimmed);

                    if let Ok(status) = serde_json::from_str::<StatusResponse>(trimmed) {
                        if status.status.as_deref() == Some("ready") {
                            log::info!("ASR engine ready");
                            break;
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                Err(e) => {
                    child.kill().ok();
                    return Err(anyhow!("Error reading from ASR engine: {}", e));
                }
            }
        }

        self.process = Some(ASRProcess {
            child,
            stdin: stdin_writer,
            stdout: stdout_reader,
        });

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

        Ok(ModelStatus {
            model_name: result.model_name.unwrap_or_else(|| "unknown".to_string()),
            loaded: result.loaded.unwrap_or(false),
            loading: result.loading.unwrap_or(false),
            error: result.error,
            available_models: result.available_models.unwrap_or_default(),
        })
    }

    /// Check if a model is cached locally
    pub fn is_model_cached(&mut self, model_name: Option<&str>) -> Result<ModelCacheStatus> {
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

    /// Set the model (requires reload)
    pub fn set_model(&mut self, model_name: &str) -> Result<ModelStatus> {
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

    // ===== Post-processing methods =====

    /// Send a JSON value command and return the result as serde_json::Value
    fn send_json_command(&mut self, request: &serde_json::Value, context: &str) -> Result<serde_json::Value> {
        let response: EngineResponse<serde_json::Value> =
            self.get_process()?.send_command(request)?;
        response.into_result(context)
    }

    /// Load the post-processing model
    pub fn load_postprocess_model(&mut self) -> Result<crate::PostProcessModelStatus> {
        let request = serde_json::json!({
            "command": "load_postprocess_model",
            "id": self.next_id(),
        });
        let result = self.send_json_command(&request, "Failed to load post-process model")?;

        Ok(crate::PostProcessModelStatus {
            model_name: result
                .get("model_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            loaded: result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            loading: false,
            error: result
                .get("error")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            available_models: vec![],
        })
    }

    /// Unload the post-processing model
    pub fn unload_postprocess_model(&mut self) -> Result<()> {
        let request = serde_json::json!({
            "command": "unload_postprocess_model",
            "id": self.next_id(),
        });
        self.send_json_command(&request, "Failed to unload post-process model")?;
        Ok(())
    }

    /// Check if the post-processing model is cached
    pub fn is_postprocess_model_cached(&mut self) -> Result<crate::PostProcessModelStatus> {
        let request = serde_json::json!({
            "command": "is_postprocess_model_cached",
            "id": self.next_id(),
        });
        let result = self.send_json_command(&request, "Failed to check post-process model cache")?;

        Ok(crate::PostProcessModelStatus {
            model_name: result
                .get("model_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            loaded: result
                .get("cached")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            loading: false,
            error: None,
            available_models: vec![],
        })
    }

    /// Set the post-processing model
    pub fn set_postprocess_model(
        &mut self,
        model_name: &str,
    ) -> Result<crate::PostProcessModelStatus> {
        let request = serde_json::json!({
            "command": "set_postprocess_model",
            "id": self.next_id(),
            "model_name": model_name,
        });
        let result = self.send_json_command(&request, "Failed to set post-process model")?;

        if result.get("success").and_then(|v| v.as_bool()) == Some(false) {
            return Err(anyhow!(
                "Failed to set post-process model: {}",
                result
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error")
            ));
        }

        Ok(crate::PostProcessModelStatus {
            model_name: result
                .get("model_name")
                .and_then(|v| v.as_str())
                .unwrap_or(model_name)
                .to_string(),
            loaded: false,
            loading: false,
            error: None,
            available_models: vec![],
        })
    }

    /// Get post-processor status
    pub fn get_postprocess_status(&mut self) -> Result<crate::PostProcessModelStatus> {
        let request = serde_json::json!({
            "command": "get_postprocess_status",
            "id": self.next_id(),
        });
        let result = self.send_json_command(&request, "Failed to get post-process status")?;

        Ok(crate::PostProcessModelStatus {
            model_name: result
                .get("model_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            loaded: result
                .get("loaded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            loading: result
                .get("loading")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            error: result
                .get("error")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            available_models: result
                .get("available_models")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
        })
    }

    /// Post-process transcribed text
    pub fn postprocess_text(
        &mut self,
        text: &str,
        app_name: Option<&str>,
        app_bundle_id: Option<&str>,
        dictionary: Option<&std::collections::HashMap<String, String>>,
        custom_prompt: Option<&str>,
    ) -> Result<crate::PostProcessResult> {
        let request = serde_json::json!({
            "command": "postprocess_text",
            "id": self.next_id(),
            "text": text,
            "app_name": app_name,
            "app_bundle_id": app_bundle_id,
            "dictionary": dictionary,
            "custom_prompt": custom_prompt,
        });
        let result = self.send_json_command(&request, "Post-processing failed")?;

        Ok(crate::PostProcessResult {
            success: result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            processed_text: result
                .get("processed_text")
                .and_then(|v| v.as_str())
                .unwrap_or(text)
                .to_string(),
            processing_time_ms: result.get("processing_time_ms").and_then(|v| v.as_f64()),
            error: result
                .get("error")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// Summarize a list of transcription entries using the LLM
    pub fn summarize_transcriptions(
        &mut self,
        texts: &[crate::SummarizeEntry],
        language_hint: Option<&str>,
        custom_prompt: Option<&str>,
    ) -> Result<crate::SummarizeResult> {
        let request = serde_json::json!({
            "command": "summarize_transcriptions",
            "id": self.next_id(),
            "texts": texts,
            "language_hint": language_hint,
            "custom_prompt": custom_prompt,
        });
        let result = self.send_json_command(&request, "Summarization failed")?;

        Ok(crate::SummarizeResult {
            success: result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            summary: result
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            processing_time_ms: result.get("processing_time_ms").and_then(|v| v.as_f64()),
            error: result
                .get("error")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            entry_count: 0, // Set by caller
        })
    }
}

impl Default for ASREngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ASREngine {
    fn drop(&mut self) {
        self.stop().ok();
    }
}
