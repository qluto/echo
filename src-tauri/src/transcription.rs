//! ASR Engine module for communicating with the Python sidecar
//!
//! This module manages the ASR (Automatic Speech Recognition) engine
//! which runs as a separate Python process using MLX-Audio.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tauri::{AppHandle, Manager};

use crate::{TranscriptionResult, TranscriptionSegment};

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

/// Response from the ASR engine
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ASRResponse {
    id: Option<u64>,
    result: Option<ASRResultInner>,
    error: Option<String>,
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

/// Status response from the ASR engine
#[derive(Debug, Deserialize)]
struct StatusResponse {
    status: Option<String>,
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

/// Generic response for model operations
#[derive(Debug, Deserialize)]
struct ModelOperationResponse {
    id: Option<u64>,
    result: Option<ModelOperationResult>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelOperationResult {
    success: Option<bool>,
    model_name: Option<String>,
    loaded: Option<bool>,
    loading: Option<bool>,
    error: Option<String>,
    available_models: Option<Vec<String>>,
    already_loaded: Option<bool>,
}

/// Process handle with I/O streams
struct ASRProcess {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
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

    /// Start the ASR engine sidecar
    pub fn start(&mut self, app: &AppHandle) -> Result<()> {
        if self.process.is_some() {
            log::info!("ASR engine already running");
            return Ok(());
        }

        // Get the resource directory for bundled Python script
        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| anyhow!("Failed to get resource dir: {}", e))?;

        // Check for bundled Python script in resources first
        let bundled_script = resource_dir.join("engine.py");

        // Determine how to run the engine
        let (program, args) = if bundled_script.exists() {
            // Bundled app mode: use Python script from resources
            log::info!("Using bundled Python script: {:?}", bundled_script);

            // Find Python executable - check common locations for macOS
            // Bundled apps don't have access to user's PATH, so we search explicitly
            // Priority: venv with mlx-audio installed > Homebrew > System Python

            // First, check for venv in common locations (these have mlx-audio pre-installed)
            let venv_candidates: Vec<std::path::PathBuf> = [
                // Check in common development locations
                dirs::home_dir()
                    .map(|h| h.join("conductor/workspaces/echo/lisbon/python-engine/venv/bin/python3")),
                // Check in home directory for a dedicated Echo venv
                dirs::home_dir().map(|h| h.join(".echo/venv/bin/python3")),
            ]
            .into_iter()
            .flatten()
            .collect();

            // Build the full list of candidates
            let mut python_candidates: Vec<String> = venv_candidates
                .iter()
                .filter(|p| p.exists())
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            // Add system Python paths as fallbacks
            python_candidates.extend([
                "/opt/homebrew/bin/python3".to_string(),
                "/usr/local/bin/python3".to_string(),
                "/usr/bin/python3".to_string(),
            ]);

            let python_executable = python_candidates
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .cloned()
                .unwrap_or_else(|| "python3".to_string());

            log::info!("Using Python executable: {}", python_executable);

            (
                python_executable,
                vec![
                    bundled_script.to_string_lossy().to_string(),
                    "daemon".to_string(),
                ],
            )
        } else {
            // Development mode: run Python script directly
            // Try multiple possible locations for the Python script
            let possible_paths = [
                // When running from src-tauri directory
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                    .map(|p| p.join("python-engine")),
                // When running from project root
                std::env::current_dir()
                    .ok()
                    .map(|p| p.join("python-engine")),
            ];

            let python_engine_dir = possible_paths
                .into_iter()
                .flatten()
                .find(|p| p.join("engine.py").exists())
                .ok_or_else(|| anyhow!("Python ASR engine directory not found"))?;

            let python_script = python_engine_dir.join("engine.py");

            // Check for virtual environment first, then fall back to system python
            let venv_python = python_engine_dir.join("venv").join("bin").join("python3");
            let python_executable = if venv_python.exists() {
                log::info!("Using venv Python: {:?}", venv_python);
                venv_python.to_string_lossy().to_string()
            } else {
                log::info!("Using system Python");
                "python3".to_string()
            };

            log::info!("Using Python script: {:?}", python_script);
            (
                python_executable,
                vec![
                    python_script.to_string_lossy().to_string(),
                    "daemon".to_string(),
                ],
            )
        };

        log::info!("Starting ASR engine: {} {:?}", program, args);

        // Build command with proper environment
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // Let stderr pass through for logging

        // For bundled apps using system Python (not venv), we need to set PYTHONPATH
        // because macOS app bundles don't inherit the user's shell environment
        // Skip this if using a venv (which has packages pre-installed)
        let is_venv = program.contains("/venv/");
        if bundled_script.exists() && !is_venv {
            // Get user's home directory
            if let Some(home) = dirs::home_dir() {
                let mut paths: Vec<String> = Vec::new();

                // Check common Python versions for user site-packages
                for version in ["3.9", "3.10", "3.11", "3.12", "3.13"] {
                    let user_site = home
                        .join("Library/Python")
                        .join(version)
                        .join("lib/python/site-packages");
                    if user_site.exists() {
                        paths.push(user_site.to_string_lossy().to_string());
                    }

                    // Also check Homebrew site-packages
                    let homebrew_site = std::path::PathBuf::from(format!(
                        "/opt/homebrew/lib/python{}/site-packages",
                        version
                    ));
                    if homebrew_site.exists() {
                        paths.push(homebrew_site.to_string_lossy().to_string());
                    }
                }

                if !paths.is_empty() {
                    let python_path = paths.join(":");
                    log::info!("Setting PYTHONPATH: {}", python_path);
                    cmd.env("PYTHONPATH", &python_path);
                }
            }
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to start ASR engine: {}", e))?;

        // Take ownership of stdin/stdout
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

        // Set a simple timeout using a loop with small reads
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(60); // Model loading can take time

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
            // Send quit command
            let request = ASRRequest {
                command: "quit".to_string(),
                id: 0,
                audio_path: None,
                language: None,
                model_name: None,
            };

            if let Ok(json) = serde_json::to_string(&request) {
                let _ = writeln!(process.stdin, "{}", json);
                let _ = process.stdin.flush();
            }

            // Wait for process to exit gracefully
            std::thread::sleep(Duration::from_millis(500));

            // Force kill if still running
            let _ = process.child.kill();
            let _ = process.child.wait();

            log::info!("ASR engine stopped");
        }
        Ok(())
    }

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

        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = ASRRequest {
            command: "get_status".to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: None,
        };

        let json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", json)?;
        process.stdin.flush()?;

        let mut line = String::new();
        process.stdout.read_line(&mut line)?;

        let response: ModelOperationResponse = serde_json::from_str(&line)?;

        if let Some(error) = response.error {
            return Err(anyhow!("Failed to get status: {}", error));
        }

        let result = response.result.ok_or_else(|| anyhow!("No result in response"))?;

        Ok(ModelStatus {
            model_name: result.model_name.unwrap_or_else(|| "unknown".to_string()),
            loaded: result.loaded.unwrap_or(false),
            loading: result.loading.unwrap_or(false),
            error: result.error,
            available_models: result.available_models.unwrap_or_default(),
        })
    }

    /// Load the model
    pub fn load_model(&mut self) -> Result<ModelStatus> {
        let process = match self.process.as_mut() {
            Some(p) => p,
            None => {
                return Err(anyhow!("Engine not running"));
            }
        };

        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = ASRRequest {
            command: "load_model".to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: None,
        };

        let json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", json)?;
        process.stdin.flush()?;

        // Model loading can take a long time, so we need to handle timeouts
        let mut line = String::new();
        process
            .stdout
            .read_line(&mut line)
            .map_err(|e| anyhow!("Failed to read response: {}", e))?;

        let response: ModelOperationResponse = serde_json::from_str(&line)?;

        if let Some(error) = response.error {
            return Err(anyhow!("Failed to load model: {}", error));
        }

        let result = response.result.ok_or_else(|| anyhow!("No result in response"))?;

        if result.success == Some(false) {
            return Err(anyhow!(
                "Failed to load model: {}",
                result.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        Ok(ModelStatus {
            model_name: result.model_name.unwrap_or_else(|| "unknown".to_string()),
            loaded: true,
            loading: false,
            error: None,
            available_models: vec![],
        })
    }

    /// Load the VAD model
    pub fn load_vad(&mut self) -> Result<()> {
        let process = match self.process.as_mut() {
            Some(p) => p,
            None => {
                return Err(anyhow!("Engine not running"));
            }
        };

        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = ASRRequest {
            command: "load_vad".to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: None,
        };

        let json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", json)?;
        process.stdin.flush()?;

        let mut line = String::new();
        process
            .stdout
            .read_line(&mut line)
            .map_err(|e| anyhow!("Failed to read response: {}", e))?;

        let response: ModelOperationResponse = serde_json::from_str(&line)?;

        if let Some(error) = response.error {
            return Err(anyhow!("Failed to load VAD: {}", error));
        }

        let result = response.result.ok_or_else(|| anyhow!("No result in response"))?;

        if result.success == Some(false) {
            return Err(anyhow!(
                "Failed to load VAD: {}",
                result.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        log::info!("VAD model loaded successfully");
        Ok(())
    }

    /// Set the model (requires reload)
    pub fn set_model(&mut self, model_name: &str) -> Result<ModelStatus> {
        let process = match self.process.as_mut() {
            Some(p) => p,
            None => {
                return Err(anyhow!("Engine not running"));
            }
        };

        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = ASRRequest {
            command: "set_model".to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: Some(model_name.to_string()),
        };

        let json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", json)?;
        process.stdin.flush()?;

        let mut line = String::new();
        process.stdout.read_line(&mut line)?;

        let response: ModelOperationResponse = serde_json::from_str(&line)?;

        if let Some(error) = response.error {
            return Err(anyhow!("Failed to set model: {}", error));
        }

        let result = response.result.ok_or_else(|| anyhow!("No result in response"))?;

        if result.success == Some(false) {
            return Err(anyhow!(
                "Failed to set model: {}",
                result.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        Ok(ModelStatus {
            model_name: result.model_name.unwrap_or_else(|| model_name.to_string()),
            loaded: false, // Model needs to be reloaded after setting
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
                // Process has exited
                self.process = None;
                return Ok(false);
            }
            Ok(None) => {} // Still running
            Err(_) => return Ok(false),
        }

        // Send ping request
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = ASRRequest {
            command: "ping".to_string(),
            id,
            audio_path: None,
            language: None,
            model_name: None,
        };

        let json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", json)?;
        process.stdin.flush()?;

        // Read response
        let mut line = String::new();
        process.stdout.read_line(&mut line)?;

        let response: ASRResponse = serde_json::from_str(&line)?;

        if response.error.is_some() {
            return Ok(false);
        }

        Ok(response.result.is_some())
    }

    /// Transcribe an audio file
    pub fn transcribe(
        &mut self,
        audio_path: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let process = match self.process.as_mut() {
            Some(p) => p,
            None => {
                return Err(anyhow!(
                    "ASR engine not running. Call start_asr_engine first."
                ));
            }
        };

        // Check if process is still alive
        match process.child.try_wait() {
            Ok(Some(status)) => {
                self.process = None;
                return Err(anyhow!("ASR engine exited unexpectedly: {:?}", status));
            }
            Ok(None) => {} // Still running
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

        log::info!("Sending transcribe request for: {} with language: {:?}", audio_path, language);

        // Send request
        let json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", json)
            .map_err(|e| anyhow!("Failed to write to ASR engine: {}", e))?;
        process
            .stdin
            .flush()
            .map_err(|e| anyhow!("Failed to flush stdin: {}", e))?;

        // Read response
        let mut line = String::new();
        process
            .stdout
            .read_line(&mut line)
            .map_err(|e| anyhow!("Failed to read from ASR engine: {}", e))?;

        log::debug!("ASR response: {}", line.trim());

        let response: ASRResponse =
            serde_json::from_str(&line).map_err(|e| anyhow!("Failed to parse response: {}", e))?;

        // Check for error
        if let Some(error) = response.error {
            return Err(anyhow!("ASR engine error: {}", error));
        }

        // Parse result
        let result = response
            .result
            .ok_or_else(|| anyhow!("No result in ASR response"))?;

        // Check for transcription error
        if !result.success {
            if let Some(error) = result.error {
                return Err(anyhow!("Transcription failed: {}", error));
            }
            return Err(anyhow!("Transcription failed: unknown error"));
        }

        // Convert segments
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
