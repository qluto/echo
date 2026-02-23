use std::sync::{Arc, Mutex};
use tauri_plugin_store::StoreExt;

use crate::continuous;
use crate::database;
use crate::transcription::ASREngine;

/// Application settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub hotkey: String,
    pub language: String,
    pub auto_insert: bool,
    pub device_name: Option<String>,
    pub model_name: Option<String>,
    #[serde(default)]
    pub postprocess: PostProcessSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CommandOrControl+Shift+Space".to_string(),
            language: "auto".to_string(),
            auto_insert: true,
            device_name: None,
            model_name: None,
            postprocess: PostProcessSettings::default(),
        }
    }
}

pub const SETTINGS_STORE_FILE: &str = "settings.json";
pub const SETTINGS_KEY: &str = "settings";

/// Load settings from persistent store
pub fn load_settings_from_store(app: &tauri::App) -> Settings {
    match app.store(SETTINGS_STORE_FILE) {
        Ok(store) => {
            match store.get(SETTINGS_KEY) {
                Some(value) => {
                    match serde_json::from_value::<Settings>(value.clone()) {
                        Ok(settings) => {
                            log::info!("Loaded settings from store: language={}, hotkey={}, model={:?}",
                                settings.language, settings.hotkey, settings.model_name);
                            settings
                        }
                        Err(e) => {
                            log::warn!("Failed to deserialize settings, using defaults: {}", e);
                            Settings::default()
                        }
                    }
                }
                None => {
                    log::info!("No saved settings found, using defaults");
                    Settings::default()
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to open settings store, using defaults: {}", e);
            Settings::default()
        }
    }
}

/// Save settings to persistent store
pub fn save_settings_to_store(app: &tauri::AppHandle, settings: &Settings) -> Result<(), String> {
    let store = app.store(SETTINGS_STORE_FILE).map_err(|e| e.to_string())?;
    let value = serde_json::to_value(settings).map_err(|e| e.to_string())?;
    store.set(SETTINGS_KEY, value);
    store.save().map_err(|e| e.to_string())?;
    log::info!("Settings saved to store");
    Ok(())
}

/// Transcription result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionResult {
    pub success: bool,
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: String,
    /// True if VAD detected no speech in the audio
    pub no_speech: Option<bool>,
}

/// Transcription segment with timestamps
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Audio device info
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
}

/// Post-processing settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostProcessSettings {
    pub enabled: bool,
    pub dictionary: std::collections::HashMap<String, String>,
    /// Custom system prompt for the LLM post-processor. If None, uses default.
    #[serde(default)]
    pub custom_prompt: Option<String>,
    /// Model name for post-processing LLM. If None, uses default.
    #[serde(default)]
    pub model_name: Option<String>,
    /// Custom system prompt for summarization. If None, uses default.
    #[serde(default)]
    pub custom_summary_prompt: Option<String>,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            dictionary: std::collections::HashMap::new(),
            custom_prompt: None,
            model_name: None,
            custom_summary_prompt: None,
        }
    }
}

/// Post-processing model status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostProcessModelStatus {
    pub model_name: String,
    pub loaded: bool,
    pub loading: bool,
    pub error: Option<String>,
    #[serde(default)]
    pub available_models: Vec<String>,
}

/// Post-processing result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostProcessResult {
    pub success: bool,
    pub processed_text: String,
    pub processing_time_ms: Option<f64>,
    pub error: Option<String>,
}

/// A transcription entry for summarization requests
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SummarizeEntry {
    pub text: String,
    pub created_at: String,
}

/// Summarization result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SummarizeResult {
    pub success: bool,
    pub summary: String,
    pub processing_time_ms: Option<f64>,
    pub error: Option<String>,
    pub entry_count: usize,
}

/// Application state
pub struct AppState {
    pub asr_engine: Arc<Mutex<ASREngine>>,
    pub settings: Mutex<Settings>,
    pub recording_state: Mutex<RecordingState>,
    pub transcription_db: Arc<Mutex<database::TranscriptionDb>>,
    pub continuous_pipeline: Mutex<Option<continuous::ContinuousPipeline>>,
}

/// Recording state
#[derive(Debug, Clone, Default)]
pub struct RecordingState {
    pub is_recording: bool,
    pub current_file: Option<String>,
    pub device_name: Option<String>,
}
