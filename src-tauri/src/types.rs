use std::sync::{Arc, Mutex};
use tauri_plugin_store::StoreExt;

use crate::continuous;
use crate::database;
use crate::memo;
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
    #[serde(default)]
    pub gated_access: GatedAccessSettings,
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
            gated_access: GatedAccessSettings::default(),
        }
    }
}

/// Settings for accessing gated HuggingFace models (e.g. Cohere Transcribe).
/// The token is stored in the same plaintext settings.json as other settings —
/// the threat model matches HuggingFace CLI's own ~/.cache/huggingface/token.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GatedAccessSettings {
    /// User explicitly opted in to use gated models.
    pub enabled: bool,
    /// HuggingFace personal access token. None = not configured.
    pub hf_token: Option<String>,
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
    /// Raw transcription before AI post-processing. Set only when post-processing
    /// changed the text, so the UI can show both (a fallback if the LLM misbehaves).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
}

/// Transcription segment with timestamps
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Structured status event ("transcription-status") shared by the hotkey and
/// always-on paths, so silence and failures are visible in the UI instead of
/// being indistinguishable from success.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptionStatusEvent {
    pub kind: StatusKind,
    pub source: StatusSource,
    /// Raw error detail for debugging/tooltips. None for no_speech.
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatusKind {
    /// VAD gate or empty ASR text — normal outcome, not an error.
    NoSpeech,
    AsrError,
    EngineBusy,
    DbError,
    AudioError,
    /// Reserved for mic-permission detection (macOS denial currently
    /// manifests as silent audio, i.e. NoSpeech); kept so the event
    /// contract is forward-compatible.
    #[allow(dead_code)]
    PermissionDenied,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatusSource {
    Hotkey,
    Continuous,
}

impl TranscriptionStatusEvent {
    pub fn emit(
        app: &tauri::AppHandle,
        source: StatusSource,
        kind: StatusKind,
        message: Option<String>,
    ) {
        use tauri::Emitter;
        let event = Self { kind, source, message };
        if let Err(e) = app.emit("transcription-status", &event) {
            log::warn!("Failed to emit transcription-status: {}", e);
        }
    }
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
    /// Per-app post-processing profiles, matched by exact bundle_id.
    #[serde(default)]
    pub app_profiles: Vec<AppProfile>,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            dictionary: std::collections::HashMap::new(),
            custom_prompt: None,
            model_name: None,
            custom_summary_prompt: None,
            app_profiles: Vec::new(),
        }
    }
}

/// A per-app post-processing profile: when the frontmost app's bundle_id
/// matches, `prompt` replaces the system prompt (e.g. Slack drafting mode).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppProfile {
    /// Exact bundle id, e.g. "com.tinyspeck.slackmacgap".
    pub bundle_id: String,
    /// Display label, e.g. "Slack".
    pub name: String,
    /// Full system prompt (always a concrete string, seeded by the UI).
    pub prompt: String,
    pub enabled: bool,
}

impl PostProcessSettings {
    /// System prompt override for the given frontmost app. An enabled profile
    /// with an exact bundle_id match takes precedence over the global
    /// custom_prompt; None means "use the built-in default prompt".
    pub fn resolve_prompt(&self, bundle_id: Option<&str>) -> Option<&str> {
        bundle_id
            .and_then(|bid| {
                self.app_profiles
                    .iter()
                    .find(|p| p.enabled && p.bundle_id == bid)
            })
            .map(|p| p.prompt.as_str())
            .or(self.custom_prompt.as_deref())
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
    /// The voice memo being recorded, if any.
    pub memo_recorder: Mutex<Option<memo::MemoRecorder>>,
}

/// Recording state
#[derive(Debug, Clone, Default)]
pub struct RecordingState {
    pub is_recording: bool,
    pub current_file: Option<String>,
    pub device_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_with(profiles: Vec<AppProfile>, custom: Option<&str>) -> PostProcessSettings {
        PostProcessSettings {
            app_profiles: profiles,
            custom_prompt: custom.map(String::from),
            ..PostProcessSettings::default()
        }
    }

    fn slack_profile(enabled: bool) -> AppProfile {
        AppProfile {
            bundle_id: "com.tinyspeck.slackmacgap".to_string(),
            name: "Slack".to_string(),
            prompt: "slack drafting prompt".to_string(),
            enabled,
        }
    }

    #[test]
    fn resolve_prompt_matches_enabled_profile() {
        let s = settings_with(vec![slack_profile(true)], Some("global"));
        assert_eq!(
            s.resolve_prompt(Some("com.tinyspeck.slackmacgap")),
            Some("slack drafting prompt")
        );
    }

    #[test]
    fn resolve_prompt_skips_disabled_profile() {
        let s = settings_with(vec![slack_profile(false)], Some("global"));
        assert_eq!(
            s.resolve_prompt(Some("com.tinyspeck.slackmacgap")),
            Some("global")
        );
    }

    #[test]
    fn resolve_prompt_falls_back_to_global_custom_prompt() {
        let s = settings_with(vec![slack_profile(true)], Some("global"));
        assert_eq!(s.resolve_prompt(Some("com.apple.Notes")), Some("global"));
        assert_eq!(s.resolve_prompt(None), Some("global"));
    }

    #[test]
    fn resolve_prompt_none_means_builtin_default() {
        let s = settings_with(vec![slack_profile(true)], None);
        assert_eq!(s.resolve_prompt(Some("com.apple.Notes")), None);
        assert_eq!(s.resolve_prompt(None), None);
    }

    #[test]
    fn old_settings_json_without_profiles_still_loads() {
        let json = r#"{"enabled":true,"dictionary":{}}"#;
        let s: PostProcessSettings = serde_json::from_str(json).unwrap();
        assert!(s.app_profiles.is_empty());
    }
}
