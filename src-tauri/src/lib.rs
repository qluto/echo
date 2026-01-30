mod audio_capture;
mod clipboard;
mod hotkey;
mod input;
mod transcription;

use std::sync::Mutex;
use tauri::Manager;

pub use input::EnigoState;
pub use transcription::ASREngine;

/// Application settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub hotkey: String,
    pub language: String,
    pub auto_insert: bool,
    pub device_name: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CommandOrControl+Shift+Space".to_string(),
            language: "auto".to_string(),
            auto_insert: true,
            device_name: None,
        }
    }
}

/// Transcription result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionResult {
    pub success: bool,
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: String,
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

/// Application state
pub struct AppState {
    pub asr_engine: Mutex<ASREngine>,
    pub settings: Mutex<Settings>,
    pub recording_state: Mutex<RecordingState>,
}

/// Recording state
#[derive(Debug, Clone, Default)]
pub struct RecordingState {
    pub is_recording: bool,
    pub current_file: Option<String>,
    pub device_name: Option<String>,
}

// Tauri commands
#[tauri::command]
fn start_recording(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut recording = state.recording_state.lock().map_err(|e| e.to_string())?;
    if recording.is_recording {
        return Err("Already recording".to_string());
    }

    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    let device_name = settings.device_name.clone();
    drop(settings);

    match audio_capture::start_recording(device_name) {
        Ok(file_path) => {
            recording.is_recording = true;
            recording.current_file = Some(file_path);
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let mut recording = state.recording_state.lock().map_err(|e| e.to_string())?;
    if !recording.is_recording {
        return Err("Not recording".to_string());
    }

    recording.is_recording = false;
    let file_path = recording
        .current_file
        .take()
        .ok_or("No recording file")?;

    audio_capture::stop_recording().map_err(|e| e.to_string())?;
    Ok(file_path)
}

#[tauri::command]
fn transcribe(
    audio_path: String,
    language: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<TranscriptionResult, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine
        .transcribe(&audio_path, language.as_deref())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn insert_text(
    text: String,
    app: tauri::AppHandle,
    enigo_state: tauri::State<'_, EnigoState>,
) -> Result<(), String> {
    clipboard::set_clipboard_text(&app, &text).map_err(|e| e.to_string())?;
    std::thread::sleep(std::time::Duration::from_millis(50));
    input::send_paste(&enigo_state).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_audio_devices() -> Result<Vec<AudioDevice>, String> {
    audio_capture::get_audio_devices().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_audio_device(
    device_name: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.device_name = Some(device_name);
    Ok(())
}

#[tauri::command]
fn get_settings(state: tauri::State<'_, AppState>) -> Result<Settings, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    Ok(settings.clone())
}

#[tauri::command]
fn update_settings(
    settings: Settings,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut current_settings = state.settings.lock().map_err(|e| e.to_string())?;
    *current_settings = settings;
    Ok(())
}

#[tauri::command]
fn register_global_hotkey(hotkey: String, app: tauri::AppHandle) -> Result<(), String> {
    hotkey::register_hotkey(&app, &hotkey).map_err(|e| e.to_string())
}

#[tauri::command]
fn unregister_global_hotkey(app: tauri::AppHandle) -> Result<(), String> {
    hotkey::unregister_all_hotkeys(&app).map_err(|e| e.to_string())
}

#[tauri::command]
fn ping_asr_engine(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    Ok(asr_engine.ping().unwrap_or(false))
}

#[tauri::command]
fn start_asr_engine(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.start(&app).map_err(|e| e.to_string())
}

#[tauri::command]
fn stop_asr_engine(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.stop().map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_log::Builder::new().build())
        .setup(|app| {
            let app_state = AppState {
                asr_engine: Mutex::new(ASREngine::new()),
                settings: Mutex::new(Settings::default()),
                recording_state: Mutex::new(RecordingState::default()),
            };
            app.manage(app_state);
            app.manage(EnigoState::new());

            // Register default hotkey
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) =
                    hotkey::register_hotkey(&app_handle, "CommandOrControl+Shift+Space")
                {
                    log::error!("Failed to register default hotkey: {}", e);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            transcribe,
            insert_text,
            get_audio_devices,
            set_audio_device,
            get_settings,
            update_settings,
            register_global_hotkey,
            unregister_global_hotkey,
            ping_asr_engine,
            start_asr_engine,
            stop_asr_engine,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
