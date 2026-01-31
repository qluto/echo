#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

mod audio_capture;
mod clipboard;
mod hotkey;
mod input;
mod transcription;

use std::sync::Mutex;
use tauri::Manager;

pub use input::EnigoState;
pub use transcription::{ASREngine, ModelStatus};

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

/// Make a window fully transparent on macOS using with_webview
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn make_window_transparent(window: &tauri::WebviewWindow) {
    use cocoa::base::{id, NO};

    // First, configure the NSWindow
    if let Ok(ns_window) = window.ns_window() {
        let ns_window = ns_window as id;
        unsafe {
            // Set background color to clear
            let clear_color: id = msg_send![class!(NSColor), clearColor];
            let _: () = msg_send![ns_window, setBackgroundColor: clear_color];

            // Make window not opaque
            let _: () = msg_send![ns_window, setOpaque: NO];
        }
    }

    // Then, configure the webview using Tauri's with_webview
    #[allow(deprecated)]
    let _ = window.with_webview(|webview| {
        use cocoa::base::{id, NO};

        unsafe {
            let wk_webview: id = webview.inner() as id;
            if !wk_webview.is_null() {
                // Set WKWebView to not be opaque
                let _: () = msg_send![wk_webview, setOpaque: NO];

                // Set background color to clear (transparent)
                let clear_color: id = msg_send![class!(NSColor), clearColor];
                let _: () = msg_send![wk_webview, setBackgroundColor: clear_color];

                // Set underPageBackgroundColor to clear for full transparency
                let _: () = msg_send![wk_webview, setUnderPageBackgroundColor: clear_color];

                // Try private API: _setDrawsBackground:NO
                // This is the most reliable way to make WKWebView fully transparent
                let _: () = msg_send![wk_webview, _setDrawsBackground: NO];

                log::info!("Configured WKWebView for transparency via with_webview");
            }
        }
    });
}

#[cfg(not(target_os = "macos"))]
fn make_window_transparent(_window: &tauri::WebviewWindow) {
    // No-op on other platforms
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
    log::info!("Updating settings: language={}, hotkey={}, auto_insert={}",
        settings.language, settings.hotkey, settings.auto_insert);
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

#[tauri::command]
fn get_model_status(state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.get_model_status().map_err(|e| e.to_string())
}

#[tauri::command]
fn load_asr_model(state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.load_model().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_asr_model(model_name: String, state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.set_model(&model_name).map_err(|e| e.to_string())
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

            // Make float window transparent on macOS
            if let Some(float_window) = app.get_webview_window("float") {
                make_window_transparent(&float_window);
            }

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
            get_model_status,
            load_asr_model,
            set_asr_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
