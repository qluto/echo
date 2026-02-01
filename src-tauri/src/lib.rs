#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

mod audio_capture;
mod clipboard;
mod handy_keys;
mod hotkey;
mod input;
mod transcription;

use std::sync::Mutex;
use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager, WindowEvent,
};
use tauri_plugin_store::StoreExt;

pub use input::EnigoState;
pub use transcription::{ASREngine, ModelStatus, WarmupResult};

/// Application settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub hotkey: String,
    pub language: String,
    pub auto_insert: bool,
    pub device_name: Option<String>,
    pub model_name: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CommandOrControl+Shift+Space".to_string(),
            language: "auto".to_string(),
            auto_insert: true,
            device_name: None,
            model_name: None,
        }
    }
}

const SETTINGS_STORE_FILE: &str = "settings.json";
const SETTINGS_KEY: &str = "settings";

/// Load settings from persistent store
fn load_settings_from_store(app: &tauri::App) -> Settings {
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
fn save_settings_to_store(app: &tauri::AppHandle, settings: &Settings) -> Result<(), String> {
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

/// Try to initialize Enigo (keyboard/mouse simulation).
/// On macOS, this will return an error if accessibility permissions are not granted.
#[tauri::command]
fn initialize_enigo(app: tauri::AppHandle) -> Result<(), String> {
    // Check if already initialized
    if app.try_state::<EnigoState>().is_some() {
        log::debug!("Enigo already initialized");
        return Ok(());
    }

    // Try to initialize
    match EnigoState::new() {
        Ok(enigo_state) => {
            app.manage(enigo_state);
            log::info!("Enigo initialized successfully after permission grant");
            Ok(())
        }
        Err(e) => {
            log::warn!(
                "Failed to initialize Enigo: {} (accessibility permissions may not be granted)",
                e
            );
            Err(format!("Failed to initialize input system: {}", e))
        }
    }
}

#[tauri::command]
fn insert_text(text: String, app: tauri::AppHandle) -> Result<(), String> {
    // Try to get or initialize EnigoState
    let enigo_state = match app.try_state::<EnigoState>() {
        Some(state) => state,
        None => {
            // Try to initialize
            initialize_enigo(app.clone())?;
            app.try_state::<EnigoState>()
                .ok_or_else(|| "Failed to initialize Enigo".to_string())?
        }
    };

    let mut enigo = enigo_state.0.lock().map_err(|e| format!("Failed to lock Enigo: {}", e))?;
    clipboard::paste_via_clipboard(&mut enigo, &text, &app)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_audio_devices() -> Result<Vec<AudioDevice>, String> {
    audio_capture::get_audio_devices().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_audio_device(
    device_name: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.device_name = Some(device_name);
    let settings_clone = settings.clone();
    drop(settings);
    save_settings_to_store(&app, &settings_clone)?;
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
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    log::info!("Updating settings: language={}, hotkey={}, auto_insert={}, model={:?}",
        settings.language, settings.hotkey, settings.auto_insert, settings.model_name);
    let mut current_settings = state.settings.lock().map_err(|e| e.to_string())?;
    *current_settings = settings.clone();
    drop(current_settings);

    // Persist to store
    save_settings_to_store(&app, &settings)?;
    Ok(())
}

#[tauri::command]
fn register_global_hotkey(hotkey: String, app: tauri::AppHandle) -> Result<(), String> {
    let result = handy_keys::register_hotkey(&app, &hotkey);
    if result.is_ok() {
        // Emit success event
        app.emit("hotkey-registered", serde_json::json!({
            "hotkey": hotkey
        })).ok();
    }
    result
}

#[tauri::command]
fn unregister_global_hotkey(app: tauri::AppHandle) -> Result<(), String> {
    handy_keys::unregister_hotkey(&app)
}

#[tauri::command]
fn start_hotkey_recording(app: tauri::AppHandle) -> Result<(), String> {
    handy_keys::start_recording(&app)
}

#[tauri::command]
fn stop_hotkey_recording(app: tauri::AppHandle) -> Result<(), String> {
    handy_keys::stop_recording(&app)
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
    let result = asr_engine.load_model().map_err(|e| e.to_string())?;

    // Also load VAD model for speech detection
    if let Err(e) = asr_engine.load_vad() {
        log::warn!("Failed to load VAD model (VAD will be disabled): {}", e);
    }

    Ok(result)
}

#[tauri::command]
fn warmup_asr_model(state: tauri::State<'_, AppState>) -> Result<WarmupResult, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;

    // Warmup ASR model
    let result = asr_engine.warmup_model().map_err(|e| e.to_string())?;

    // Also warmup VAD if loaded
    if let Err(e) = asr_engine.warmup_vad() {
        log::warn!("Failed to warmup VAD (non-critical): {}", e);
    }

    Ok(result)
}

#[tauri::command]
fn set_asr_model(model_name: String, app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    let result = asr_engine.set_model(&model_name).map_err(|e| e.to_string())?;
    drop(asr_engine);

    // Update and save settings with new model name
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.model_name = Some(model_name);
    let settings_clone = settings.clone();
    drop(settings);
    save_settings_to_store(&app, &settings_clone)?;

    Ok(result)
}

/// Check if accessibility permissions are granted
#[tauri::command]
fn check_accessibility_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "ApplicationServices", kind = "framework")]
        extern "C" {
            fn AXIsProcessTrusted() -> bool;
        }

        unsafe { AXIsProcessTrusted() }
    }

    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Request accessibility permissions (shows system prompt on macOS)
#[tauri::command]
fn request_accessibility_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::c_void;
        use std::ptr;

        #[link(name = "ApplicationServices", kind = "framework")]
        extern "C" {
            fn AXIsProcessTrustedWithOptions(options: *const c_void) -> bool;
        }

        #[link(name = "CoreFoundation", kind = "framework")]
        extern "C" {
            fn CFDictionaryCreate(
                allocator: *const c_void,
                keys: *const *const c_void,
                values: *const *const c_void,
                num_values: isize,
                key_callbacks: *const c_void,
                value_callbacks: *const c_void,
            ) -> *const c_void;
            fn CFRelease(cf: *const c_void);
            static kCFTypeDictionaryKeyCallBacks: c_void;
            static kCFTypeDictionaryValueCallBacks: c_void;
            static kCFBooleanTrue: *const c_void;
        }

        // kAXTrustedCheckOptionPrompt key
        extern "C" {
            static kAXTrustedCheckOptionPrompt: *const c_void;
        }

        unsafe {
            let keys = [kAXTrustedCheckOptionPrompt];
            let values = [kCFBooleanTrue];

            let options = CFDictionaryCreate(
                ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );

            let result = AXIsProcessTrustedWithOptions(options);
            CFRelease(options);
            result
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Open System Settings to Accessibility pane
#[tauri::command]
fn open_accessibility_settings() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
            .spawn()
            .map_err(|e| format!("Failed to open System Settings: {}", e))?;
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(())
    }
}

/// Restart the application
#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    app.restart();
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
            // Load settings from persistent store
            let settings = load_settings_from_store(app);
            let hotkey = settings.hotkey.clone();
            let model_name = settings.model_name.clone();

            // Create ASR engine and apply saved model name if present
            let mut asr_engine = ASREngine::new();
            if let Some(ref model) = model_name {
                log::info!("Applying saved model: {}", model);
                if let Err(e) = asr_engine.set_model(model) {
                    log::warn!("Failed to set saved model '{}': {}", model, e);
                }
            }

            let app_state = AppState {
                asr_engine: Mutex::new(asr_engine),
                settings: Mutex::new(settings),
                recording_state: Mutex::new(RecordingState::default()),
            };
            app.manage(app_state);

            // Try to initialize Enigo, but don't fail if permissions are not granted
            match EnigoState::new() {
                Ok(enigo_state) => {
                    app.manage(enigo_state);
                    log::info!("Enigo initialized successfully");
                }
                Err(e) => {
                    log::warn!(
                        "Enigo initialization deferred: {} (will initialize when accessibility permission is granted)",
                        e
                    );
                }
            }

            // Make float window transparent on macOS
            if let Some(float_window) = app.get_webview_window("float") {
                make_window_transparent(&float_window);
            }

            // Setup system tray
            setup_system_tray(app)?;

            // Initialize handy-keys and register saved hotkey
            // Use a blocking spawn to ensure initialization completes before the app is ready
            let app_handle = app.handle().clone();
            let hotkey_clone = hotkey.clone();
            std::thread::spawn(move || {
                // Small delay to ensure the app is fully initialized
                std::thread::sleep(std::time::Duration::from_millis(100));

                if let Err(e) = handy_keys::init(&app_handle) {
                    log::error!("Failed to initialize handy-keys: {}", e);
                    // Emit error to frontend
                    app_handle.emit("hotkey-init-error", serde_json::json!({
                        "error": format!("Failed to initialize hotkey system: {}. Please ensure Echo has accessibility permissions in System Settings > Privacy & Security > Accessibility.", e)
                    })).ok();
                    return;
                }

                // Retry hotkey registration a few times in case of timing issues
                let mut last_error = None;
                for attempt in 0..3 {
                    if attempt > 0 {
                        log::info!("Retrying hotkey registration (attempt {})", attempt + 1);
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }

                    match handy_keys::register_hotkey(&app_handle, &hotkey_clone) {
                        Ok(()) => {
                            log::info!("Hotkey '{}' registered successfully", hotkey_clone);
                            // Emit success to frontend
                            app_handle.emit("hotkey-registered", serde_json::json!({
                                "hotkey": hotkey_clone
                            })).ok();
                            return;
                        }
                        Err(e) => {
                            last_error = Some(e);
                        }
                    }
                }

                // All attempts failed
                if let Some(e) = last_error {
                    log::error!("Failed to register hotkey '{}' after 3 attempts: {}", hotkey_clone, e);
                    app_handle.emit("hotkey-init-error", serde_json::json!({
                        "error": format!("Failed to register hotkey '{}': {}. Please check accessibility permissions.", hotkey_clone, e)
                    })).ok();
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Hide main window instead of closing the app
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    // Prevent default close behavior
                    api.prevent_close();
                    // Hide the window instead
                    if let Err(e) = window.hide() {
                        log::error!("Failed to hide window: {}", e);
                    }
                    log::info!("Main window hidden, app continues running in tray");
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            transcribe,
            initialize_enigo,
            insert_text,
            get_audio_devices,
            set_audio_device,
            get_settings,
            update_settings,
            register_global_hotkey,
            unregister_global_hotkey,
            start_hotkey_recording,
            stop_hotkey_recording,
            ping_asr_engine,
            start_asr_engine,
            stop_asr_engine,
            get_model_status,
            load_asr_model,
            warmup_asr_model,
            set_asr_model,
            check_accessibility_permission,
            request_accessibility_permission,
            open_accessibility_settings,
            restart_app,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Setup system tray icon and menu
fn setup_system_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // Create menu items
    let show_item = MenuItem::with_id(app, "show", "ウィンドウを表示", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "終了", true, None::<&str>)?;

    // Create menu
    let menu = Menu::with_items(app, &[&show_item, &quit_item])?;

    // Load tray icon using include_image! macro (embeds at compile time)
    // Custom Echo icon with sound wave arcs
    const TRAY_ICON: Image<'_> = tauri::include_image!("icons/tray-icon.png");

    // Build tray icon
    let _tray = TrayIconBuilder::new()
        .icon(TRAY_ICON)
        .icon_as_template(true) // Use as template for macOS (adapts to light/dark mode)
        .tooltip("Echo - 音声入力")
        .menu(&menu)
        .show_menu_on_left_click(false) // Show menu on right-click only
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "quit" => {
                log::info!("Quit requested from tray menu");
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Show window on left-click
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                if let Some(window) = tray.app_handle().get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    log::info!("System tray initialized");
    Ok(())
}
