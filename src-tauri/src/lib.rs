#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

mod active_app;
mod audio_capture;
mod clipboard;
mod commands;
mod continuous;
mod database;
mod handy_keys;
mod hotkey;
mod input;
mod transcription;
mod types;
mod vad;

use std::sync::{Arc, Mutex};
use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager, WindowEvent,
};

pub use input::EnigoState;
pub use transcription::{ASREngine, ModelCacheStatus, ModelStatus, WarmupResult};
pub use types::*;

/// Make a window fully transparent on macOS using with_webview
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn make_window_transparent(window: &tauri::WebviewWindow) {
    use cocoa::base::{id, NO, YES};

    // First, configure the NSWindow
    if let Ok(ns_window) = window.ns_window() {
        let ns_window = ns_window as id;
        unsafe {
            // Set background color to clear
            let clear_color: id = msg_send![class!(NSColor), clearColor];
            let _: () = msg_send![ns_window, setBackgroundColor: clear_color];

            // Make window not opaque
            let _: () = msg_send![ns_window, setOpaque: NO];

            // Keep float window interactive and visible even when other apps are active.
            // NSStatusWindowLevel keeps it above normal/floating app windows.
            let status_window_level: i64 = 25;
            let _: () = msg_send![ns_window, setLevel: status_window_level];
            let _: () = msg_send![ns_window, setIgnoresMouseEvents: NO];
            let _: () = msg_send![ns_window, setHidesOnDeactivate: NO];
            let _: () = msg_send![ns_window, setAcceptsMouseMovedEvents: YES];

            // Keep window present across spaces/fullscreen apps as an auxiliary overlay.
            // NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
            let collection_behavior: u64 = (1 << 0) | (1 << 8);
            let _: () = msg_send![ns_window, setCollectionBehavior: collection_behavior];
            let _: () = msg_send![ns_window, orderFrontRegardless];
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Debug)
                .level_for("tao", log::LevelFilter::Warn)
                .level_for("tao::platform_impl", log::LevelFilter::Warn)
                .build(),
        )
        .setup(|app| {
            // Load settings from persistent store
            let settings = load_settings_from_store(app);
            let hotkey = settings.hotkey.clone();

            // Create ASR engine. Saved model selection is applied when the engine starts.
            let asr_engine = ASREngine::new();

            // Initialize transcription database
            let data_dir = app.path().app_data_dir().map_err(|e| {
                anyhow::anyhow!("Failed to get app data dir: {}", e)
            })?;
            std::fs::create_dir_all(&data_dir).map_err(|e| {
                anyhow::anyhow!("Failed to create data dir: {}", e)
            })?;
            let db_path = data_dir.join("transcriptions.db");
            let db = database::TranscriptionDb::open(&db_path).map_err(|e| {
                anyhow::anyhow!("Failed to open transcription database: {}", e)
            })?;
            log::info!("Transcription database opened at {:?}", db_path);

            let app_state = AppState {
                asr_engine: Arc::new(Mutex::new(asr_engine)),
                settings: Mutex::new(settings),
                recording_state: Mutex::new(RecordingState::default()),
                transcription_db: Arc::new(Mutex::new(db)),
                continuous_pipeline: Mutex::new(None),
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
                let _ = float_window.set_always_on_top(true);
                let _ = float_window.set_visible_on_all_workspaces(true);
            }

            // Setup system tray
            setup_system_tray(app)?;

            // Initialize handy-keys and register saved hotkey
            let app_handle = app.handle().clone();
            let hotkey_clone = hotkey.clone();
            std::thread::spawn(move || {
                // Small delay to ensure the app is fully initialized
                std::thread::sleep(std::time::Duration::from_millis(100));

                if let Err(e) = handy_keys::init(&app_handle) {
                    log::error!("Failed to initialize handy-keys: {}", e);
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
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    if let Err(e) = window.hide() {
                        log::error!("Failed to hide window: {}", e);
                    }
                    log::info!("Main window hidden, app continues running in tray");
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_recording,
            commands::stop_recording,
            commands::transcribe,
            commands::initialize_enigo,
            commands::insert_text,
            commands::get_audio_devices,
            commands::set_audio_device,
            commands::get_settings,
            commands::update_settings,
            commands::register_global_hotkey,
            commands::unregister_global_hotkey,
            commands::start_hotkey_recording,
            commands::stop_hotkey_recording,
            commands::ping_asr_engine,
            commands::start_asr_engine,
            commands::stop_asr_engine,
            commands::get_model_status,
            commands::load_asr_model,
            commands::load_asr_model_async,
            commands::warmup_asr_model,
            commands::set_asr_model,
            commands::is_model_cached,
            commands::check_accessibility_permission,
            commands::request_accessibility_permission,
            commands::open_accessibility_settings,
            commands::restart_app,
            commands::get_frontmost_app,
            commands::get_postprocess_settings,
            commands::update_postprocess_settings,
            commands::load_postprocess_model,
            commands::unload_postprocess_model,
            commands::is_postprocess_model_cached,
            commands::postprocess_text,
            commands::set_postprocess_model,
            commands::get_postprocess_model_status,
            commands::start_continuous_listening,
            commands::stop_continuous_listening,
            commands::get_continuous_listening_status,
            commands::get_transcription_history,
            commands::search_transcription_history,
            commands::delete_transcription_entry,
            commands::clear_transcription_history,
            commands::summarize_recent_transcriptions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Setup system tray icon and menu
fn setup_system_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let show_item = MenuItem::with_id(app, "show", "ウィンドウを表示", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "終了", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_item, &quit_item])?;

    const TRAY_ICON: Image<'_> = tauri::include_image!("icons/tray-icon.png");

    let _tray = TrayIconBuilder::new()
        .icon(TRAY_ICON)
        .icon_as_template(true)
        .tooltip("Echo - 音声入力")
        .menu(&menu)
        .show_menu_on_left_click(false)
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
