//! Hotkey management module
//!
//! This module handles global hotkey registration and events
//! using tauri-plugin-global-shortcut.

use anyhow::Result;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Register a global hotkey
pub fn register_hotkey(app: &AppHandle, hotkey: &str) -> Result<()> {
    let shortcut: Shortcut = hotkey
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid hotkey: {}", e))?;

    let app_handle = app.clone();
    app.global_shortcut()
        .on_shortcut(shortcut, move |_app, _shortcut, event| {
            match event.state() {
                ShortcutState::Pressed => {
                    handle_hotkey_pressed(&app_handle);
                }
                ShortcutState::Released => {
                    handle_hotkey_released(&app_handle);
                }
            }
        })?;

    app.global_shortcut().register(shortcut)?;
    log::info!("Registered hotkey: {}", hotkey);

    Ok(())
}

fn handle_hotkey_pressed(app: &AppHandle) {
    log::info!("Hotkey pressed - starting recording");
    app.emit(
        "recording-state-change",
        serde_json::json!({"state": "recording"}),
    )
    .ok();

    // Get device name from settings
    let device_name = if let Some(state) = app.try_state::<crate::AppState>() {
        state
            .settings
            .lock()
            .ok()
            .and_then(|s| s.device_name.clone())
    } else {
        None
    };

    // Start recording
    match crate::audio_capture::start_recording(device_name) {
        Ok(file_path) => {
            log::info!("Recording started: {}", file_path);

            // Store file path in recording state
            if let Some(state) = app.try_state::<crate::AppState>() {
                if let Ok(mut recording) = state.recording_state.lock() {
                    recording.is_recording = true;
                    recording.current_file = Some(file_path);
                }
            }
        }
        Err(e) => {
            log::error!("Failed to start recording: {}", e);
            app.emit("error", e.to_string()).ok();
            app.emit(
                "recording-state-change",
                serde_json::json!({"state": "idle"}),
            )
            .ok();
        }
    }
}

fn handle_hotkey_released(app: &AppHandle) {
    log::info!("Hotkey released - stopping recording");
    app.emit(
        "recording-state-change",
        serde_json::json!({"state": "transcribing"}),
    )
    .ok();

    // Get the recorded file path and settings
    let (file_path, language, auto_insert) =
        if let Some(state) = app.try_state::<crate::AppState>() {
            let file_path = state
                .recording_state
                .lock()
                .ok()
                .and_then(|mut r| {
                    r.is_recording = false;
                    r.current_file.take()
                });
            let (language, auto_insert) = state
                .settings
                .lock()
                .ok()
                .map(|s| (s.language.clone(), s.auto_insert))
                .unwrap_or(("auto".to_string(), true));
            (file_path, language, auto_insert)
        } else {
            (None, "auto".to_string(), true)
        };

    let app_clone = app.clone();
    std::thread::spawn(move || {
        // Stop recording
        if let Err(e) = crate::audio_capture::stop_recording() {
            log::error!("Failed to stop recording: {}", e);
            app_clone.emit("error", e.to_string()).ok();
            app_clone
                .emit(
                    "recording-state-change",
                    serde_json::json!({"state": "idle"}),
                )
                .ok();
            return;
        }

        // Get file path
        let file_path = match file_path {
            Some(path) => path,
            None => {
                log::error!("No recording file path");
                app_clone.emit("error", "No recording file path").ok();
                app_clone
                    .emit(
                        "recording-state-change",
                        serde_json::json!({"state": "idle"}),
                    )
                    .ok();
                return;
            }
        };

        log::info!("Transcribing: {} with language setting: {}", file_path, language);

        // Call ASR engine to transcribe
        let transcription_result =
            if let Some(state) = app_clone.try_state::<crate::AppState>() {
                if let Ok(mut asr_engine) = state.asr_engine.lock() {
                    let lang = if language == "auto" {
                        log::info!("Language is auto, passing None to ASR engine");
                        None
                    } else {
                        log::info!("Passing language '{}' to ASR engine", language);
                        Some(language.as_str())
                    };
                    asr_engine.transcribe(&file_path, lang)
                } else {
                    Err(anyhow::anyhow!("Failed to lock ASR engine"))
                }
            } else {
                Err(anyhow::anyhow!("AppState not available"))
            };

        match transcription_result {
            Ok(result) => {
                log::info!("Transcription complete: {} chars", result.text.len());

                // Emit transcription result
                app_clone
                    .emit(
                        "transcription-complete",
                        serde_json::json!({
                            "result": result,
                            "error": null
                        }),
                    )
                    .ok();

                // Auto-insert if enabled
                if auto_insert && !result.text.is_empty() {
                    log::info!("Auto-inserting text via main thread");

                    let text = result.text.clone();
                    let app_for_paste = app_clone.clone();

                    // Execute paste on main thread for reliability
                    if let Err(e) = app_clone.run_on_main_thread(move || {
                        if let Some(enigo_state) = app_for_paste.try_state::<crate::EnigoState>() {
                            if let Err(e) = crate::clipboard::paste_with_restore(
                                &app_for_paste,
                                &text,
                                &enigo_state,
                            ) {
                                log::error!("Failed to paste: {}", e);
                            } else {
                                log::info!("Text inserted successfully");
                            }
                        } else {
                            log::error!("EnigoState not available");
                        }
                    }) {
                        log::error!("Failed to run paste on main thread: {:?}", e);
                    }
                }
            }
            Err(e) => {
                log::error!("Transcription failed: {}", e);
                app_clone
                    .emit(
                        "transcription-complete",
                        serde_json::json!({
                            "result": null,
                            "error": e.to_string()
                        }),
                    )
                    .ok();
            }
        }

        // Clean up temp file
        if let Err(e) = std::fs::remove_file(&file_path) {
            log::warn!("Failed to delete temp file: {}", e);
        }

        app_clone
            .emit(
                "recording-state-change",
                serde_json::json!({"state": "idle"}),
            )
            .ok();
    });
}

/// Unregister all hotkeys
pub fn unregister_all_hotkeys(app: &AppHandle) -> Result<()> {
    app.global_shortcut().unregister_all()?;
    log::info!("Unregistered all hotkeys");
    Ok(())
}
