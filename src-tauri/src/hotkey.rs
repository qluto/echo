//! Hotkey management module
//!
//! This module handles global hotkey registration and events
//! using tauri-plugin-global-shortcut.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Flag to track whether continuous listening was active when hotkey was pressed.
/// Used to auto-resume after hotkey transcription completes.
static WAS_LISTENING: AtomicBool = AtomicBool::new(false);

/// Register a global hotkey
#[allow(dead_code)]
pub fn register_hotkey(app: &AppHandle, hotkey: &str) -> Result<()> {
    // Unregister all existing hotkeys first
    if let Err(e) = app.global_shortcut().unregister_all() {
        log::warn!("Failed to unregister existing hotkeys: {}", e);
    }

    let shortcut: Shortcut = hotkey
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid hotkey '{}': {}", hotkey, e))?;

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

/// Trigger hotkey pressed event (can be called from handy_keys module)
pub fn trigger_hotkey_pressed(app: &AppHandle) {
    handle_hotkey_pressed(app);
}

/// Trigger hotkey released event (can be called from handy_keys module)
pub fn trigger_hotkey_released(app: &AppHandle) {
    handle_hotkey_released(app);
}

fn handle_hotkey_pressed(app: &AppHandle) {
    log::info!("Hotkey pressed - starting recording");

    // Pause continuous listening if active
    if let Some(state) = app.try_state::<crate::AppState>() {
        if let Ok(mut pipeline) = state.continuous_pipeline.lock() {
            if let Some(mut p) = pipeline.take() {
                log::info!("Pausing continuous listening for hotkey recording");
                p.stop();
                WAS_LISTENING.store(true, Ordering::SeqCst);
            }
        }
    }

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
    let (file_path, language, auto_insert, postprocess_settings) =
        if let Some(state) = app.try_state::<crate::AppState>() {
            let file_path = state
                .recording_state
                .lock()
                .ok()
                .and_then(|mut r| {
                    r.is_recording = false;
                    r.current_file.take()
                });
            let (language, auto_insert, postprocess) = state
                .settings
                .lock()
                .ok()
                .map(|s| (s.language.clone(), s.auto_insert, s.postprocess.clone()))
                .unwrap_or((
                    "auto".to_string(),
                    true,
                    crate::PostProcessSettings::default(),
                ));
            (file_path, language, auto_insert, postprocess)
        } else {
            (
                None,
                "auto".to_string(),
                true,
                crate::PostProcessSettings::default(),
            )
        };

    // Get frontmost app info for post-processing (before spawning thread to capture while app is still focused)
    let active_app = if postprocess_settings.enabled {
        Some(crate::active_app::get_frontmost_app())
    } else {
        None
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
                // Check if no speech was detected
                let is_no_speech = result.no_speech.unwrap_or(false);
                if is_no_speech {
                    log::info!("No speech detected in recording, skipping transcription");
                } else {
                    log::info!("Transcription complete: {} chars", result.text.len());
                }

                // Apply post-processing if enabled
                let final_text = if postprocess_settings.enabled
                    && !result.text.is_empty()
                    && !is_no_speech
                {
                    log::info!("Post-processing enabled, applying...");
                    if let Some(state) = app_clone.try_state::<crate::AppState>() {
                        if let Ok(mut asr_engine) = state.asr_engine.lock() {
                            let app_info = active_app.as_ref();
                            let dictionary = if postprocess_settings.dictionary.is_empty() {
                                None
                            } else {
                                Some(&postprocess_settings.dictionary)
                            };

                            match asr_engine.postprocess_text(
                                &result.text,
                                app_info.and_then(|a| a.app_name.as_deref()),
                                app_info.and_then(|a| a.bundle_id.as_deref()),
                                dictionary,
                                postprocess_settings.custom_prompt.as_deref(),
                            ) {
                                Ok(pp_result) => {
                                    if pp_result.success {
                                        log::info!(
                                            "Post-processing complete in {:?}ms: '{}' -> '{}'",
                                            pp_result.processing_time_ms,
                                            &result.text[..result.text.len().min(30)],
                                            &pp_result.processed_text[..pp_result.processed_text.len().min(30)]
                                        );
                                        pp_result.processed_text
                                    } else {
                                        log::warn!(
                                            "Post-processing failed: {:?}, using original text",
                                            pp_result.error
                                        );
                                        result.text.clone()
                                    }
                                }
                                Err(e) => {
                                    log::error!(
                                        "Post-processing error: {}, using original text",
                                        e
                                    );
                                    result.text.clone()
                                }
                            }
                        } else {
                            log::error!("Failed to lock ASR engine for post-processing");
                            result.text.clone()
                        }
                    } else {
                        result.text.clone()
                    }
                } else {
                    result.text.clone()
                };

                // Create result with possibly processed text for emission
                let emit_result = crate::TranscriptionResult {
                    success: result.success,
                    text: final_text.clone(),
                    segments: result.segments.clone(),
                    language: result.language.clone(),
                    no_speech: result.no_speech,
                };

                // Emit transcription result
                app_clone
                    .emit(
                        "transcription-complete",
                        serde_json::json!({
                            "result": emit_result,
                            "no_speech": is_no_speech,
                            "error": null
                        }),
                    )
                    .ok();

                // Auto-insert if enabled (skip if no speech detected)
                if auto_insert && !final_text.is_empty() && !is_no_speech {
                    log::info!("Auto-inserting text");

                    // Try to get or initialize EnigoState
                    let enigo_state = match app_clone.try_state::<crate::EnigoState>() {
                        Some(state) => Some(state),
                        None => {
                            // Try to initialize
                            log::info!("EnigoState not available, trying to initialize...");
                            match crate::EnigoState::new() {
                                Ok(state) => {
                                    app_clone.manage(state);
                                    log::info!("Enigo initialized successfully");
                                    app_clone.try_state::<crate::EnigoState>()
                                }
                                Err(e) => {
                                    log::error!("Failed to initialize Enigo: {} (accessibility permissions may not be granted)", e);
                                    None
                                }
                            }
                        }
                    };

                    if let Some(enigo_state) = enigo_state {
                        match enigo_state.0.lock() {
                            Ok(mut enigo) => {
                                if let Err(e) = crate::clipboard::paste_via_clipboard(
                                    &mut enigo,
                                    &final_text,
                                    &app_clone,
                                ) {
                                    log::error!("Failed to paste: {}", e);
                                } else {
                                    log::info!("Text inserted successfully");
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to lock Enigo: {}", e);
                            }
                        }
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

        // Resume continuous listening if it was active before hotkey press
        if WAS_LISTENING.swap(false, Ordering::SeqCst) {
            log::info!("Resuming continuous listening after hotkey transcription");
            if let Some(state) = app_clone.try_state::<crate::AppState>() {
                let settings = state.settings.lock().ok();
                let language = settings
                    .as_ref()
                    .map(|s| {
                        if s.language == "auto" {
                            None
                        } else {
                            Some(s.language.clone())
                        }
                    })
                    .unwrap_or(None);
                let device_name = settings.and_then(|s| s.device_name.clone());

                let asr_engine = Arc::clone(&state.asr_engine);
                let db = Arc::clone(&state.transcription_db);

                match crate::continuous::ContinuousPipeline::start(
                    app_clone.clone(),
                    asr_engine,
                    db,
                    language,
                    device_name,
                    1.5,
                    60,
                ) {
                    Ok(pipeline) => {
                        if let Ok(mut p) = state.continuous_pipeline.lock() {
                            *p = Some(pipeline);
                        }
                        log::info!("Continuous listening resumed");
                    }
                    Err(e) => {
                        log::error!("Failed to resume continuous listening: {}", e);
                    }
                }
            }
        }
    });
}

/// Unregister all hotkeys
#[allow(dead_code)]
pub fn unregister_all_hotkeys(app: &AppHandle) -> Result<()> {
    app.global_shortcut().unregister_all()?;
    log::info!("Unregistered all hotkeys");
    Ok(())
}
