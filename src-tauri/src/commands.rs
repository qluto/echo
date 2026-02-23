use std::sync::Arc;
use tauri::{Emitter, Manager};

use crate::active_app;
use crate::audio_capture;
use crate::clipboard;
use crate::continuous;
use crate::database;
use crate::handy_keys;
use crate::input::EnigoState;
use crate::transcription::{ModelCacheStatus, ModelStatus, WarmupResult};
use crate::types::*;

#[tauri::command]
pub fn start_recording(state: tauri::State<'_, AppState>) -> Result<(), String> {
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
pub fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String, String> {
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
pub fn transcribe(
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
pub fn initialize_enigo(app: tauri::AppHandle) -> Result<(), String> {
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
pub fn insert_text(text: String, app: tauri::AppHandle) -> Result<(), String> {
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
pub fn get_audio_devices() -> Result<Vec<AudioDevice>, String> {
    audio_capture::get_audio_devices().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_audio_level() -> f32 {
    audio_capture::get_audio_level()
}

#[tauri::command]
pub fn set_audio_device(
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
pub fn get_settings(state: tauri::State<'_, AppState>) -> Result<Settings, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    Ok(settings.clone())
}

#[tauri::command]
pub fn update_settings(
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
pub fn register_global_hotkey(hotkey: String, app: tauri::AppHandle) -> Result<(), String> {
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
pub fn unregister_global_hotkey(app: tauri::AppHandle) -> Result<(), String> {
    handy_keys::unregister_hotkey(&app)
}

#[tauri::command]
pub fn start_hotkey_recording(app: tauri::AppHandle) -> Result<(), String> {
    handy_keys::start_recording(&app)
}

#[tauri::command]
pub fn stop_hotkey_recording(app: tauri::AppHandle) -> Result<(), String> {
    handy_keys::stop_recording(&app)
}

#[tauri::command]
pub fn ping_asr_engine(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    Ok(asr_engine.ping().unwrap_or(false))
}

#[tauri::command]
pub fn start_asr_engine(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.start(&app).map_err(|e| e.to_string())?;

    // Apply saved model after engine is running.
    // setup() cannot apply it because the sidecar process does not exist yet.
    let saved_model = {
        let settings = state.settings.lock().map_err(|e| e.to_string())?;
        settings.model_name.clone()
    };
    if let Some(model_name) = saved_model {
        if let Err(e) = asr_engine.set_model(&model_name) {
            log::warn!("Failed to apply saved model '{}' on engine start: {}", model_name, e);
        } else {
            log::info!("Applied saved model '{}' on engine start", model_name);
        }
    }

    Ok(())
}

#[tauri::command]
pub fn stop_asr_engine(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.stop().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_model_status(state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.get_model_status().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn load_asr_model(state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    let result = asr_engine.load_model().map_err(|e| e.to_string())?;

    // Also load VAD model for speech detection
    if let Err(e) = asr_engine.load_vad() {
        log::warn!("Failed to load VAD model (VAD will be disabled): {}", e);
    }

    Ok(result)
}

/// Async version that runs model loading in background thread
/// Emits "model-load-complete" or "model-load-error" events when done
#[tauri::command]
pub async fn load_asr_model_async(app: tauri::AppHandle) -> Result<(), String> {
    let app_handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let state = app_handle.state::<AppState>();
        let mut asr_engine = match state.asr_engine.lock() {
            Ok(e) => e,
            Err(e) => {
                log::error!("Failed to lock ASR engine: {}", e);
                let _ = app_handle.emit(
                    "model-load-error",
                    serde_json::json!({ "error": e.to_string() }),
                );
                return;
            }
        };

        log::info!("Starting background model load...");
        match asr_engine.load_model() {
            Ok(status) => {
                // Also load VAD model
                if let Err(e) = asr_engine.load_vad() {
                    log::warn!("Failed to load VAD model (VAD will be disabled): {}", e);
                }

                log::info!("Model loaded successfully: {}", status.model_name);
                let _ = app_handle.emit("model-load-complete", &status);
            }
            Err(e) => {
                log::error!("Failed to load model: {}", e);
                let _ = app_handle.emit(
                    "model-load-error",
                    serde_json::json!({ "error": e.to_string() }),
                );
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub fn warmup_asr_model(state: tauri::State<'_, AppState>) -> Result<WarmupResult, String> {
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
pub fn set_asr_model(model_name: String, app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<ModelStatus, String> {
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

#[tauri::command]
pub fn is_model_cached(model_name: Option<String>, state: tauri::State<'_, AppState>) -> Result<ModelCacheStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine
        .is_model_cached(model_name.as_deref())
        .map_err(|e| e.to_string())
}

// ===== Post-processing commands =====

#[tauri::command]
pub fn get_frontmost_app() -> active_app::ActiveAppInfo {
    active_app::get_frontmost_app()
}

#[tauri::command]
pub fn get_postprocess_settings(state: tauri::State<'_, AppState>) -> Result<PostProcessSettings, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    Ok(settings.postprocess.clone())
}

#[tauri::command]
pub fn update_postprocess_settings(
    postprocess: PostProcessSettings,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.postprocess = postprocess;
    let settings_clone = settings.clone();
    drop(settings);
    save_settings_to_store(&app, &settings_clone)?;
    Ok(())
}

#[tauri::command]
pub fn load_postprocess_model(state: tauri::State<'_, AppState>) -> Result<PostProcessModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.load_postprocess_model().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn unload_postprocess_model(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.unload_postprocess_model().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn is_postprocess_model_cached(state: tauri::State<'_, AppState>) -> Result<PostProcessModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.is_postprocess_model_cached().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn postprocess_text(
    text: String,
    app_name: Option<String>,
    app_bundle_id: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<PostProcessResult, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    let dictionary = if settings.postprocess.dictionary.is_empty() {
        None
    } else {
        Some(settings.postprocess.dictionary.clone())
    };
    let custom_prompt = settings.postprocess.custom_prompt.clone();
    drop(settings);

    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine
        .postprocess_text(
            &text,
            app_name.as_deref(),
            app_bundle_id.as_deref(),
            dictionary.as_ref(),
            custom_prompt.as_deref(),
        )
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_postprocess_model(
    model_name: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<PostProcessModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    let result = asr_engine.set_postprocess_model(&model_name).map_err(|e| e.to_string())?;
    drop(asr_engine);

    // Update and save settings with new model name
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.postprocess.model_name = Some(model_name);
    let settings_clone = settings.clone();
    drop(settings);
    save_settings_to_store(&app, &settings_clone)?;

    Ok(result)
}

#[tauri::command]
pub fn get_postprocess_model_status(state: tauri::State<'_, AppState>) -> Result<PostProcessModelStatus, String> {
    let mut asr_engine = state.asr_engine.lock().map_err(|e| e.to_string())?;
    asr_engine.get_postprocess_status().map_err(|e| e.to_string())
}

// ===== Continuous listening commands =====

#[tauri::command]
pub fn start_continuous_listening(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut pipeline = state.continuous_pipeline.lock().map_err(|e| e.to_string())?;
    if pipeline.is_some() {
        return Err("Already listening".to_string());
    }

    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    let language = if settings.language == "auto" {
        None
    } else {
        Some(settings.language.clone())
    };
    let device_name = settings.device_name.clone();
    drop(settings);

    let asr_engine = Arc::clone(&state.asr_engine);
    let db = Arc::clone(&state.transcription_db);

    let p = continuous::ContinuousPipeline::start(
        app,
        asr_engine,
        db,
        language,
        device_name,
        1.5,  // silence_sec
        60,   // max_segment_sec
    )
    .map_err(|e| e.to_string())?;

    *pipeline = Some(p);
    Ok(())
}

#[tauri::command]
pub fn stop_continuous_listening(
    state: tauri::State<'_, AppState>,
) -> Result<u32, String> {
    let mut pipeline = state.continuous_pipeline.lock().map_err(|e| e.to_string())?;
    match pipeline.take() {
        Some(mut p) => Ok(p.stop()),
        None => Err("Not listening".to_string()),
    }
}

#[tauri::command]
pub fn get_continuous_listening_status(
    state: tauri::State<'_, AppState>,
) -> Result<continuous::ContinuousListeningStatus, String> {
    let pipeline = state.continuous_pipeline.lock().map_err(|e| e.to_string())?;
    Ok(continuous::ContinuousListeningStatus {
        is_listening: pipeline.is_some(),
        segment_count: pipeline.as_ref().map_or(0, |p| p.segment_count()),
    })
}

// ===== Transcription history commands =====

#[tauri::command]
pub fn get_transcription_history(
    limit: Option<u32>,
    offset: Option<u32>,
    state: tauri::State<'_, AppState>,
) -> Result<database::HistoryPage, String> {
    let db = state.transcription_db.lock().map_err(|e| e.to_string())?;
    db.get_all(limit.unwrap_or(20), offset.unwrap_or(0))
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_transcription_history(
    query: String,
    limit: Option<u32>,
    offset: Option<u32>,
    state: tauri::State<'_, AppState>,
) -> Result<database::HistoryPage, String> {
    let db = state.transcription_db.lock().map_err(|e| e.to_string())?;
    db.search(&query, limit.unwrap_or(20), offset.unwrap_or(0))
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_transcription_entry(
    id: i64,
    state: tauri::State<'_, AppState>,
) -> Result<bool, String> {
    let db = state.transcription_db.lock().map_err(|e| e.to_string())?;
    db.delete(id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn clear_transcription_history(
    state: tauri::State<'_, AppState>,
) -> Result<u32, String> {
    let db = state.transcription_db.lock().map_err(|e| e.to_string())?;
    db.delete_all().map_err(|e| e.to_string())
}

// ===== Summarization commands =====

#[tauri::command]
pub async fn summarize_recent_transcriptions(
    minutes: Option<u32>,
    app: tauri::AppHandle,
) -> Result<SummarizeResult, String> {
    let state = app.state::<AppState>();
    let minutes = minutes.unwrap_or(30);

    // Query DB for recent entries
    let entries = {
        let db = state.transcription_db.lock().map_err(|e| e.to_string())?;
        db.get_recent(minutes).map_err(|e| e.to_string())?
    };

    if entries.is_empty() {
        return Ok(SummarizeResult {
            success: true,
            summary: String::new(),
            processing_time_ms: Some(0.0),
            error: None,
            entry_count: 0,
        });
    }

    let entry_count = entries.len();

    // Convert to SummarizeEntry
    let texts: Vec<SummarizeEntry> = entries
        .iter()
        .map(|e| SummarizeEntry {
            text: e.text.clone(),
            created_at: e.created_at.clone(),
        })
        .collect();

    // Detect dominant language from entries
    let language_hint = entries
        .iter()
        .filter_map(|e| e.language.as_deref())
        .next()
        .map(|s| s.to_string());

    // Read custom summary prompt from settings
    let custom_summary_prompt = state
        .settings
        .lock()
        .ok()
        .and_then(|s| s.postprocess.custom_summary_prompt.clone());

    // Run summarization in blocking thread (LLM inference takes seconds)
    let asr_engine = Arc::clone(&state.asr_engine);
    tauri::async_runtime::spawn_blocking(move || {
        let mut engine = asr_engine.lock().map_err(|e| e.to_string())?;
        let mut result = engine
            .summarize_transcriptions(&texts, language_hint.as_deref(), custom_summary_prompt.as_deref())
            .map_err(|e| e.to_string())?;
        result.entry_count = entry_count;
        Ok(result)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Check if accessibility permissions are granted
#[tauri::command]
pub fn check_accessibility_permission() -> bool {
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
pub fn request_accessibility_permission() -> bool {
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
pub fn open_accessibility_settings() -> Result<(), String> {
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
pub fn restart_app(app: tauri::AppHandle) {
    app.restart();
}
