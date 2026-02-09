//! Wake word listener module
//!
//! Provides continuous audio monitoring with energy-based speech detection.
//! When a speech segment is detected and transcribed, checks for a wake word prefix.
//! If matched, extracts the command text and auto-inserts it into the target app.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// RMS energy threshold to consider as speech (tunable)
const ENERGY_THRESHOLD: f32 = 0.015;
/// Silence duration (ms) before considering speech ended
const SILENCE_DURATION_MS: u64 = 800;
/// Minimum speech duration (ms) to bother transcribing
const MIN_SPEECH_DURATION_MS: u64 = 500;
/// Maximum speech duration (ms) to prevent runaway recording
const MAX_SPEECH_DURATION_MS: u64 = 30_000;
/// How often to check the buffer state (ms)
const POLL_INTERVAL_MS: u64 = 100;
/// Delay after activating target app before pasting (ms)
const APP_SWITCH_DELAY_MS: u64 = 300;
/// Delay after pasting before pressing Enter (ms)
const SEND_DELAY_MS: u64 = 150;

/// Wake word listener that continuously monitors audio for a trigger keyword
pub struct WakeWordListener {
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl WakeWordListener {
    /// Start the wake word listener in a background thread
    pub fn start(
        app: AppHandle,
        keyword: String,
        auto_send: bool,
        target_mode: String,
        device_name: Option<String>,
    ) -> Result<Self> {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let thread = thread::spawn(move || {
            log::info!(
                "Wake word listener started: keyword='{}', auto_send={}, target='{}'",
                keyword,
                auto_send,
                target_mode
            );
            app.emit(
                "wake-word-state",
                serde_json::json!({"state": "listening"}),
            )
            .ok();

            if let Err(e) = wake_word_loop(
                &app,
                &keyword,
                auto_send,
                &target_mode,
                device_name.as_deref(),
                &running_clone,
            ) {
                log::error!("Wake word listener error: {}", e);
                app.emit(
                    "wake-word-state",
                    serde_json::json!({"state": "error", "error": e.to_string()}),
                )
                .ok();
            }

            app.emit(
                "wake-word-state",
                serde_json::json!({"state": "stopped"}),
            )
            .ok();
            log::info!("Wake word listener stopped");
        });

        Ok(Self {
            running,
            thread: Some(thread),
        })
    }

    /// Stop the wake word listener
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Drop for WakeWordListener {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Shared state for the audio callback to communicate with the main loop
struct AudioState {
    /// Accumulated audio samples (mono, f32)
    buffer: Vec<f32>,
    /// Whether speech is currently being detected
    speech_active: bool,
    /// Last time speech energy was above threshold
    last_speech_time: Instant,
    /// Time when current speech segment started
    speech_start_time: Option<Instant>,
    /// Native sample rate of the audio device
    sample_rate: u32,
}

/// Main loop for wake word detection
fn wake_word_loop(
    app: &AppHandle,
    keyword: &str,
    auto_send: bool,
    target_mode: &str,
    device_name: Option<&str>,
    running: &AtomicBool,
) -> Result<()> {
    let host = cpal::default_host();

    let device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("Device not found: {}", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device"))?
    };

    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;

    log::info!(
        "Wake word audio: {} Hz, {} ch, format: {:?}",
        sample_rate,
        channels,
        supported_config.sample_format()
    );

    let state = Arc::new(Mutex::new(AudioState {
        buffer: Vec::new(),
        speech_active: false,
        last_speech_time: Instant::now(),
        speech_start_time: None,
        sample_rate,
    }));

    let config = supported_config.config();

    // Build the audio stream depending on sample format
    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => build_stream_f32(&device, &config, channels, state.clone())?,
        SampleFormat::I16 => build_stream_i16(&device, &config, channels, state.clone())?,
        fmt => return Err(anyhow!("Unsupported sample format: {:?}", fmt)),
    };

    stream.play()?;

    let keyword_lower = keyword.to_lowercase();

    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));

        // Skip processing if a hotkey recording is in progress
        if is_hotkey_recording_active(app) {
            // Discard any buffered audio during hotkey recording
            if let Ok(mut audio) = state.lock() {
                audio.buffer.clear();
                audio.speech_active = false;
                audio.speech_start_time = None;
            }
            continue;
        }

        let should_process = {
            let audio = state.lock().map_err(|e| anyhow!("{}", e))?;
            if !audio.speech_active {
                false
            } else {
                let silence = audio.last_speech_time.elapsed().as_millis() as u64;
                let speech_duration = audio
                    .speech_start_time
                    .map(|t| t.elapsed().as_millis() as u64)
                    .unwrap_or(0);

                silence > SILENCE_DURATION_MS || speech_duration > MAX_SPEECH_DURATION_MS
            }
        };

        if should_process {
            let (samples, sr) = {
                let mut audio = state.lock().map_err(|e| anyhow!("{}", e))?;
                let speech_duration = audio
                    .speech_start_time
                    .map(|t| t.elapsed().as_millis() as u64)
                    .unwrap_or(0);

                // Reset state
                let samples = std::mem::take(&mut audio.buffer);
                audio.speech_active = false;
                audio.speech_start_time = None;

                if speech_duration < MIN_SPEECH_DURATION_MS {
                    continue;
                }

                (samples, audio.sample_rate)
            };

            if samples.is_empty() {
                continue;
            }

            app.emit(
                "wake-word-state",
                serde_json::json!({"state": "processing"}),
            )
            .ok();

            match process_speech_segment(app, &samples, sr, &keyword_lower, auto_send, target_mode)
            {
                Ok(true) => {
                    log::info!("Wake word command processed successfully");
                }
                Ok(false) => {
                    log::debug!("Speech detected but no wake word match");
                }
                Err(e) => {
                    log::error!("Failed to process speech segment: {}", e);
                }
            }

            app.emit(
                "wake-word-state",
                serde_json::json!({"state": "listening"}),
            )
            .ok();
        }
    }

    drop(stream);
    Ok(())
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<Mutex<AudioState>>,
) -> Result<cpal::Stream> {
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            process_audio_callback(data, channels, &state);
        },
        |err| log::error!("Wake word audio stream error: {}", err),
        None,
    )?;
    Ok(stream)
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<Mutex<AudioState>>,
) -> Result<cpal::Stream> {
    let stream = device.build_input_stream(
        config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            // Convert i16 to f32
            let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
            process_audio_callback(&f32_data, channels, &state);
        },
        |err| log::error!("Wake word audio stream error: {}", err),
        None,
    )?;
    Ok(stream)
}

/// Process incoming audio samples: mix to mono, compute energy, buffer if speech detected
fn process_audio_callback(data: &[f32], channels: usize, state: &Arc<Mutex<AudioState>>) {
    // Mix to mono
    let mono: Vec<f32> = data
        .chunks(channels)
        .map(|ch| ch.iter().sum::<f32>() / channels as f32)
        .collect();

    // Compute RMS energy
    let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len().max(1) as f32).sqrt();

    if let Ok(mut audio) = state.lock() {
        if rms > ENERGY_THRESHOLD {
            audio.last_speech_time = Instant::now();
            if !audio.speech_active {
                audio.speech_active = true;
                audio.speech_start_time = Some(Instant::now());
                log::debug!("Speech onset detected (RMS: {:.4})", rms);
            }
        }

        if audio.speech_active {
            audio.buffer.extend_from_slice(&mono);
        }
    }
}

/// Check if a hotkey-based recording is currently active
fn is_hotkey_recording_active(app: &AppHandle) -> bool {
    app.try_state::<crate::AppState>()
        .and_then(|state| state.recording_state.lock().ok().map(|r| r.is_recording))
        .unwrap_or(false)
}

/// Process a detected speech segment: save to WAV, transcribe, check for wake word
fn process_speech_segment(
    app: &AppHandle,
    samples: &[f32],
    sample_rate: u32,
    keyword: &str,
    auto_send: bool,
    target_mode: &str,
) -> Result<bool> {
    // Save samples to a temporary WAV file
    let temp_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("echo");
    std::fs::create_dir_all(&temp_dir)?;

    let file_name = format!("wake_{}.wav", uuid::Uuid::new_v4());
    let file_path = temp_dir.join(&file_name);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    {
        let mut writer = hound::WavWriter::create(&file_path, spec)?;
        for &sample in samples {
            let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(s16)?;
        }
        writer.finalize()?;
    }

    let file_path_str = file_path.to_string_lossy().to_string();

    // Get language setting
    let language = app
        .try_state::<crate::AppState>()
        .and_then(|state| state.settings.lock().ok().map(|s| s.language.clone()));

    // Transcribe via ASR engine
    let result = {
        let state = app
            .try_state::<crate::AppState>()
            .ok_or_else(|| anyhow!("AppState not available"))?;
        let mut asr_engine = state.asr_engine.lock().map_err(|e| anyhow!("{}", e))?;

        let lang = match language.as_deref() {
            Some("auto") | None => None,
            Some(l) => Some(l),
        };
        asr_engine.transcribe(&file_path_str, lang)?
    };

    // Clean up temp file
    std::fs::remove_file(&file_path).ok();

    if result.no_speech.unwrap_or(false) || result.text.is_empty() {
        return Ok(false);
    }

    // Check for wake word prefix and extract command
    let command = match strip_wake_word(&result.text, keyword) {
        Some(cmd) => cmd,
        None => return Ok(false),
    };

    log::info!(
        "Wake word matched! Full: '{}', Command: '{}'",
        result.text,
        command
    );

    // Apply post-processing if enabled
    let final_text = apply_postprocessing(app, &command, target_mode)?;

    // Emit transcription event
    app.emit(
        "wake-word-transcription",
        serde_json::json!({
            "text": final_text,
            "full_text": result.text,
            "command": command,
        }),
    )
    .ok();

    // Determine target app and switch to it
    let target_app = determine_target_app(app, target_mode);
    if let Some(ref bundle_id) = target_app {
        log::info!("Switching to target app: {}", bundle_id);
        crate::active_app::activate_app_by_bundle_id(bundle_id);
        thread::sleep(Duration::from_millis(APP_SWITCH_DELAY_MS));
    }

    // Auto-insert the text
    insert_and_send(app, &final_text, auto_send)?;

    Ok(true)
}

/// Strip the wake word prefix from the transcription and return the command portion.
/// Handles punctuation and whitespace between the keyword and the command.
fn strip_wake_word(text: &str, keyword: &str) -> Option<String> {
    let text_trimmed = text.trim();
    let text_lower = text_trimmed.to_lowercase();
    let keyword_lower = keyword.to_lowercase();

    if !text_lower.starts_with(&keyword_lower) {
        return None;
    }

    // Skip the keyword bytes in the original text
    let after_keyword = &text_trimmed[keyword_lower.len()..];

    // Strip leading punctuation and whitespace
    let command = after_keyword
        .trim_start_matches(|c: char| {
            matches!(
                c,
                ',' | '.'
                    | '!'
                    | '?'
                    | ':'
                    | ';'
                    | '、'
                    | '。'
                    | '！'
                    | '？'
                    | ' '
                    | '\u{3000}'
                    | '\n'
                    | '\r'
            )
        })
        .trim();

    if command.is_empty() {
        return None;
    }

    Some(command.to_string())
}

/// Optionally apply LLM post-processing to the command text
fn apply_postprocessing(app: &AppHandle, text: &str, target_mode: &str) -> Result<String> {
    let state = match app.try_state::<crate::AppState>() {
        Some(s) => s,
        None => return Ok(text.to_string()),
    };

    let settings = state
        .settings
        .lock()
        .map_err(|e| anyhow!("{}", e))?
        .clone();

    if !settings.postprocess.enabled {
        return Ok(text.to_string());
    }

    // Determine target app info for context-aware post-processing
    let target_bundle_id = determine_target_app(app, target_mode);
    let target_app_info = target_bundle_id.as_ref().map(|bid| {
        crate::active_app::ActiveAppInfo {
            bundle_id: Some(bid.clone()),
            app_name: None, // Could be resolved from bundle_id if needed
        }
    });

    let mut asr_engine = state.asr_engine.lock().map_err(|e| anyhow!("{}", e))?;
    let dictionary = if settings.postprocess.dictionary.is_empty() {
        None
    } else {
        Some(&settings.postprocess.dictionary)
    };

    match asr_engine.postprocess_text(
        text,
        target_app_info
            .as_ref()
            .and_then(|a| a.app_name.as_deref()),
        target_app_info
            .as_ref()
            .and_then(|a| a.bundle_id.as_deref()),
        dictionary,
        settings.postprocess.custom_prompt.as_deref(),
    ) {
        Ok(pp) if pp.success => Ok(pp.processed_text),
        _ => Ok(text.to_string()),
    }
}

/// Determine which app to target for auto-insert
fn determine_target_app(_app: &AppHandle, target_mode: &str) -> Option<String> {
    match target_mode {
        "notification" => {
            // Try to get the last notification source app
            if let Some(bundle_id) = crate::active_app::get_last_notification_source_app() {
                return Some(bundle_id);
            }
            // Fall back to current frontmost app
            let frontmost = crate::active_app::get_frontmost_app();
            frontmost.bundle_id
        }
        "last_app" => {
            // Use the current frontmost app (the one the user is looking at)
            let frontmost = crate::active_app::get_frontmost_app();
            frontmost.bundle_id
        }
        _ => {
            // Check if target_mode is a specific bundle_id
            if !target_mode.is_empty() && target_mode.contains('.') {
                Some(target_mode.to_string())
            } else {
                // Default: target the frontmost app
                let frontmost = crate::active_app::get_frontmost_app();
                frontmost.bundle_id
            }
        }
    }
}

/// Insert text via clipboard paste and optionally press Enter to send
fn insert_and_send(app: &AppHandle, text: &str, auto_send: bool) -> Result<()> {
    // Get or initialize EnigoState
    let enigo_state = match app.try_state::<crate::EnigoState>() {
        Some(state) => state,
        None => {
            log::info!("Initializing Enigo for wake word insert...");
            match crate::EnigoState::new() {
                Ok(s) => {
                    app.manage(s);
                    app.try_state::<crate::EnigoState>()
                        .ok_or_else(|| anyhow!("Failed to initialize Enigo"))?
                }
                Err(e) => {
                    return Err(anyhow!("Cannot initialize Enigo: {}", e));
                }
            }
        }
    };

    let mut enigo = enigo_state.0.lock().map_err(|e| anyhow!("{}", e))?;

    // Paste text
    crate::clipboard::paste_via_clipboard(&mut enigo, text, app)?;

    // Auto-send: press Enter
    if auto_send {
        thread::sleep(Duration::from_millis(SEND_DELAY_MS));
        crate::input::send_return_key(&mut enigo).map_err(|e| anyhow!("{}", e))?;
        log::info!("Auto-send: Enter key pressed");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_wake_word_english() {
        assert_eq!(
            strip_wake_word("Echo, hello world", "echo"),
            Some("hello world".to_string())
        );
        assert_eq!(
            strip_wake_word("echo hello world", "echo"),
            Some("hello world".to_string())
        );
        assert_eq!(
            strip_wake_word("ECHO! Send this message", "echo"),
            Some("Send this message".to_string())
        );
        assert_eq!(strip_wake_word("Echo", "echo"), None);
        assert_eq!(strip_wake_word("Something else", "echo"), None);
    }

    #[test]
    fn test_strip_wake_word_japanese() {
        assert_eq!(
            strip_wake_word("エコー、今から帰ります", "エコー"),
            Some("今から帰ります".to_string())
        );
        assert_eq!(
            strip_wake_word("エコー今から帰ります", "エコー"),
            Some("今から帰ります".to_string())
        );
        assert_eq!(strip_wake_word("エコー", "エコー"), None);
    }
}
