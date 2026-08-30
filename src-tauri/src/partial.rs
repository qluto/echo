//! In-progress ("partial") decoding while the hotkey is held.
//!
//! Follows hayamimi's draft scheme: every ~0.5 s of new audio, re-decode the
//! last ≤8 s of the recording and show the result as a draft. The window cap
//! keeps each decode O(1) regardless of how long the key is held; the final
//! transcription on release still uses the full recording as before.
//!
//! The decoder only ever `try_lock`s the ASR engine and skips a tick when it
//! is busy, so it can never delay the final decode or the always-on pipeline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel as channel;
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::transcription::ASREngine;
use crate::vad::VAD_SAMPLE_RATE;

const PARTIAL_EVERY_SEC: f64 = 0.5;
const PARTIAL_WINDOW_SEC: f64 = 8.0;
const PARTIAL_MIN_SEC: f64 = 0.6;

/// Draft text of the recording in progress (empty = cleared).
#[derive(Debug, Clone, Serialize)]
pub struct HotkeyPartialEvent {
    pub text: String,
}

pub struct PartialDecoder {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PartialDecoder {
    /// Spawn the decoder; `frame_rx` carries 16 kHz mono frames from the
    /// recording tap (see `audio_capture::start_recording_with_tap`).
    pub fn start(
        app: AppHandle,
        asr_engine: Arc<Mutex<ASREngine>>,
        language: Option<String>,
        frame_rx: channel::Receiver<Vec<f32>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("echo-partial".into())
            .spawn(move || run(app, asr_engine, language.as_deref(), frame_rx, stop_flag))
            .ok();
        Self { stop, handle }
    }

    /// Stop and wait for any in-flight draft decode, then clear the draft.
    /// Called on hotkey release *before* the final decode so the two never
    /// interleave and no stale draft lands after the final.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(
    app: AppHandle,
    asr_engine: Arc<Mutex<ASREngine>>,
    language: Option<&str>,
    frame_rx: channel::Receiver<Vec<f32>>,
    stop: Arc<AtomicBool>,
) {
    let every = (PARTIAL_EVERY_SEC * VAD_SAMPLE_RATE as f64) as usize;
    let window = (PARTIAL_WINDOW_SEC * VAD_SAMPLE_RATE as f64) as usize;
    let min = (PARTIAL_MIN_SEC * VAD_SAMPLE_RATE as f64) as usize;
    let mut audio: Vec<f32> = Vec::with_capacity(window * 2);
    let mut decoded_upto = 0usize;
    let mut last_text = String::new();
    let mut enabled: Option<bool> = None; // resolved lazily from the loaded engine

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        match frame_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => {
                audio.extend_from_slice(&frame);
                // Keep only what a draft can use (plus slack) so a long hold
                // doesn't grow memory.
                if audio.len() > window * 2 {
                    let drop = audio.len() - window;
                    audio.drain(..drop);
                    decoded_upto = decoded_upto.saturating_sub(drop);
                }
            }
            Err(channel::RecvTimeoutError::Timeout) => continue,
            Err(channel::RecvTimeoutError::Disconnected) => break,
        }

        if audio.len() < min || audio.len() - decoded_upto < every {
            continue;
        }

        // Engine busy (final decode, always-on, model load): skip this tick,
        // the next one will cover the same audio anyway.
        let Ok(mut engine) = asr_engine.try_lock() else {
            continue;
        };
        let ok = *enabled.get_or_insert_with(|| engine.supports_partial());
        if !ok {
            log::info!("Partial decoding disabled for the active model");
            break;
        }
        decoded_upto = audio.len();
        let start = audio.len().saturating_sub(window);
        let t0 = std::time::Instant::now();
        let result = engine.transcribe_samples(&audio[start..], language, false);
        drop(engine);

        match result {
            Ok(r) if r.success => {
                let text = r.text.trim().to_string();
                log::debug!(
                    "partial: {:.1}s window in {} ms: {}",
                    (audio.len() - start) as f64 / VAD_SAMPLE_RATE as f64,
                    t0.elapsed().as_millis(),
                    text
                );
                if !text.is_empty() && text != last_text {
                    last_text = text.clone();
                    let _ = app.emit("hotkey-partial", HotkeyPartialEvent { text });
                }
            }
            Ok(_) => {}
            Err(e) => log::warn!("partial decode failed: {}", e),
        }
    }

    let _ = app.emit("hotkey-partial", HotkeyPartialEvent { text: String::new() });
    log::debug!("Partial decoder exiting");
}
