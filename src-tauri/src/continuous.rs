//! Always-on logging: the microphone and system audio, transcribed
//! continuously into the transcription history.
//!
//! Same capture → chunk → ASR pipeline as the meeting companion (`live`):
//! chunks are cut at pauses (4–15 s) rather than picked out by a VAD, and the
//! other side's voice picked up by the mic is dropped as echo. Unlike a
//! meeting there are no speaker labels, no recording and no LLM notes: each
//! transcribed chunk becomes one history row.
//!
//! The mic is muted (its audio not transcribed) while hotkey dictation, a
//! voice memo or a meeting has it; system audio too during a meeting, which
//! transcribes it itself.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{anyhow, Result};
use crossbeam_channel as channel;
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::database::{TranscriptionDb, TranscriptionEntry};
use crate::live::{self, AudioObserver, Captures, Chunk, Source};
use crate::transcription::ASREngine;
use crate::types::{StatusKind, StatusSource, TranscriptionStatusEvent};

/// Something else is using the audio always-on would transcribe.
#[derive(Debug, Clone, Copy)]
pub enum Mute {
    /// Hotkey dictation (mic).
    Dictation = 1,
    /// A voice memo (mic).
    Memo = 2,
    /// A meeting (mic and system audio).
    Meeting = 4,
}

/// Bit set of active `Mute` reasons. Global so that muting works whether or
/// not always-on is running (it may start mid-meeting).
static MUTED: AtomicU8 = AtomicU8::new(0);

pub fn set_muted(reason: Mute, on: bool) {
    if on {
        MUTED.fetch_or(reason as u8, Ordering::SeqCst);
    } else {
        MUTED.fetch_and(!(reason as u8), Ordering::SeqCst);
    }
}

fn is_muted(source: Source) -> bool {
    let muted = MUTED.load(Ordering::SeqCst);
    match source {
        Source::Me => muted != 0,
        Source::Others => muted & Mute::Meeting as u8 != 0,
    }
}

/// When always-on last heard sound (Unix seconds; 0 = never), so that an
/// update does not restart the app while there is something to transcribe.
static LAST_SOUND: AtomicU64 = AtomicU64::new(0);

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Seconds since always-on last heard sound (very large if never).
pub fn seconds_since_sound() -> f64 {
    now_secs().saturating_sub(LAST_SOUND.load(Ordering::Relaxed)) as f64
}

/// A frame this loud counts as sound for the activity indicator…
const ACTIVE_DBFS: f32 = -40.0;
/// …which stays on this many frames (~0.8 s) after the last one.
const ACTIVE_HOLD_FRAMES: u32 = 25;

/// Event emitted to frontend when a continuous transcription completes.
#[derive(Debug, Clone, Serialize)]
pub struct ContinuousTranscriptionEvent {
    pub id: i64,
    pub text: String,
    pub created_at: String,
    pub duration_seconds: Option<f64>,
    pub language: Option<String>,
    pub model_name: Option<String>,
}

/// Status of the continuous listening pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct ContinuousListeningStatus {
    pub is_listening: bool,
    /// Lines saved since listening started.
    pub segment_count: u32,
    /// System audio is captured along with the mic.
    pub system_audio: bool,
    /// Why system audio isn't captured, if it couldn't start.
    pub warning: Option<String>,
}

impl ContinuousListeningStatus {
    pub fn idle() -> Self {
        Self { is_listening: false, segment_count: 0, system_audio: false, warning: None }
    }
}

/// Always-on listening in progress.
pub struct ContinuousPipeline {
    captures: Captures,
    line_count: Arc<AtomicU32>,
    warning: Option<String>,
}

impl ContinuousPipeline {
    /// Start capturing and transcribing. Fails if the microphone delivers no
    /// audio; without system audio it logs the mic only (see `status().warning`).
    pub fn start(
        app: AppHandle,
        asr_engine: Arc<Mutex<ASREngine>>,
        db: Arc<Mutex<TranscriptionDb>>,
        language: Option<String>,
        device_name: Option<String>,
    ) -> Result<Self> {
        let (captures, inputs, system_error) = live::start_captures(device_name)?;
        let warning = system_error.map(|e| format!("システム音声を取得できないため、マイクのみ記録します: {e}"));
        let line_count = Arc::new(AtomicU32::new(0));

        let (chunk_tx, chunk_rx) = channel::unbounded::<Chunk>();
        {
            let app = app.clone();
            thread::Builder::new().name("echo-always-on-audio".into()).spawn(move || {
                let observer = Activity::new(app);
                if let Err(e) = live::audio_loop(inputs, chunk_tx, None, &observer) {
                    log::error!("Always-on audio thread failed: {e:#}");
                }
                observer.set_active(false);
            })?;
        }
        {
            let line_count = Arc::clone(&line_count);
            thread::Builder::new().name("echo-always-on-asr".into()).spawn(move || {
                // Set while the engine is locked for a chunk, read when its line is saved.
                let model_name: RefCell<Option<String>> = RefCell::new(None);
                live::asr_worker(
                    chunk_rx,
                    |samples| Ok(asr_engine.lock().map_err(|e| anyhow!("{e}"))?.has_speech(samples)),
                    |samples| {
                        // Lock per chunk so hotkey dictation can run in between.
                        let mut engine = asr_engine.lock().map_err(|e| anyhow!("{e}"))?;
                        *model_name.borrow_mut() = Some(engine.active_model_name().to_string()).filter(|m| !m.is_empty());
                        let result = engine.transcribe_samples(samples, language.as_deref(), false);
                        if let Err(e) = &result {
                            TranscriptionStatusEvent::emit(&app, StatusSource::Continuous, StatusKind::AsrError, Some(format!("{e:#}")));
                        }
                        Ok(result?.text)
                    },
                    |line| {
                        if save_line(&app, &db, &line, language.as_deref(), model_name.borrow().clone()) {
                            line_count.fetch_add(1, Ordering::SeqCst);
                        }
                    },
                );
            })?;
        }

        log::info!("Always-on listening started (system audio: {})", captures.has_system());
        Ok(Self { captures, line_count, warning })
    }

    /// Stop capturing and return the number of lines saved so far. What is
    /// still queued is transcribed and saved in the background.
    pub fn stop(mut self) -> u32 {
        self.captures.stop();
        let count = self.line_count.load(Ordering::SeqCst);
        log::info!("Always-on listening stopped ({count} lines so far; finishing the rest)");
        count
    }

    pub fn status(&self) -> ContinuousListeningStatus {
        ContinuousListeningStatus {
            is_listening: true,
            segment_count: self.line_count.load(Ordering::SeqCst),
            system_audio: self.captures.has_system(),
            warning: self.warning.clone(),
        }
    }
}

/// Mutes per `MUTED` and drives the frontend's activity indicator
/// (`continuous-activity`: some unmuted source is making sound).
struct Activity {
    app: AppHandle,
    /// Frames left before each source counts as quiet.
    hold: [Cell<u32>; 2],
    active: Cell<bool>,
}

impl Activity {
    fn new(app: AppHandle) -> Self {
        Self { app, hold: [Cell::new(0), Cell::new(0)], active: Cell::new(false) }
    }

    fn set_active(&self, active: bool) {
        if self.active.replace(active) != active {
            if let Err(e) = self.app.emit("continuous-activity", serde_json::json!({ "active": active })) {
                log::warn!("Failed to emit continuous-activity: {e}");
            }
        }
    }
}

impl AudioObserver for Activity {
    fn on_frame(&self, source: Source, _level: f32, dbfs: f32) {
        let hold = &self.hold[source as usize];
        if dbfs > ACTIVE_DBFS && !is_muted(source) {
            hold.set(ACTIVE_HOLD_FRAMES);
            LAST_SOUND.store(now_secs(), Ordering::Relaxed);
        } else {
            hold.set(hold.get().saturating_sub(1));
        }
        self.set_active(self.hold.iter().any(|h| h.get() > 0));
    }

    fn muted(&self, source: Source) -> bool {
        is_muted(source)
    }
}

/// Store a line in the history and tell the frontend. Returns whether it was saved.
fn save_line(
    app: &AppHandle,
    db: &Mutex<TranscriptionDb>,
    line: &live::Line,
    language: Option<&str>,
    model_name: Option<String>,
) -> bool {
    let entry = TranscriptionEntry {
        id: None,
        created_at: String::new(), // DB generates the real value (returned by insert)
        duration_seconds: Some(line.end - line.start),
        text: line.text.clone(),
        raw_text: None,
        language: language.map(str::to_string),
        model_name,
        segments_json: None,
    };
    let inserted = db.lock().map_err(|e| anyhow!("{e}")).and_then(|db| db.insert(&entry));
    let (id, created_at) = match inserted {
        Ok(v) => v,
        Err(e) => {
            log::error!("Failed to save always-on line: {e:#}");
            TranscriptionStatusEvent::emit(app, StatusSource::Continuous, StatusKind::DbError, Some(format!("{e:#}")));
            return false;
        }
    };

    let event = ContinuousTranscriptionEvent {
        id,
        text: entry.text,
        created_at,
        duration_seconds: entry.duration_seconds,
        language: entry.language,
        model_name: entry.model_name,
    };
    if let Err(e) = app.emit("continuous-transcription", &event) {
        log::error!("Failed to emit continuous transcription event: {e}");
    }
    true
}
