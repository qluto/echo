//! Continuous listening pipeline: streaming audio → VAD → segment detection → ASR → DB.
//!
//! Ports echo-cli's recorder.py + listener.py logic to Rust.
//! Python is only used for ASR inference (via existing JSON-RPC sidecar).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::Result;
use crossbeam_channel as channel;
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::audio_capture::StreamingCapture;
use crate::database::{TranscriptionDb, TranscriptionEntry};
use crate::transcription::ASREngine;
use crate::vad::{VadEvent, VadProcessor, VAD_FRAME_SIZE, VAD_SAMPLE_RATE};

/// A speech segment detected by VAD, ready for transcription.
struct SpeechSegment {
    audio_path: PathBuf,
    duration_seconds: f64,
}

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
    pub segment_count: u32,
}

/// The continuous listening pipeline.
///
/// Architecture:
/// ```text
/// [cpal callback thread]  →  channel<Vec<f32>>  →  [VAD processor thread]
///   captures audio                                    runs Silero VAD per frame
///   resamples to 16kHz                                detects speech segments
///                                                     saves WAV files
///                                                  →  channel<SpeechSegment>  →  [transcription worker]
///                                                                                  calls Python ASR
///                                                                                  saves to DB
///                                                                                  emits events
/// ```
pub struct ContinuousPipeline {
    streaming_capture: Option<StreamingCapture>,
    vad_handle: Option<JoinHandle<()>>,
    worker_handle: Option<JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
    segment_count: Arc<AtomicU32>,
}

impl ContinuousPipeline {
    /// Start the continuous listening pipeline.
    pub fn start(
        app: AppHandle,
        asr_engine: Arc<Mutex<ASREngine>>,
        db: Arc<Mutex<TranscriptionDb>>,
        language: Option<String>,
        device_name: Option<String>,
        silence_sec: f64,
        max_segment_sec: u32,
    ) -> Result<Self> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let segment_count = Arc::new(AtomicU32::new(0));

        // Create temp directory for speech segment WAV files
        let tmp_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("echo")
            .join("vad_segments");
        std::fs::create_dir_all(&tmp_dir)?;

        // Channels
        let (frame_tx, frame_rx) = channel::bounded::<Vec<f32>>(256);
        let (segment_tx, segment_rx) = channel::bounded::<SpeechSegment>(10);

        // 1. Start streaming audio capture
        let capture = StreamingCapture::start(device_name, frame_tx)?;

        // 2. Start VAD processor thread
        let vad_stop = Arc::clone(&stop_flag);
        let vad_tmp_dir = tmp_dir.clone();
        let vad_handle = thread::Builder::new()
            .name("echo-vad".into())
            .spawn(move || {
                if let Err(e) = vad_thread(
                    frame_rx,
                    segment_tx,
                    vad_stop,
                    silence_sec,
                    max_segment_sec,
                    &vad_tmp_dir,
                ) {
                    log::error!("VAD thread error: {}", e);
                }
            })?;

        // 3. Start transcription worker thread
        let worker_stop = Arc::clone(&stop_flag);
        let worker_count = Arc::clone(&segment_count);
        let worker_handle = thread::Builder::new()
            .name("echo-transcribe-worker".into())
            .spawn(move || {
                transcription_worker(
                    segment_rx,
                    asr_engine,
                    db,
                    language,
                    worker_stop,
                    worker_count,
                    app,
                );
            })?;

        log::info!("Continuous pipeline started (silence={}s, max_segment={}s)", silence_sec, max_segment_sec);

        Ok(Self {
            streaming_capture: Some(capture),
            vad_handle: Some(vad_handle),
            worker_handle: Some(worker_handle),
            stop_flag,
            segment_count,
        })
    }

    /// Stop the pipeline and wait for threads to finish.
    /// Returns the number of segments transcribed.
    pub fn stop(&mut self) -> u32 {
        self.stop_flag.store(true, Ordering::SeqCst);

        // Stop audio capture first (no more frames)
        if let Some(mut capture) = self.streaming_capture.take() {
            capture.stop();
        }

        // Wait for threads
        if let Some(handle) = self.vad_handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }

        let count = self.segment_count.load(Ordering::SeqCst);
        log::info!("Continuous pipeline stopped, {} segments transcribed", count);
        count
    }

    /// Get the current segment count.
    pub fn segment_count(&self) -> u32 {
        self.segment_count.load(Ordering::SeqCst)
    }
}

impl Drop for ContinuousPipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// VAD thread: receives audio frames, detects speech segments, saves WAV files
// ---------------------------------------------------------------------------

fn vad_thread(
    frame_rx: channel::Receiver<Vec<f32>>,
    segment_tx: channel::Sender<SpeechSegment>,
    stop_flag: Arc<AtomicBool>,
    silence_sec: f64,
    max_segment_sec: u32,
    tmp_dir: &Path,
) -> Result<()> {
    let mut vad = VadProcessor::new(0.5)?;
    let mut state = VadStateMachine::new(silence_sec, max_segment_sec);

    loop {
        if stop_flag.load(Ordering::SeqCst) {
            // Flush remaining speech on stop
            if state.is_speaking {
                if let Some(segment) = state.finalize_segment(tmp_dir) {
                    let _ = segment_tx.send(segment);
                }
                // Reset VAD LSTM state after segment (matches Python's vad.reset_states())
                vad.reset();
            }
            break;
        }

        match frame_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(frame) => {
                let event = vad.process_frame(&frame);
                if let Some(segment) = state.process(event, frame, tmp_dir) {
                    // Reset VAD LSTM state after each segment (matches Python's vad.reset_states())
                    vad.reset();
                    if segment_tx.send(segment).is_err() {
                        log::warn!("Segment channel closed, stopping VAD");
                        break;
                    }
                }
            }
            Err(channel::RecvTimeoutError::Timeout) => continue,
            Err(channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    log::info!("VAD thread exiting");
    Ok(())
}

/// VAD state machine: IDLE → SPEAKING → SEGMENT_COMPLETE.
/// Mirrors echo-cli recorder.py logic.
struct VadStateMachine {
    is_speaking: bool,
    silence_count: u32,
    silence_threshold: u32,
    max_frames: u32,
    speech_frames: Vec<Vec<f32>>,
    pre_buffer: VecDeque<Vec<f32>>,
    pre_buffer_max: usize,
    speech_frame_count: u32,
    segment_counter: u32,
}

impl VadStateMachine {
    fn new(silence_sec: f64, max_segment_sec: u32) -> Self {
        let frames_per_sec = VAD_SAMPLE_RATE as f64 / VAD_FRAME_SIZE as f64;
        let pre_buffer_sec = 0.3;

        Self {
            is_speaking: false,
            silence_count: 0,
            silence_threshold: (silence_sec * frames_per_sec) as u32,
            max_frames: (max_segment_sec as f64 * frames_per_sec) as u32,
            speech_frames: Vec::new(),
            pre_buffer: VecDeque::new(),
            pre_buffer_max: (pre_buffer_sec * frames_per_sec) as usize,
            speech_frame_count: 0,
            segment_counter: 0,
        }
    }

    /// Process a VAD event for one frame. Returns a segment if one completes.
    fn process(
        &mut self,
        event: VadEvent,
        frame: Vec<f32>,
        tmp_dir: &Path,
    ) -> Option<SpeechSegment> {
        let is_speech = matches!(event, VadEvent::Speech { .. });

        if !self.is_speaking {
            // Maintain pre-buffer
            self.pre_buffer.push_back(frame.clone());
            if self.pre_buffer.len() > self.pre_buffer_max {
                self.pre_buffer.pop_front();
            }

            if is_speech {
                // Speech started
                self.is_speaking = true;
                self.silence_count = 0;
                self.speech_frame_count = 0;

                // Include pre-buffer
                self.speech_frames = self.pre_buffer.drain(..).collect();
                self.speech_frames.push(frame);
                self.speech_frame_count += 1;

                log::debug!("Speech started");
            }

            None
        } else {
            // Currently speaking
            self.speech_frames.push(frame);
            self.speech_frame_count += 1;

            if !is_speech {
                self.silence_count += 1;
            } else {
                self.silence_count = 0;
            }

            // Check end conditions
            if self.silence_count >= self.silence_threshold {
                log::debug!("Speech ended after silence");
                self.finalize_segment(tmp_dir)
            } else if self.speech_frame_count >= self.max_frames {
                log::info!("Max segment duration reached, forcing split");
                self.finalize_segment(tmp_dir)
            } else {
                None
            }
        }
    }

    /// Save accumulated speech as WAV and reset state.
    fn finalize_segment(&mut self, tmp_dir: &Path) -> Option<SpeechSegment> {
        if self.speech_frames.is_empty() {
            self.reset();
            return None;
        }

        // Concatenate all frames
        let total_samples: usize = self.speech_frames.iter().map(|f| f.len()).sum();
        let duration = total_samples as f64 / VAD_SAMPLE_RATE as f64;

        // Skip very short segments (< 0.5s)
        if duration < 0.5 {
            log::debug!("Skipping short segment ({:.2}s)", duration);
            self.reset();
            return None;
        }

        // Save to WAV
        self.segment_counter += 1;
        let filename = format!("segment_{}.wav", self.segment_counter);
        let filepath = tmp_dir.join(&filename);

        match save_frames_to_wav(&self.speech_frames, &filepath) {
            Ok(()) => {
                log::info!("Segment saved: {:.1}s → {:?}", duration, filepath);
                let segment = SpeechSegment {
                    audio_path: filepath,
                    duration_seconds: duration,
                };
                self.reset();
                Some(segment)
            }
            Err(e) => {
                log::error!("Failed to save segment WAV: {}", e);
                self.reset();
                None
            }
        }
    }

    fn reset(&mut self) {
        self.is_speaking = false;
        self.silence_count = 0;
        self.speech_frames.clear();
        self.speech_frame_count = 0;
    }
}

/// Save f32 audio frames to a 16kHz mono WAV file.
fn save_frames_to_wav(frames: &[Vec<f32>], path: &Path) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: VAD_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: HoundSampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for frame in frames {
        for &sample in frame {
            let s16 = (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer.write_sample(s16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Transcription worker: dequeues segments, calls ASR, saves to DB, emits events
// ---------------------------------------------------------------------------

fn transcription_worker(
    segment_rx: channel::Receiver<SpeechSegment>,
    asr_engine: Arc<Mutex<ASREngine>>,
    db: Arc<Mutex<TranscriptionDb>>,
    language: Option<String>,
    stop_flag: Arc<AtomicBool>,
    segment_count: Arc<AtomicU32>,
    app: AppHandle,
) {
    loop {
        match segment_rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(segment) => {
                process_segment(&segment, &asr_engine, &db, language.as_deref(), &segment_count, &app);
                // Clean up temp WAV file
                let _ = std::fs::remove_file(&segment.audio_path);
            }
            Err(channel::RecvTimeoutError::Timeout) => {
                if stop_flag.load(Ordering::SeqCst) && segment_rx.is_empty() {
                    break;
                }
            }
            Err(channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    log::info!("Transcription worker exiting");
}

fn process_segment(
    segment: &SpeechSegment,
    asr_engine: &Arc<Mutex<ASREngine>>,
    db: &Arc<Mutex<TranscriptionDb>>,
    language: Option<&str>,
    segment_count: &Arc<AtomicU32>,
    app: &AppHandle,
) {
    let audio_path = segment.audio_path.to_string_lossy();
    log::info!("Processing segment: {:.1}s", segment.duration_seconds);

    // Call ASR engine
    let result = {
        let mut engine = match asr_engine.lock() {
            Ok(e) => e,
            Err(e) => {
                log::error!("Failed to lock ASR engine: {}", e);
                return;
            }
        };
        engine.transcribe(&audio_path, language)
    };

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            log::error!("Transcription failed: {}", e);
            return;
        }
    };

    if !result.success {
        log::warn!("Transcription returned success=false");
        return;
    }

    let text = result.text.trim().to_string();
    if text.is_empty() {
        log::info!("Empty transcription, skipping");
        return;
    }

    // Save to database
    let entry = TranscriptionEntry {
        id: None,
        created_at: String::new(), // DB default
        duration_seconds: Some(segment.duration_seconds),
        text: text.clone(),
        raw_text: None,
        language: if result.language.is_empty() {
            None
        } else {
            Some(result.language.clone())
        },
        model_name: None, // TODO: get from engine settings
        segments_json: if result.segments.is_empty() {
            None
        } else {
            serde_json::to_string(&result.segments).ok()
        },
    };

    let entry_id = match db.lock() {
        Ok(db) => match db.insert(&entry) {
            Ok(id) => id,
            Err(e) => {
                log::error!("Failed to save to DB: {}", e);
                return;
            }
        },
        Err(e) => {
            log::error!("Failed to lock DB: {}", e);
            return;
        }
    };

    segment_count.fetch_add(1, Ordering::SeqCst);

    // Emit event to frontend
    let event = ContinuousTranscriptionEvent {
        id: entry_id,
        text,
        created_at: entry.created_at,
        duration_seconds: Some(segment.duration_seconds),
        language: if result.language.is_empty() {
            None
        } else {
            Some(result.language)
        },
        model_name: None,
    };

    if let Err(e) = app.emit("continuous-transcription", &event) {
        log::error!("Failed to emit continuous transcription event: {}", e);
    }
}
