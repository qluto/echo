//! Voice memos: record a long session to disk, then — once the recording is
//! stopped — transcribe it and turn the transcript into meeting minutes.
//!
//! ```text
//! recording:   [StreamingCapture] → channel<Vec<f32>> → [writer thread] → 16 kHz mono WAV
//! processing:  WAV → 20–28 s chunks cut at the quietest point
//!                  → ASR per chunk → timestamped transcript → LLM → minutes
//! ```
//!
//! The recording uses its own capture stream, so hotkey dictation keeps working
//! while a memo is being recorded. Processing takes the `ASREngine` lock one
//! chunk / one LLM call at a time: dictation gets in between chunks, but has
//! to wait out a minutes LLM call (seconds to tens of seconds) in progress.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel as channel;
use hound::{SampleFormat as HoundSampleFormat, WavReader, WavSpec, WavWriter};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::audio_capture::StreamingCapture;
use crate::database::{MemoStatus, TranscriptionDb};
use crate::types::{AppState, TranscriptionSegment};
use crate::partial::quietest_cut;
use crate::vad::VAD_SAMPLE_RATE;

/// How long to wait for the first audio frame before declaring the input dead.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(3);
/// The WAV header is rewritten this often, so a crash loses at most this much.
const FLUSH_INTERVAL_SAMPLES: usize = 10 * VAD_SAMPLE_RATE as usize;

/// dB range mapped to 0.0–1.0 for the recording level meter.
const LEVEL_DB_FLOOR: f32 = -55.0;
const LEVEL_DB_CEIL: f32 = -12.0;

const fn samples(sec: f64) -> usize {
    (sec * VAD_SAMPLE_RATE as f64) as usize
}

/// A chunk is cut at the quietest point between these two lengths. The upper
/// bound keeps a chunk inside Whisper's 30 s window; the lower one keeps
/// chunks long, since every engine does better with more context.
const MIN_CHUNK_SAMPLES: usize = samples(20.0);
const MAX_CHUNK_SAMPLES: usize = samples(28.0);
/// Granularity for locating the quietest point: long enough that a stop
/// consonant inside a word doesn't pass for a pause.
const CUT_PROBE_SAMPLES: usize = samples(0.2);
/// A final chunk shorter than this is the tail of the stop click, not words.
const MIN_FINAL_CHUNK_SAMPLES: usize = samples(0.3);

/// Transcript / notes are fed to the LLM in pieces of at most this many chars.
const MAX_LLM_INPUT_CHARS: usize = 3500;
const PART_NOTES_MAX_TOKENS: usize = 1024;
const MINUTES_MAX_TOKENS: usize = 2048;

const MINUTES_PROMPT: &str = "You write meeting minutes from a speech-recognition transcript of a recording.

## Input
Chronological transcript lines with [MM:SS] timestamps, or notes extracted from consecutive parts of a long recording. There are no speaker labels, and the text may contain recognition errors.

## Rules
- Use ONLY what is in the input. Never add facts, names, numbers, dates, owners or deadlines that were not said.
- Do not attribute statements to a person unless their name is said in the input.
- Fix an obvious recognition error only when the intended word is clear from context.
- Write in the same language as the input, including the section headings (the headings below are for Japanese input).
- If a section has nothing to report, write \"なし\".

## Output format (Markdown only, no preamble)
## 概要
What the recording was about, in 2-3 sentences.

## 議題と要点
- One bullet per topic, with the key points discussed.

## 決定事項
- What was decided.

## アクションアイテム
- What needs to be done (owner and deadline only if they were said).";

const PART_NOTES_PROMPT: &str = "You extract notes from one part of a long recording's speech-recognition transcript. The notes from all parts will later be merged into meeting minutes.

## Rules
- Use ONLY what is in the input. Never add facts, names, numbers, dates, owners or deadlines that were not said.
- The text may contain recognition errors; fix one only when the intended word is clear from context.
- Write in the same language as the input.

## Output (no preamble)
A concise bullet list covering: topics discussed and their key points, decisions made, and things that need to be done. Keep names, numbers and dates exactly as said.";

/// Payload of the `memo-progress` event.
#[derive(Debug, Clone, Serialize)]
pub struct MemoProgressEvent {
    pub id: i64,
    pub status: MemoStatus,
    /// 0.0–1.0 within the current stage, when known.
    pub progress: Option<f64>,
}

/// Live state of the in-progress recording.
#[derive(Debug, Clone, Serialize)]
pub struct MemoRecordingStatus {
    pub memo_id: i64,
    pub elapsed_seconds: f64,
    /// Input level, 0.0–1.0.
    pub level: f32,
}

/// Directory holding memo recordings (`<app data>/recordings`).
pub fn recordings_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {e}"))?
        .join("recordings");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/// An in-progress memo recording, streamed to a 16 kHz mono WAV file.
pub struct MemoRecorder {
    memo_id: i64,
    capture: StreamingCapture,
    writer_handle: JoinHandle<Result<()>>,
    samples_written: Arc<AtomicU64>,
    /// Latest input level as `f32` bits.
    level: Arc<AtomicU32>,
}

impl MemoRecorder {
    /// Start recording into `audio_path`. Fails if the input device delivers
    /// no audio (missing device, denied permission, …).
    pub fn start(memo_id: i64, device_name: Option<String>, audio_path: &Path) -> Result<Self> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: VAD_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };
        let writer = WavWriter::create(audio_path, spec)
            .with_context(|| format!("Failed to create {audio_path:?}"))?;

        let (frame_tx, frame_rx) = channel::bounded::<Vec<f32>>(256);
        let mut capture = StreamingCapture::start(device_name, frame_tx)?;

        // The capture thread only logs device errors, so confirm audio is
        // actually flowing before reporting the recording as started.
        let first = match frame_rx.recv_timeout(FIRST_FRAME_TIMEOUT) {
            Ok(frame) => frame,
            Err(_) => {
                capture.stop();
                return Err(anyhow!(
                    "No audio from the input device (check the microphone permission)"
                ));
            }
        };

        let samples_written = Arc::new(AtomicU64::new(0));
        let level = Arc::new(AtomicU32::new(0f32.to_bits()));
        let writer_handle = {
            let samples_written = Arc::clone(&samples_written);
            let level = Arc::clone(&level);
            thread::Builder::new()
                .name("echo-memo-writer".into())
                .spawn(move || writer_loop(writer, first, frame_rx, &samples_written, &level))?
        };

        log::info!("Memo {memo_id} recording started: {audio_path:?}");
        Ok(Self {
            memo_id,
            capture,
            writer_handle,
            samples_written,
            level,
        })
    }

    pub fn status(&self) -> MemoRecordingStatus {
        MemoRecordingStatus {
            memo_id: self.memo_id,
            elapsed_seconds: self.elapsed_seconds(),
            level: f32::from_bits(self.level.load(Ordering::Relaxed)),
        }
    }

    fn elapsed_seconds(&self) -> f64 {
        self.samples_written.load(Ordering::Relaxed) as f64 / VAD_SAMPLE_RATE as f64
    }

    /// Stop capturing and finalize the WAV. Returns the memo id and the
    /// recorded duration in seconds.
    pub fn stop(mut self) -> (i64, f64) {
        // Stopping the capture drops the frame sender; the writer drains what
        // is left in the channel and finalizes the file.
        self.capture.stop();
        match self.writer_handle.join() {
            Ok(Ok(())) => {}
            // Audio up to the last flush is still on disk and usable.
            Ok(Err(e)) => log::error!("Memo {} writer failed: {e}", self.memo_id),
            Err(_) => log::error!("Memo {} writer thread panicked", self.memo_id),
        }
        let duration = self.samples_written.load(Ordering::Relaxed) as f64 / VAD_SAMPLE_RATE as f64;
        log::info!("Memo {} recording stopped: {duration:.1}s", self.memo_id);
        (self.memo_id, duration)
    }
}

fn writer_loop(
    mut writer: WavWriter<BufWriter<File>>,
    first: Vec<f32>,
    frame_rx: channel::Receiver<Vec<f32>>,
    samples_written: &AtomicU64,
    level: &AtomicU32,
) -> Result<()> {
    let mut since_flush = 0usize;
    for frame in std::iter::once(first).chain(frame_rx.iter()) {
        let mut sum_sq = 0.0f32;
        for &s in &frame {
            writer.write_sample((s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)?;
            sum_sq += s * s;
        }
        let db = 20.0 * ((sum_sq / frame.len().max(1) as f32).sqrt() + 1e-9).log10();
        let norm = ((db - LEVEL_DB_FLOOR) / (LEVEL_DB_CEIL - LEVEL_DB_FLOOR)).clamp(0.0, 1.0);
        level.store(norm.to_bits(), Ordering::Relaxed);
        samples_written.fetch_add(frame.len() as u64, Ordering::Relaxed);

        since_flush += frame.len();
        if since_flush >= FLUSH_INTERVAL_SAMPLES {
            writer.flush()?;
            since_flush = 0;
        }
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Chunking + transcription
// ---------------------------------------------------------------------------

/// Transcribe a 16 kHz mono 16-bit WAV chunk by chunk. `transcribe` gets one
/// chunk at a time; `on_progress` gets the fraction of the file done. The file
/// is streamed, so memory stays bounded however long the recording.
///
/// Every sample reaches the engine: chunks are contiguous and nothing is
/// filtered out as silence. A VAD is deliberately not used to pick out the
/// speech — Silero scores stretches of clearly audible speech near zero on
/// some microphones, and whatever it misses is gone from the transcript,
/// while the engines cope with pauses on their own. Chunking only bounds the
/// length of one ASR call.
pub fn transcribe_wav_chunked(
    path: &Path,
    mut transcribe: impl FnMut(&[f32]) -> Result<String>,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<TranscriptionSegment>> {
    let mut reader = WavReader::open(path).with_context(|| format!("Failed to open {path:?}"))?;
    let spec = reader.spec();
    if spec.sample_rate != VAD_SAMPLE_RATE
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != HoundSampleFormat::Int
    {
        bail!("Unexpected memo audio format: {spec:?}");
    }
    let total_samples = (reader.duration() as usize).max(1);

    let mut segments = Vec::new();
    let mut run = |start: usize, chunk: &[f32]| -> Result<()> {
        let text = transcribe(chunk)?;
        let text = text.trim();
        if !text.is_empty() {
            let rate = VAD_SAMPLE_RATE as f64;
            segments.push(TranscriptionSegment {
                start: start as f64 / rate,
                end: (start + chunk.len()) as f64 / rate,
                text: text.to_string(),
            });
        }
        Ok(())
    };

    // Offset of `chunk`'s first sample within the recording.
    let mut start = 0;
    let mut chunk = Vec::with_capacity(MAX_CHUNK_SAMPLES);
    for sample in reader.samples::<i16>() {
        let sample = match sample {
            Ok(s) => s,
            Err(e) => {
                // An interrupted recording can end mid-sample; use what's there.
                log::warn!("Memo audio ends early ({e}); transcribing what was read");
                break;
            }
        };
        chunk.push(sample as f32 / 32768.0);
        if chunk.len() < MAX_CHUNK_SAMPLES {
            continue;
        }
        let cut = quietest_cut(&chunk, MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES, CUT_PROBE_SAMPLES);
        let rest = chunk.split_off(cut);
        run(start, &chunk)?;
        start += cut;
        chunk = rest;
        on_progress((start as f64 / total_samples as f64).min(1.0));
    }
    if chunk.len() >= MIN_FINAL_CHUNK_SAMPLES {
        run(start, &chunk)?;
    }
    on_progress(1.0);
    Ok(segments)
}

/// `MM:SS`, or `H:MM:SS` from one hour on.
fn format_timestamp(seconds: f64) -> String {
    let total = seconds as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn transcript_lines(segments: &[TranscriptionSegment]) -> Vec<String> {
    segments
        .iter()
        .map(|s| format!("[{}] {}", format_timestamp(s.start), s.text))
        .collect()
}

// ---------------------------------------------------------------------------
// Minutes
// ---------------------------------------------------------------------------

/// Pack consecutive blocks into newline-joined groups of at most `max_chars`.
/// Blocks too big to share a group are paired up anyway when nothing else
/// packs, so there are always fewer groups than blocks (given two or more).
fn group_by_chars(blocks: &[String], max_chars: usize) -> Vec<String> {
    let mut groups: Vec<String> = Vec::new();
    let mut current_chars = 0;
    for block in blocks {
        let chars = block.chars().count();
        match groups.last_mut() {
            Some(group) if current_chars + chars + 1 <= max_chars => {
                group.push('\n');
                group.push_str(block);
                current_chars += chars + 1;
            }
            _ => {
                groups.push(block.clone());
                current_chars = chars;
            }
        }
    }
    if groups.len() == blocks.len() && blocks.len() > 1 {
        return blocks.chunks(2).map(|pair| pair.join("\n")).collect();
    }
    groups
}

/// Turn transcript lines into meeting minutes.
///
/// `llm(system, user, max_tokens)` runs one completion. A transcript
/// that fits one LLM input is turned into minutes directly; a longer one is
/// first reduced part by part into notes (repeatedly, if the notes are still
/// too long), which are then merged into the minutes.
pub fn generate_minutes(
    lines: &[String],
    language: Option<&str>,
    mut llm: impl FnMut(&str, &str, usize) -> Result<String>,
    mut on_progress: impl FnMut(f64),
) -> Result<String> {
    let with_language = |text: &str| match language {
        Some(lang) => format!("{text}\n\n(Primary language: {lang})"),
        None => text.to_string(),
    };
    let mut call = |system: &str, user: &str, max_tokens: usize| -> Result<String> {
        let out = llm(system, &with_language(user), max_tokens)?;
        let out = out.trim();
        if out.is_empty() {
            bail!("The language model returned no text");
        }
        Ok(out.to_string())
    };

    let mut blocks = lines.to_vec();
    loop {
        let groups = group_by_chars(&blocks, MAX_LLM_INPUT_CHARS);
        if groups.len() <= 1 {
            break;
        }
        let total = groups.len();
        let mut notes = Vec::with_capacity(total);
        for (i, group) in groups.iter().enumerate() {
            let part = call(PART_NOTES_PROMPT, group, PART_NOTES_MAX_TOKENS)?;
            notes.push(format!("## Part {}/{}\n{}", i + 1, total, part));
            on_progress((i + 1) as f64 / (total + 1) as f64);
        }
        blocks = notes;
    }

    let minutes = call(MINUTES_PROMPT, &blocks.join("\n"), MINUTES_MAX_TOKENS)?;
    on_progress(1.0);
    Ok(minutes)
}

// ---------------------------------------------------------------------------
// Processing (transcription + minutes) on a background thread
// ---------------------------------------------------------------------------

fn with_db<T>(app: &AppHandle, f: impl FnOnce(&TranscriptionDb) -> Result<T>) -> Result<T> {
    let state = app.state::<AppState>();
    let db = state
        .transcription_db
        .lock()
        .map_err(|e| anyhow!("Failed to lock DB: {e}"))?;
    f(&db)
}

fn emit_progress(app: &AppHandle, id: i64, status: MemoStatus, progress: Option<f64>) {
    let event = MemoProgressEvent { id, status, progress };
    if let Err(e) = app.emit("memo-progress", &event) {
        log::warn!("Failed to emit memo-progress: {e}");
    }
}

/// Kick off background processing of a memo: transcription followed by
/// minutes, or minutes only when a transcript exists and `retranscribe` is
/// false. Fails if the memo is already being processed.
pub fn start_processing(app: &AppHandle, id: i64, retranscribe: bool) -> Result<()> {
    let status = with_db(app, |db| {
        let memo = db.get_memo(id)?.ok_or_else(|| anyhow!("Memo {id} not found"))?;
        if memo.status.is_processing() {
            bail!("Memo {id} is already being processed");
        }
        let status = if retranscribe || memo.segments_json.is_none() {
            MemoStatus::Transcribing
        } else {
            MemoStatus::Summarizing
        };
        db.set_memo_status(id, status, None)?;
        Ok(status)
    })?;
    emit_progress(app, id, status, None);

    let app = app.clone();
    thread::Builder::new().name("echo-memo".into()).spawn(move || {
        let status = match process_memo(&app, id) {
            Ok(()) => MemoStatus::Done,
            Err(e) => {
                log::error!("Memo {id} processing failed: {e:#}");
                if let Err(db_err) = with_db(&app, |db| {
                    db.set_memo_status(id, MemoStatus::Error, Some(&format!("{e:#}")))
                }) {
                    log::error!("Failed to record memo {id} error: {db_err}");
                }
                MemoStatus::Error
            }
        };
        emit_progress(&app, id, status, None);
    })?;
    Ok(())
}

/// Run the stage(s) selected by `start_processing` (read back from the memo's
/// status) and leave the memo `Done`.
fn process_memo(app: &AppHandle, id: i64) -> Result<()> {
    let state = app.state::<AppState>();
    let memo = with_db(app, |db| db.get_memo(id))?.ok_or_else(|| anyhow!("Memo {id} not found"))?;
    let language = {
        let settings = state.settings.lock().map_err(|e| anyhow!("{e}"))?;
        (settings.language != "auto").then(|| settings.language.clone())
    };

    let segments = if memo.status == MemoStatus::Transcribing {
        let mut model_name = String::new();
        let segments = transcribe_wav_chunked(
            Path::new(&memo.audio_path),
            |chunk| {
                // Lock per chunk so hotkey dictation can run in between.
                let mut engine = state.asr_engine.lock().map_err(|e| anyhow!("{e}"))?;
                model_name = engine.active_model_name().to_string();
                // No VAD gate: it would drop speech the VAD fails to detect.
                Ok(engine.transcribe_samples(chunk, language.as_deref(), false)?.text)
            },
            |p| emit_progress(app, id, MemoStatus::Transcribing, Some(p)),
        )?;
        log::info!("Memo {id}: transcribed {} chunks", segments.len());

        let transcript = transcript_lines(&segments).join("\n");
        let segments_json = serde_json::to_string(&segments)?;
        with_db(app, |db| {
            if memo.duration_seconds.is_none() {
                // Interrupted recording: the duration was never stored.
                let reader = WavReader::open(&memo.audio_path)?;
                db.set_memo_duration(id, reader.duration() as f64 / VAD_SAMPLE_RATE as f64)?;
            }
            db.set_memo_transcript(
                id,
                &transcript,
                Some(&segments_json),
                language.as_deref(),
                Some(&model_name).filter(|m| !m.is_empty()).map(String::as_str),
            )?;
            // Minutes of a previous transcript no longer apply.
            db.set_memo_minutes(id, None)?;
            if segments.is_empty() {
                db.set_memo_status(id, MemoStatus::Done, None)
            } else {
                db.set_memo_status(id, MemoStatus::Summarizing, None)
            }
        })?;
        if segments.is_empty() {
            log::info!("Memo {id}: no speech found");
            return Ok(());
        }
        emit_progress(app, id, MemoStatus::Summarizing, None);
        segments
    } else {
        let json = memo.segments_json.as_deref().unwrap_or("[]");
        serde_json::from_str::<Vec<TranscriptionSegment>>(json)?
    };

    let minutes = generate_minutes(
        &transcript_lines(&segments),
        language.as_deref(),
        |system, user, max_tokens| {
            let mut engine = state.asr_engine.lock().map_err(|e| anyhow!("{e}"))?;
            engine.llm_chat(system, user, max_tokens)
        },
        |p| emit_progress(app, id, MemoStatus::Summarizing, Some(p)),
    )
    .context("Minutes generation failed")?;

    with_db(app, |db| {
        db.set_memo_minutes(id, Some(&minutes))?;
        db.set_memo_status(id, MemoStatus::Done, None)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::VAD_FRAME_SIZE;

    /// Write a 16 kHz mono WAV of a steady tone, silent in `gaps` (seconds).
    fn tone_wav(path: &Path, seconds: f64, gaps: &[(f64, f64)]) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: VAD_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for i in 0..samples(seconds) {
            let t = i as f64 / VAD_SAMPLE_RATE as f64;
            let silent = gaps.iter().any(|&(from, to)| (from..to).contains(&t));
            let s = if silent { 0.0 } else { (t * 220.0 * std::f64::consts::TAU).sin() * 0.3 };
            writer.write_sample((s * i16::MAX as f64) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    /// Chunk `path`, returning each chunk's `(start, end)` in seconds.
    fn chunk_spans(path: &Path) -> Vec<(f64, f64)> {
        let segments = transcribe_wav_chunked(path, |_| Ok("x".to_string()), |_| {}).unwrap();
        segments.iter().map(|s| (s.start, s.end)).collect()
    }

    #[test]
    fn chunks_cover_the_whole_recording_without_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo.wav");
        // The long silences must not be skipped.
        tone_wav(&path, 70.0, &[(5.0, 15.0), (40.0, 44.0)]);
        let spans = chunk_spans(&path);
        assert_eq!(spans[0].0, 0.0);
        for pair in spans.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
        assert_eq!(spans.last().unwrap().1, 70.0);
        assert!(spans.iter().all(|(start, end)| end - start <= 28.0));
    }

    #[test]
    fn chunks_are_cut_at_the_quietest_point() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo.wav");
        tone_wav(&path, 40.0, &[(23.0, 23.6)]);
        let spans = chunk_spans(&path);
        assert_eq!(spans.len(), 2);
        assert!((23.0..23.6).contains(&spans[0].1), "{}", spans[0].1);
    }

    #[test]
    fn short_recording_is_one_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo.wav");
        tone_wav(&path, 3.0, &[]);
        assert_eq!(chunk_spans(&path), vec![(0.0, 3.0)]);
    }

    #[test]
    fn timestamps_switch_to_hours() {
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(754.9), "12:34");
        assert_eq!(format_timestamp(3725.0), "1:02:05");
    }

    #[test]
    fn groups_respect_char_budget() {
        let blocks: Vec<String> = ["aaaa", "bbbb", "cccc", "dddddddddddd"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            group_by_chars(&blocks, 9),
            vec!["aaaa\nbbbb", "cccc", "dddddddddddd"]
        );
        // Nothing fits together: pair up rather than make no progress.
        assert_eq!(group_by_chars(&blocks[1..], 4), vec!["bbbb\ncccc", "dddddddddddd"]);
        // Counted in chars, not bytes.
        let ja = vec!["あいうえ".to_string(), "かきくけ".to_string()];
        assert_eq!(group_by_chars(&ja, 9), vec!["あいうえ\nかきくけ"]);
    }

    #[test]
    fn short_transcript_is_one_llm_call() {
        let lines = vec!["[00:00] こんにちは".to_string()];
        let mut calls = Vec::new();
        let minutes = generate_minutes(
            &lines,
            Some("ja"),
            |system, user, _| {
                calls.push((system.to_string(), user.to_string()));
                Ok("## 概要\nあいさつ".to_string())
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(minutes, "## 概要\nあいさつ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, MINUTES_PROMPT);
        assert_eq!(calls[0].1, "[00:00] こんにちは\n\n(Primary language: ja)");
    }

    #[test]
    fn long_transcript_is_reduced_to_notes_then_merged() {
        // 30 lines × ~310 chars ≈ 9300 chars → 3 parts of ≤3500 chars.
        let lines: Vec<String> = (0..30)
            .map(|i| format!("[{:02}:00] {}", i, "あ".repeat(300)))
            .collect();
        let mut systems = Vec::new();
        let mut last_user = String::new();
        let mut progress = Vec::new();
        generate_minutes(
            &lines,
            None,
            |system, user, _| {
                assert!(user.chars().count() <= MAX_LLM_INPUT_CHARS);
                systems.push(system.to_string());
                last_user = user.to_string();
                Ok(format!("notes {}", systems.len()))
            },
            |p| progress.push(p),
        )
        .unwrap();
        assert_eq!(
            systems,
            vec![PART_NOTES_PROMPT, PART_NOTES_PROMPT, PART_NOTES_PROMPT, MINUTES_PROMPT]
        );
        assert_eq!(
            last_user,
            "## Part 1/3\nnotes 1\n## Part 2/3\nnotes 2\n## Part 3/3\nnotes 3"
        );
        assert_eq!(progress, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn oversized_notes_still_converge() {
        // Every note is over half the budget, so no two pack together; the
        // reduction must keep merging instead of dumping them all into the
        // final call.
        let lines: Vec<String> = (0..40).map(|i| format!("[{i:02}:00] {}", "a".repeat(600))).collect();
        let mut max_input = 0;
        generate_minutes(
            &lines,
            None,
            |system, user, _| {
                max_input = max_input.max(user.chars().count());
                Ok(if system == MINUTES_PROMPT { "## 概要".into() } else { "n".repeat(2000) })
            },
            |_| {},
        )
        .unwrap();
        assert!(max_input <= 2 * 2100, "{max_input}");
    }

    #[test]
    fn empty_llm_output_is_an_error() {
        let lines = vec!["[00:00] テスト".to_string()];
        let err = generate_minutes(&lines, None, |_, _, _| Ok("  ".into()), |_| {}).unwrap_err();
        assert!(err.to_string().contains("no text"), "{err}");
    }

    /// End-to-end on real audio with the cached models:
    /// `ECHO_MEMO_WAV=<16k mono wav> [ECHO_MEMO_MODEL=<asr model id>] \
    ///  cargo test memo_end_to_end -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn memo_end_to_end() {
        let wav = std::env::var("ECHO_MEMO_WAV").expect("set ECHO_MEMO_WAV");
        let model = std::env::var("ECHO_MEMO_MODEL")
            .unwrap_or_else(|_| "mlx-community/parakeet-tdt_ctc-0.6b-ja".to_string());
        let home = std::env::var("HOME").unwrap();
        let hub = PathBuf::from(home).join("Library/Caches/io.qluto.echo/huggingface/hub");
        let mut engine = crate::transcription::ASREngine::with_hub(hub);
        engine.set_model(&model).unwrap();

        let t = std::time::Instant::now();
        let segments = transcribe_wav_chunked(
            Path::new(&wav),
            |chunk| Ok(engine.transcribe_samples(chunk, Some("ja"), false)?.text),
            |_| {},
        )
        .unwrap();
        let lines = transcript_lines(&segments);
        println!("--- transcript ({} chunks, {:?}) ---", segments.len(), t.elapsed());
        println!("{}", lines.join("\n"));
        assert!(!segments.is_empty());
        assert!(segments.iter().all(|s| s.end - s.start <= 28.5));

        let t = std::time::Instant::now();
        let mut calls = 0;
        let minutes = generate_minutes(
            &lines,
            Some("ja"),
            |system, user, max_tokens| {
                calls += 1;
                engine.llm_chat(system, user, max_tokens)
            },
            |_| {},
        )
        .unwrap();
        println!("--- minutes ({calls} LLM calls, {:?}) ---\n{minutes}", t.elapsed());
        assert!(minutes.contains("##"));
    }

    #[test]
    fn writer_streams_frames_to_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: VAD_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };
        let writer = WavWriter::create(&path, spec).unwrap();
        let (tx, rx) = channel::unbounded();
        let samples = Arc::new(AtomicU64::new(0));
        let level = Arc::new(AtomicU32::new(0f32.to_bits()));
        let handle = {
            let (samples, level) = (Arc::clone(&samples), Arc::clone(&level));
            thread::spawn(move || writer_loop(writer, vec![0.5; VAD_FRAME_SIZE], rx, &samples, &level))
        };

        // Enough frames to cross a flush boundary: the file must already be a
        // valid WAV while the recording is still running.
        let n = FLUSH_INTERVAL_SAMPLES / VAD_FRAME_SIZE + 10;
        for _ in 0..n {
            tx.send(vec![0.5; VAD_FRAME_SIZE]).unwrap();
        }
        let expected = ((n + 1) * VAD_FRAME_SIZE) as u64;
        while samples.load(Ordering::Relaxed) < expected {
            thread::sleep(Duration::from_millis(5));
        }
        let flushed = WavReader::open(&path).unwrap().duration() as usize;
        assert!(flushed >= FLUSH_INTERVAL_SAMPLES, "{flushed}");
        // 0.5 amplitude ≈ -6 dBFS, above the meter's ceiling.
        assert_eq!(f32::from_bits(level.load(Ordering::Relaxed)), 1.0);

        drop(tx);
        handle.join().unwrap().unwrap();
        let mut reader = WavReader::open(&path).unwrap();
        assert_eq!(reader.duration() as u64, expected);
        assert_eq!(reader.samples::<i16>().next().unwrap().unwrap(), i16::MAX / 2);
    }

    /// Records 2 s from the default input device (needs microphone access):
    /// `cargo test memo_recorder_writes_wav -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn memo_recorder_writes_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo.wav");
        let recorder = MemoRecorder::start(1, None, &path).unwrap();
        thread::sleep(Duration::from_secs(2));
        let live = recorder.status();
        assert!(live.elapsed_seconds > 1.0, "{live:?}");
        let (id, duration) = recorder.stop();
        assert_eq!(id, 1);

        let reader = WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, VAD_SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        let file_secs = reader.duration() as f64 / VAD_SAMPLE_RATE as f64;
        println!("recorded {duration:.2}s, file {file_secs:.2}s, level {:.2}", live.level);
        assert!((duration - file_secs).abs() < 1e-6);
        assert!((1.5..3.5).contains(&duration), "{duration}");
    }
}
