//! In-progress ("partial") decoding while the hotkey is held.
//!
//! Follows hayamimi's draft scheme: every ~0.5 s of new audio, re-decode the
//! last ≤8 s of the recording and show the result as a draft. The window cap
//! keeps each decode O(1) regardless of how long the key is held; the final
//! transcription on release still uses the full recording as before.
//!
//! So that a long hold still shows *everything* said so far, audio that
//! falls out of the window is not dropped: once the buffer exceeds the
//! window, it is cut at the quietest point in the window's second half, the
//! part before the cut is decoded once more and appended to a `committed`
//! prefix, and drafting continues on the remainder. The draft shown is
//! `committed + current window`.
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
/// Granularity (100 ms) for locating the quietest point to commit at.
const CUT_PROBE_SEC: f64 = 0.1;

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
    let mut committed = String::new();
    let mut last_text = String::new();
    let mut enabled: Option<bool> = None; // resolved lazily from the loaded engine

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        match frame_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => audio.extend_from_slice(&frame),
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

        // Buffer outgrew the window: commit the head so it stays visible.
        if audio.len() > window {
            let cut = quietest_cut(&audio, window / 2, window, (CUT_PROBE_SEC * VAD_SAMPLE_RATE as f64) as usize);
            let t0 = std::time::Instant::now();
            match engine.transcribe_samples(&audio[..cut], language, false) {
                Ok(r) if r.success => {
                    log::debug!(
                        "partial: committed {:.1}s in {} ms: {}",
                        cut as f64 / VAD_SAMPLE_RATE as f64,
                        t0.elapsed().as_millis(),
                        r.text.trim()
                    );
                    append_text(&mut committed, r.text.trim());
                }
                Ok(_) => {}
                Err(e) => log::warn!("partial commit decode failed: {}", e),
            }
            audio.drain(..cut);
            decoded_upto = audio.len();
        }

        let t0 = std::time::Instant::now();
        let result = engine.transcribe_samples(&audio, language, false);
        drop(engine);

        match result {
            Ok(r) if r.success => {
                let mut text = committed.clone();
                append_text(&mut text, r.text.trim());
                log::debug!(
                    "partial: {:.1}s window in {} ms: {}",
                    audio.len() as f64 / VAD_SAMPLE_RATE as f64,
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

/// Index in `[lo, hi)` (aligned to `probe`-sample steps) of the quietest
/// `probe`-long stretch — the least likely place to be mid-word.
fn quietest_cut(audio: &[f32], lo: usize, hi: usize, probe: usize) -> usize {
    let hi = hi.min(audio.len());
    let mut best = hi;
    let mut best_energy = f32::INFINITY;
    let mut i = lo;
    while i + probe <= hi {
        let e: f32 = audio[i..i + probe].iter().map(|s| s * s).sum();
        if e < best_energy {
            best_energy = e;
            best = i + probe / 2;
        }
        i += probe;
    }
    best
}

/// Join draft pieces: no separator between CJK text, a space otherwise.
fn append_text(acc: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    if let Some(last) = acc.chars().last() {
        let first = piece.chars().next().unwrap_or(' ');
        if !(is_cjk(last) || is_cjk(first)) {
            acc.push(' ');
        }
    }
    acc.push_str(piece);
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0xAC00..=0xD7AF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quietest_cut_finds_the_silent_gap() {
        let mut audio = vec![0.5f32; 4000];
        for s in &mut audio[2500..2700] {
            *s = 0.0;
        }
        let cut = quietest_cut(&audio, 1000, 4000, 100);
        assert!((2500..2700).contains(&cut), "cut at {cut}");
    }

    #[test]
    fn append_text_joins_cjk_without_space_and_latin_with_space() {
        let mut s = String::from("今日は");
        append_text(&mut s, "いい天気");
        assert_eq!(s, "今日はいい天気");
        let mut e = String::from("hello");
        append_text(&mut e, "world");
        assert_eq!(e, "hello world");
        let mut m = String::new();
        append_text(&mut m, "");
        assert_eq!(m, "");
    }
}
