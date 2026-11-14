//! Meeting standby: listen for a conversation and start a meeting by itself,
//! from shortly before the conversation began.
//!
//! ```text
//! live captures (mic + system audio)
//!   └→ [standby thread] ─ every frame → 5 min ring buffer (memory only)
//!                       └ Silero VAD or a sustained rise over the
//!                         noise floor → TalkDetector (10 s buckets)
//!          talk in 14 of the last 18 buckets (3 min), 6 of them from the mic
//!            → the captures and the buffered audio from the conversation's
//!              start are handed to a meeting (`MeetingSession::start_with`)
//! ```
//!
//! Detection only runs the VAD (on the CPU); nothing is transcribed and
//! nothing is written to disk until a meeting starts. The captures are handed
//! over rather than restarted, so no audio is lost at the switch.
//!
//! The mic only counts as talk while system audio is quiet, so a video playing
//! through the speakers doesn't start a meeting, and it doesn't count while
//! hotkey dictation or a voice memo has it. Standby pauses while a meeting
//! runs and resumes when the meeting is done.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::continuous;
use crate::live::{self, Captures, Inputs, Source};
use crate::meeting::{Handoff, MeetingSession};
use crate::memo::frame_dbfs;
use crate::types::AppState;
use crate::vad::{VadProcessor, VAD_FRAME_SIZE, VAD_SAMPLE_RATE};

/// The ring buffer keeps this much audio (per source)…
const RING_FRAMES: usize = live::samples(300.0) / VAD_FRAME_SIZE;
/// …and talk is judged in buckets this long.
const BUCKET_FRAMES: usize = live::samples(10.0) / VAD_FRAME_SIZE;
const RING_BUCKETS: usize = RING_FRAMES / BUCKET_FRAMES;
/// A bucket is talk when a source has speech in this share of its frames (1 s of 10).
const TALK_SHARE: f64 = 0.10;
/// A meeting starts when, of the last `WINDOW_BUCKETS` (3 min)…
const WINDOW_BUCKETS: usize = 18;
/// …this many are talk (lulls are fine: real conversations scored 11 of 14 to
/// all buckets, scattered remarks half)…
const MIN_TALK_BUCKETS: usize = 14;
/// …and the user (the mic) takes part: talking in this many buckets, for this
/// share of the window in all (27 s). Listening to a recording or a webinar,
/// with a remark now and then ("なるほど"), is not a meeting.
const MIN_MIC_BUCKETS: usize = 6;
const MIN_MIC_SHARE: f64 = 0.15;
/// The conversation began after the last run of more than this many quiet buckets…
const MAX_GAP_BUCKETS: usize = 2;
/// …and the meeting starts this many buckets before it.
const LEAD_IN_BUCKETS: usize = 1;

const VAD_THRESHOLD: f32 = 0.5;
/// Frames quieter than this are never speech (and aren't run through the VAD).
const SILENT_DBFS: f32 = -60.0;
/// Silero misses audible speech on some microphones (it heard almost none of
/// two recorded meetings), so a frame also counts when it is this far above
/// the noise floor (5th percentile of the last `NOISE_WINDOW_FRAMES`, updated
/// once a bucket; a shorter window rises with continuous talk)…
const PROMINENCE_DB: f32 = 12.0;
const NOISE_WINDOW_FRAMES: usize = live::samples(120.0) / VAD_FRAME_SIZE;
const NOISE_PERCENTILE: usize = 5;
/// …within a run of at least this many such frames (~130 ms): a voice is
/// sustained, keyboard clicks are not.
const MIN_PROMINENT_RUN: usize = 4;
/// While system audio is louder than this (and for `ECHO_HOLD_FRAMES` after),
/// the mic may be hearing the speakers, so it doesn't count as talk.
const OTHERS_LOUD_DBFS: f32 = -45.0;
const ECHO_HOLD_FRAMES: u32 = 10;

/// When standby last heard speech (Unix seconds; 0 = never), so that an
/// update does not restart the app in the middle of a conversation it is
/// buffering.
static LAST_TALK: AtomicU64 = AtomicU64::new(0);

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Seconds since standby last heard speech (very large if never).
pub fn seconds_since_talk() -> f64 {
    now_secs().saturating_sub(LAST_TALK.load(Ordering::Relaxed)) as f64
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Bucket {
    /// Some source has speech.
    talk: bool,
    /// Speech frames from the mic.
    mic_speech: usize,
}

impl Bucket {
    fn mic_talk(&self) -> bool {
        is_talk(self.mic_speech)
    }
}

fn is_talk(speech_frames: usize) -> bool {
    speech_frames as f64 >= TALK_SHARE * BUCKET_FRAMES as f64
}

/// Decides from per-frame "speech or not" whether a conversation is going on.
/// The mic's frames are the clock (system audio may be absent).
#[derive(Default)]
struct TalkDetector {
    mic_frames: usize,
    /// Speech frames per source in the bucket being filled.
    speech: [usize; 2],
    /// Finished buckets, newest last, at most `RING_BUCKETS`.
    buckets: VecDeque<Bucket>,
}

impl TalkDetector {
    /// Add one frame. When a conversation has gone on for 3 min, returns how
    /// many mic frames back the meeting should start (its beginning, plus a
    /// lead-in).
    fn push(&mut self, source: Source, speech: bool) -> Option<usize> {
        if speech {
            self.speech[source.index()] += 1;
        }
        if source != Source::Me {
            return None;
        }
        self.mic_frames += 1;
        if !self.mic_frames.is_multiple_of(BUCKET_FRAMES) {
            return None;
        }
        let [mic, others] = std::mem::take(&mut self.speech);
        self.buckets.push_back(Bucket { talk: is_talk(mic) || is_talk(others), mic_speech: mic });
        if self.buckets.len() > RING_BUCKETS {
            self.buckets.pop_front();
        }
        let start = self.conversation_start();
        self.conversation(start).then(|| (self.buckets.len() - start.saturating_sub(LEAD_IN_BUCKETS)) * BUCKET_FRAMES)
    }

    /// A conversation beginning at bucket `start` has gone on for 3 min, with
    /// enough talk and enough of it from the user.
    fn conversation(&self, start: usize) -> bool {
        if self.buckets.len() - start < WINDOW_BUCKETS || !self.buckets.back().is_some_and(|b| b.talk) {
            return false;
        }
        let window = || self.buckets.iter().rev().take(WINDOW_BUCKETS);
        let talk = window().filter(|b| b.talk).count();
        let mic_buckets = window().filter(|b| b.mic_talk()).count();
        let mic_speech: usize = window().map(|b| b.mic_speech).sum();
        talk >= MIN_TALK_BUCKETS
            && mic_buckets >= MIN_MIC_BUCKETS
            && mic_speech as f64 >= MIN_MIC_SHARE * (WINDOW_BUCKETS * BUCKET_FRAMES) as f64
    }

    /// The first bucket of the current conversation: the earliest talk not
    /// separated from now by more than `MAX_GAP_BUCKETS` quiet ones (at most
    /// the whole ring).
    fn conversation_start(&self) -> usize {
        let mut start = self.buckets.len();
        let mut gap = 0;
        for (i, bucket) in self.buckets.iter().enumerate().rev() {
            if bucket.talk {
                start = i;
                gap = 0;
            } else {
                gap += 1;
                if gap > MAX_GAP_BUCKETS {
                    break;
                }
            }
        }
        start
    }
}

/// The last `RING_FRAMES` mic frames and the system audio frames that came
/// with them, in arrival order (the order a meeting would have seen them in).
#[derive(Default)]
struct Ring {
    frames: VecDeque<(Source, Vec<f32>)>,
    mic_frames: usize,
}

impl Ring {
    fn push(&mut self, source: Source, frame: Vec<f32>) {
        self.frames.push_back((source, frame));
        if source == Source::Me {
            self.mic_frames += 1;
        }
        self.keep_last(RING_FRAMES);
    }

    /// Drop all but the last `mic_frames` mic frames (and what came with them).
    fn keep_last(&mut self, mic_frames: usize) {
        while self.mic_frames > mic_frames {
            if let Some((Source::Me, _)) = self.frames.pop_front() {
                self.mic_frames -= 1;
            }
        }
    }
}

/// Loudness of one source relative to its noise floor.
struct Prominence {
    recent_db: VecDeque<f32>,
    floor: f32,
    frames: usize,
    run: usize,
}

impl Default for Prominence {
    fn default() -> Self {
        // Nothing stands out until there is a floor to stand out from.
        Self { recent_db: VecDeque::with_capacity(NOISE_WINDOW_FRAMES + 1), floor: f32::INFINITY, frames: 0, run: 0 }
    }
}

impl Prominence {
    /// Add a frame's level; whether it is in a sustained run above the floor.
    fn push(&mut self, dbfs: f32) -> bool {
        self.recent_db.push_back(dbfs);
        if self.recent_db.len() > NOISE_WINDOW_FRAMES {
            self.recent_db.pop_front();
        }
        self.frames += 1;
        if self.frames.is_multiple_of(BUCKET_FRAMES) {
            let mut sorted: Vec<f32> = self.recent_db.iter().copied().collect();
            sorted.sort_by(f32::total_cmp);
            self.floor = sorted[sorted.len() * NOISE_PERCENTILE / 100];
        }
        self.run = if dbfs >= self.floor + PROMINENCE_DB { self.run + 1 } else { 0 };
        self.run >= MIN_PROMINENT_RUN
    }
}

/// Whether one frame is speech, per source (each with its own VAD state).
struct Listener {
    vad: [VadProcessor; 2],
    prominence: [Prominence; 2],
    /// Frames of system audio left before the mic counts again.
    echo_hold: u32,
}

impl Listener {
    fn new() -> Result<Self> {
        Ok(Self {
            vad: [VadProcessor::new(VAD_THRESHOLD)?, VadProcessor::new(VAD_THRESHOLD)?],
            prominence: Default::default(),
            echo_hold: 0,
        })
    }

    fn is_speech(&mut self, source: Source, frame: &[f32]) -> bool {
        let dbfs = frame_dbfs(frame);
        let i = source.index();
        let prominent = self.prominence[i].push(dbfs);
        if source == Source::Others {
            self.echo_hold = if dbfs > OTHERS_LOUD_DBFS { ECHO_HOLD_FRAMES } else { self.echo_hold.saturating_sub(1) };
        }
        if frame.len() != VAD_FRAME_SIZE || dbfs < SILENT_DBFS || continuous::is_muted(source) {
            return false;
        }
        if source == Source::Me && self.echo_hold > 0 {
            return false;
        }
        prominent || self.vad[i].predict(frame) > VAD_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// Running standby
// ---------------------------------------------------------------------------

/// Standby listening; its captures move to a meeting when one starts.
struct Running {
    captures: Arc<Mutex<Option<Captures>>>,
    warning: Option<String>,
}

impl Running {
    fn start(app: &AppHandle, device_name: Option<String>) -> Result<Self> {
        let (captures, inputs, system_error) = live::start_captures(device_name)?;
        let warning = system_error.clone().map(|e| format!("システム音声を取得できないため、マイクだけで検知します: {e}"));
        let listener = Listener::new()?;
        let captures = Arc::new(Mutex::new(Some(captures)));
        {
            let (app, captures) = (app.clone(), Arc::clone(&captures));
            thread::Builder::new()
                .name("echo-standby".into())
                .spawn(move || listen(&app, &captures, inputs, listener, system_error))?;
        }
        log::info!("Meeting standby listening");
        Ok(Self { captures, warning })
    }

    /// Still listening (not stopped, not handed to a meeting).
    fn is_listening(&self) -> bool {
        self.captures.lock().is_ok_and(|c| c.is_some())
    }

    fn stop(&self) {
        // Closing the frame channels ends the thread and frees the ring.
        if let Some(mut captures) = self.captures.lock().ok().and_then(|mut c| c.take()) {
            captures.stop();
            log::info!("Meeting standby stopped");
        }
    }
}

fn listen(
    app: &AppHandle,
    captures: &Mutex<Option<Captures>>,
    mut inputs: Inputs,
    mut listener: Listener,
    system_error: Option<String>,
) {
    let mut ring = Ring::default();
    let mut detector = TalkDetector::default();
    while let Some((source, frame)) = inputs.next_frame() {
        let speech = listener.is_speech(source, &frame);
        if speech {
            LAST_TALK.store(now_secs(), Ordering::Relaxed);
        }
        ring.push(source, frame);
        let Some(back) = detector.push(source, speech) else { continue };

        let Some(captures) = captures.lock().ok().and_then(|mut c| c.take()) else {
            return; // stopped meanwhile
        };
        ring.keep_last(back);
        let preroll = Duration::from_secs_f64((back * VAD_FRAME_SIZE) as f64 / VAD_SAMPLE_RATE as f64);
        log::info!("Meeting standby: conversation detected; starting a meeting {:.0}s back", preroll.as_secs_f64());
        inputs.prepend(std::mem::take(&mut ring.frames));
        let notice = format!(
            "会議・立ち話を検知しました。{}前から記録しています（会話が3分途切れると自動で終了します）",
            format_minutes(preroll)
        );
        start_meeting(app, Handoff { captures, inputs, system_error, preroll, notice });
        return;
    }
    log::info!("Meeting standby thread exiting");
}

fn format_minutes(d: Duration) -> String {
    let secs = d.as_secs();
    match (secs / 60, secs % 60) {
        (0, s) => format!("{s}秒"),
        (m, 0) => format!("{m}分"),
        (m, s) => format!("{m}分{s}秒"),
    }
}

fn start_meeting(app: &AppHandle, handoff: Handoff) {
    let state = app.state::<AppState>();
    let started = (|| -> Result<bool> {
        let mut current = state.meeting.lock().map_err(|e| anyhow!("{e}"))?;
        if current.as_ref().is_some_and(|m| m.is_active()) {
            // Started by hand just now; it resumes standby when it is done.
            return Ok(false);
        }
        let session = MeetingSession::start_with(app, Some(handoff))?;
        continuous::set_muted(continuous::Mute::Meeting, true);
        *current = Some(session);
        Ok(true)
    })();
    match started {
        Ok(false) => emit_status(app),
        Ok(true) => {
            if let Some(window) = app.get_webview_window("meeting") {
                // Shown, not focused: the user may be typing elsewhere.
                if let Err(e) = window.show() {
                    log::warn!("Failed to show the meeting window: {e}");
                }
            }
            emit_status(app);
        }
        Err(e) => {
            log::error!("Meeting standby could not start a meeting: {e:#}");
            set_error(app, Some(format!("会議を開始できませんでした: {e:#}")));
            resume(app);
        }
    }
}

// ---------------------------------------------------------------------------
// App-facing control
// ---------------------------------------------------------------------------

/// Standby as the app holds it (`AppState::standby`).
#[derive(Default)]
pub struct Standby {
    running: Option<Running>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StandbyStatus {
    /// The setting.
    pub enabled: bool,
    /// Listening now (not while a meeting runs).
    pub listening: bool,
    pub warning: Option<String>,
    pub error: Option<String>,
}

pub fn status(app: &AppHandle) -> StandbyStatus {
    let state = app.state::<AppState>();
    let enabled = state.settings.lock().is_ok_and(|s| s.meeting_standby);
    let standby = state.standby.lock().unwrap_or_else(|e| e.into_inner());
    let running = standby.running.as_ref().filter(|r| r.is_listening());
    StandbyStatus {
        enabled,
        listening: running.is_some(),
        warning: running.and_then(|r| r.warning.clone()),
        error: standby.error.clone(),
    }
}

fn emit_status(app: &AppHandle) {
    if let Err(e) = app.emit("meeting-standby", status(app)) {
        log::warn!("Failed to emit meeting-standby: {e}");
    }
}

fn set_error(app: &AppHandle, error: Option<String>) {
    app.state::<AppState>().standby.lock().unwrap_or_else(|e| e.into_inner()).error = error;
    emit_status(app);
}

/// Forget the last error (standby was turned off or on).
pub fn clear_error(app: &AppHandle) {
    app.state::<AppState>().standby.lock().unwrap_or_else(|e| e.into_inner()).error = None;
}

/// Stop listening (a meeting is starting, or standby was turned off).
pub fn pause(app: &AppHandle) {
    let state = app.state::<AppState>();
    let mut standby = state.standby.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(running) = standby.running.take() {
        running.stop();
    }
    drop(standby);
    emit_status(app);
}

/// Start listening if standby is on, not already listening and no meeting is
/// running. In the background: starting waits for the mic's first audio.
pub fn resume(app: &AppHandle) {
    let app = app.clone();
    let spawned = thread::Builder::new().name("echo-standby-start".into()).spawn(move || {
        let state = app.state::<AppState>();
        let (enabled, device_name) = match state.settings.lock() {
            Ok(s) => (s.meeting_standby, s.device_name.clone()),
            Err(_) => return,
        };
        // Lock order: standby, then meeting.
        let mut standby = state.standby.lock().unwrap_or_else(|e| e.into_inner());
        if !enabled || standby.running.as_ref().is_some_and(Running::is_listening) {
            return;
        }
        if state.meeting.lock().map_or(true, |m| m.as_ref().is_some_and(|m| m.is_active())) {
            return; // resumed when the meeting is done
        }
        match Running::start(&app, device_name) {
            // An earlier error stays until standby is turned off or on.
            Ok(running) => standby.running = Some(running),
            Err(e) => {
                log::error!("Meeting standby failed to start: {e:#}");
                standby.running = None;
                standby.error = Some(format!("待機を開始できませんでした: {e:#}"));
            }
        }
        drop(standby);
        emit_status(&app);
    });
    if let Err(e) = spawned {
        log::error!("Failed to spawn the standby start thread: {e}");
    }
}

/// Listen afresh (the input device changed).
pub fn restart(app: &AppHandle) {
    pause(app);
    resume(app);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed one bucket's worth of mic frames (speech in the first `mic_share`
    /// of them) and system audio frames (`others_share`); returns the detector's answer.
    fn bucket(d: &mut TalkDetector, mic_share: f64, others_share: f64) -> Option<usize> {
        let mut result = None;
        for i in 0..BUCKET_FRAMES {
            let share = i as f64 / BUCKET_FRAMES as f64;
            d.push(Source::Others, share < others_share);
            result = d.push(Source::Me, share < mic_share).or(result);
        }
        result
    }

    #[test]
    fn three_minutes_of_conversation_start_a_meeting() {
        let mut d = TalkDetector::default();
        for _ in 0..12 {
            assert_eq!(bucket(&mut d, 0.0, 0.0), None); // 2 min of quiet
        }
        let mut detected = None;
        for i in 0..WINDOW_BUCKETS {
            // Two people at the mic, with a quiet bucket now and then.
            let talk = if i % 6 == 2 { 0.0 } else { 0.4 };
            if let Some(back) = bucket(&mut d, talk, 0.0) {
                detected = Some((i, back));
                break;
            }
        }
        let (at, back) = detected.expect("no meeting started");
        assert_eq!(at, WINDOW_BUCKETS - 1, "starts once 3 min are in");
        // From the first talk bucket, plus one bucket of lead-in.
        assert_eq!(back, (WINDOW_BUCKETS + LEAD_IN_BUCKETS) * BUCKET_FRAMES);
    }

    #[test]
    fn continuous_talk_must_go_on_for_three_minutes() {
        let mut d = TalkDetector::default();
        for _ in 0..12 {
            bucket(&mut d, 0.0, 0.0);
        }
        let at = (0..WINDOW_BUCKETS * 2).find(|_| bucket(&mut d, 0.5, 0.0).is_some());
        assert_eq!(at, Some(WINDOW_BUCKETS - 1));
    }

    #[test]
    fn talk_on_system_audio_alone_is_not_a_meeting() {
        let mut d = TalkDetector::default();
        for _ in 0..RING_BUCKETS * 2 {
            assert_eq!(bucket(&mut d, 0.0, 0.6), None);
        }
    }

    #[test]
    fn a_call_where_the_user_talks_now_and_then_is_a_meeting() {
        let mut d = TalkDetector::default();
        let detected = (0..WINDOW_BUCKETS).find_map(|i| {
            // The user speaks in a third of the buckets, ~30 s in all.
            let mine = if i % 3 == 0 { 0.5 } else { 0.0 };
            bucket(&mut d, mine, 0.5)
        });
        assert!(detected.is_some());
    }

    #[test]
    fn listening_to_a_recording_with_remarks_now_and_then_is_not_a_meeting() {
        let mut d = TalkDetector::default();
        for i in 0..RING_BUCKETS * 2 {
            // A second or so from the user in every other bucket.
            let mine = if i % 2 == 0 { 0.12 } else { 0.0 };
            assert_eq!(bucket(&mut d, mine, 0.6), None);
        }
    }

    #[test]
    fn scattered_remarks_are_not_a_meeting() {
        let mut d = TalkDetector::default();
        for i in 0..RING_BUCKETS * 2 {
            let talk = if i % 2 == 0 { 0.5 } else { 0.0 };
            assert_eq!(bucket(&mut d, talk, 0.0), None);
        }
    }

    #[test]
    fn the_start_goes_back_over_short_lulls_but_not_long_silences() {
        let mut d = TalkDetector::default();
        for _ in 0..3 {
            bucket(&mut d, 0.5, 0.0); // an earlier, separate remark
        }
        for _ in 0..MAX_GAP_BUCKETS + 1 {
            bucket(&mut d, 0.0, 0.0);
        }
        for _ in 0..3 {
            bucket(&mut d, 0.5, 0.0); // the conversation begins…
        }
        for _ in 0..MAX_GAP_BUCKETS {
            bucket(&mut d, 0.0, 0.0); // …with a lull
        }
        let back = (0..WINDOW_BUCKETS).find_map(|_| bucket(&mut d, 0.5, 0.0)).expect("no meeting started");
        // The conversation's first bucket, after the remark and the silence.
        let start = 3 + MAX_GAP_BUCKETS + 1;
        assert_eq!(back, (d.buckets.len() - start + LEAD_IN_BUCKETS) * BUCKET_FRAMES);
    }

    #[test]
    fn the_start_is_capped_at_the_ring() {
        let mut d = TalkDetector::default();
        // Talk for longer than the ring, with too little of the user to start a meeting…
        for _ in 0..RING_BUCKETS {
            bucket(&mut d, 0.0, 0.5);
        }
        // …until the user joins in.
        let back = (0..WINDOW_BUCKETS).find_map(|_| bucket(&mut d, 0.5, 0.5)).expect("no meeting started");
        assert_eq!(back, RING_BUCKETS * BUCKET_FRAMES);
    }

    /// Recordings through the real VAD and detector, as if heard by the mic:
    /// prints each bucket's speech share and where a meeting would start.
    /// `ECHO_STANDBY_WAVS=<a.wav>[,<b.wav>…] (16 kHz mono) \
    ///  cargo test standby_replay -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn standby_replay() {
        for path in std::env::var("ECHO_STANDBY_WAVS").expect("set ECHO_STANDBY_WAVS").split(',') {
            let mut reader = hound::WavReader::open(path).unwrap();
            assert_eq!(reader.spec().sample_rate, VAD_SAMPLE_RATE);
            let audio: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
            println!("\n{path} ({:.0}s)", audio.len() as f64 / VAD_SAMPLE_RATE as f64);
            let mut listener = Listener::new().unwrap();
            let mut detector = TalkDetector::default();
            let mut speech = 0;
            let mut shares = String::new();
            for (i, frame) in audio.chunks_exact(VAD_FRAME_SIZE).enumerate() {
                let heard = listener.is_speech(Source::Me, frame);
                speech += heard as usize;
                let detected = detector.push(Source::Me, heard);
                if (i + 1) % BUCKET_FRAMES == 0 {
                    shares.push_str(&format!("{:3.0}", 100.0 * speech as f64 / BUCKET_FRAMES as f64));
                    speech = 0;
                }
                if let Some(back) = detected {
                    let at = (i + 1) as f64 * VAD_FRAME_SIZE as f64 / VAD_SAMPLE_RATE as f64;
                    let from = at - (back * VAD_FRAME_SIZE) as f64 / VAD_SAMPLE_RATE as f64;
                    println!("speech % per 10 s: {shares}");
                    println!("=> meeting starts at {at:.0}s, recorded from {from:.0}s");
                    break;
                }
            }
            if detector.mic_frames * VAD_FRAME_SIZE >= audio.len() - VAD_FRAME_SIZE {
                println!("speech % per 10 s: {shares}");
                println!("=> no meeting");
            }
        }
    }

    #[test]
    fn a_sustained_voice_stands_out_but_clicks_do_not() {
        let mut p = Prominence::default();
        for _ in 0..NOISE_WINDOW_FRAMES {
            assert!(!p.push(-60.0));
        }
        // Typing: one loud frame at a time.
        let clicks = (0..300).filter(|i| p.push(if i % 3 == 0 { -30.0 } else { -60.0 })).count();
        assert_eq!(clicks, 0);
        // A word: a quarter of a second of voice.
        let voice = (0..8).filter(|_| p.push(-40.0)).count();
        assert_eq!(voice, 8 - (MIN_PROMINENT_RUN - 1));
    }

    /// A recording played through the speakers: it is the system audio, and
    /// the mic hears it too (`ECHO_STANDBY_ECHO_DB` quieter, default 20 dB,
    /// `ECHO_STANDBY_ECHO_MS` later, default 80 ms) over room noise. With
    /// `ECHO_STANDBY_MIC=<voice.wav>`, the user's own voice is added to the mic.
    /// `ECHO_STANDBY_WAVS=<a.wav>[,…] cargo test standby_playback -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn standby_playback() {
        let read = |path: &str| -> Vec<f32> {
            let mut reader = hound::WavReader::open(path).unwrap();
            assert_eq!(reader.spec().sample_rate, VAD_SAMPLE_RATE);
            reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
        };
        let env = |k: &str, d: f32| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
        let gain = 10f32.powf(-env("ECHO_STANDBY_ECHO_DB", 20.0) / 20.0);
        let delay = live::samples(env("ECHO_STANDBY_ECHO_MS", 80.0) as f64 / 1000.0);
        let voice = std::env::var("ECHO_STANDBY_MIC").ok().map(|p| read(&p));
        for path in std::env::var("ECHO_STANDBY_WAVS").expect("set ECHO_STANDBY_WAVS").split(',') {
            let sys = read(path);
            let mut x = 0x2545_f491u32;
            let mic: Vec<f32> = (0..sys.len())
                .map(|i| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    let noise = (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.002; // ~ -60 dBFS
                    let echo = if i >= delay { sys[i - delay] * gain } else { 0.0 };
                    noise + echo + voice.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0)
                })
                .collect();
            let mut listener = Listener::new().unwrap();
            let mut detector = TalkDetector::default();
            let (mut heard, mut detected) = ([0usize; 2], None);
            for (i, (m, o)) in mic.chunks_exact(VAD_FRAME_SIZE).zip(sys.chunks_exact(VAD_FRAME_SIZE)).enumerate() {
                for (source, frame) in [(Source::Others, o), (Source::Me, m)] {
                    let speech = listener.is_speech(source, frame);
                    heard[source.index()] += speech as usize;
                    if let Some(back) = detector.push(source, speech) {
                        detected.get_or_insert((i, back));
                    }
                }
            }
            let sec = |frames: usize| (frames * VAD_FRAME_SIZE) as f64 / VAD_SAMPLE_RATE as f64;
            println!(
                "{path} ({:.0}s): speech heard on system audio {:.0}s, on the mic {:.0}s → {}",
                sec(sys.len() / VAD_FRAME_SIZE),
                sec(heard[1]),
                sec(heard[0]),
                detected.map_or("no meeting".to_string(), |(i, back)| format!("meeting at {:.0}s, from {:.0}s", sec(i + 1), sec((i + 1).saturating_sub(back))))
            );
        }
    }

    /// How long a meeting standby starts takes to catch up: detect on a
    /// recording as standby would, then time the meeting's audio thread + ASR
    /// worker over the buffered audio with a real engine (loaded and warmed
    /// up first, as in the app).
    /// `ECHO_CATCHUP_MIC=<mic.wav> [ECHO_CATCHUP_SYS=<system.wav>] [ECHO_CATCHUP_MODEL=<id>] \
    ///  cargo test --release standby_catch_up -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn standby_catch_up() {
        use std::time::Instant;
        let read = |path: &str| -> Vec<f32> {
            let mut reader = hound::WavReader::open(path).unwrap();
            assert_eq!(reader.spec().sample_rate, VAD_SAMPLE_RATE);
            reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
        };
        let mic = read(&std::env::var("ECHO_CATCHUP_MIC").expect("set ECHO_CATCHUP_MIC"));
        let sys = std::env::var("ECHO_CATCHUP_SYS").ok().map(|p| read(&p));
        let model = std::env::var("ECHO_CATCHUP_MODEL").unwrap_or_else(|_| rust_asr::haas::HAAS_MODEL_ID.to_string());
        let sec = |frames: usize| (frames * VAD_FRAME_SIZE) as f64 / VAD_SAMPLE_RATE as f64;

        // Standby: listen until a meeting would start.
        let mut listener = Listener::new().unwrap();
        let mut detector = TalkDetector::default();
        let mut ring = Ring::default();
        let mut handoff = None;
        'frames: for (i, m) in mic.chunks_exact(VAD_FRAME_SIZE).enumerate() {
            let o = sys.as_ref().and_then(|s| s.get(i * VAD_FRAME_SIZE..(i + 1) * VAD_FRAME_SIZE));
            for (source, frame) in o.map(|o| (Source::Others, o)).into_iter().chain([(Source::Me, m)]) {
                let speech = listener.is_speech(source, frame);
                ring.push(source, frame.to_vec());
                if let Some(back) = detector.push(source, speech) {
                    ring.keep_last(back);
                    handoff = Some((i + 1, back));
                    break 'frames;
                }
            }
        }
        let (at, back) = handoff.expect("no meeting started");
        println!("meeting starts at {:.0}s with {:.0}s of buffered audio (from {:.0}s)", sec(at), sec(back), sec(at - back));

        let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Caches/io.qluto.echo/huggingface/hub");
        let mut engine = crate::transcription::ASREngine::with_hub(hub);
        engine.set_model(&model).unwrap();
        let t = Instant::now();
        engine.load_model().unwrap();
        engine.warmup_model().unwrap();
        engine.has_speech(&vec![0.0; VAD_SAMPLE_RATE as usize]);
        println!("{model}: loaded + warmed up in {:.1}s (before standby, in the app)", t.elapsed().as_secs_f64());

        // The meeting: buffered audio through the audio thread and ASR worker.
        let inputs = live::Inputs::from_frames(std::mem::take(&mut ring.frames));
        let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();
        let start = Instant::now();
        let audio = thread::spawn(move || live::audio_loop(inputs, chunk_tx, None, &NoObserver).unwrap());
        let engine = std::cell::RefCell::new(engine);
        let (mut chunks, mut vad_time, mut asr_time) = (0usize, 0.0f64, 0.0f64);
        let mut lines = Vec::new();
        live::asr_worker(
            chunk_rx,
            |samples| {
                chunks += 1;
                let t = Instant::now();
                let heard = engine.borrow_mut().has_speech(samples);
                vad_time += t.elapsed().as_secs_f64();
                Ok(heard)
            },
            |samples| {
                let t = Instant::now();
                let text = engine.borrow_mut().transcribe_samples(samples, Some("ja"), false)?.text;
                asr_time += t.elapsed().as_secs_f64();
                Ok(text)
            },
            |line| lines.push((start.elapsed().as_secs_f64(), line)),
        );
        audio.join().unwrap();
        let total = start.elapsed().as_secs_f64();
        println!(
            "caught up in {total:.1}s: {chunks} chunks (both sources), {} lines; ASR {asr_time:.1}s, VAD {vad_time:.2}s",
            lines.len()
        );
        if let Some((t, l)) = lines.first() {
            println!("first line after {t:.2}s: [{:.0}s] {}", l.start, l.text);
        }
        for (t, l) in &lines {
            println!("  {t:5.1}s  [{:5.1}-{:5.1}] {:?}: {}", l.start, l.end, l.source, l.text);
        }
    }

    struct NoObserver;
    impl live::AudioObserver for NoObserver {}

    #[test]
    fn the_ring_keeps_the_last_frames_in_arrival_order() {
        let mut ring = Ring::default();
        for i in 0..RING_FRAMES + 10 {
            ring.push(Source::Me, vec![i as f32]);
            ring.push(Source::Others, vec![-(i as f32)]);
        }
        assert_eq!(ring.mic_frames, RING_FRAMES);
        assert_eq!(ring.frames.front().unwrap(), &(Source::Others, vec![-9.0]));
        ring.keep_last(2);
        let kept: Vec<_> = ring.frames.iter().map(|(s, f)| (*s, f[0])).collect();
        let last = (RING_FRAMES + 9) as f32;
        // System audio that came in after a dropped mic frame stays: it is
        // from the same moment as the first kept one.
        assert_eq!(
            kept,
            vec![
                (Source::Others, -(last - 2.0)),
                (Source::Me, last - 1.0),
                (Source::Others, -(last - 1.0)),
                (Source::Me, last),
                (Source::Others, -last)
            ]
        );
    }
}
