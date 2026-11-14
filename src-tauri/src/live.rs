//! Live capture shared by the meeting companion and always-on logging:
//! microphone + system audio, cut into chunks at pauses, transcribed as they
//! come, with the other side's voice picked up by the mic dropped as echo.
//!
//! ```text
//! [mic StreamingCapture]    ─┐
//! [SystemAudioCapture tap]  ─┴→ [audio_loop] → Chunker per source → chunks
//!                                    └→ (optional) mixed 16 kHz WAV      ↓
//!                                                  [asr_worker] → lines (source, start, end, text)
//! ```
//!
//! Chunks are cut at pauses, not picked out by a VAD (it misses audible speech
//! on some microphones). But the mic sits on room noise for minutes at a time,
//! and the engines hallucinate on noise (Whisper: "ご視聴ありがとうございました"),
//! so a chunk only reaches ASR when it stands out from the room's noise floor
//! or the VAD hears speech in it: speech is lost only if both miss it.

use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::time::Duration;

use anyhow::{anyhow, Result};
use crossbeam_channel as channel;
use hound::WavWriter;
use serde::Serialize;

use crate::audio_capture::StreamingCapture;
use crate::memo::{self, frame_dbfs, meter_level, to_i16};
use crate::partial::{append_text, quietest_cut};
use crate::system_audio::SystemAudioCapture;
use crate::vad::{VAD_FRAME_SIZE, VAD_SAMPLE_RATE};

pub const fn samples(sec: f64) -> usize {
    (sec * VAD_SAMPLE_RATE as f64) as usize
}

/// A chunk is cut at the first pause once it is this long…
const MIN_CHUNK_SAMPLES: usize = samples(4.0);
/// …and at the quietest point once it reaches this length.
const MAX_CHUNK_SAMPLES: usize = samples(15.0);
/// A pause is this long stretch at least `PAUSE_DROP_DB` below the chunk's speech level.
const PAUSE_FRAMES: usize = samples(0.5) / VAD_FRAME_SIZE;
const PAUSE_DROP_DB: f32 = 18.0;
const CUT_PROBE_SAMPLES: usize = samples(0.2);
const MIN_FINAL_CHUNK_SAMPLES: usize = samples(0.3);
/// A chunk with no frame above this is not sent to ASR at all.
const SILENT_DBFS: f32 = -55.0;
/// The noise floor is the 10th percentile frame level over this window…
const NOISE_WINDOW_FRAMES: usize = samples(30.0) / VAD_FRAME_SIZE;
/// …and a chunk "stands out" when this many frames (~160 ms) are this far above it.
const PROMINENCE_DB: f32 = 15.0;
const MIN_PROMINENT_FRAMES: usize = 5;
/// A prominent chunk the VAD hears no speech in is still transcribed (Silero
/// misses audible speech on some microphones), but its line is kept only with
/// at least this many letters: missed speech gives a full sentence, while
/// keyboard clicks and other noise come back as a word or two ("キュウリ").
const MIN_UNHEARD_LETTERS: usize = 6;
/// The other side counts as talking during a mic chunk when this share of
/// its frames is louder than `OTHERS_ACTIVE_DBFS`.
const OTHERS_ACTIVE_DBFS: f32 = -45.0;
const OTHERS_ACTIVE_SHARE: f64 = 0.3;
/// How much of the other side's loudness history is kept for that check
/// (mic chunks are at most `MAX_CHUNK_SAMPLES` long and arrive in step).
const OTHERS_HISTORY_FRAMES: usize = samples(120.0) / VAD_FRAME_SIZE;
/// A mic line counts as an echo of the speakers when this share of its
/// character bigrams appears in what the other side said around it.
const ECHO_CONTAINMENT: f64 = 0.6;
const ECHO_MARGIN_SEC: f64 = 3.0;
/// The other side's lines are kept this long (behind its transcript's end)
/// for checking held mic lines against.
const OTHERS_LINES_KEEP_SEC: f64 = 120.0;

const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// The microphone: the user.
    Me,
    /// System audio: whatever the Mac plays (the other side of a call, a video…).
    Others,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Self::Me => "自分",
            Self::Others => "相手",
        }
    }

    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// The running captures. Stopping (or dropping) them closes their frame
/// channels, which lets `audio_loop` flush and the rest of the pipeline drain.
pub struct Captures {
    mic: Option<StreamingCapture>,
    system: Option<SystemAudioCapture>,
}

impl Captures {
    pub fn has_system(&self) -> bool {
        self.system.is_some()
    }

    pub fn stop(&mut self) {
        if let Some(mut mic) = self.mic.take() {
            mic.stop();
        }
        self.system.take();
    }
}

impl Drop for Captures {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The frame streams of `Captures`, consumed by `audio_loop` (or by meeting
/// standby, which hands them on to a meeting).
pub struct Inputs {
    /// Frames already received, delivered before the live streams.
    backlog: VecDeque<(Source, Vec<f32>)>,
    mic_rx: channel::Receiver<Vec<f32>>,
    sys_rx: channel::Receiver<Vec<f32>>,
    has_system: bool,
    mic_open: bool,
    sys_open: bool,
    /// Frames delivered per source.
    delivered: [usize; 2],
}

impl Inputs {
    fn new(
        first: Vec<f32>,
        mic_rx: channel::Receiver<Vec<f32>>,
        sys_rx: channel::Receiver<Vec<f32>>,
        has_system: bool,
    ) -> Self {
        Self {
            backlog: VecDeque::from([(Source::Me, first)]),
            mic_rx,
            sys_rx,
            has_system,
            mic_open: true,
            sys_open: has_system,
            delivered: [0, 0],
        }
    }

    /// Recorded streams fed as if captured; `first` is the mic's first frame.
    #[cfg(test)]
    pub fn for_test(
        first: Vec<f32>,
        mic_rx: channel::Receiver<Vec<f32>>,
        sys_rx: channel::Receiver<Vec<f32>>,
    ) -> Self {
        Self::new(first, mic_rx, sys_rx, true)
    }

    /// Only these frames (in arrival order), as if the captures then stopped.
    #[cfg(test)]
    pub fn from_frames(frames: impl IntoIterator<Item = (Source, Vec<f32>)>) -> Self {
        Self {
            backlog: frames.into_iter().collect(),
            mic_rx: channel::never(),
            sys_rx: channel::never(),
            has_system: true,
            mic_open: false,
            sys_open: false,
            delivered: [0, 0],
        }
    }

    /// The next frame, in arrival order; None once both captures have stopped.
    pub fn next_frame(&mut self) -> Option<(Source, Vec<f32>)> {
        let next = self.receive();
        if let Some((source, _)) = &next {
            self.delivered[source.index()] += 1;
        }
        next
    }

    fn receive(&mut self) -> Option<(Source, Vec<f32>)> {
        if let Some(frame) = self.backlog.pop_front() {
            return Some(frame);
        }
        // When frames have piled up in both channels (while a meeting starts),
        // take them from the source that is behind: `select!` picks at random,
        // and a source a second ahead looks to the mixer like the other stalled.
        let behind = if self.delivered[0] <= self.delivered[1] { [Source::Me, Source::Others] } else { [Source::Others, Source::Me] };
        for source in behind {
            let (open, rx) = match source {
                Source::Me => (self.mic_open, &self.mic_rx),
                Source::Others => (self.sys_open, &self.sys_rx),
            };
            if let Some(frame) = open.then(|| rx.try_recv().ok()).flatten() {
                return Some((source, frame));
            }
        }
        let never = channel::never::<Vec<f32>>();
        while self.mic_open || self.sys_open {
            channel::select! {
                recv(if self.mic_open { &self.mic_rx } else { &never }) -> msg => match msg {
                    Ok(frame) => return Some((Source::Me, frame)),
                    Err(_) => self.mic_open = false,
                },
                recv(if self.sys_open { &self.sys_rx } else { &never }) -> msg => match msg {
                    Ok(frame) => return Some((Source::Others, frame)),
                    Err(_) => self.sys_open = false,
                },
            }
        }
        None
    }

    /// Deliver `frames` (in arrival order) before anything not yet delivered.
    pub fn prepend(&mut self, frames: impl IntoIterator<Item = (Source, Vec<f32>)>) {
        let rest = std::mem::take(&mut self.backlog);
        self.backlog = frames.into_iter().chain(rest).collect();
    }
}

/// Start the microphone and the system audio tap. Fails if the microphone
/// delivers no audio; system audio is optional: when it can't start, the
/// third value says why and only the mic is captured.
pub fn start_captures(device_name: Option<String>) -> Result<(Captures, Inputs, Option<String>)> {
    let (mic_tx, mic_rx) = channel::bounded::<Vec<f32>>(256);
    let mut mic = StreamingCapture::start(device_name, mic_tx)?;
    let first = match mic_rx.recv_timeout(FIRST_FRAME_TIMEOUT) {
        Ok(frame) => frame,
        Err(_) => {
            mic.stop();
            return Err(anyhow!("No audio from the input device (check the microphone permission)"));
        }
    };

    let (sys_tx, sys_rx) = channel::bounded::<Vec<f32>>(256);
    let (system, sys_rx, system_error) = match SystemAudioCapture::start(sys_tx) {
        Ok(capture) => (Some(capture), sys_rx, None),
        Err(e) => {
            log::warn!("System audio capture unavailable: {e:#}");
            (None, channel::never(), Some(format!("{e:#}")))
        }
    };
    let has_system = system.is_some();
    Ok((Captures { mic: Some(mic), system }, Inputs::new(first, mic_rx, sys_rx, has_system), system_error))
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

pub struct Chunk {
    source: Source,
    /// Offset of the first sample within this source's stream.
    start: usize,
    samples: Vec<f32>,
    /// No frame is loud enough to hold speech.
    silent: bool,
    /// Clearly louder than the noise floor; otherwise ASR runs only if the VAD hears speech.
    prominent: bool,
    /// (Mic chunks) the other side was audible during it, so it may contain
    /// their voice coming out of the speakers.
    others_active: bool,
}

impl Chunk {
    fn start_sec(&self) -> f64 {
        self.start as f64 / VAD_SAMPLE_RATE as f64
    }

    fn end_sec(&self) -> f64 {
        (self.start + self.samples.len()) as f64 / VAD_SAMPLE_RATE as f64
    }
}

/// Cuts one source's frame stream into contiguous chunks at pauses.
struct Chunker {
    source: Source,
    start: usize,
    buf: Vec<f32>,
    /// dBFS of each `VAD_FRAME_SIZE` frame in `buf`.
    frame_db: Vec<f32>,
    /// dBFS of the last `NOISE_WINDOW_FRAMES` frames, across chunks.
    recent_db: VecDeque<f32>,
}

impl Chunker {
    fn new(source: Source) -> Self {
        Self {
            source,
            start: 0,
            buf: Vec::with_capacity(MAX_CHUNK_SAMPLES),
            frame_db: Vec::new(),
            recent_db: VecDeque::with_capacity(NOISE_WINDOW_FRAMES + 1),
        }
    }

    /// Add one frame (`VAD_FRAME_SIZE` samples); returns a chunk when one is cut.
    fn push(&mut self, frame: &[f32]) -> Option<Chunk> {
        self.buf.extend_from_slice(frame);
        let db = frame_dbfs(frame);
        self.frame_db.push(db);
        self.recent_db.push_back(db);
        if self.recent_db.len() > NOISE_WINDOW_FRAMES {
            self.recent_db.pop_front();
        }
        if self.buf.len() >= MAX_CHUNK_SAMPLES {
            let cut = quietest_cut(&self.buf, MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES, CUT_PROBE_SAMPLES);
            return Some(self.cut(cut));
        }
        if self.buf.len() >= MIN_CHUNK_SAMPLES && self.ends_in_pause() {
            // Keep half the pause as the next chunk's lead-in.
            let cut = self.buf.len() - PAUSE_FRAMES * VAD_FRAME_SIZE / 2;
            return Some(self.cut(cut));
        }
        None
    }

    /// The rest of the stream, if it is long enough to hold a word.
    fn flush(&mut self) -> Option<Chunk> {
        if self.buf.len() < MIN_FINAL_CHUNK_SAMPLES {
            return None;
        }
        Some(self.cut(self.buf.len()))
    }

    /// Everything buffered so far, however short.
    fn take_all(&mut self) -> Option<Chunk> {
        (!self.buf.is_empty()).then(|| self.cut(self.buf.len()))
    }

    fn ends_in_pause(&self) -> bool {
        if self.frame_db.len() <= PAUSE_FRAMES {
            return false;
        }
        let speech = percentile(&self.frame_db, 0.9);
        if speech < SILENT_DBFS {
            return false;
        }
        let threshold = speech - PAUSE_DROP_DB;
        self.frame_db[self.frame_db.len() - PAUSE_FRAMES..]
            .iter()
            .all(|&db| db < threshold)
    }

    fn cut(&mut self, at: usize) -> Chunk {
        let rest = self.buf.split_off(at);
        let samples = std::mem::replace(&mut self.buf, rest);
        let frames = at.div_ceil(VAD_FRAME_SIZE).min(self.frame_db.len());
        let chunk_db = &self.frame_db[..frames];
        let silent = chunk_db.iter().all(|&db| db < SILENT_DBFS);
        let floor = percentile(self.recent_db.make_contiguous(), 0.1);
        let prominent = chunk_db.iter().filter(|&&db| db >= floor + PROMINENCE_DB).count() >= MIN_PROMINENT_FRAMES;
        // Frames straddling the cut are counted on both sides; harmless for a level.
        self.frame_db.drain(..at / VAD_FRAME_SIZE);
        let chunk = Chunk { source: self.source, start: self.start, samples, silent, prominent, others_active: false };
        self.start += at;
        chunk
    }
}

fn percentile(values: &[f32], p: f64) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Per-frame "the other side is audible", for the recent past.
#[derive(Default)]
struct OthersLoudness {
    /// Frame index of `loud[0]` within the system audio stream.
    base: usize,
    loud: VecDeque<bool>,
}

impl OthersLoudness {
    fn push(&mut self, loud: bool) {
        self.loud.push_back(loud);
        if self.loud.len() > OTHERS_HISTORY_FRAMES {
            self.loud.pop_front();
            self.base += 1;
        }
    }

    /// Whether the other side was talking during frames `from..to`.
    fn active(&self, from: usize, to: usize) -> bool {
        let from = from.max(self.base);
        let to = to.min(self.base + self.loud.len());
        if from >= to {
            return false;
        }
        let loud = self.loud.range(from - self.base..to - self.base).filter(|&&l| l).count();
        loud as f64 / (to - from) as f64 >= OTHERS_ACTIVE_SHARE
    }
}

// ---------------------------------------------------------------------------
// Audio thread: levels, chunking, optional mixed recording
// ---------------------------------------------------------------------------

/// What the audio thread reports to, and asks of, its owner.
pub trait AudioObserver {
    /// Every captured frame, before chunking: its meter level and dBFS.
    fn on_frame(&self, _source: Source, _level: f32, _dbfs: f32) {}

    /// While a source is muted its audio is not transcribed (it is still
    /// recorded). The chunk in progress is cut where muting starts.
    fn muted(&self, _source: Source) -> bool {
        false
    }
}

/// Chunk both sources until their captures stop, sending each chunk to the
/// ASR worker. With a `writer`, also record both sources mixed into it.
/// Never waits on ASR (`chunk_tx` should be unbounded).
pub fn audio_loop(
    mut inputs: Inputs,
    chunk_tx: channel::Sender<Chunk>,
    mut writer: Option<WavWriter<BufWriter<File>>>,
    observer: &impl AudioObserver,
) -> Result<()> {
    let has_system = inputs.has_system;
    let mut chunkers = [Chunker::new(Source::Me), Chunker::new(Source::Others)];
    let mut muted = [false, false];
    let mut others = OthersLoudness::default();
    // Samples not yet mixed into the recording, per source.
    let mut pending: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut since_flush = 0usize;

    let send = |mut chunk: Chunk, others: &OthersLoudness| {
        if chunk.source == Source::Me && !chunk.silent {
            let from = chunk.start / VAD_FRAME_SIZE;
            let to = (chunk.start + chunk.samples.len()) / VAD_FRAME_SIZE;
            chunk.others_active = others.active(from, to);
        }
        let _ = chunk_tx.send(chunk);
    };

    while let Some((source, frame)) = inputs.next_frame() {
        let i = source.index();
        let dbfs = frame_dbfs(&frame);
        observer.on_frame(source, meter_level(&frame), dbfs);
        if source == Source::Others {
            others.push(dbfs > OTHERS_ACTIVE_DBFS);
        }

        let now_muted = observer.muted(source);
        if now_muted != muted[i] {
            muted[i] = now_muted;
            if now_muted {
                // What was said up to now still counts.
                if let Some(chunk) = chunkers[i].flush() {
                    send(chunk, &others);
                }
            } else {
                // Discard what was heard while muted.
                chunkers[i].take_all();
            }
        }
        if let Some(chunk) = chunkers[i].push(&frame) {
            if !muted[i] {
                send(chunk, &others);
            }
        }

        let Some(writer) = writer.as_mut() else { continue };
        pending[i].extend_from_slice(&frame);
        // Mix what both sources have delivered. A source that falls more than
        // a second behind (stalled tap, no system audio) is padded with silence.
        let [mic_pending, sys_pending] = &mut pending;
        let lag_limit = samples(1.0);
        if !has_system || sys_pending.len() + lag_limit < mic_pending.len() {
            sys_pending.resize(mic_pending.len(), 0.0);
        } else if mic_pending.len() + lag_limit < sys_pending.len() {
            mic_pending.resize(sys_pending.len(), 0.0);
        }
        let n = mic_pending.len().min(sys_pending.len());
        if n > 0 {
            for (m, s) in mic_pending.drain(..n).zip(sys_pending.drain(..n)) {
                writer.write_sample(to_i16(m + s))?;
            }
            since_flush += n;
            if since_flush >= memo::FLUSH_INTERVAL_SAMPLES {
                writer.flush()?;
                since_flush = 0;
            }
        }
    }

    for (chunker, muted) in chunkers.iter_mut().zip(muted) {
        if let Some(chunk) = chunker.flush().filter(|_| !muted) {
            send(chunk, &others);
        }
    }
    if let Some(mut writer) = writer {
        let [mic_pending, sys_pending] = pending;
        let n = mic_pending.len().max(sys_pending.len());
        let pad = |mut v: Vec<f32>| {
            v.resize(n, 0.0);
            v
        };
        for (m, s) in pad(mic_pending).into_iter().zip(pad(sys_pending)) {
            writer.write_sample(to_i16(m + s))?;
        }
        writer.finalize()?;
    }
    log::info!("Live audio thread exiting");
    Ok(())
}

// ---------------------------------------------------------------------------
// ASR worker
// ---------------------------------------------------------------------------

/// One transcribed chunk.
#[derive(Debug, Clone)]
pub struct Line {
    pub source: Source,
    /// Seconds since capture started.
    pub start: f64,
    pub end: f64,
    pub text: String,
}

fn bigrams(text: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Whether `mine` (a mic line) is mostly the other side's speech picked up
/// from the speakers: most of its character bigrams occur in `theirs`.
fn is_echo(mine: &str, theirs: &str) -> bool {
    let mine = bigrams(mine);
    if mine.is_empty() {
        return false;
    }
    let theirs: std::collections::HashSet<_> = bigrams(theirs).into_iter().collect();
    let shared = mine.iter().filter(|b| theirs.contains(b)).count();
    shared as f64 / mine.len() as f64 >= ECHO_CONTAINMENT
}

/// The text of a non-silent chunk, or "" when it is not speech.
/// `has_speech` runs the VAD over 16 kHz mono samples and `transcribe` the ASR.
///
/// The VAD is a second opinion, not a gate: a chunk it hears speech in is
/// always kept. One it hears none in is skipped if it doesn't stand out from
/// the noise floor, and otherwise transcribed but kept only with at least
/// `MIN_UNHEARD_LETTERS` letters.
fn recognize(
    chunk: &Chunk,
    has_speech: &mut impl FnMut(&[f32]) -> Result<bool>,
    transcribe: &mut impl FnMut(&[f32]) -> Result<String>,
) -> Result<String> {
    let speech = has_speech(&chunk.samples).unwrap_or_else(|e| {
        log::warn!("VAD failed ({e:#}); treating the chunk as speech");
        true
    });
    if !speech && !chunk.prominent {
        return Ok(String::new());
    }
    let text = transcribe(&chunk.samples)?.trim().to_string();
    let letters = text.chars().filter(|c| c.is_alphanumeric()).count();
    if !speech && letters > 0 && letters < MIN_UNHEARD_LETTERS {
        log::info!("Dropped as noise (VAD heard no speech): {text:?}");
        return Ok(String::new());
    }
    Ok(text)
}

/// Transcribe chunks as they come until the audio thread is done (see
/// `recognize` for `has_speech` and `transcribe`), handing each non-empty
/// line to `on_line`. A mic line said while the other side was audible is
/// held until the other side's transcript covers its time, and dropped if it
/// is an echo of it.
pub fn asr_worker(
    chunks: impl IntoIterator<Item = Chunk>,
    mut has_speech: impl FnMut(&[f32]) -> Result<bool>,
    mut transcribe: impl FnMut(&[f32]) -> Result<String>,
    mut on_line: impl FnMut(Line),
) {
    let mut held: Vec<Line> = Vec::new();
    // The other side's recent lines, and how far its stream is transcribed (seconds).
    let mut theirs: VecDeque<Line> = VecDeque::new();
    let mut others_done = 0.0f64;

    // Publish held lines the other side's transcript now covers, dropping echoes.
    let release = |held: &mut Vec<Line>, theirs: &VecDeque<Line>, upto: f64, on_line: &mut dyn FnMut(Line)| {
        let ready: Vec<Line>;
        (ready, *held) = std::mem::take(held).into_iter().partition(|h| h.end <= upto);
        for h in ready {
            let mut around = String::new();
            for l in theirs
                .iter()
                .filter(|l| l.end > h.start - ECHO_MARGIN_SEC && l.start < h.end + ECHO_MARGIN_SEC)
            {
                append_text(&mut around, &l.text);
            }
            if is_echo(&h.text, &around) {
                log::info!("Dropped mic line as speaker echo: {}", h.text);
            } else {
                on_line(h);
            }
        }
    };

    for chunk in chunks {
        let (start, end) = (chunk.start_sec(), chunk.end_sec());
        let text = if chunk.silent {
            String::new()
        } else {
            match recognize(&chunk, &mut has_speech, &mut transcribe) {
                Ok(text) => text,
                Err(e) => {
                    log::error!("Chunk transcription failed: {e:#}");
                    String::new()
                }
            }
        };
        let line = Line { source: chunk.source, start, end, text };

        match chunk.source {
            Source::Others => {
                if !line.text.is_empty() {
                    theirs.push_back(line.clone());
                    on_line(line);
                }
                others_done = others_done.max(end);
                release(&mut held, &theirs, others_done, &mut on_line);
                while theirs.front().is_some_and(|l| l.end < others_done - OTHERS_LINES_KEEP_SEC) {
                    theirs.pop_front();
                }
            }
            Source::Me if line.text.is_empty() => {}
            Source::Me if chunk.others_active => held.push(line),
            Source::Me => on_line(line),
        }
    }
    // Everything is transcribed: check what is still held against what we have.
    release(&mut held, &theirs, f64::INFINITY, &mut on_line);
    log::info!("Live ASR worker exiting");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn tone(seconds: f64, amplitude: f32) -> Vec<f32> {
        (0..samples(seconds))
            .map(|i| amplitude * (i as f32 / VAD_SAMPLE_RATE as f32 * 220.0 * std::f32::consts::TAU).sin())
            .collect()
    }

    /// Feed `audio` frame by frame; return every chunk's `(start, end, silent, prominent)`.
    fn chunks_of(audio: &[f32]) -> Vec<(f64, f64, bool, bool)> {
        let mut chunker = Chunker::new(Source::Me);
        let mut out = Vec::new();
        let mut take = |c: Chunk| out.push((c.start_sec(), c.end_sec(), c.silent, c.prominent));
        for frame in audio.chunks(VAD_FRAME_SIZE) {
            if let Some(c) = chunker.push(frame) {
                take(c);
            }
        }
        if let Some(c) = chunker.flush() {
            take(c);
        }
        out
    }

    fn chunk_all(audio: &[f32]) -> Vec<(f64, f64, bool)> {
        chunks_of(audio).into_iter().map(|(s, e, silent, _)| (s, e, silent)).collect()
    }

    /// Steady noise at about `db` dBFS RMS.
    fn noise(seconds: f64, db: f32) -> Vec<f32> {
        let amp = 10f32.powf(db / 20.0) * 3f32.sqrt();
        let mut x = 0x2545_f491u32;
        (0..samples(seconds))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    #[test]
    fn room_noise_does_not_stand_out_but_speech_over_it_does() {
        let mut audio = noise(40.0, -50.0);
        let quiet = chunks_of(&audio);
        assert!(quiet.iter().all(|c| !c.2 && !c.3), "{quiet:?}");
        // A "voice" 20 dB above the noise, after 30 s of room tone.
        for (s, t) in audio[samples(31.0)..samples(33.0)].iter_mut().zip(tone(2.0, 0.03)) {
            *s += t;
        }
        let chunks = chunks_of(&audio);
        let with_voice = chunks.iter().find(|c| c.0 <= 31.5 && c.1 >= 32.5).unwrap();
        assert!(with_voice.3, "{chunks:?}");
        assert_eq!(chunks.iter().filter(|c| c.3).count(), 1, "{chunks:?}");
    }

    #[test]
    fn chunks_are_cut_at_a_pause_after_the_minimum_length() {
        let mut audio = tone(5.0, 0.3);
        audio.extend(vec![0.0; samples(1.0)]);
        audio.extend(tone(3.0, 0.3));
        let chunks = chunk_all(&audio);
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        // Cut inside the pause, which starts at 5.0 s.
        assert!((5.2..5.6).contains(&chunks[0].1), "{chunks:?}");
        assert_eq!(chunks[0].1, chunks[1].0);
        assert!((chunks[1].1 - 9.0).abs() < 0.05);
    }

    #[test]
    fn a_short_pause_before_the_minimum_length_does_not_cut() {
        let mut audio = tone(2.0, 0.3);
        audio.extend(vec![0.0; samples(0.8)]);
        audio.extend(tone(2.0, 0.3));
        assert_eq!(chunk_all(&audio).len(), 1);
    }

    #[test]
    fn continuous_speech_is_cut_at_the_maximum_length() {
        let chunks = chunk_all(&tone(32.0, 0.3));
        assert!(chunks.iter().all(|(s, e, _)| e - s <= 15.0 + 1e-9), "{chunks:?}");
        assert_eq!(chunks.first().unwrap().0, 0.0);
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
        assert!((chunks.last().unwrap().1 - 32.0).abs() < 0.05);
    }

    #[test]
    fn silence_is_chunked_but_marked_silent() {
        let chunks = chunk_all(&vec![0.0; samples(20.0)]);
        assert!(chunks.iter().all(|c| c.2), "{chunks:?}");
        // Quiet speech is not "silent".
        let quiet = chunk_all(&tone(3.0, 0.005));
        assert!(!quiet[0].2, "{quiet:?}");
    }

    #[test]
    fn echo_is_detected_by_shared_bigrams() {
        let theirs = "来週の月曜までに見積もりを出してもらえますか。はい、お願いします";
        assert!(is_echo("来週の月曜までに見積もりを出して", theirs));
        assert!(!is_echo("わかりました、私が担当します", theirs));
        assert!(!is_echo("", theirs));
        assert!(is_echo("Send the Quote by Monday", "please send the quote by monday, thanks"));
    }

    #[test]
    fn others_loudness_keeps_frame_indices_after_pruning() {
        let mut others = OthersLoudness::default();
        for i in 0..OTHERS_HISTORY_FRAMES + 100 {
            others.push(i >= OTHERS_HISTORY_FRAMES + 50);
        }
        let end = OTHERS_HISTORY_FRAMES + 100;
        assert!(others.active(end - 50, end));
        assert!(!others.active(end - 150, end - 100));
        // Pruned frames are unknown, not "active".
        assert!(!others.active(0, 10));
    }

    /// Mutes the mic while the frame counter is in `range`.
    struct MuteMic {
        frames: Cell<usize>,
        range: std::ops::Range<usize>,
    }

    impl AudioObserver for MuteMic {
        fn on_frame(&self, source: Source, _level: f32, _dbfs: f32) {
            if source == Source::Me {
                self.frames.set(self.frames.get() + 1);
            }
        }

        fn muted(&self, source: Source) -> bool {
            source == Source::Me && self.range.contains(&(self.frames.get() - 1))
        }
    }

    /// Run `audio_loop` over a mic-only stream; return the `(start, end)` of the chunks sent.
    fn mic_chunks(audio: &[f32], observer: &impl AudioObserver) -> Vec<(f64, f64)> {
        let mut frames = audio.chunks(VAD_FRAME_SIZE).map(<[f32]>::to_vec);
        let (mic_tx, mic_rx) = channel::unbounded();
        let first = frames.next().unwrap();
        for f in frames {
            mic_tx.send(f).unwrap();
        }
        drop(mic_tx);
        let inputs = Inputs::new(first, mic_rx, channel::never(), false);
        let (chunk_tx, chunk_rx) = channel::unbounded();
        audio_loop(inputs, chunk_tx, None, observer).unwrap();
        chunk_rx.iter().map(|c| (c.start_sec(), c.end_sec())).collect()
    }

    #[test]
    fn audio_heard_while_muted_is_not_transcribed() {
        // Speech throughout; the mic is muted from 6 s to 9 s (hotkey dictation).
        let audio = tone(20.0, 0.3);
        let frame_at = |sec: f64| samples(sec) / VAD_FRAME_SIZE;
        let observer = MuteMic { frames: Cell::new(0), range: frame_at(6.0)..frame_at(9.0) };
        let chunks = mic_chunks(&audio, &observer);
        assert!(!chunks.is_empty());
        for &(s, e) in &chunks {
            assert!(e <= 6.0 + 0.05 || s >= 9.0 - 0.05, "chunk {s}..{e} overlaps the muted span: {chunks:?}");
        }
        // Speech before and after muting is kept, up to the cut.
        assert!(chunks.iter().any(|&(_, e)| (e - 6.0).abs() < 0.05), "{chunks:?}");
        assert!(chunks.iter().any(|&(s, _)| (s - 9.0).abs() < 0.05), "{chunks:?}");
        assert!((chunks.last().unwrap().1 - 20.0).abs() < 0.05);
    }

    #[test]
    fn buffered_audio_prepended_to_live_audio_is_one_recording() {
        // Standby buffered 20 s of both sources (in arrival order), then the
        // live captures go on for 10 s more.
        let mic = tone(30.0, 0.3);
        let sys = tone(30.0, 0.1);
        let frames = |a: &[f32]| a.chunks(VAD_FRAME_SIZE).map(<[f32]>::to_vec).collect::<Vec<_>>();
        let (mic_frames, sys_frames) = (frames(&mic), frames(&sys));
        let split = samples(20.0) / VAD_FRAME_SIZE;
        let (mic_tx, mic_rx) = channel::unbounded();
        let (sys_tx, sys_rx) = channel::unbounded();
        let mut inputs = Inputs::new(mic_frames[split].clone(), mic_rx, sys_rx, true);
        let backlog = (0..split).flat_map(|i| [(Source::Me, mic_frames[i].clone()), (Source::Others, sys_frames[i].clone())]);
        inputs.prepend(backlog);
        for i in split..mic_frames.len() {
            if i > split {
                mic_tx.send(mic_frames[i].clone()).unwrap();
            }
            sys_tx.send(sys_frames[i].clone()).unwrap();
        }
        drop((mic_tx, sys_tx));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.wav");
        let spec = hound::WavSpec { channels: 1, sample_rate: VAD_SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let writer = WavWriter::create(&path, spec).unwrap();
        let (chunk_tx, chunk_rx) = channel::unbounded();
        audio_loop(inputs, chunk_tx, Some(writer), &MuteMic { frames: Cell::new(0), range: 0..0 }).unwrap();

        let mixed: Vec<i16> = hound::WavReader::open(&path).unwrap().samples().map(Result::unwrap).collect();
        assert_eq!(mixed.len(), mic.len());
        // Both sources everywhere: no stretch of either was padded with silence.
        let expected = |i: usize| to_i16(mic[i] + sys[i]);
        assert!((0..mixed.len()).step_by(997).all(|i| mixed[i] == expected(i)));
        // Each source is chunked contiguously from 0 s to 30 s.
        let chunks: Vec<Chunk> = chunk_rx.iter().collect();
        for source in [Source::Me, Source::Others] {
            let spans: Vec<_> = chunks.iter().filter(|c| c.source == source).map(|c| (c.start_sec(), c.end_sec())).collect();
            assert_eq!(spans[0].0, 0.0, "{source:?}");
            assert!(spans.windows(2).all(|w| w[0].1 == w[1].0), "{source:?}: {spans:?}");
            assert!((spans.last().unwrap().1 - 30.0).abs() < 0.05, "{source:?}: {spans:?}");
        }
    }

    /// The audio thread + ASR worker with a real engine on a two-track recording:
    /// `ECHO_LIVE_DIR=<dir with me.wav, others.wav (16 kHz mono)> [ECHO_LIVE_MIC=me_echo.wav]
    ///  [ECHO_LIVE_MODEL=<asr model>] [ECHO_LIVE_MUTE=<from>-<to> (mic, seconds)] \
    ///  cargo test live_end_to_end -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn live_end_to_end() {
        let read_wav = |path: std::path::PathBuf| -> Vec<f32> {
            let mut reader = hound::WavReader::open(&path).unwrap();
            assert_eq!(reader.spec().sample_rate, VAD_SAMPLE_RATE);
            reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
        };
        let dir = std::path::PathBuf::from(std::env::var("ECHO_LIVE_DIR").expect("set ECHO_LIVE_DIR"));
        let mic = read_wav(dir.join(std::env::var("ECHO_LIVE_MIC").unwrap_or_else(|_| "me.wav".into())));
        let others = read_wav(dir.join("others.wav"));
        let model = std::env::var("ECHO_LIVE_MODEL")
            .unwrap_or_else(|_| "mlx-community/parakeet-tdt_ctc-0.6b-ja".to_string());
        let mute = std::env::var("ECHO_LIVE_MUTE").ok().map(|r| {
            let (a, b) = r.split_once('-').unwrap();
            let frame = |s: &str| samples(s.parse().unwrap()) / VAD_FRAME_SIZE;
            frame(a)..frame(b)
        });

        let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Caches/io.qluto.echo/huggingface/hub");
        let mut engine = crate::transcription::ASREngine::with_hub(hub);
        engine.set_model(&model).unwrap();

        let first = mic[..VAD_FRAME_SIZE].to_vec();
        let (mic_tx, mic_rx) = channel::bounded(2);
        let (sys_tx, sys_rx) = channel::bounded(2);
        let feeder = std::thread::spawn(move || {
            let mic_frames = mic[VAD_FRAME_SIZE..].chunks_exact(VAD_FRAME_SIZE);
            for (m, o) in mic_frames.zip(others.chunks_exact(VAD_FRAME_SIZE)) {
                mic_tx.send(m.to_vec()).unwrap();
                sys_tx.send(o.to_vec()).unwrap();
            }
        });
        let (chunk_tx, chunk_rx) = channel::unbounded();
        let audio = std::thread::spawn(move || {
            let observer = MuteMic { frames: Cell::new(0), range: mute.unwrap_or(0..0) };
            audio_loop(Inputs::for_test(first, mic_rx, sys_rx), chunk_tx, None, &observer).unwrap()
        });
        let engine = std::cell::RefCell::new(engine);
        let mut lines = Vec::new();
        asr_worker(
            chunk_rx,
            |samples| {
                let heard = engine.borrow_mut().has_speech(samples);
                println!("chunk {:4.1}s: VAD {}", samples.len() as f64 / VAD_SAMPLE_RATE as f64, if heard { "speech" } else { "no speech" });
                Ok(heard)
            },
            |samples| {
                let text = engine.borrow_mut().transcribe_samples(samples, Some("ja"), false)?.text;
                println!("  ASR: {text:?}");
                Ok(text)
            },
            |line| lines.push(line),
        );
        feeder.join().unwrap();
        audio.join().unwrap();
        lines.sort_by(|a, b| a.start.total_cmp(&b.start));
        println!();
        for l in &lines {
            println!("[{:5.1}-{:5.1}] {:?}: {}", l.start, l.end, l.source, l.text);
        }
        assert!(!lines.is_empty());
    }

    fn chunk(source: Source, start_sec: f64, end_sec: f64, others_active: bool) -> Chunk {
        let start = samples(start_sec);
        Chunk {
            source,
            start,
            samples: vec![0.1; samples(end_sec) - start],
            silent: false,
            prominent: true,
            others_active,
        }
    }

    #[test]
    fn mic_lines_echoing_the_other_side_are_dropped() {
        let (tx, rx) = channel::unbounded();
        // Mic chunks arrive before the system audio covering the same time.
        tx.send(chunk(Source::Me, 0.0, 4.0, true)).unwrap(); // echo
        tx.send(chunk(Source::Me, 4.0, 8.0, true)).unwrap(); // the user, over the other side
        tx.send(chunk(Source::Me, 8.0, 12.0, false)).unwrap(); // the user alone
        tx.send(chunk(Source::Others, 0.0, 8.0, false)).unwrap();
        drop(tx);
        let mut texts = vec![
            "来週の月曜までに見積もりを出して",
            "わかりました、私が担当します",
            "では次の議題に移りましょう",
            "来週の月曜までに見積もりを出してもらえますか",
        ]
        .into_iter();
        let mut lines = Vec::new();
        asr_worker(rx, |_| Ok(true), |_| Ok(texts.next().unwrap().to_string()), |l| lines.push((l.source, l.text)));
        assert_eq!(
            lines,
            vec![
                (Source::Me, "では次の議題に移りましょう".to_string()),
                (Source::Others, "来週の月曜までに見積もりを出してもらえますか".to_string()),
                (Source::Me, "わかりました、私が担当します".to_string()),
            ]
        );
    }

    #[test]
    fn short_lines_the_vad_heard_no_speech_in_are_dropped() {
        let speech = |text: &str, prominent: bool, heard: bool| {
            let mut c = chunk(Source::Me, 0.0, 4.0, false);
            c.prominent = prominent;
            let mut called = false;
            let text = recognize(
                &c,
                &mut |_| Ok(heard),
                &mut |_| {
                    called = true;
                    Ok(text.to_string())
                },
            )
            .unwrap();
            (text, called)
        };
        // Heard: kept whatever its length.
        assert_eq!(speech("はい", true, true), ("はい".to_string(), true));
        assert_eq!(speech("はい", false, true), ("はい".to_string(), true));
        // Not heard and not prominent: not even transcribed.
        assert_eq!(speech("はい", false, false), (String::new(), false));
        // Not heard but prominent: a word or two is noise…
        assert_eq!(speech("キュウリ", true, false), (String::new(), true));
        assert_eq!(speech("フッ。", true, false), (String::new(), true));
        // …a sentence is speech the VAD missed.
        let missed = "後からデータが取れるようになるといいな";
        assert_eq!(speech(missed, true, false), (missed.to_string(), true));
    }
}
