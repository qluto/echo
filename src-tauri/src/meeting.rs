//! Meeting companion: live transcript of a call plus a running list of its
//! decisions, action items and open questions.
//!
//! ```text
//! live capture (mic + system audio → chunks → ASR, see `live`)
//!   ├→ mixed 16 kHz WAV (saved as a voice memo)
//!   └→ lines ("自分" / "相手") → `meeting-line`
//!                                   ↓
//!      [notes thread] every 3 min: items + new lines (+ 60 s of already
//!      reviewed context) → LLM → ADD / UPDATE / DELETE → `meeting-items`
//! ```
//!
//! Items are never edited by hand: a wrong item is corrected by saying so in
//! the meeting, which the next round picks up. On stop, the remaining audio is
//! transcribed, a final round runs, and the meeting is stored as a voice memo
//! (mixed audio, speaker-labelled transcript, items as minutes).

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel as channel;
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::database::MemoStatus;
use crate::live::{self, AudioObserver, Captures, Chunk};
pub use crate::live::Source;
use crate::memo::{self, format_timestamp, with_db};
use crate::transcription::ASREngine;
use crate::types::{AppState, TranscriptionSegment};
use crate::vad::VAD_SAMPLE_RATE;

/// The item list is revised this often (wall clock) by the local notes model…
const UPDATE_EVERY: Duration = Duration::from_secs(180);
/// …and this often by Claude Code, which answers in seconds and runs off the GPU.
const CLAUDE_UPDATE_EVERY: Duration = Duration::from_secs(60);
/// …showing the model this much of the already-reviewed transcript as context.
const UPDATE_OVERLAP_SEC: f64 = 60.0;
/// A round is skipped while less than this much new text has come in.
const MIN_NEW_CHARS: usize = 20;
const ITEMS_MAX_TOKENS: usize = 2048;

const ITEMS_PROMPT: &str = "You keep the live notes of a meeting: the main topics discussed, its decisions, action items and open questions. Every few minutes you get the current notes and the newest part of the speech-recognition transcript, and you rewrite the notes so they are up to date.

## Sections
- TOPIC: a main subject the meeting talked about, with the gist of what was said about it (key points, proposals, concerns, conclusions). One item per subject, one or two sentences. Every substantial discussion belongs here, even when nothing was decided, so this section is rarely empty.
- DECISION: what the participants agreed on, including agreed dates and schedules. Not a topic or a mere proposal.
- ACTION: a concrete task someone will do. Include the owner and deadline only if they were said.
- OPEN: a specific question or issue that was raised and still needs an answer. Not the meeting's agenda or a topic in general.

## Input
- Current notes: each item is `- [<id>] <text>` under its section.
- Transcript lines: `[MM:SS] <speaker>: <utterance>`. 自分 is the user; 相手 is everyone else (several people share it). Lines under \"Already reviewed\" are context only; work from the lines under \"New\". Expect recognition errors.

## Output
First `Facts:`: one short line for every subject discussed and every decision, task, open question, correction, retraction and answer in the New lines (`- none` if there is none).
Then `Notes:`: the complete updated notes, in the same format as the input, with all four sections.
- Copy every current item that is still valid, with its [id], unchanged unless the New lines change it.
- When the New lines revise an item (a changed date, a new owner, a reversed decision), rewrite that item's text and keep its [id]. Never keep both an old and a new version.
- When the New lines go on with a subject already under TOPIC, keep its [id] and rewrite its text to cover the new points. Add a TOPIC only for a new subject. Never leave out a TOPIC: a subject that was settled or dropped was still discussed.
- When an OPEN item is answered, move it (with its [id]) to DECISION or ACTION and rewrite it as the answer.
- Leave out an item only when it was retracted, said to be wrong, cancelled or found unnecessary, or is an ACTION reported as done.
- The outcome that settles an item is itself a DECISION: when a question or task is closed because, say, something turned out unnecessary, write that outcome as a DECISION rather than only dropping the item.
- Add each new fact as `- [new] <text>` in its section.
- Write `- none` under an empty section.

## Rules
- Use ONLY what was said. Never invent decisions, tasks, owners, deadlines, numbers or names.
- One short sentence per item (a TOPIC may take two), in the transcript's language. Never put ids in the text. For an owner, use the name if it was said; otherwise write 自分 for the user and 相手 for the other side (they are labels: no さん).
- No comments or explanations.

## Example
Current notes:
TOPIC:
- [T1] 次回デモの日程と準備の分担を確認した
DECISION:
- [D1] 次回のデモは10日に行う
ACTION:
- [A1] 山田さんがデモ環境を準備する
- [A2] 会場を予約する
OPEN:
- [Q1] デモに顧客を招待するか
New lines:
[12:01] 相手: デモですが、10日は都合が悪いので12日にずらします。
[12:20] 自分: 顧客は呼ばずに社内だけでやりましょう。
[12:35] 相手: デモ環境は山田さんではなく佐藤さんが準備します。来週までにお願いします。
[12:41] 相手: 会場はオンラインにするので予約は不要です。
[12:50] 相手: 資料は誰が作りますか。
[13:05] 相手: 別件ですが、問い合わせが増えていてサポートの負荷が高くなっています。
[13:20] 自分: FAQを充実させれば減らせるかもしれませんね。
Output:
Facts:
- デモは10日から12日に変更
- 顧客は呼ばず社内だけで行う（招待の論点に答えが出た）
- デモ環境の準備は佐藤さんに変更、期限は来週
- 会場はオンラインで予約不要
- 資料の作成者が未定
- 問い合わせ増加でサポート負荷が高い、FAQ充実で減らせるかもしれない
Notes:
TOPIC:
- [T1] 次回デモの日程、参加者、準備の分担と会場を見直した
- [new] 問い合わせの増加でサポートの負荷が高く、FAQの充実で減らせるかもしれないという話が出た
DECISION:
- [D1] 次回のデモは12日に行う
- [Q1] デモは顧客を呼ばず社内だけで行う
- [new] デモはオンラインで行う
ACTION:
- [A1] 佐藤さんが来週までにデモ環境を準備する
OPEN:
- [new] デモの資料を誰が作るか";

// ---------------------------------------------------------------------------
// Data shared with the frontend
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct MeetingLine {
    pub id: u64,
    pub source: Source,
    /// Seconds since the meeting started.
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemKind {
    Topic,
    Decision,
    Action,
    Open,
}

impl ItemKind {
    const ALL: [Self; 4] = [Self::Topic, Self::Decision, Self::Action, Self::Open];

    fn keyword(self) -> &'static str {
        match self {
            Self::Topic => "TOPIC",
            Self::Decision => "DECISION",
            Self::Action => "ACTION",
            Self::Open => "OPEN",
        }
    }

    fn id_prefix(self) -> char {
        match self {
            Self::Topic => 'T',
            Self::Decision => 'D',
            Self::Action => 'A',
            Self::Open => 'Q',
        }
    }

    fn heading(self) -> &'static str {
        match self {
            Self::Topic => "主に話されたこと",
            Self::Decision => "決定事項",
            Self::Action => "ネクストアクション",
            Self::Open => "残論点",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemChange {
    Added,
    Updated,
    Deleted,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeetingItem {
    /// `T1`, `D1`, `A2`, `Q3`, …: stable for the item's lifetime.
    pub id: String,
    pub kind: ItemKind,
    pub text: String,
    /// What the latest round did to the item. Items deleted by it stay in the
    /// list (as `Deleted`) until the next round, so the UI can show them.
    pub change: Option<ItemChange>,
    pub updated_round: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MeetingPhase {
    Running,
    /// Stopped; the last audio is being transcribed and the final round run.
    Finalizing,
    Done,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeetingSnapshot {
    pub phase: MeetingPhase,
    pub memo_id: i64,
    pub started_at: String,
    pub duration_seconds: f64,
    pub lines: Vec<MeetingLine>,
    pub items: Vec<MeetingItem>,
    pub round: u32,
    pub warning: Option<String>,
    pub error: Option<String>,
}

/// Polled while the meeting window is open.
#[derive(Debug, Clone, Serialize)]
pub struct MeetingStatus {
    pub phase: MeetingPhase,
    pub elapsed_seconds: f64,
    pub mic_level: f32,
    pub system_level: f32,
    /// An item revision is running.
    pub updating: bool,
    /// Seconds until the next revision is due.
    pub next_update_in: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LineEvent<'a> {
    memo_id: i64,
    #[serde(flatten)]
    line: &'a MeetingLine,
}

#[derive(Debug, Clone, Serialize)]
struct ItemsEvent<'a> {
    memo_id: i64,
    items: &'a [MeetingItem],
    round: u32,
}

#[derive(Debug, Clone, Serialize)]
struct StateEvent<'a> {
    phase: MeetingPhase,
    memo_id: i64,
    error: Option<&'a str>,
    warning: Option<&'a str>,
}

// ---------------------------------------------------------------------------
// Item revisions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Add(ItemKind, String),
    /// A new kind moves the item to that list (under a new id).
    Update(String, Option<ItemKind>, String),
    Delete(String),
}

fn clean_id(raw: &str) -> Option<String> {
    let id = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric()).to_ascii_uppercase();
    let mut chars = id.chars();
    let valid = matches!(chars.next(), Some('T' | 'D' | 'A' | 'Q'))
        && id.len() > 1
        && chars.all(|c| c.is_ascii_digit());
    valid.then_some(id)
}

/// One item of the model's rewritten notes: its id (`None` for `[new]` or
/// an unknown id), section and text.
#[derive(Debug, Clone, PartialEq)]
struct NoteLine {
    id: Option<String>,
    kind: ItemKind,
    text: String,
}

/// Parse the rewritten notes that follow the `Notes:` heading (the `Facts:`
/// before it are the model's scratch work). `None` when there is no notes
/// section at all, so a malformed answer can't wipe the list.
fn parse_notes(output: &str) -> Option<Vec<NoteLine>> {
    let is_heading = |line: &str, word: &str| {
        line.trim_matches(|c: char| !c.is_alphabetic()).eq_ignore_ascii_case(word)
    };
    let lines: Vec<&str> = output.lines().collect();
    let start = lines.iter().position(|l| is_heading(l, "notes"))? + 1;

    let mut notes = Vec::new();
    let mut section = None;
    let mut saw_section = false;
    for line in &lines[start..] {
        let line = line.trim();
        if let Some(kind) = ItemKind::ALL.into_iter().find(|k| is_heading(line, k.keyword())) {
            section = Some(kind);
            saw_section = true;
            continue;
        }
        let Some(kind) = section else { continue };
        let Some(item) = line.strip_prefix(['-', '*', '•']) else { continue };
        let item = item.trim().replace("**", "");
        let (id, text) = match item.strip_prefix(['[', '［']).and_then(|r| r.split_once([']', '］'])) {
            Some((id, text)) => (clean_id(id), text.trim()),
            None => (None, item.as_str()),
        };
        let text = text.trim_start_matches([':', '：']).trim();
        if text.is_empty() || text.eq_ignore_ascii_case("none") || text == "なし" {
            continue;
        }
        notes.push(NoteLine { id, kind, text: text.to_string() });
    }
    saw_section.then_some(notes)
}

/// The edits that turn `board`'s live items into the rewritten `notes`:
/// items kept under their id are updated or moved, `[new]` ones added, and
/// ids the model left out deleted.
fn notes_to_ops(board: &ItemBoard, notes: &[NoteLine]) -> Vec<Op> {
    let mut ops = Vec::new();
    let mut kept = std::collections::HashSet::new();
    for note in notes {
        let item = note.id.as_deref().and_then(|id| board.find_live(id)).map(|i| &board.items[i]);
        match item {
            // An id listed twice: the first occurrence wins.
            Some(item) if kept.insert(item.id.clone()) => {
                if item.kind != note.kind {
                    ops.push(Op::Update(item.id.clone(), Some(note.kind), note.text.clone()));
                } else if normalized(&item.text) != normalized(&note.text) {
                    ops.push(Op::Update(item.id.clone(), None, note.text.clone()));
                }
            }
            Some(_) => {}
            None => ops.push(Op::Add(note.kind, note.text.clone())),
        }
    }
    for item in board.live() {
        if !kept.contains(&item.id) {
            ops.push(Op::Delete(item.id.clone()));
        }
    }
    ops
}

fn normalized(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The meeting's items and their revision history.
#[derive(Debug, Default)]
struct ItemBoard {
    items: Vec<MeetingItem>,
    next_number: [u32; ItemKind::ALL.len()],
    round: u32,
}

impl ItemBoard {
    fn live(&self) -> impl Iterator<Item = &MeetingItem> {
        self.items.iter().filter(|i| i.change != Some(ItemChange::Deleted))
    }

    fn find_live(&self, id: &str) -> Option<usize> {
        self.items
            .iter()
            .position(|i| i.id == id && i.change != Some(ItemChange::Deleted))
    }

    /// Start a new round and apply `ops` to it. Returns how many took effect.
    fn apply(&mut self, ops: &[Op]) -> usize {
        self.round += 1;
        let round = self.round;
        self.items.retain(|i| i.change != Some(ItemChange::Deleted));
        for item in &mut self.items {
            item.change = None;
        }

        let mut applied = 0;
        for op in ops {
            match op {
                Op::Add(kind, text) => {
                    if !self.add(*kind, text) {
                        continue;
                    }
                }
                Op::Update(id, Some(kind), text) if self.find_live(id).is_some_and(|i| self.items[i].kind != *kind) => {
                    // Settled into another kind: retire the old item, list the new one.
                    self.delete(id);
                    self.add(*kind, text);
                }
                Op::Update(id, _, text) => {
                    let Some(i) = self.find_live(id) else { continue };
                    let item = &mut self.items[i];
                    if normalized(&item.text) == normalized(text) {
                        continue;
                    }
                    item.text = text.clone();
                    if item.change != Some(ItemChange::Added) {
                        item.change = Some(ItemChange::Updated);
                    }
                    item.updated_round = round;
                }
                Op::Delete(id) => {
                    if !self.delete(id) {
                        continue;
                    }
                }
            }
            applied += 1;
        }
        applied
    }

    /// Add an item to the current round, unless the same text is listed already.
    fn add(&mut self, kind: ItemKind, text: &str) -> bool {
        let key = normalized(text);
        if key.is_empty() || self.live().any(|i| i.kind == kind && normalized(&i.text) == key) {
            return false;
        }
        let n = &mut self.next_number[kind as usize];
        *n += 1;
        self.items.push(MeetingItem {
            id: format!("{}{}", kind.id_prefix(), n),
            kind,
            text: text.to_string(),
            change: Some(ItemChange::Added),
            updated_round: self.round,
        });
        true
    }

    fn delete(&mut self, id: &str) -> bool {
        let Some(i) = self.find_live(id) else { return false };
        if self.items[i].change == Some(ItemChange::Added) {
            // Added and dropped in the same round: never shown.
            self.items.remove(i);
        } else {
            self.items[i].change = Some(ItemChange::Deleted);
            self.items[i].updated_round = self.round;
        }
        true
    }

    /// The live items as the model sees them, by section.
    fn listing(&self) -> String {
        ItemKind::ALL
            .iter()
            .map(|&kind| {
                let items: Vec<String> = self
                    .live()
                    .filter(|i| i.kind == kind)
                    .map(|i| format!("- [{}] {}", i.id, i.text))
                    .collect();
                let body = if items.is_empty() { "- none".to_string() } else { items.join("\n") };
                format!("{}:\n{}", kind.keyword(), body)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The live items as Markdown minutes.
    fn markdown(&self) -> String {
        ItemKind::ALL
            .iter()
            .map(|&kind| {
                let bullets: Vec<String> = self
                    .live()
                    .filter(|i| i.kind == kind)
                    .map(|i| format!("- {}", i.text))
                    .collect();
                let body = if bullets.is_empty() { "なし".to_string() } else { bullets.join("\n") };
                format!("## {}\n{}", kind.heading(), body)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

fn transcript_line(line: &MeetingLine) -> String {
    format!("[{}] {}: {}", format_timestamp(line.start), line.source.label(), line.text)
}

/// The user message of one revision round: the current items, then the lines
/// not yet reviewed (`id > reviewed_id`) preceded by the last
/// `UPDATE_OVERLAP_SEC` of reviewed ones as context. `None` when there is too
/// little new text to be worth a round.
fn revision_input(board: &ItemBoard, lines: &[MeetingLine], reviewed_id: u64, language: Option<&str>) -> Option<String> {
    let (reviewed, new): (Vec<&MeetingLine>, Vec<&MeetingLine>) = lines.iter().partition(|l| l.id <= reviewed_id);
    if new.iter().map(|l| l.text.chars().count()).sum::<usize>() < MIN_NEW_CHARS {
        return None;
    }
    let reviewed_end = reviewed.iter().map(|l| l.end).fold(0.0, f64::max);
    let mut context: Vec<&MeetingLine> = reviewed
        .into_iter()
        .filter(|l| l.end > reviewed_end - UPDATE_OVERLAP_SEC)
        .collect();
    context.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut new = new;
    new.sort_by(|a, b| a.start.total_cmp(&b.start));

    let mut input = format!("## Current notes\n{}\n\n## Transcript\n", board.listing());
    if !context.is_empty() {
        input.push_str("### Already reviewed\n");
        for line in context {
            input.push_str(&transcript_line(line));
            input.push('\n');
        }
    }
    input.push_str("### New\n");
    for line in new {
        input.push_str(&transcript_line(line));
        input.push('\n');
    }
    if let Some(lang) = language {
        input.push_str(&format!("\n(Primary language: {lang})"));
    }
    Some(input)
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

struct State {
    phase: MeetingPhase,
    lines: Vec<MeetingLine>,
    board: ItemBoard,
    /// Lines up to this id have been through a revision round.
    reviewed_id: u64,
    next_update: Instant,
    ended_after: Option<Duration>,
    warning: Option<String>,
    error: Option<String>,
}

struct Shared {
    memo_id: i64,
    started_at: String,
    started: Instant,
    state: Mutex<State>,
    mic_level: AtomicU32,
    system_level: AtomicU32,
    updating: AtomicBool,
    /// The ASR worker has transcribed everything; the final round can run.
    transcribed: AtomicBool,
    update_every: Duration,
}

impl Shared {
    fn new(memo_id: i64, started_at: String, warning: Option<String>, update_every: Duration) -> Self {
        Self {
            memo_id,
            started_at,
            started: Instant::now(),
            state: Mutex::new(State {
                phase: MeetingPhase::Running,
                lines: Vec::new(),
                board: ItemBoard::default(),
                reviewed_id: 0,
                next_update: Instant::now() + update_every,
                ended_after: None,
                warning,
                error: None,
            }),
            mic_level: AtomicU32::new(0f32.to_bits()),
            system_level: AtomicU32::new(0f32.to_bits()),
            updating: AtomicBool::new(false),
            transcribed: AtomicBool::new(false),
            update_every,
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // A panicked thread can't leave `State` inconsistent in a way that
        // matters for display, so keep going on poison.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn elapsed(&self, state: &State) -> Duration {
        state.ended_after.unwrap_or_else(|| self.started.elapsed())
    }

    /// Number a transcribed line and add it to the meeting.
    fn push_line(&self, line: live::Line) -> MeetingLine {
        let mut state = self.state();
        let line = MeetingLine {
            id: state.lines.len() as u64 + 1,
            source: line.source,
            start: line.start,
            end: line.end,
            text: line.text,
        };
        state.lines.push(line.clone());
        line
    }

    fn emit_state(&self, app: &AppHandle) {
        let state = self.state();
        let event = StateEvent {
            phase: state.phase,
            memo_id: self.memo_id,
            error: state.error.as_deref(),
            warning: state.warning.as_deref(),
        };
        if let Err(e) = app.emit("meeting-state", &event) {
            log::warn!("Failed to emit meeting-state: {e}");
        }
    }
}

impl AudioObserver for Shared {
    fn on_frame(&self, source: Source, level: f32, _dbfs: f32) {
        let meter = match source {
            Source::Me => &self.mic_level,
            Source::Others => &self.system_level,
        };
        meter.store(level.to_bits(), Ordering::Relaxed);
    }
}

/// A meeting in progress, or the last one (kept for the window to show).
pub struct MeetingSession {
    shared: Arc<Shared>,
    captures: Captures,
}

impl MeetingSession {
    /// Start capturing and transcribing. Fails if the microphone delivers no
    /// audio; system audio is optional (mic only, with a warning, without it).
    pub fn start(app: &AppHandle) -> Result<Self> {
        let state = app.state::<AppState>();
        let (device_name, language) = {
            let settings = state.settings.lock().map_err(|e| anyhow!("{e}"))?;
            (settings.device_name.clone(), (settings.language != "auto").then(|| settings.language.clone()))
        };
        let dir = memo::recordings_dir(app)?;
        let new_memo = with_db(app, |db| db.insert_memo(&dir))?;
        let audio_path = Path::new(&new_memo.audio_path).to_path_buf();

        let started = (|| -> Result<Self> {
            let (captures, inputs, system_error) = live::start_captures(device_name)?;
            let warning = system_error
                .map(|e| format!("相手の音声（システム音声）を取得できないため、マイクのみで記録します: {e}"));

            let writer = WavWriter::create(
                &audio_path,
                WavSpec {
                    channels: 1,
                    sample_rate: VAD_SAMPLE_RATE,
                    bits_per_sample: 16,
                    sample_format: HoundSampleFormat::Int,
                },
            )
            .with_context(|| format!("Failed to create {audio_path:?}"))?;

            let claude = state.asr_engine.lock().map_err(|e| anyhow!("{e}"))?.notes_use_claude();
            let update_every = if claude { CLAUDE_UPDATE_EVERY } else { UPDATE_EVERY };
            let shared = Arc::new(Shared::new(new_memo.id, new_memo.created_at.clone(), warning, update_every));

            let (chunk_tx, chunk_rx) = channel::unbounded::<Chunk>();
            {
                let shared = Arc::clone(&shared);
                thread::Builder::new().name("echo-meeting-audio".into()).spawn(move || {
                    if let Err(e) = live::audio_loop(inputs, chunk_tx, Some(writer), &*shared) {
                        log::error!("Meeting audio thread failed: {e:#}");
                    }
                })?;
            }
            let worker: JoinHandle<()> = {
                let (shared, app, language) = (Arc::clone(&shared), app.clone(), language.clone());
                thread::Builder::new()
                    .name("echo-meeting-asr".into())
                    .spawn(move || {
                        let engine = &app.state::<AppState>().asr_engine;
                        live::asr_worker(
                            chunk_rx,
                            |samples| Ok(engine.lock().map_err(|e| anyhow!("{e}"))?.has_speech(samples)),
                            |samples| {
                                // Lock per chunk so hotkey dictation can run in between.
                                let mut engine = engine.lock().map_err(|e| anyhow!("{e}"))?;
                                Ok(engine.transcribe_samples(samples, language.as_deref(), false)?.text)
                            },
                            |line| {
                                let line = shared.push_line(line);
                                let event = LineEvent { memo_id: shared.memo_id, line: &line };
                                if let Err(e) = app.emit("meeting-line", &event) {
                                    log::warn!("Failed to emit meeting-line: {e}");
                                }
                            },
                        );
                        shared.transcribed.store(true, Ordering::SeqCst);
                    })?
            };
            {
                let (shared, app) = (Arc::clone(&shared), app.clone());
                thread::Builder::new()
                    .name("echo-meeting-notes".into())
                    .spawn(move || notes_loop(worker, &shared, &app, language.as_deref()))?;
            }

            log::info!("Meeting {} started (system audio: {})", new_memo.id, captures.has_system());
            Ok(Self { shared, captures })
        })();

        if started.is_err() {
            let _ = std::fs::remove_file(&audio_path);
            let _ = with_db(app, |db| db.delete_memo(new_memo.id).map(|_| ()));
        }
        let session = started?;
        session.shared.emit_state(app);
        Ok(session)
    }

    pub fn memo_id(&self) -> i64 {
        self.shared.memo_id
    }

    pub fn is_active(&self) -> bool {
        matches!(self.shared.state().phase, MeetingPhase::Running | MeetingPhase::Finalizing)
    }

    /// Stop capturing. Transcription of the remaining audio, the final round
    /// and saving continue in the background (`meeting-state` reports `Done`).
    pub fn stop(&mut self, app: &AppHandle) {
        {
            let mut state = self.shared.state();
            if state.phase != MeetingPhase::Running {
                return;
            }
            state.phase = MeetingPhase::Finalizing;
            state.ended_after = Some(self.shared.started.elapsed());
        }
        // Closing the frame channels lets the audio thread flush and the rest
        // of the pipeline drain.
        self.captures.stop();
        self.shared.emit_state(app);
        log::info!("Meeting {} stopped; finalizing", self.shared.memo_id);
    }

    pub fn snapshot(&self) -> MeetingSnapshot {
        let state = self.shared.state();
        MeetingSnapshot {
            phase: state.phase,
            memo_id: self.shared.memo_id,
            started_at: self.shared.started_at.clone(),
            duration_seconds: self.shared.elapsed(&state).as_secs_f64(),
            lines: state.lines.clone(),
            items: state.board.items.clone(),
            round: state.board.round,
            warning: state.warning.clone(),
            error: state.error.clone(),
        }
    }

    pub fn status(&self) -> MeetingStatus {
        let state = self.shared.state();
        let running = state.phase == MeetingPhase::Running;
        MeetingStatus {
            phase: state.phase,
            elapsed_seconds: self.shared.elapsed(&state).as_secs_f64(),
            mic_level: if running { f32::from_bits(self.shared.mic_level.load(Ordering::Relaxed)) } else { 0.0 },
            system_level: if running { f32::from_bits(self.shared.system_level.load(Ordering::Relaxed)) } else { 0.0 },
            updating: self.shared.updating.load(Ordering::Relaxed),
            next_update_in: running
                .then(|| state.next_update.saturating_duration_since(Instant::now()).as_secs_f64()),
        }
    }
}

// ---------------------------------------------------------------------------
// Notes thread: periodic item revisions, final round, saving
// ---------------------------------------------------------------------------

/// Run one revision round if there is enough new text. Returns whether the
/// model was called.
fn revise(shared: &Shared, app: &AppHandle, language: Option<&str>) -> Result<bool> {
    let (input, upto) = {
        let state = shared.state();
        let upto = state.lines.last().map_or(state.reviewed_id, |l| l.id);
        match revision_input(&state.board, &state.lines, state.reviewed_id, language) {
            Some(input) => (input, upto),
            None => return Ok(false),
        }
    };

    shared.updating.store(true, Ordering::Relaxed);
    let t = Instant::now();
    // Without the engine lock: the meeting keeps being transcribed meanwhile.
    let output = ASREngine::notes_llm(&app.state::<AppState>().asr_engine)
        .and_then(|llm| llm.chat(ITEMS_PROMPT, &input, ITEMS_MAX_TOKENS));
    shared.updating.store(false, Ordering::Relaxed);
    let output = output?;

    let mut state = shared.state();
    let Some(notes) = parse_notes(&output) else {
        log::warn!("Meeting round: no notes in the model's answer:\n{output}");
        return Ok(true);
    };
    let ops = notes_to_ops(&state.board, &notes);
    let applied = state.board.apply(&ops);
    state.reviewed_id = upto;
    log::info!(
        "Meeting round {}: {} of {} ops applied in {:.1}s\n{}",
        state.board.round,
        applied,
        ops.len(),
        t.elapsed().as_secs_f64(),
        output.trim()
    );
    let event = ItemsEvent { memo_id: shared.memo_id, items: &state.board.items, round: state.board.round };
    if let Err(e) = app.emit("meeting-items", &event) {
        log::warn!("Failed to emit meeting-items: {e}");
    }
    Ok(true)
}

fn notes_loop(worker: JoinHandle<()>, shared: &Shared, app: &AppHandle, language: Option<&str>) {
    // Load the notes model now (downloading it if needed) and read its weights
    // in with a tiny generation, rather than in the first round minutes later.
    let t = Instant::now();
    match ASREngine::notes_llm(&app.state::<AppState>().asr_engine).and_then(|llm| llm.warmup()) {
        Ok(()) => log::info!("Meeting notes model ready in {:.1}s", t.elapsed().as_secs_f64()),
        // The first round retries and reports the failure.
        Err(e) => log::warn!("Preparing the notes model failed: {e:#}"),
    }
    loop {
        thread::sleep(Duration::from_millis(500));
        if shared.transcribed.load(Ordering::SeqCst) {
            break;
        }
        let due = {
            let state = shared.state();
            state.phase == MeetingPhase::Running && Instant::now() >= state.next_update
        };
        if !due {
            continue;
        }
        if let Err(e) = revise(shared, app, language) {
            log::error!("Meeting item revision failed: {e:#}");
            shared.state().warning = Some(format!("議事メモの更新に失敗しました: {e:#}"));
            shared.emit_state(app);
        }
        shared.state().next_update = Instant::now() + shared.update_every;
    }
    let _ = worker.join();

    let mut failure = None;
    if let Err(e) = revise(shared, app, language) {
        log::error!("Meeting final revision failed: {e:#}");
        failure = Some(format!("最後の議事メモ更新に失敗しました: {e:#}"));
    }
    if let Ok(mut engine) = app.state::<AppState>().asr_engine.lock() {
        engine.release_notes_model();
    }
    if let Err(e) = save(shared, app, language) {
        log::error!("Saving meeting {} failed: {e:#}", shared.memo_id);
        failure = Some(format!("会議の保存に失敗しました: {e:#}"));
    }
    {
        let mut state = shared.state();
        state.phase = if failure.is_some() { MeetingPhase::Error } else { MeetingPhase::Done };
        state.error = failure;
    }
    memo::emit_progress(app, shared.memo_id, MemoStatus::Done, None);
    shared.emit_state(app);
    log::info!("Meeting {} finished", shared.memo_id);
}

/// Store the meeting in its voice memo row: transcript, items as minutes.
fn save(shared: &Shared, app: &AppHandle, language: Option<&str>) -> Result<()> {
    let (mut lines, minutes, duration) = {
        let state = shared.state();
        (state.lines.clone(), state.board.markdown(), shared.elapsed(&state).as_secs_f64())
    };
    lines.sort_by(|a, b| a.start.total_cmp(&b.start));
    let transcript = lines.iter().map(transcript_line).collect::<Vec<_>>().join("\n");
    let segments: Vec<TranscriptionSegment> = lines
        .iter()
        .map(|l| TranscriptionSegment {
            start: l.start,
            end: l.end,
            text: format!("{}: {}", l.source.label(), l.text),
        })
        .collect();
    let model_name = app
        .state::<AppState>()
        .asr_engine
        .lock()
        .map(|e| e.active_model_name().to_string())
        .unwrap_or_default();

    with_db(app, |db| {
        db.set_memo_duration(shared.memo_id, duration)?;
        if !lines.is_empty() {
            db.set_memo_transcript(
                shared.memo_id,
                &transcript,
                Some(&serde_json::to_string(&segments)?),
                language,
                Some(model_name.as_str()).filter(|m| !m.is_empty()),
            )?;
            db.set_memo_minutes(shared.memo_id, Some(&minutes))?;
        }
        db.set_memo_status(shared.memo_id, MemoStatus::Done, None)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::VAD_FRAME_SIZE;

    #[test]
    fn notes_are_parsed_leniently() {
        let out = "Facts:\n- ACTION: these are scratch notes\n\
                   ## Notes:\n\
                   TOPIC:\n\
                   - [T1] リリース日程を話した\n\
                   **DECISION:**\n\
                   - [D1] リリースは来週木曜\n\
                   - [Q1]: 予算は100万円\n\
                   ACTION:\n\
                   * [new] 鈴木さんがリリースノートを書く\n\
                   - 担当者未定で告知文を作る\n\
                   OPEN:\n\
                   - none\n\
                   - [X9] 存在しないID";
        assert_eq!(
            parse_notes(out).unwrap(),
            vec![
                NoteLine { id: Some("T1".into()), kind: ItemKind::Topic, text: "リリース日程を話した".into() },
                NoteLine { id: Some("D1".into()), kind: ItemKind::Decision, text: "リリースは来週木曜".into() },
                NoteLine { id: Some("Q1".into()), kind: ItemKind::Decision, text: "予算は100万円".into() },
                NoteLine { id: None, kind: ItemKind::Action, text: "鈴木さんがリリースノートを書く".into() },
                NoteLine { id: None, kind: ItemKind::Action, text: "担当者未定で告知文を作る".into() },
                NoteLine { id: None, kind: ItemKind::Open, text: "存在しないID".into() },
            ]
        );
        // No notes section: no answer, rather than an empty list.
        assert!(parse_notes("Facts:\n- none").is_none());
        assert_eq!(parse_notes("Notes:\nDECISION:\n- none\nACTION:\n- none\nOPEN:\n- none"), Some(vec![]));
    }

    #[test]
    fn rewritten_notes_become_edits() {
        let mut board = ItemBoard::default();
        board.apply(&[
            Op::Add(ItemKind::Decision, "リリースは水曜".into()),
            Op::Add(ItemKind::Action, "田中さんがノートを書く".into()),
            Op::Add(ItemKind::Open, "価格改定を含めるか".into()),
            Op::Add(ItemKind::Open, "サーバー増強".into()),
        ]);
        let notes = parse_notes(
            "Notes:\nDECISION:\n- [D1] リリースは木曜\n- [Q1] 価格改定は見送り\n\
             ACTION:\n- [A1] 田中さんがノートを書く。\n- [new] 告知文を作る\nOPEN:\n- none",
        )
        .unwrap();
        assert_eq!(
            notes_to_ops(&board, &notes),
            vec![
                Op::Update("D1".into(), None, "リリースは木曜".into()),
                Op::Update("Q1".into(), Some(ItemKind::Decision), "価格改定は見送り".into()),
                Op::Add(ItemKind::Action, "告知文を作る".into()),
                Op::Delete("Q2".into()),
            ]
        );
    }

    #[test]
    fn board_tracks_adds_updates_and_deletes_per_round() {
        let mut board = ItemBoard::default();
        board.apply(&[
            Op::Add(ItemKind::Open, "リリース日".into()),
            Op::Add(ItemKind::Action, "見積もりを作る".into()),
            Op::Add(ItemKind::Action, "見積もりを作る。".into()), // duplicate
        ]);
        let ids: Vec<_> = board.items.iter().map(|i| (i.id.as_str(), i.change)).collect();
        assert_eq!(ids, vec![("Q1", Some(ItemChange::Added)), ("A1", Some(ItemChange::Added))]);

        // The open question is settled; the action gets a deadline.
        board.apply(&[
            Op::Delete("Q1".into()),
            Op::Add(ItemKind::Decision, "リリースは来週月曜".into()),
            Op::Update("A1".into(), None, "金曜までに見積もりを作る".into()),
            Op::Update("A9".into(), None, "存在しない".into()),
        ]);
        let changes: Vec<_> = board.items.iter().map(|i| (i.id.as_str(), i.change)).collect();
        assert_eq!(
            changes,
            vec![
                ("Q1", Some(ItemChange::Deleted)),
                ("A1", Some(ItemChange::Updated)),
                ("D1", Some(ItemChange::Added)),
            ]
        );
        assert_eq!(
            board.listing(),
            "TOPIC:\n- none\nDECISION:\n- [D1] リリースは来週月曜\nACTION:\n- [A1] 金曜までに見積もりを作る\nOPEN:\n- none"
        );

        // Next round: the deleted item is gone and flags reset; ids keep counting.
        board.apply(&[Op::Add(ItemKind::Open, "予算".into()), Op::Delete("Q1".into())]);
        let changes: Vec<_> = board.items.iter().map(|i| (i.id.as_str(), i.change)).collect();
        assert_eq!(changes, vec![("A1", None), ("D1", None), ("Q2", Some(ItemChange::Added))]);
        assert_eq!(board.round, 3);

        // Added and deleted in the same round: never shown.
        board.apply(&[Op::Add(ItemKind::Decision, "仮".into()), Op::Delete("D2".into())]);
        assert!(board.items.iter().all(|i| i.id != "D2"));

        // An open question settled into a decision moves lists; same kind is a plain update.
        board.apply(&[
            Op::Update("Q2".into(), Some(ItemKind::Decision), "予算は100万円".into()),
            Op::Update("A1".into(), Some(ItemKind::Action), "木曜までに見積もりを作る".into()),
        ]);
        let changes: Vec<_> = board.items.iter().map(|i| (i.id.as_str(), i.change)).collect();
        assert_eq!(
            changes,
            vec![
                ("A1", Some(ItemChange::Updated)),
                ("D1", None),
                ("Q2", Some(ItemChange::Deleted)),
                ("D3", Some(ItemChange::Added)),
            ]
        );

        assert_eq!(
            board.markdown(),
            "## 主に話されたこと\nなし\n\n## 決定事項\n- リリースは来週月曜\n- 予算は100万円\n\n## ネクストアクション\n- 木曜までに見積もりを作る\n\n## 残論点\nなし"
        );
    }

    fn read_wav(path: &Path) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().sample_rate, VAD_SAMPLE_RATE);
        reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
    }

    /// The real audio thread + ASR worker on a two-track recording, then
    /// revision rounds as in `meeting_rounds_replay`:
    /// `ECHO_MEETING_DIR=<dir with me.wav, others.wav> [ECHO_MEETING_MIC=me_echo.wav]
    ///  [ECHO_MEETING_MODEL=<asr model>] [ECHO_MEETING_LLM=<notes model>] [ECHO_MEETING_ROUND_SEC=35] \
    ///  cargo test meeting_end_to_end -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn meeting_end_to_end() {
        let dir = std::path::PathBuf::from(std::env::var("ECHO_MEETING_DIR").expect("set ECHO_MEETING_DIR"));
        let mic_file = std::env::var("ECHO_MEETING_MIC").unwrap_or_else(|_| "me.wav".into());
        let model = std::env::var("ECHO_MEETING_MODEL")
            .unwrap_or_else(|_| "mlx-community/parakeet-tdt_ctc-0.6b-ja".to_string());
        let mic = read_wav(&dir.join(&mic_file));
        let others = read_wav(&dir.join("others.wav"));

        let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Caches/io.qluto.echo/huggingface/hub");
        let mut engine = crate::transcription::ASREngine::with_hub(hub);
        engine.set_model(&model).unwrap();

        // Feed both tracks in lockstep, as live capture would.
        let shared = Arc::new(Shared::new(0, String::new(), None, UPDATE_EVERY));
        let (mic_tx, mic_rx) = channel::bounded(2);
        let (sys_tx, sys_rx) = channel::bounded(2);
        let (chunk_tx, chunk_rx) = channel::unbounded();
        let feeder = thread::spawn(move || {
            for (m, o) in mic.chunks(VAD_FRAME_SIZE).zip(others.chunks(VAD_FRAME_SIZE)) {
                if m.len() < VAD_FRAME_SIZE {
                    break;
                }
                mic_tx.send(m.to_vec()).unwrap();
                sys_tx.send(o.to_vec()).unwrap();
            }
        });
        let tmp = tempfile::tempdir().unwrap();
        let wav = tmp.path().join("mix.wav");
        let spec = WavSpec { channels: 1, sample_rate: VAD_SAMPLE_RATE, bits_per_sample: 16, sample_format: HoundSampleFormat::Int };
        let writer = WavWriter::create(&wav, spec).unwrap();
        let audio = {
            let shared = Arc::clone(&shared);
            let inputs = live::Inputs::for_test(vec![0.0; VAD_FRAME_SIZE], mic_rx, sys_rx);
            thread::spawn(move || live::audio_loop(inputs, chunk_tx, Some(writer), &*shared).unwrap())
        };
        let t = Instant::now();
        let engine = std::cell::RefCell::new(engine);
        live::asr_worker(
            chunk_rx,
            |samples| Ok(engine.borrow_mut().has_speech(samples)),
            |samples| Ok(engine.borrow_mut().transcribe_samples(samples, Some("ja"), false)?.text),
            |line| {
                shared.push_line(line);
            },
        );
        feeder.join().unwrap();
        audio.join().unwrap();
        println!("--- transcript ({:?}) ---", t.elapsed());

        let mut lines = shared.state().lines.clone();
        lines.sort_by(|a, b| a.end.total_cmp(&b.end));
        for (i, l) in lines.iter_mut().enumerate() {
            l.id = i as u64 + 1;
            println!("{}", transcript_line(l));
        }
        if let Ok(path) = std::env::var("ECHO_MEETING_DUMP") {
            // For evaluating other LLMs on the same transcript.
            std::fs::write(&path, serde_json::to_string_pretty(&lines).unwrap()).unwrap();
        }
        assert!(!lines.is_empty());
        let board = replay_rounds(engine.into_inner(), &lines, &round_times(&lines));
        assert!(board.round > 0);
    }

    /// Revision rounds on a saved transcript (`ECHO_MEETING_DUMP` output, or
    /// lines from a memo), skipping ASR — for comparing notes models:
    /// `ECHO_MEETING_LINES=<lines.json> [ECHO_MEETING_LLM=<notes model> | ECHO_MEETING_CLAUDE=<sonnet|opus|…>]
    ///  [ECHO_MEETING_ROUNDS=182,366 | ECHO_MEETING_ROUND_SEC=35] \
    ///  cargo test meeting_rounds_replay -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn meeting_rounds_replay() {
        let path = std::env::var("ECHO_MEETING_LINES").expect("set ECHO_MEETING_LINES");
        let lines: Vec<MeetingLine> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Caches/io.qluto.echo/huggingface/hub");
        let board = replay_rounds(ASREngine::with_hub(hub), &lines, &round_times(&lines));
        assert!(board.round > 0);
    }

    /// `ECHO_MEETING_ROUNDS` (seconds, comma-separated), else every
    /// `ECHO_MEETING_ROUND_SEC`; the last round is always at the end.
    fn round_times(lines: &[MeetingLine]) -> Vec<f64> {
        let last_end = lines.iter().map(|l| l.end).fold(0.0, f64::max);
        let mut times: Vec<f64> = match std::env::var("ECHO_MEETING_ROUNDS") {
            Ok(list) => list.split(',').map(|t| t.trim().parse().unwrap()).filter(|&t| t < last_end).collect(),
            Err(_) => {
                let every: f64 = std::env::var("ECHO_MEETING_ROUND_SEC").map_or(35.0, |v| v.parse().unwrap());
                (1..).map(|i| i as f64 * every).take_while(|&t| t < last_end).collect()
            }
        };
        times.push(last_end);
        times
    }

    /// Run the revision rounds at `times` with the notes model
    /// (`ECHO_MEETING_LLM` overrides it): lines become visible once they end.
    fn replay_rounds(mut engine: ASREngine, lines: &[MeetingLine], times: &[f64]) -> ItemBoard {
        if let Ok(llm) = std::env::var("ECHO_MEETING_LLM") {
            engine.set_notes_model(&llm).unwrap();
        }
        if let Ok(model) = std::env::var("ECHO_MEETING_CLAUDE") {
            engine.set_notes_claude(Some(&model)).unwrap();
        }
        let engine = Mutex::new(engine);
        let mut board = ItemBoard::default();
        let mut reviewed = 0u64;
        for &at in times {
            let visible: Vec<MeetingLine> = lines.iter().filter(|l| l.end <= at).cloned().collect();
            let Some(input) = revision_input(&board, &visible, reviewed, Some("ja")) else { continue };
            let t = Instant::now();
            let output = ASREngine::notes_llm(&engine).unwrap().chat(ITEMS_PROMPT, &input, ITEMS_MAX_TOKENS).unwrap();
            let ops = parse_notes(&output).map(|n| notes_to_ops(&board, &n)).unwrap_or_default();
            board.apply(&ops);
            reviewed = visible.last().map_or(reviewed, |l| l.id);
            println!("\n=== round {} at {at:.0}s ({:.1}s) ===\n{}\n---", board.round, t.elapsed().as_secs_f64(), output.trim());
            for i in &board.items {
                println!("{} {:?} {}", i.id, i.change, i.text);
            }
        }
        println!("\n{}", board.markdown());
        board
    }

    fn line(id: u64, source: Source, start: f64, text: &str) -> MeetingLine {
        MeetingLine { id, source, start, end: start + 5.0, text: text.into() }
    }

    #[test]
    fn revision_input_has_new_lines_and_recent_context() {
        let lines = vec![
            line(1, Source::Me, 0.0, "古い発言なので文脈にも入らない発言です"),
            line(2, Source::Others, 100.0, "前回までに確認済みの直近の発言です"),
            line(4, Source::Me, 130.0, "新しい発言その二、見積もりは金曜まで"),
            line(3, Source::Others, 120.0, "新しい発言その一、リリースは来週"),
        ];
        let board = ItemBoard::default();
        let input = revision_input(&board, &lines, 2, Some("ja")).unwrap();
        assert_eq!(
            input,
            "## Current notes\nTOPIC:\n- none\nDECISION:\n- none\nACTION:\n- none\nOPEN:\n- none\n\n## Transcript\n\
             ### Already reviewed\n[01:40] 相手: 前回までに確認済みの直近の発言です\n\
             ### New\n[02:00] 相手: 新しい発言その一、リリースは来週\n\
             [02:10] 自分: 新しい発言その二、見積もりは金曜まで\n\n(Primary language: ja)"
        );
        // Nothing (or too little) new: no round.
        assert!(revision_input(&board, &lines, 4, None).is_none());
        assert!(revision_input(&board, &[line(1, Source::Me, 0.0, "はい")], 0, None).is_none());
    }
}
