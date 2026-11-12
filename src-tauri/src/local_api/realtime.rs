//! Streaming transcription over WebSocket, shaped like OpenAI's Realtime
//! transcription sessions: `GET /v1/realtime?intent=transcription`.
//!
//! Client → server:
//! - `session.update` — `session.audio.input.format` (`audio/pcm`, any rate
//!   from 8 to 48 kHz; default 24 kHz like OpenAI) and
//!   `session.audio.input.transcription.language` / `languages[0]`
//! - `input_audio_buffer.append` — `audio`: base64 PCM16 LE mono. A binary
//!   frame carrying the raw PCM16 bytes does the same, without the base64
//! - `input_audio_buffer.commit` — end of the turn, like releasing the hotkey
//! - `input_audio_buffer.clear` — drop the turn
//!
//! Server → client: `session.created` / `session.updated`,
//! `input_audio_buffer.committed` / `.cleared`,
//! `conversation.item.input_audio_transcription.delta` / `.completed` /
//! `.failed`, `error`, and Echo's own `echo.transcription.draft`.
//!
//! A turn runs through the hotkey's `PartialDecoder`: while audio arrives,
//! the last ≤ 8 s are re-decoded every 0.5 s and the head is committed at a
//! quiet point. Committed text is final and goes out as `delta`s (append-only,
//! as OpenAI's are) — unlike OpenAI, they start before the commit, with the
//! turn's `item_id`. The draft can still change, so it only goes out as
//! `echo.transcription.draft`, each one replacing the last. On commit only
//! the tail is decoded, then `completed` carries the whole transcript.
//!
//! Engine sharing is the hotkey's: drafts `try_lock` and skip a tick when
//! the engine is busy; the final decode waits its turn.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use serde_json::{json, Value};
use tauri::{AppHandle, Manager, Runtime};
use tokio::sync::mpsc;

use super::{bad_request, resolve_language, transcribe_samples};
use crate::audio_capture::FrameAccumulator;
use crate::partial::{append_text, PartialDecoder};
use crate::vad::VAD_SAMPLE_RATE;
use crate::AppState;

/// OpenAI's default for `audio/pcm`.
const DEFAULT_RATE: u32 = 24_000;
/// One turn's audio is kept for the final decode; cap it (≈ 230 MB).
const MAX_TURN_SAMPLES: usize = 60 * 60 * VAD_SAMPLE_RATE as usize;

#[derive(serde::Deserialize)]
pub(super) struct Params {
    intent: Option<String>,
}

pub(super) async fn handler<R: Runtime>(
    State(app): State<AppHandle<R>>,
    Query(params): Query<Params>,
    ws: WebSocketUpgrade,
) -> Response {
    if params.intent.as_deref().is_some_and(|i| i != "transcription") {
        return bad_request("Only transcription sessions are supported (intent=transcription)")
            .into_response();
    }
    // Browsers ask for "realtime" next to the token subprotocol; echo it back.
    ws.protocols(["realtime"])
        .max_message_size(16 * 1024 * 1024)
        .on_upgrade(move |socket| run(app, socket))
}

type Events = mpsc::UnboundedSender<Value>;

fn send(events: &Events, mut event: Value) {
    event["event_id"] = json!(format!("event_{}", uuid::Uuid::new_v4().simple()));
    let _ = events.send(event);
}

fn send_error(events: &Events, code: &str, message: impl std::fmt::Display, client_event: Option<&str>) {
    send(
        events,
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": code,
                "message": message.to_string(),
                "event_id": client_event,
            }
        }),
    );
}

async fn run<R: Runtime>(app: AppHandle<R>, mut socket: WebSocket) {
    let (events, mut outgoing) = mpsc::unbounded_channel::<Value>();
    log::info!("Local API: realtime session opened");

    // Load the model up front: drafts only run on a loaded engine.
    let loader = app.clone();
    let loaded = tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<()> {
        let state = loader.state::<AppState>();
        let mut engine = state.asr_engine.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        if !engine.get_model_status()?.loaded {
            engine.load_model()?;
        }
        Ok(())
    })
    .await;
    let mut session = Session::new(app, events.clone());
    match loaded {
        Ok(Ok(())) => {}
        Ok(Err(e)) => send_error(&events, "model_not_loaded", format!("{e:#}"), None),
        Err(e) => send_error(&events, "model_not_loaded", e, None),
    }
    send(&events, json!({ "type": "session.created", "session": session.describe() }));

    loop {
        tokio::select! {
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Text(text))) => session.handle_event(text.as_str()),
                Some(Ok(Message::Binary(bytes))) => session.append_pcm(&bytes),
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    log::warn!("Local API: realtime socket error: {e}");
                    break;
                }
            },
            Some(event) = outgoing.recv() => {
                if socket.send(Message::Text(event.to_string().into())).await.is_err() {
                    break;
                }
            }
        }
    }
    session.clear();
    log::info!("Local API: realtime session closed");
}

/// The audio of the turn in progress and its draft decoder.
struct Turn {
    item_id: String,
    accumulator: FrameAccumulator,
    /// 16 kHz frames out of the accumulator.
    frames: crossbeam_channel::Receiver<Vec<f32>>,
    /// Into the decoder; dropped on commit, which ends the decoder.
    decoder_tx: crossbeam_channel::Sender<Vec<f32>>,
    decoder: PartialDecoder,
    /// The whole turn at 16 kHz, for the final decode when the active
    /// model can't draft (Whisper).
    audio: Vec<f32>,
    /// Bytes of committed text already sent as deltas.
    sent: Arc<AtomicUsize>,
    language: String,
    overflowed: bool,
}

impl Turn {
    /// Move resampled frames into the turn's audio and the decoder.
    fn pump(&mut self) {
        for frame in self.frames.try_iter() {
            if self.audio.len() + frame.len() > MAX_TURN_SAMPLES {
                self.overflowed = true;
                break;
            }
            self.audio.extend_from_slice(&frame);
            // Err: the decoder gave up (model can't draft); `audio` has it all.
            let _ = self.decoder_tx.send(frame);
        }
    }
}

struct Session<R: Runtime> {
    app: AppHandle<R>,
    events: Events,
    rate: u32,
    /// None = the app's language setting.
    language: Option<String>,
    turn: Option<Turn>,
    /// Odd byte left over from the last PCM chunk.
    pending_byte: Option<u8>,
    last_item: Option<String>,
}

impl<R: Runtime> Session<R> {
    fn new(app: AppHandle<R>, events: Events) -> Self {
        Self { app, events, rate: DEFAULT_RATE, language: None, turn: None, pending_byte: None, last_item: None }
    }

    fn describe(&self) -> Value {
        let model = self
            .app
            .state::<AppState>()
            .asr_engine
            .try_lock()
            .map(|e| e.active_model_name().to_string())
            .ok();
        json!({
            "type": "transcription",
            "audio": { "input": {
                "format": { "type": "audio/pcm", "rate": self.rate },
                "transcription": { "model": model, "language": self.language },
                "turn_detection": null,
            }},
        })
    }

    fn handle_event(&mut self, text: &str) {
        let event: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => return send_error(&self.events, "invalid_json", e, None),
        };
        let id = event["event_id"].as_str();
        match event["type"].as_str().unwrap_or("") {
            "session.update" => self.update(&event["session"], id),
            "input_audio_buffer.append" => {
                let Some(audio) = event["audio"].as_str() else {
                    return send_error(&self.events, "missing_audio", "'audio' must be base64 PCM16", id);
                };
                match base64::engine::general_purpose::STANDARD.decode(audio) {
                    Ok(bytes) => self.append_pcm(&bytes),
                    Err(e) => send_error(&self.events, "invalid_audio", e, id),
                }
            }
            "input_audio_buffer.commit" => self.commit(id),
            "input_audio_buffer.clear" => {
                self.clear();
                send(&self.events, json!({ "type": "input_audio_buffer.cleared" }));
            }
            other => send_error(&self.events, "unsupported_event", format!("Unsupported event type '{other}'"), id),
        }
    }

    fn update(&mut self, session: &Value, id: Option<&str>) {
        let input = &session["audio"]["input"];
        let format = &input["format"];
        if !format.is_null() {
            if let Some(kind) = format["type"].as_str().filter(|t| *t != "audio/pcm") {
                return send_error(&self.events, "unsupported_format", format!("Unsupported format '{kind}' (use audio/pcm)"), id);
            }
            if let Some(rate) = format["rate"].as_u64() {
                if !(8_000..=48_000).contains(&rate) {
                    return send_error(&self.events, "unsupported_rate", format!("Unsupported rate {rate}"), id);
                }
                if self.turn.is_some() && rate as u32 != self.rate {
                    return send_error(&self.events, "turn_in_progress", "Commit or clear the audio before changing the rate", id);
                }
                self.rate = rate as u32;
            }
        }
        let transcription = &input["transcription"];
        let language = transcription["language"]
            .as_str()
            .or_else(|| transcription["languages"][0].as_str());
        if let Some(language) = language {
            self.language = Some(language.to_string()).filter(|l| !l.is_empty());
        }
        send(&self.events, json!({ "type": "session.updated", "session": self.describe() }));
    }

    fn append_pcm(&mut self, bytes: &[u8]) {
        let mut data = Vec::with_capacity(bytes.len() + 1);
        data.extend(self.pending_byte.take());
        data.extend_from_slice(bytes);
        if data.len() % 2 == 1 {
            self.pending_byte = data.pop();
        }
        let samples: Vec<f32> = data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        if samples.is_empty() {
            return;
        }
        if self.turn.is_none() {
            match self.start_turn() {
                Ok(turn) => self.turn = Some(turn),
                Err(e) => return send_error(&self.events, "internal_error", format!("{e:#}"), None),
            }
        }
        let turn = self.turn.as_mut().expect("turn started");
        if turn.overflowed {
            return;
        }
        turn.accumulator.push_samples(&samples);
        turn.pump();
        if turn.overflowed {
            send_error(&self.events, "buffer_too_long", "Turn is over 60 minutes; further audio is ignored until commit", None);
        }
    }

    fn start_turn(&self) -> anyhow::Result<Turn> {
        let item_id = format!("item_{}", uuid::Uuid::new_v4().simple());
        let language = resolve_language(&self.app, self.language.clone())?;
        let (frames_tx, frames) = crossbeam_channel::unbounded();
        let accumulator = FrameAccumulator::new(self.rate, 1, frames_tx)?;
        let (decoder_tx, decoder_rx) = crossbeam_channel::unbounded();
        let sent = Arc::new(AtomicUsize::new(0));

        let events = self.events.clone();
        let (sent_by_decoder, item) = (sent.clone(), item_id.clone());
        let on_update = move |update: crate::partial::HotkeyPartialEvent| {
            if update.committed.is_empty() && update.draft.is_empty() {
                return; // the decoder's exit notice
            }
            let already = sent_by_decoder.load(Ordering::SeqCst);
            if update.committed.len() > already {
                send(&events, json!({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item,
                    "content_index": 0,
                    "delta": &update.committed[already..],
                }));
                sent_by_decoder.store(update.committed.len(), Ordering::SeqCst);
            }
            let mut text = update.committed.clone();
            append_text(&mut text, &update.draft);
            send(&events, json!({
                "type": "echo.transcription.draft",
                "item_id": item,
                "committed": update.committed,
                "draft": update.draft,
                "text": text,
            }));
        };
        let engine = self.app.state::<AppState>().asr_engine.clone();
        let lang = (language != "auto").then(|| language.clone());
        let decoder = PartialDecoder::start(on_update, engine, lang, decoder_rx);
        Ok(Turn {
            item_id,
            accumulator,
            frames,
            decoder_tx,
            decoder,
            audio: Vec::new(),
            sent,
            language,
            overflowed: false,
        })
    }

    fn commit(&mut self, id: Option<&str>) {
        let Some(mut turn) = self.turn.take() else {
            return send_error(&self.events, "input_audio_buffer_commit_empty", "No audio to commit", id);
        };
        turn.accumulator.flush();
        turn.pump();
        send(&self.events, json!({
            "type": "input_audio_buffer.committed",
            "item_id": turn.item_id,
            "previous_item_id": self.last_item,
        }));
        self.last_item = Some(turn.item_id.clone());
        self.pending_byte = None;

        let app = self.app.clone();
        let events = self.events.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let item_id = turn.item_id.clone();
            let started = std::time::Instant::now();
            let seconds = turn.audio.len() as f64 / VAD_SAMPLE_RATE as f64;
            match finish(&app, turn, &events) {
                Ok(text) => {
                    log::info!(
                        "Local API: realtime turn of {seconds:.1} s finished {} ms after commit ({} chars)",
                        started.elapsed().as_millis(),
                        text.chars().count()
                    );
                    send(&events, json!({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "content_index": 0,
                        "transcript": text,
                    }));
                }
                Err(e) => {
                    log::warn!("Local API: realtime turn failed: {e:#}");
                    send(&events, json!({
                        "type": "conversation.item.input_audio_transcription.failed",
                        "item_id": item_id,
                        "content_index": 0,
                        "error": { "type": "transcription_error", "message": format!("{e:#}") },
                    }));
                }
            }
        });
    }

    /// Drop the turn in progress without transcribing it.
    fn clear(&mut self) {
        if let Some(turn) = self.turn.take() {
            drop(turn.decoder_tx);
            turn.decoder.stop();
        }
        self.pending_byte = None;
    }
}

/// The final transcript of a committed turn; sends the last delta.
fn finish<R: Runtime>(app: &AppHandle<R>, turn: Turn, events: &Events) -> anyhow::Result<String> {
    drop(turn.decoder_tx); // the decoder drains what it has, then exits
    let text = match turn.decoder.finish() {
        // Like a hotkey release: only the tail after the last cut is left.
        Some(draft) => {
            let lang = (turn.language != "auto").then_some(turn.language.as_str());
            let state = app.state::<AppState>();
            let mut engine = state.asr_engine.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
            let r = engine.transcribe_samples(&draft.tail, lang, draft.committed.is_empty())?;
            let mut text = draft.committed;
            append_text(&mut text, r.text.trim());
            text
        }
        // The model doesn't draft: transcribe the whole turn.
        None => transcribe_samples(app, &turn.audio, &turn.language)?.text,
    };
    let sent = turn.sent.load(Ordering::SeqCst);
    let rest = text.get(sent..).unwrap_or("");
    if !rest.is_empty() {
        send(events, json!({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": turn.item_id,
            "content_index": 0,
            "delta": rest,
        }));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use futures_util::{SinkExt, StreamExt};
    use serde_json::{json, Value};
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, Message};

    use crate::local_api::tests::start_test_server;

    /// PCM16 mono samples of a WAV, at its own rate.
    fn pcm(path: &str) -> (Vec<u8>, u32) {
        let mut r = hound::WavReader::open(path).unwrap();
        let spec = r.spec();
        assert_eq!((spec.channels, spec.bits_per_sample), (1, 16), "{path}: mono PCM16 please");
        let bytes = r.samples::<i16>().flat_map(|s| s.unwrap().to_le_bytes()).collect();
        (bytes, spec.sample_rate)
    }

    /// Streams speech through `/v1/realtime` at 4× real time while another
    /// thread takes the engine lock like hotkey dictation, then checks the
    /// events. Audio as for `api_end_to_end`, plus the long one as a 24 kHz WAV
    /// (`afconvert -f WAVE -d LEI16@24000 -c 1`).
    ///
    /// ECHO_API_SHORT_WAV=… ECHO_API_LONG_WAV24=… cargo test realtime_end_to_end -- --ignored --nocapture
    #[test]
    #[ignore]
    fn realtime_end_to_end() {
        let short = std::env::var("ECHO_API_SHORT_WAV").expect("set ECHO_API_SHORT_WAV");
        let long = std::env::var("ECHO_API_LONG_WAV24").expect("set ECHO_API_LONG_WAV24");
        let port = 47_798;
        let (_app, engine, token, _dir) = start_test_server(port);
        let drafts_expected = !rust_asr::is_whisper_model(engine.lock().unwrap().active_model_name());
        let url = format!("ws://127.0.0.1:{port}/v1/realtime?intent=transcription");

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Auth: none, from a web page, wrong; then the subprotocol form.
            let status = |r: Result<_, tokio_tungstenite::tungstenite::Error>| match r {
                Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => resp.status().as_u16(),
                Err(e) => panic!("{e}"),
                Ok(_) => 101,
            };
            assert_eq!(status(tokio_tungstenite::connect_async(&url).await.map(|_| ())), 401);
            let mut req = url.as_str().into_client_request().unwrap();
            req.headers_mut().insert("Authorization", format!("Bearer {token}").parse().unwrap());
            req.headers_mut().insert("Origin", "https://example.com".parse().unwrap());
            assert_eq!(status(tokio_tungstenite::connect_async(req).await.map(|_| ())), 403);
            let mut req = url.as_str().into_client_request().unwrap();
            req.headers_mut().insert(
                "Sec-WebSocket-Protocol",
                format!("realtime, openai-insecure-api-key.{token}").parse().unwrap(),
            );
            let (ws, resp) = tokio_tungstenite::connect_async(req).await.unwrap();
            assert_eq!(resp.headers()["sec-websocket-protocol"], "realtime");
            let (mut tx, mut rx) = ws.split();

            let events: Arc<Mutex<Vec<(Instant, Value)>>> = Arc::default();
            let collector = {
                let events = events.clone();
                tokio::spawn(async move {
                    while let Some(Ok(Message::Text(t))) = rx.next().await {
                        events.lock().unwrap().push((Instant::now(), serde_json::from_str(&t).unwrap()));
                    }
                })
            };
            let wait_for = |pred: Box<dyn Fn(&Value) -> bool>| {
                let events = events.clone();
                async move {
                    let t = Instant::now();
                    loop {
                        if let Some((_, v)) = events.lock().unwrap().iter().find(|(_, v)| pred(v)) {
                            return v.clone();
                        }
                        assert!(t.elapsed() < Duration::from_secs(120), "timed out");
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                }
            };
            let send = |v: Value| Message::Text(v.to_string().into());

            let created = wait_for(Box::new(|v| v["type"] == "session.created")).await;
            println!("session.created: {}", created["session"]);

            // Nothing to commit yet.
            tx.send(send(json!({"type": "input_audio_buffer.commit", "event_id": "c0"}))).await.unwrap();
            let err = wait_for(Box::new(|v| v["type"] == "error")).await;
            assert_eq!(err["error"]["code"], "input_audio_buffer_commit_empty");
            assert_eq!(err["error"]["event_id"], "c0");

            // Turn 1: the long clip as base64 appends at 24 kHz, 4× real time.
            let (long_pcm, rate) = pcm(&long);
            tx.send(send(json!({"type": "session.update", "session": {"type": "transcription",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": rate},
                "transcription": {"language": "ja"}}}}}))).await.unwrap();
            wait_for(Box::new(|v| v["type"] == "session.updated")).await;

            let short_samples = rust_asr::read_wav_16k_mono(&short).unwrap();
            let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let dictation = {
                let (engine, done) = (engine.clone(), done.clone());
                std::thread::spawn(move || {
                    let mut waits = Vec::new();
                    while !done.load(std::sync::atomic::Ordering::Relaxed) {
                        let t = Instant::now();
                        let mut e = engine.lock().unwrap();
                        waits.push(t.elapsed());
                        e.transcribe_samples(&short_samples, Some("ja"), true).unwrap();
                        drop(e);
                        std::thread::sleep(Duration::from_millis(700));
                    }
                    waits
                })
            };
            let chunk = (rate as usize / 10) * 2; // 100 ms
            let started = Instant::now();
            for c in long_pcm.chunks(chunk) {
                let audio = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, c);
                tx.send(send(json!({"type": "input_audio_buffer.append", "audio": audio}))).await.unwrap();
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            let commit_at = Instant::now();
            tx.send(send(json!({"type": "input_audio_buffer.commit"}))).await.unwrap();
            let committed = wait_for(Box::new(|v| v["type"] == "input_audio_buffer.committed")).await;
            let item = committed["item_id"].as_str().unwrap().to_string();
            let it = item.clone();
            let completed = wait_for(Box::new(move |v| {
                v["type"] == "conversation.item.input_audio_transcription.completed" && v["item_id"] == it
            }))
            .await;
            done.store(true, std::sync::atomic::Ordering::Relaxed);
            let waits = dictation.join().unwrap();

            let all = events.lock().unwrap().clone();
            let of = |kind: &str| -> Vec<(Instant, Value)> {
                all.iter().filter(|(_, v)| v["type"] == kind && v["item_id"] == item.as_str()).cloned().collect()
            };
            let drafts = of("echo.transcription.draft");
            let deltas = of("conversation.item.input_audio_transcription.delta");
            let early = deltas.iter().filter(|(t, _)| *t < commit_at).count();
            let joined: String = deltas.iter().map(|(_, v)| v["delta"].as_str().unwrap()).collect();
            let transcript = completed["transcript"].as_str().unwrap();
            let done_at = all.iter().find(|(_, v)| v["type"] == completed["type"]).unwrap().0;
            println!(
                "turn 1: streamed {:?}, {} drafts, {} deltas ({early} before commit), completed {:?} after commit",
                commit_at - started,
                drafts.len(),
                deltas.len(),
                done_at - commit_at
            );
            if let Some((_, d)) = drafts.get(drafts.len() / 2) {
                println!("  a draft midway: {}", d["text"]);
            }
            println!("  transcript: {transcript}");
            println!("  dictation: {} runs, longest wait {:?}", waits.len(), waits.iter().max().unwrap());
            assert_eq!(joined, transcript, "deltas add up to the transcript");
            assert!(transcript.contains("トークン"));
            if drafts_expected {
                assert!(!drafts.is_empty());
                assert!(early > 0, "committed text streams before the commit");
            }
            assert!(waits.len() >= 3);

            // Turn 2: the short clip as binary frames at 16 kHz.
            let (short_pcm, rate) = pcm(&short);
            assert_eq!(rate, 16_000);
            tx.send(send(json!({"type": "session.update", "session": {"type": "transcription",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}))).await.unwrap();
            for c in short_pcm.chunks(3201) {
                tx.send(Message::Binary(c.to_vec().into())).await.unwrap();
            }
            let commit_at = Instant::now();
            tx.send(send(json!({"type": "input_audio_buffer.commit"}))).await.unwrap();
            let completed = wait_for(Box::new(move |v| {
                v["type"] == "conversation.item.input_audio_transcription.completed" && v["item_id"] != item.as_str()
            }))
            .await;
            println!("turn 2 ({:?} after commit): {}", commit_at.elapsed(), completed["transcript"]);
            assert!(completed["transcript"].as_str().unwrap().contains("会議"));

            tx.send(Message::Close(None)).await.ok();
            collector.await.ok();
        });
    }
}
