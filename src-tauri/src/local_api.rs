//! Local transcription API: an HTTP server on 127.0.0.1 so other apps on this
//! Mac can use Echo's loaded ASR engine. Off by default (`LocalApiSettings`).
//!
//! - `POST /v1/audio/transcriptions` — OpenAI-compatible multipart upload:
//!   `file` (required; WAV, or anything `afconvert` reads: m4a, mp3, aac,
//!   flac, caf…), `language` (ISO code; defaults to the app's language
//!   setting), `response_format` (`json` | `text` | `verbose_json`). `model`
//!   is accepted and ignored: the app's active ASR model is used.
//! - `GET /v1/realtime?intent=transcription` — streaming PCM over WebSocket,
//!   shaped like OpenAI's Realtime transcription (`realtime.rs`).
//! - `GET /health` — the active model, and whether the engine is busy.
//!
//! Every request needs the token: `Authorization: Bearer <token>`, or for
//! WebSocket clients that can't set headers, the subprotocol
//! `openai-insecure-api-key.<token>` (as OpenAI accepts). Requests carrying
//! an `Origin` header are refused: that is a web page, and while browsers
//! guard cross-origin HTTP with CORS, they don't guard WebSockets.
//!
//! Requests share the engine with dictation, always-on, memos and meetings
//! through `AppState::asr_engine`'s lock, taken once per chunk of at most
//! 28 s. So a long upload never holds up the hotkey for more than one chunk,
//! and the app never waits on the API for longer either.

use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{anyhow, bail, Context};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tauri::{AppHandle, Manager, Runtime};
use tokio::sync::oneshot;

mod realtime;

use crate::{memo, partial, AppState, LocalApiSettings, TranscriptionSegment};

pub const DEFAULT_PORT: u16 = 47_700;
/// Upload size limit: about 1.5 h of 16 kHz 16-bit WAV, more in compressed formats.
const MAX_UPLOAD_BYTES: usize = 200 * 1024 * 1024;

struct Server {
    port: u16,
    shutdown: oneshot::Sender<()>,
}

/// The running server, if any.
static SERVER: Mutex<Option<Server>> = Mutex::new(None);
/// Why the server isn't running although it is enabled (e.g. port in use).
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// What the settings UI shows.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LocalApiStatus {
    pub enabled: bool,
    pub port: u16,
    pub token: String,
    pub running: bool,
    pub error: Option<String>,
}

pub fn status(settings: &LocalApiSettings) -> LocalApiStatus {
    let running = SERVER.lock().map(|s| s.is_some()).unwrap_or(false);
    LocalApiStatus {
        enabled: settings.enabled,
        port: settings.port,
        token: settings.token.clone(),
        running,
        error: LAST_ERROR.lock().ok().and_then(|e| e.clone()),
    }
}

pub fn new_token() -> String {
    format!("echo_{}", uuid::Uuid::new_v4().simple())
}

/// Start, stop or move the server to match `settings`. The token is read per
/// request, so changing it needs no restart.
pub fn apply<R: Runtime>(app: &AppHandle<R>, settings: &LocalApiSettings) -> anyhow::Result<()> {
    let result = apply_inner(app, settings);
    if let Ok(mut last) = LAST_ERROR.lock() {
        *last = result.as_ref().err().map(|e| format!("{e:#}"));
    }
    result
}

fn apply_inner<R: Runtime>(app: &AppHandle<R>, settings: &LocalApiSettings) -> anyhow::Result<()> {
    let mut server = SERVER.lock().map_err(|e| anyhow!("{e}"))?;
    let want = settings.enabled.then_some(settings.port);
    if server.as_ref().map(|s| s.port) == want {
        return Ok(());
    }
    if let Some(old) = server.take() {
        let _ = old.shutdown.send(());
        log::info!("Local API stopped (port {})", old.port);
    }
    let Some(port) = want else { return Ok(()) };
    if settings.token.is_empty() {
        bail!("No API token");
    }

    let listener = bind(port)?;
    let (shutdown, stop) = oneshot::channel::<()>();
    let router = router(app.clone());
    tauri::async_runtime::spawn(async move {
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(l) => l,
            Err(e) => {
                log::error!("Local API: {e}");
                return;
            }
        };
        let serve = axum::serve(listener, router).with_graceful_shutdown(async {
            stop.await.ok();
        });
        if let Err(e) = serve.await {
            log::error!("Local API server failed: {e}");
        }
    });
    log::info!("Local API listening on http://127.0.0.1:{port}");
    *server = Some(Server { port, shutdown });
    Ok(())
}

/// Bind 127.0.0.1:`port`, retrying briefly: a server just told to stop on the
/// same port may not have closed its socket yet.
fn bind(port: u16) -> anyhow::Result<std::net::TcpListener> {
    let mut attempt = 0;
    loop {
        match std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, port)) {
            Ok(l) => {
                l.set_nonblocking(true)?;
                return Ok(l);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse && attempt < 10 => {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(e).with_context(|| format!("Can't listen on port {port}")),
        }
    }
}

fn router<R: Runtime>(app: AppHandle<R>) -> Router {
    Router::new()
        .route("/health", get(health::<R>))
        .route("/v1/audio/transcriptions", post(transcriptions::<R>))
        .route("/v1/realtime", get(realtime::handler::<R>))
        .layer(middleware::from_fn_with_state(app.clone(), auth::<R>))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .with_state(app)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.0.is_client_error() { "invalid_request_error" } else { "server_error" };
        (self.0, Json(json!({ "error": { "message": self.1, "type": kind } }))).into_response()
    }
}

fn bad_request(msg: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.to_string())
}

fn internal(msg: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
}

async fn auth<R: Runtime>(State(app): State<AppHandle<R>>, req: Request, next: Next) -> Response {
    let expected = app
        .state::<AppState>()
        .settings
        .lock()
        .map(|s| s.local_api.token.clone())
        .unwrap_or_default();
    let headers = req.headers();
    if headers.contains_key(header::ORIGIN) {
        return ApiError(StatusCode::FORBIDDEN, "Requests from web pages are not allowed".into()).into_response();
    }
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let subprotocol = headers
        .get_all(header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .find_map(|p| p.trim().strip_prefix("openai-insecure-api-key."));
    let given = bearer.or(subprotocol).unwrap_or("");
    if expected.is_empty() || !constant_time_eq(given.as_bytes(), expected.as_bytes()) {
        return ApiError(StatusCode::UNAUTHORIZED, "Missing or wrong API token".into()).into_response();
    }
    next.run(req).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn health<R: Runtime>(State(app): State<AppHandle<R>>) -> Response {
    let state = app.state::<AppState>();
    // Don't wait for a transcription in progress: report it as busy.
    let body = match state.asr_engine.try_lock() {
        Ok(engine) => json!({
            "status": "ok",
            "model": engine.active_model_name(),
            "busy": false,
        }),
        Err(_) => json!({ "status": "ok", "busy": true }),
    };
    Json(body).into_response()
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Json,
    Text,
    VerboseJson,
}

async fn transcriptions<R: Runtime>(
    State(app): State<AppHandle<R>>,
    mut form: Multipart,
) -> Result<Response, ApiError> {
    let mut audio = None;
    let mut language = None;
    let mut format = Format::Json;
    while let Some(field) = form.next_field().await.map_err(bad_request)? {
        match field.name() {
            Some("file") => audio = Some(field.bytes().await.map_err(bad_request)?),
            Some("language") => language = Some(field.text().await.map_err(bad_request)?),
            Some("response_format") => {
                format = match field.text().await.map_err(bad_request)?.as_str() {
                    "json" => Format::Json,
                    "text" => Format::Text,
                    "verbose_json" => Format::VerboseJson,
                    other => {
                        return Err(bad_request(format!(
                            "Unsupported response_format '{other}' (use json, text or verbose_json)"
                        )))
                    }
                }
            }
            _ => {}
        }
    }
    let audio = audio.ok_or_else(|| bad_request("Missing 'file' field"))?;

    let started = std::time::Instant::now();
    let out = tauri::async_runtime::spawn_blocking(move || transcribe(&app, &audio, language))
        .await
        .map_err(internal)??;
    log::info!(
        "Local API: {:.1} s of audio transcribed in {} ms ({} chars)",
        out.duration,
        started.elapsed().as_millis(),
        out.text.chars().count()
    );

    Ok(match format {
        Format::Text => out.text.into_response(),
        Format::Json => Json(json!({ "text": out.text })).into_response(),
        Format::VerboseJson => {
            let segments: Vec<_> = out
                .segments
                .iter()
                .enumerate()
                .map(|(id, s)| json!({ "id": id, "start": s.start, "end": s.end, "text": s.text }))
                .collect();
            Json(json!({
                "task": "transcribe",
                "language": out.language,
                "duration": out.duration,
                "model": out.model,
                "text": out.text,
                "segments": segments,
            }))
            .into_response()
        }
    })
}

struct Transcript {
    text: String,
    language: String,
    duration: f64,
    model: String,
    segments: Vec<TranscriptionSegment>,
}

fn transcribe<R: Runtime>(app: &AppHandle<R>, audio: &[u8], language: Option<String>) -> Result<Transcript, ApiError> {
    let samples = decode(audio).map_err(|e| bad_request(format!("Can't read the audio: {e:#}")))?;
    let language = resolve_language(app, language).map_err(internal)?;
    transcribe_samples(app, &samples, &language).map_err(internal)
}

/// The requested language, or the app's setting when none was given.
fn resolve_language<R: Runtime>(app: &AppHandle<R>, language: Option<String>) -> anyhow::Result<String> {
    match language.filter(|l| !l.is_empty()) {
        Some(l) => Ok(l),
        None => Ok(app.state::<AppState>().settings.lock().map_err(|e| anyhow!("{e}"))?.language.clone()),
    }
}

/// Transcribe a whole 16 kHz mono buffer, sharing the engine: up to 28 s is
/// one VAD-gated call (like a hotkey recording, so a silent clip comes back
/// empty instead of hallucinated); longer audio is cut like a voice memo,
/// without the VAD, taking the engine lock per chunk.
fn transcribe_samples<R: Runtime>(
    app: &AppHandle<R>,
    samples: &[f32],
    language: &str,
) -> anyhow::Result<Transcript> {
    let state = app.state::<AppState>();
    let lang = (language != "auto").then_some(language);
    let mut model = String::new();
    let mut detected = None;
    let rate = crate::vad::VAD_SAMPLE_RATE as f64;
    let segments = if samples.len() <= memo::MAX_CHUNK_SAMPLES {
        let mut engine = state.asr_engine.lock().map_err(|e| anyhow!("{e}"))?;
        model = engine.active_model_name().to_string();
        let r = engine.transcribe_samples(samples, lang, true)?;
        detected = Some(r.language);
        let text = r.text.trim();
        if text.is_empty() {
            Vec::new()
        } else {
            vec![TranscriptionSegment { start: 0.0, end: samples.len() as f64 / rate, text: text.to_string() }]
        }
    } else {
        memo::transcribe_chunked(
            samples.iter().copied(),
            samples.len(),
            |chunk| {
                let mut engine = state.asr_engine.lock().map_err(|e| anyhow!("{e}"))?;
                model = engine.active_model_name().to_string();
                let r = engine.transcribe_samples(chunk, lang, false)?;
                detected.get_or_insert(r.language);
                Ok(r.text)
            },
            |_| {},
        )?
    };

    let mut text = String::new();
    for s in &segments {
        partial::append_text(&mut text, &s.text);
    }
    Ok(Transcript {
        text,
        language: detected.unwrap_or_else(|| language.to_string()),
        duration: samples.len() as f64 / rate,
        model,
        segments,
    })
}

// ---------------------------------------------------------------------------
// Audio decoding
// ---------------------------------------------------------------------------

/// Removes the file when dropped.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn temp_path(ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("echo-api-{}.{ext}", uuid::Uuid::new_v4().simple()))
}

/// Uploaded audio → 16 kHz mono f32. WAV is read directly; anything else goes
/// through macOS's `afconvert` first.
fn decode(audio: &[u8]) -> anyhow::Result<Vec<f32>> {
    let input = TempFile(temp_path("audio"));
    std::fs::write(&input.0, audio)?;
    if let Ok(samples) = read_wav(&input.0) {
        return Ok(samples);
    }
    let wav = TempFile(temp_path("wav"));
    let out = std::process::Command::new("/usr/bin/afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
        .arg(&input.0)
        .arg(&wav.0)
        .output()
        .context("Failed to run afconvert")?;
    if !out.status.success() {
        bail!(
            "unsupported format (use WAV, m4a, mp3, aac, flac or caf): {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    read_wav(&wav.0)
}

fn read_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let path = path.to_str().ok_or_else(|| anyhow!("non-UTF-8 temp path"))?;
    rust_asr::read_wav_16k_mono(path).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison() {
        assert!(constant_time_eq(b"echo_abc", b"echo_abc"));
        assert!(!constant_time_eq(b"echo_abc", b"echo_abd"));
        assert!(!constant_time_eq(b"echo_abc", b"echo_ab"));
        assert!(!constant_time_eq(b"", b"x"));
    }

    #[test]
    fn decodes_wav_and_converts_other_formats() {
        // 1 s of a 440 Hz tone at 48 kHz stereo WAV → 16 kHz mono.
        let path = temp_path("wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..48_000 {
            let s = ((i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 8000.0) as i16;
            w.write_sample(s).unwrap();
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        let wav = std::fs::read(&path).unwrap();
        let _cleanup = TempFile(path.clone());
        let samples = decode(&wav).unwrap();
        assert!((samples.len() as i64 - 16_000).abs() < 200, "{}", samples.len());

        // The same audio as AAC in m4a, through afconvert.
        let m4a = TempFile(temp_path("m4a"));
        let ok = std::process::Command::new("/usr/bin/afconvert")
            .args(["-f", "m4af", "-d", "aac"])
            .arg(&path)
            .arg(&m4a.0)
            .status()
            .unwrap()
            .success();
        assert!(ok);
        let samples = decode(&std::fs::read(&m4a.0).unwrap()).unwrap();
        assert!((samples.len() as i64 - 16_000).abs() < 2_000, "{}", samples.len());

        assert!(decode(b"not audio").is_err());
    }

    /// A mock app with a real engine (`ECHO_API_MODEL`, default Parakeet) and
    /// the API running on `port`. Returns the app, its engine, the token and
    /// the temp dir holding its database.
    pub(crate) fn start_test_server(
        port: u16,
    ) -> (
        tauri::App<tauri::test::MockRuntime>,
        std::sync::Arc<Mutex<crate::ASREngine>>,
        String,
        tempfile::TempDir,
    ) {
        use std::sync::Arc;
        let model = std::env::var("ECHO_API_MODEL")
            .unwrap_or_else(|_| "mlx-community/parakeet-tdt_ctc-0.6b-ja".to_string());
        let hub = PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Caches/io.qluto.echo/huggingface/hub");
        let mut engine = crate::ASREngine::with_hub(hub);
        engine.set_model(&model).unwrap();
        engine.load_model().unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let dir = tempfile::tempdir().unwrap();
        let token = new_token();
        let mut settings = crate::Settings::default();
        settings.language = "ja".to_string();
        settings.local_api = LocalApiSettings { enabled: true, port, token: token.clone() };
        let app = tauri::test::mock_app();
        app.manage(AppState {
            asr_engine: engine.clone(),
            settings: Mutex::new(settings.clone()),
            recording_state: Mutex::new(Default::default()),
            transcription_db: Arc::new(Mutex::new(
                crate::database::TranscriptionDb::open(&dir.path().join("t.db")).unwrap(),
            )),
            continuous_pipeline: Mutex::new(None),
            memo_recorder: Mutex::new(None),
            meeting: Mutex::new(None),
        });
        apply(app.handle(), &settings.local_api).unwrap();
        assert!(status(&settings.local_api).running);
        (app, engine, token, dir)
    }

    /// Starts the server on a mock app with a real engine and sends it audio
    /// with curl, while another thread takes the engine lock the way hotkey
    /// dictation does. Audio: `say -v Kyoko -o x.aiff "…"`, then
    /// `afconvert -f WAVE -d LEI16@16000 -c 1` (short WAV) or
    /// `afconvert -f m4af -d aac` (long m4a, over 28 s).
    ///
    /// ECHO_API_SHORT_WAV=… ECHO_API_LONG_AUDIO=… cargo test api_end_to_end -- --ignored --nocapture
    #[test]
    #[ignore]
    fn api_end_to_end() {
        use std::sync::Arc;
        use std::time::Instant;

        let short = std::env::var("ECHO_API_SHORT_WAV").expect("set ECHO_API_SHORT_WAV");
        let long = std::env::var("ECHO_API_LONG_AUDIO").expect("set ECHO_API_LONG_AUDIO");
        let port = 47_799;
        let (app, engine, token, dir) = start_test_server(port);
        let settings_api = LocalApiSettings { enabled: true, port, token: token.clone() };
        let url = format!("http://127.0.0.1:{port}");
        let curl = |args: &[&str]| -> (u16, String) {
            let out = std::process::Command::new("curl")
                .args(["-s", "-w", "\n%{http_code}"])
                .args(args)
                .output()
                .unwrap();
            let out = String::from_utf8(out.stdout).unwrap();
            let (body, code) = out.rsplit_once('\n').unwrap();
            (code.parse().unwrap(), body.to_string())
        };
        let bearer = format!("Authorization: Bearer {token}");
        let transcriptions = format!("{url}/v1/audio/transcriptions");

        // Auth.
        assert_eq!(curl(&[&format!("{url}/health")]).0, 401);
        assert_eq!(curl(&["-H", "Authorization: Bearer nope", &format!("{url}/health")]).0, 401);
        let (code, body) = curl(&["-H", &bearer, &format!("{url}/health")]);
        println!("health: {body}");
        assert_eq!(code, 200);

        // Short WAV, plain text.
        let t = Instant::now();
        let (code, body) = curl(&[
            "-H", &bearer, "-F", &format!("file=@{short}"), "-F", "model=whisper-1",
            "-F", "response_format=text", &transcriptions,
        ]);
        println!("short ({:?}): {code} {body}", t.elapsed());
        assert_eq!(code, 200);
        assert!(!body.trim().is_empty());

        // Bad requests.
        assert_eq!(curl(&["-H", &bearer, "-F", "language=ja", &transcriptions]).0, 400);
        let junk = dir.path().join("junk.bin");
        std::fs::write(&junk, b"not audio").unwrap();
        let (code, body) = curl(&["-H", &bearer, "-F", &format!("file=@{}", junk.display()), &transcriptions]);
        println!("junk: {code} {body}");
        assert_eq!(code, 400);

        // Long audio, while "dictation" keeps taking the engine lock.
        let short_samples = rust_asr::read_wav_16k_mono(&short).unwrap();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dictation = {
            let engine = engine.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                let mut waits = Vec::new();
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    let t = Instant::now();
                    let mut e = engine.lock().unwrap();
                    waits.push(t.elapsed());
                    let r = e.transcribe_samples(&short_samples, Some("ja"), true).unwrap();
                    assert!(!r.text.trim().is_empty());
                    drop(e);
                    std::thread::sleep(Duration::from_millis(300));
                }
                waits
            })
        };
        let t = Instant::now();
        let (code, body) = curl(&[
            "-H", &bearer, "-F", &format!("file=@{long}"), "-F", "response_format=verbose_json",
            &transcriptions,
        ]);
        let api_time = t.elapsed();
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let waits = dictation.join().unwrap();
        println!("long ({api_time:?}): {code} {body}");
        println!(
            "dictation during the request: {} runs, longest wait for the engine {:?}",
            waits.len(),
            waits.iter().max().unwrap()
        );
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["segments"].as_array().unwrap().len() >= 2);
        assert!(waits.len() >= 2, "dictation should run while the API works");

        // Off: the port closes.
        apply(app.handle(), &LocalApiSettings { enabled: false, ..settings_api }).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(curl(&["-H", &bearer, &format!("{url}/health")]).0, 0);
    }
}
