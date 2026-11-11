//! Updates the installed app from GitHub Releases (`latest.json`, signed with
//! the updater key). The download and the install happen in the background;
//! the restart waits until nothing would be cut off: no dictation, memo or
//! meeting, no memo still being processed, no sound for always-on to
//! transcribe, and no input from the user for a while. Always-on listening
//! that was on before the restart is turned back on after it.

use std::{path::PathBuf, process::Command, sync::Mutex, thread, time::Duration};

use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_store::StoreExt;
use tauri_plugin_updater::UpdaterExt;

use crate::types::SETTINGS_STORE_FILE;
use crate::AppState;

const FIRST_CHECK: Duration = Duration::from_secs(60);
const CHECK_EVERY: Duration = Duration::from_secs(6 * 3600);
const RETRY_RESTART: Duration = Duration::from_secs(60);
/// No keyboard or mouse input — and no sound for always-on — for this long counts as away.
const QUIET_SECONDS: f64 = 120.0;
/// Store keys: the version the app last ran as, and whether to restart always-on.
const VERSION_KEY: &str = "last_version";
const RESUME_LISTENING_KEY: &str = "resume_listening";
/// The version installed into the bundle and waiting for a restart.
static READY: Mutex<Option<String>> = Mutex::new(None);

/// Only the app in /Applications updates itself: a dev build or an app still
/// running from the dmg would replace the wrong copy.
pub fn start(app: AppHandle) {
    if !bundle().is_some_and(|b| b.starts_with("/Applications")) {
        log::info!("Auto-update off: not running from /Applications");
        return;
    }
    thread::spawn(move || {
        thread::sleep(FIRST_CHECK);
        loop {
            match tauri::async_runtime::block_on(install(&app)) {
                Ok(Some(version)) => {
                    *READY.lock().unwrap() = Some(version.clone());
                    let _ = app.emit("update-ready", version);
                    return restart_when_quiet(&app);
                }
                Ok(None) => {}
                Err(e) => log::warn!("Update check failed: {e}"),
            }
            thread::sleep(CHECK_EVERY);
        }
    });
}

/// The newer version, once it is installed into the app bundle.
async fn install(app: &AppHandle) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(update) = app.updater()?.check().await? else { return Ok(None) };
    log::info!("Installing update {}", update.version);
    update.download_and_install(|_, _| {}, || {}).await?;
    Ok(Some(update.version))
}

fn restart_when_quiet(app: &AppHandle) {
    while !quiet(app) {
        thread::sleep(RETRY_RESTART);
    }
    if let Err(e) = relaunch(app) {
        log::error!("Update restart failed: {e}");
    }
}

fn quiet(app: &AppHandle) -> bool {
    idle_seconds() >= QUIET_SECONDS
        && crate::continuous::seconds_since_sound() >= QUIET_SECONDS
        && !busy(app)
}

/// Something a restart would cut off.
fn busy(app: &AppHandle) -> bool {
    let state = app.state::<AppState>();
    let dictating = state.recording_state.lock().map_or(true, |r| r.is_recording);
    let memo_recording = state.memo_recorder.lock().map_or(true, |r| r.is_some());
    let in_meeting = state.meeting.lock().map_or(true, |m| m.as_ref().is_some_and(|m| m.is_active()));
    let memo_processing = state.transcription_db.lock().map_or(true, |db| db.has_memo_in_progress().unwrap_or(true));
    dictating || memo_recording || in_meeting || memo_processing
}

/// Quit and start the bundle again. The new copy is started once this
/// process has exited, so the two never hold the hotkey or the database at
/// the same time; `open` starts it through Launch Services like a normal launch.
fn relaunch(app: &AppHandle) -> Result<(), String> {
    let bundle = bundle().ok_or("not running from an app bundle")?;
    let listening = app.state::<AppState>().continuous_pipeline.lock().is_ok_and(|p| p.is_some());
    save(app, RESUME_LISTENING_KEY, serde_json::json!(listening))?;
    Command::new("/bin/sh")
        .args(["-c", "while kill -0 \"$1\" 2>/dev/null; do sleep 0.2; done; open \"$2\"", "sh"])
        .arg(std::process::id().to_string())
        .arg(&bundle)
        .spawn()
        .map_err(|e| format!("could not start the relauncher: {e}"))?;
    log::info!("Restarting into the updated app");
    app.exit(0);
    Ok(())
}

fn bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().find(|p| p.extension().is_some_and(|e| e == "app")).map(PathBuf::from)
}

fn idle_seconds() -> f64 {
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(state: i32, event_type: u32) -> f64;
    }
    const COMBINED_SESSION_STATE: i32 = 0;
    const ANY_INPUT_EVENT: u32 = u32::MAX;
    unsafe { CGEventSourceSecondsSinceLastEventType(COMBINED_SESSION_STATE, ANY_INPUT_EVENT) }
}

fn save(app: &AppHandle, key: &str, value: serde_json::Value) -> Result<(), String> {
    let store = app.store(SETTINGS_STORE_FILE).map_err(|e| e.to_string())?;
    store.set(key, value);
    store.save().map_err(|e| e.to_string())
}

/// After a restart for an update, turn always-on listening back on. It waits
/// for the microphone, so it runs off the setup thread; the frontend learns
/// the new status from `continuous-listening-resumed`.
pub fn resume_listening(app: &AppHandle) {
    let resume = app
        .store(SETTINGS_STORE_FILE)
        .ok()
        .and_then(|s| s.get(RESUME_LISTENING_KEY))
        .is_some_and(|v| v.as_bool() == Some(true));
    if !resume {
        return;
    }
    if let Err(e) = save(app, RESUME_LISTENING_KEY, serde_json::json!(false)) {
        log::warn!("Failed to clear {RESUME_LISTENING_KEY}: {e}");
    }
    let app = app.clone();
    thread::spawn(move || {
        match crate::commands::start_continuous_listening(app.clone(), app.state()) {
            Ok(status) => {
                log::info!("Always-on listening resumed after the update");
                let _ = app.emit("continuous-listening-resumed", status);
            }
            Err(e) => log::error!("Failed to resume always-on listening after the update: {e}"),
        }
    });
}

/// The installed version waiting for a restart, if any.
#[tauri::command]
pub fn update_ready() -> Option<String> {
    READY.lock().unwrap().clone()
}

/// Switch to the installed version now instead of waiting for a quiet moment.
#[tauri::command]
pub fn restart_for_update(app: AppHandle) -> Result<(), String> {
    if READY.lock().unwrap().is_none() {
        return Err("No update is waiting".into());
    }
    if busy(&app) {
        return Err("Finish the recording, meeting or memo first".into());
    }
    relaunch(&app)
}

/// The version the app just moved to, once, so the UI can say so after an
/// update. A first install says nothing.
#[tauri::command]
pub fn just_updated(app: AppHandle) -> Option<String> {
    let current = app.package_info().version.to_string();
    let store = app.store(SETTINGS_STORE_FILE).ok()?;
    let previous = store.get(VERSION_KEY).and_then(|v| v.as_str().map(String::from));
    if previous.as_deref() != Some(current.as_str()) {
        if let Err(e) = save(&app, VERSION_KEY, serde_json::json!(current)) {
            log::warn!("Failed to save {VERSION_KEY}: {e}");
        }
    }
    changed(previous.as_deref(), &current)
}

fn changed(previous: Option<&str>, current: &str) -> Option<String> {
    previous.filter(|p| *p != current).map(|_| current.to_string())
}

#[cfg(test)]
mod tests {
    use super::changed;

    #[test]
    fn only_a_version_change_is_announced() {
        assert_eq!(changed(Some("0.13.2"), "0.13.3"), Some("0.13.3".into()));
        assert_eq!(changed(Some("0.13.3"), "0.13.3"), None);
        assert_eq!(changed(None, "0.13.3"), None);
    }
}
