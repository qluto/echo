//! Updates the installed app from GitHub Releases (`latest.json`, signed with
//! the updater key). The download and the install happen in the background;
//! the restart waits until nothing would be cut off: no dictation or meeting,
//! no memo still being processed, no talk heard by meeting standby, and no
//! input from the user for a while. The user can also check now (tray menu,
//! Settings); the answer comes as a notice under the menu bar icon.

use std::{path::PathBuf, process::Command, sync::Mutex, thread, time::Duration};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_store::StoreExt;
use tauri_plugin_updater::UpdaterExt;

use crate::types::SETTINGS_STORE_FILE;
use crate::AppState;

const FIRST_CHECK: Duration = Duration::from_secs(60);
const CHECK_EVERY: Duration = Duration::from_secs(6 * 3600);
const RETRY_RESTART: Duration = Duration::from_secs(60);
/// No keyboard or mouse input — and no talk for meeting standby — for this long counts as away.
const QUIET_SECONDS: f64 = 120.0;
/// Store key: the version the app last ran as.
const VERSION_KEY: &str = "last_version";
/// The version installed into the bundle and waiting for a restart.
static READY: Mutex<Option<String>> = Mutex::new(None);
/// Held while checking / installing, so a check by hand and the periodic one
/// never download twice.
static CHECKING: Mutex<()> = Mutex::new(());
/// Set at startup when this run is a newer version than the last one.
static JUST_UPDATED: Mutex<Option<JustUpdated>> = Mutex::new(None);

/// Only the app in /Applications updates itself: a dev build or an app still
/// running from the dmg would replace the wrong copy.
fn installed() -> bool {
    bundle().is_some_and(|b| b.starts_with("/Applications"))
}

pub fn start(app: AppHandle) {
    if !installed() {
        log::info!("Auto-update off: not running from /Applications");
        return;
    }
    thread::spawn(move || {
        thread::sleep(FIRST_CHECK);
        loop {
            match check(&app) {
                Ok(CheckResult::Installed { .. }) => return, // restarting when quiet
                Ok(_) => {}
                Err(e) => log::warn!("Update check failed: {e}"),
            }
            thread::sleep(CHECK_EVERY);
        }
    });
}

/// The outcome of an update check.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CheckResult {
    /// No newer release.
    UpToDate { current: String },
    /// A newer version is installed and waits for a restart.
    Installed { version: String },
    /// Not the copy in /Applications, which is the only one that updates.
    Unsupported,
}

/// Check now, and install a newer version (once: a later check reports the
/// one already waiting). Blocks until the download is done.
fn check(app: &AppHandle) -> Result<CheckResult, String> {
    if !installed() {
        return Ok(CheckResult::Unsupported);
    }
    let _checking = CHECKING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(version) = READY.lock().unwrap().clone() {
        return Ok(CheckResult::Installed { version });
    }
    match tauri::async_runtime::block_on(install(app)).map_err(|e| e.to_string())? {
        Some(version) => {
            *READY.lock().unwrap() = Some(version.clone());
            let _ = app.emit("update-ready", version.clone());
            let app = app.clone();
            thread::spawn(move || restart_when_quiet(&app));
            Ok(CheckResult::Installed { version })
        }
        None => Ok(CheckResult::UpToDate { current: app.package_info().version.to_string() }),
    }
}

/// Check for an update now (Settings). Off the main thread: it may download.
#[tauri::command(async)]
pub fn check_for_update(app: AppHandle) -> Result<CheckResult, String> {
    check(&app)
}

/// Check from the tray menu and answer with a notice under the icon.
pub fn check_and_notify(app: &AppHandle) {
    let app = app.clone();
    thread::spawn(move || {
        use crate::notice::{show, NoticeAction};
        match check(&app) {
            Ok(CheckResult::UpToDate { current }) => {
                show(&app, "Echo は最新です", Some(format!("バージョン {current}")), None, 5000)
            }
            Ok(CheckResult::Installed { version }) => show(
                &app,
                &format!("Echo {version} をインストールしました"),
                Some("再起動すると新しいバージョンになります。手が空いたときに自動でも再起動します。".into()),
                Some(NoticeAction::RestartForUpdate),
                12000,
            ),
            Ok(CheckResult::Unsupported) => show(
                &app,
                "アップデートを確認できません",
                Some("/Applications にインストールした Echo だけがアップデートできます。".into()),
                None,
                6000,
            ),
            Err(e) => {
                log::warn!("Update check failed: {e}");
                show(&app, "アップデートを確認できませんでした", Some(e), None, 8000)
            }
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
        && crate::standby::seconds_since_talk() >= QUIET_SECONDS
        && !busy(app)
}

/// Something a restart would cut off.
fn busy(app: &AppHandle) -> bool {
    let state = app.state::<AppState>();
    let dictating = state.recording_state.lock().map_or(true, |r| r.is_recording);
    let in_meeting = state.meeting.lock().map_or(true, |m| m.as_ref().is_some_and(|m| m.is_active()));
    let memo_processing = state.transcription_db.lock().map_or(true, |db| db.has_memo_in_progress().unwrap_or(true));
    dictating || in_meeting || memo_processing
}

/// Quit and start the bundle again. The new copy is started once this
/// process has exited, so the two never hold the hotkey or the database at
/// the same time; `open` starts it through Launch Services like a normal launch.
fn relaunch(app: &AppHandle) -> Result<(), String> {
    let bundle = bundle().ok_or("not running from an app bundle")?;
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

/// This run is a newer version than the last: what the UI says after an update.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JustUpdated {
    pub version: String,
    pub previous: String,
}

/// At startup: remember the version, and whether it is newer than the last
/// run's (then say so under the menu bar icon, with the changelog a click away).
/// A first install says nothing, and nor does running an older build.
pub fn note_version(app: &AppHandle) {
    let current = app.package_info().version.to_string();
    let Ok(store) = app.store(SETTINGS_STORE_FILE) else { return };
    let previous = store.get(VERSION_KEY).and_then(|v| v.as_str().map(String::from));
    if previous.as_deref() != Some(current.as_str()) {
        if let Err(e) = save(app, VERSION_KEY, serde_json::json!(current)) {
            log::warn!("Failed to save {VERSION_KEY}: {e}");
        }
    }
    let Some(updated) = changed(previous.as_deref(), &current) else { return };
    log::info!("Updated from {} to {}", updated.previous, updated.version);
    *JUST_UPDATED.lock().unwrap() = Some(updated.clone());
    let app = app.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(3)); // the menu bar icon and the popup load first
        crate::notice::show(
            &app,
            &format!("Echo {} にアップデートしました", updated.version),
            Some("変更点を見るには、ここをクリックしてください。".into()),
            Some(crate::notice::NoticeAction::ShowChangelog),
            15000,
        );
    });
}

/// The version the app just moved to (this run), so the UI can say so.
#[tauri::command]
pub fn just_updated() -> Option<JustUpdated> {
    JUST_UPDATED.lock().unwrap().clone()
}

fn changed(previous: Option<&str>, current: &str) -> Option<JustUpdated> {
    let previous = previous?;
    (parse(current)? > parse(previous)?).then(|| JustUpdated { version: current.into(), previous: previous.into() })
}

fn parse(version: &str) -> Option<(u32, u32, u32)> {
    let mut parts = version.trim_start_matches('v').split('.').map(|p| p.parse().ok());
    Some((parts.next()??, parts.next()??, parts.next()??))
}

#[cfg(test)]
mod tests {
    use super::{changed, JustUpdated};

    #[test]
    fn only_a_newer_version_is_announced() {
        let updated = |v: &str, p: &str| Some(JustUpdated { version: v.into(), previous: p.into() });
        assert_eq!(changed(Some("0.13.2"), "0.13.3"), updated("0.13.3", "0.13.2"));
        assert_eq!(changed(Some("0.9.0"), "0.10.0"), updated("0.10.0", "0.9.0"));
        assert_eq!(changed(Some("0.13.3"), "0.13.3"), None);
        assert_eq!(changed(None, "0.13.3"), None);
        // A local build (0.1.0) run after a release
        assert_eq!(changed(Some("0.15.0"), "0.1.0"), None);
    }
}
