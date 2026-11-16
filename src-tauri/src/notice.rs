//! A small popup just below Echo's menu bar icon, for things the user would
//! otherwise miss: a meeting standby started, an update. It shows on every
//! desktop (Space) and over full-screen apps, never takes focus, and hides
//! itself after a while (`NoticeApp.tsx`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, LogicalSize, Manager, PhysicalPosition, WebviewWindow};

/// The menu bar icon's id (`setup_system_tray`).
pub const TRAY_ID: &str = "echo";
const LABEL: &str = "notice";
/// Window size (logical px). The card inside leaves room for its shadow.
const WIDTH: f64 = 340.0;
const HEIGHT: f64 = 128.0;
/// Distance below the menu bar icon (logical px).
const GAP: f64 = 2.0;

/// What the popup's button does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoticeAction {
    OpenMeeting,
    RestartForUpdate,
    ShowChangelog,
}

#[derive(Debug, Clone, Serialize)]
pub struct Notice {
    /// Tells notices apart, so a timer of an older one can't hide a newer one.
    pub id: u64,
    pub title: String,
    pub body: Option<String>,
    pub action: Option<NoticeAction>,
    /// Hide after this long unless the pointer is over it.
    pub timeout_ms: u64,
}

static CURRENT: Mutex<Option<Notice>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Show a notice under the menu bar icon (replacing one already shown).
pub fn show(app: &AppHandle, title: &str, body: Option<String>, action: Option<NoticeAction>, timeout_ms: u64) {
    let notice = Notice {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        title: title.to_string(),
        body,
        action,
        timeout_ms,
    };
    log::info!("Notice: {}", notice.title);
    *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(notice.clone());
    let handle = app.clone();
    let shown = app.run_on_main_thread(move || {
        let Some(window) = handle.get_webview_window(LABEL) else {
            log::error!("Notice window not found");
            return;
        };
        place(&handle, &window);
        if let Err(e) = handle.emit_to(LABEL, "notice", notice) {
            log::warn!("Failed to emit notice: {e}");
        }
        order_front(&window);
    });
    if let Err(e) = shown {
        log::warn!("Failed to show notice: {e}");
    }
}

/// Centered under the menu bar icon, kept on its screen; top right of the
/// primary screen if the icon can't be found.
fn place(app: &AppHandle, window: &WebviewWindow) {
    let _ = window.set_size(LogicalSize::new(WIDTH, HEIGHT));
    let scale = window.scale_factor().unwrap_or(2.0);
    let (w, h) = (WIDTH * scale, HEIGHT * scale);
    let icon = app.tray_by_id(TRAY_ID).and_then(|t| t.rect().ok().flatten());
    let (mut x, mut y) = match icon {
        Some(rect) => {
            let pos = rect.position.to_physical::<f64>(scale);
            let size = rect.size.to_physical::<f64>(scale);
            (pos.x + size.width / 2.0 - w / 2.0, pos.y + size.height + GAP * scale)
        }
        None => match window.primary_monitor().ok().flatten() {
            Some(m) => {
                let (mx, my) = (m.position().x as f64, m.position().y as f64);
                (mx + m.size().width as f64 - w - 8.0 * scale, my + 32.0 * scale)
            }
            None => (0.0, 0.0),
        },
    };
    if let Ok(Some(m)) = window.monitor_from_point(x + w / 2.0, y) {
        let (mx, my) = (m.position().x as f64, m.position().y as f64);
        let (mw, mh) = (m.size().width as f64, m.size().height as f64);
        x = x.clamp(mx, (mx + mw - w).max(mx));
        y = y.clamp(my, (my + mh - h).max(my));
    }
    let _ = window.set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32));
}

/// Bring the window up without activating Echo or taking focus.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn order_front(window: &WebviewWindow) {
    use cocoa::base::id;
    if let Ok(ns_window) = window.ns_window() {
        unsafe {
            let _: () = msg_send![ns_window as id, orderFrontRegardless];
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn order_front(window: &WebviewWindow) {
    let _ = window.show();
}

fn hide(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(LABEL) {
        let _ = window.hide();
    }
}

/// The notice shown now, for a popup that (re)loaded after it was sent.
#[tauri::command]
pub fn get_notice() -> Option<Notice> {
    CURRENT.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Hide the popup, if `id` is still the notice it shows.
#[tauri::command]
pub fn dismiss_notice(id: u64, app: AppHandle) {
    let mut current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
    if current.as_ref().is_some_and(|n| n.id == id) {
        *current = None;
        drop(current);
        hide(&app);
    }
}

/// The popup's button: do it and hide the popup.
#[tauri::command(async)]
pub fn notice_action(action: NoticeAction, app: AppHandle) -> Result<(), String> {
    *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = None;
    hide(&app);
    match action {
        NoticeAction::OpenMeeting => crate::commands::open_meeting_window(app),
        NoticeAction::RestartForUpdate => crate::update::restart_for_update(app),
        NoticeAction::ShowChangelog => {
            crate::show_main_window(&app);
            app.emit_to("main", "show-changelog", ()).map_err(|e| e.to_string())
        }
    }
}
