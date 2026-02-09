//! Active application detection for macOS
//!
//! Detects the frontmost application using NSWorkspace API.

use serde::{Deserialize, Serialize};

/// Information about the currently active (frontmost) application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAppInfo {
    /// Bundle identifier (e.g., "com.apple.mail", "notion.id")
    pub bundle_id: Option<String>,
    /// Application name (e.g., "Mail", "Notion")
    pub app_name: Option<String>,
}

impl Default for ActiveAppInfo {
    fn default() -> Self {
        Self {
            bundle_id: None,
            app_name: None,
        }
    }
}

/// Get information about the frontmost application on macOS
#[cfg(target_os = "macos")]
pub fn get_frontmost_app() -> ActiveAppInfo {
    use cocoa::base::{id, nil};

    unsafe {
        // Get shared NSWorkspace
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        if workspace == nil {
            log::warn!("Failed to get NSWorkspace");
            return ActiveAppInfo::default();
        }

        // Get frontmost application (NSRunningApplication)
        let frontmost_app: id = msg_send![workspace, frontmostApplication];
        if frontmost_app == nil {
            log::warn!("Failed to get frontmost application");
            return ActiveAppInfo::default();
        }

        // Get bundle identifier
        let bundle_id_ns: id = msg_send![frontmost_app, bundleIdentifier];
        let bundle_id = nsstring_to_string(bundle_id_ns);

        // Get localized name
        let app_name_ns: id = msg_send![frontmost_app, localizedName];
        let app_name = nsstring_to_string(app_name_ns);

        log::debug!(
            "Frontmost app: name={:?}, bundle_id={:?}",
            app_name,
            bundle_id
        );

        ActiveAppInfo {
            bundle_id,
            app_name,
        }
    }
}

/// Helper to convert NSString to Rust String
#[cfg(target_os = "macos")]
unsafe fn nsstring_to_string(ns_string: cocoa::base::id) -> Option<String> {
    use cocoa::base::nil;
    use std::ffi::CStr;

    if ns_string == nil {
        return None;
    }

    let c_str: *const i8 = msg_send![ns_string, UTF8String];
    if c_str.is_null() {
        return None;
    }

    CStr::from_ptr(c_str).to_str().ok().map(|s| s.to_string())
}

/// Fallback for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub fn get_frontmost_app() -> ActiveAppInfo {
    ActiveAppInfo::default()
}

/// Activate (bring to front) an application by its bundle identifier.
/// Returns true if the app was found and activated.
#[cfg(target_os = "macos")]
pub fn activate_app_by_bundle_id(bundle_id: &str) -> bool {
    use cocoa::base::{id, nil};

    unsafe {
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        if workspace == nil {
            log::warn!("Failed to get NSWorkspace for app activation");
            return false;
        }

        let running_apps: id = msg_send![workspace, runningApplications];
        if running_apps == nil {
            return false;
        }

        let count: usize = msg_send![running_apps, count];
        for i in 0..count {
            let app: id = msg_send![running_apps, objectAtIndex: i];
            let bid: id = msg_send![app, bundleIdentifier];
            if let Some(bid_str) = nsstring_to_string(bid) {
                if bid_str == bundle_id {
                    // NSApplicationActivateIgnoringOtherApps = 1 << 1 = 2
                    let activated: bool = msg_send![app, activateWithOptions: 2u64];
                    log::info!(
                        "Activated app: {} (bundle: {}), success={}",
                        nsstring_to_string(msg_send![app, localizedName])
                            .unwrap_or_default(),
                        bundle_id,
                        activated
                    );
                    return activated;
                }
            }
        }

        log::warn!("App not found for activation: {}", bundle_id);
        false
    }
}

#[cfg(not(target_os = "macos"))]
pub fn activate_app_by_bundle_id(_bundle_id: &str) -> bool {
    false
}

/// Try to determine the source app of the most recent desktop notification.
/// Uses the macOS notification center database (best-effort).
/// Returns the bundle_id of the notification source, or None if unavailable.
#[cfg(target_os = "macos")]
pub fn get_last_notification_source_app() -> Option<String> {
    let home = std::env::var("HOME").ok()?;

    // macOS 13+ (Ventura): notification database location
    let db_path = format!(
        "{}/Library/Group Containers/group.com.apple.usernoted/db2/db",
        home
    );

    if !std::path::Path::new(&db_path).exists() {
        log::debug!("Notification database not found at: {}", db_path);
        return None;
    }

    // Use sqlite3 CLI to query the most recent notification source
    let output = std::process::Command::new("sqlite3")
        .arg(&db_path)
        .arg("-separator")
        .arg("|")
        .arg("SELECT app_id FROM record ORDER BY delivered_date DESC LIMIT 1;")
        .output()
        .ok()?;

    if output.status.success() {
        let app_id = String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .to_string();
        if !app_id.is_empty() {
            log::info!("Last notification source app: {}", app_id);
            return Some(app_id);
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!("Failed to query notification database: {}", stderr);
    }

    None
}

#[cfg(not(target_os = "macos"))]
pub fn get_last_notification_source_app() -> Option<String> {
    None
}

/// Known app profiles for post-processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppCategory {
    /// Email applications (Mail, Gmail in browser, etc.)
    Email,
    /// Note-taking with Markdown support (Notion, Obsidian, etc.)
    Markdown,
    /// Code editors (VS Code, Xcode, etc.)
    CodeEditor,
    /// Terminal applications
    Terminal,
    /// Default/unknown applications
    Default,
}

impl AppCategory {
    /// Determine the category from bundle ID
    pub fn from_bundle_id(bundle_id: &str) -> Self {
        match bundle_id {
            // Email apps
            "com.apple.mail" | "com.microsoft.Outlook" | "com.readdle.smartemail-Mac"
            | "com.google.Chrome" // Could be Gmail, need URL check
            => AppCategory::Email,

            // Markdown/Note apps
            "notion.id" | "md.obsidian" | "com.electron.logseq" | "abnerworks.Typora"
            | "com.bear-writer.bear" | "com.ulyssesapp.mac"
            => AppCategory::Markdown,

            // Code editors
            "com.microsoft.VSCode" | "com.apple.dt.Xcode" | "com.jetbrains.intellij"
            | "com.sublimetext.4" | "com.github.atom" | "com.panic.Nova"
            => AppCategory::CodeEditor,

            // Terminal
            "com.apple.Terminal" | "com.googlecode.iterm2" | "dev.warp.Warp-Stable"
            | "com.mitchellh.ghostty"
            => AppCategory::Terminal,

            // Default
            _ => AppCategory::Default,
        }
    }
}

impl ActiveAppInfo {
    /// Get the category of this application
    pub fn category(&self) -> AppCategory {
        self.bundle_id
            .as_ref()
            .map(|id| AppCategory::from_bundle_id(id))
            .unwrap_or(AppCategory::Default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_category_from_bundle_id() {
        assert_eq!(
            AppCategory::from_bundle_id("com.apple.mail"),
            AppCategory::Email
        );
        assert_eq!(
            AppCategory::from_bundle_id("notion.id"),
            AppCategory::Markdown
        );
        assert_eq!(
            AppCategory::from_bundle_id("com.microsoft.VSCode"),
            AppCategory::CodeEditor
        );
        assert_eq!(
            AppCategory::from_bundle_id("com.apple.Terminal"),
            AppCategory::Terminal
        );
        assert_eq!(
            AppCategory::from_bundle_id("com.unknown.app"),
            AppCategory::Default
        );
    }
}
