//! Clipboard management module
//!
//! This module provides clipboard read/write functionality
//! using tauri-plugin-clipboard-manager.

use anyhow::Result;
use enigo::Enigo;
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;

/// Set text to clipboard
pub fn set_clipboard_text(app: &AppHandle, text: &str) -> Result<()> {
    app.clipboard()
        .write_text(text)
        .map_err(|e| anyhow::anyhow!("Failed to write to clipboard: {}", e))?;
    log::debug!("Text copied to clipboard: {} chars", text.len());
    Ok(())
}

/// Get text from clipboard
#[allow(dead_code)]
pub fn get_clipboard_text(app: &AppHandle) -> Result<String> {
    let text = app
        .clipboard()
        .read_text()
        .map_err(|e| anyhow::anyhow!("Failed to read from clipboard: {}", e))?;
    Ok(text)
}

/// Clipboard manager wrapper
pub struct ClipboardManager;

impl ClipboardManager {
    pub fn new() -> Self {
        Self
    }

    #[allow(dead_code)]
    pub fn set_text(&self, app: &AppHandle, text: &str) -> Result<()> {
        set_clipboard_text(app, text)
    }

    #[allow(dead_code)]
    pub fn get_text(&self, app: &AppHandle) -> Result<String> {
        get_clipboard_text(app)
    }
}

impl Default for ClipboardManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Pastes text using the clipboard: saves current content, writes text, sends paste keystroke, restores clipboard.
/// This matches the Handy implementation.
pub fn paste_via_clipboard(
    enigo: &mut Enigo,
    text: &str,
    app: &AppHandle,
) -> Result<()> {
    // Save original clipboard content
    let clipboard_content = get_clipboard_text(app).unwrap_or_default();

    // Write text to clipboard
    set_clipboard_text(app, text)?;

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Send paste key combo (Cmd+V on macOS)
    crate::input::send_paste_ctrl_v(enigo)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Restore original clipboard content
    set_clipboard_text(app, &clipboard_content)?;

    log::info!("Text pasted with clipboard restoration");
    Ok(())
}
