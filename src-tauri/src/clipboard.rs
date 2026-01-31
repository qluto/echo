//! Clipboard management module
//!
//! This module provides clipboard read/write functionality
//! using tauri-plugin-clipboard-manager.

use anyhow::Result;
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;

use crate::input::EnigoState;

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

/// Paste text with clipboard preservation
///
/// This function:
/// 1. Saves the original clipboard content
/// 2. Writes new text to clipboard
/// 3. Waits for clipboard to be ready
/// 4. Sends paste command (Cmd+V / Ctrl+V)
/// 5. Waits for paste to complete
/// 6. Restores original clipboard content
pub fn paste_with_restore(
    app: &AppHandle,
    text: &str,
    enigo_state: &EnigoState,
) -> Result<()> {
    // 1. Save original clipboard content
    let original = get_clipboard_text(app).unwrap_or_default();

    // 2. Write new text to clipboard
    set_clipboard_text(app, text)?;

    // 3. Wait for clipboard to be ready
    std::thread::sleep(std::time::Duration::from_millis(50));

    // 4. Send paste command
    crate::input::send_paste(enigo_state)?;

    // 5. Wait for paste to complete
    std::thread::sleep(std::time::Duration::from_millis(50));

    // 6. Restore original clipboard content (only if there was content)
    if !original.is_empty() {
        set_clipboard_text(app, &original)?;
        log::debug!("Restored original clipboard content");
    }

    log::info!("Text pasted with clipboard restoration");
    Ok(())
}
