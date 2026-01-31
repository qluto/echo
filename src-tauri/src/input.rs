//! Input simulation module using enigo
//!
//! This module handles keyboard input simulation for pasting text
//! from the clipboard into the active application.
//!
//! Note: Enigo is not Send-safe on macOS due to CGEventSource,
//! so we create it on-demand rather than storing in Tauri state.

use anyhow::Result;
use enigo::{Enigo, Key, Keyboard, Settings};
use std::time::Duration;

/// Dummy state for Tauri compatibility
/// The actual Enigo instance is created on-demand in each function
pub struct EnigoState;

impl EnigoState {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EnigoState {
    fn default() -> Self {
        Self::new()
    }
}

/// Send paste command (Cmd+V on macOS, Ctrl+V on others)
/// Creates Enigo on-demand to avoid Send/Sync requirements
pub fn send_paste(_enigo_state: &EnigoState) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default())?;
    send_paste_impl(&mut enigo)
}

fn send_paste_impl(enigo: &mut Enigo) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        // macOS: Use Key::Other(9) for 'v' keycode
        let v_key = Key::Other(9);

        enigo.key(Key::Meta, enigo::Direction::Press)?;
        enigo.key(v_key, enigo::Direction::Click)?;
        std::thread::sleep(Duration::from_millis(100));
        enigo.key(Key::Meta, enigo::Direction::Release)?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        enigo.key(Key::Control, enigo::Direction::Press)?;
        enigo.key(Key::Unicode('v'), enigo::Direction::Click)?;
        std::thread::sleep(Duration::from_millis(100));
        enigo.key(Key::Control, enigo::Direction::Release)?;
    }

    log::info!("Paste command sent");
    Ok(())
}

/// Send Ctrl+Shift+V paste command (for terminals)
#[allow(dead_code)]
pub fn send_paste_ctrl_shift_v(_enigo_state: &EnigoState) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default())?;

    #[cfg(target_os = "macos")]
    {
        let v_key = Key::Other(9);

        enigo.key(Key::Meta, enigo::Direction::Press)?;
        enigo.key(Key::Shift, enigo::Direction::Press)?;
        enigo.key(v_key, enigo::Direction::Click)?;
        std::thread::sleep(Duration::from_millis(100));
        enigo.key(Key::Shift, enigo::Direction::Release)?;
        enigo.key(Key::Meta, enigo::Direction::Release)?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        enigo.key(Key::Control, enigo::Direction::Press)?;
        enigo.key(Key::Shift, enigo::Direction::Press)?;
        enigo.key(Key::Unicode('v'), enigo::Direction::Click)?;
        std::thread::sleep(Duration::from_millis(100));
        enigo.key(Key::Shift, enigo::Direction::Release)?;
        enigo.key(Key::Control, enigo::Direction::Release)?;
    }

    Ok(())
}

/// Send Shift+Insert paste command (legacy method)
#[allow(dead_code)]
pub fn send_paste_shift_insert(_enigo_state: &EnigoState) -> Result<()> {
    log::warn!("Shift+Insert not supported, use send_paste instead");
    Ok(())
}
