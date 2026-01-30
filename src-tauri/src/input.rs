//! Input simulation module using enigo
//!
//! This module handles keyboard input simulation for pasting text
//! from the clipboard into the active application.
//! Uses a dedicated thread to handle the non-Send Enigo type.

use anyhow::{anyhow, Result};
use enigo::{Enigo, Key, Keyboard, Settings};
use std::sync::{mpsc, Mutex, OnceLock};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;

/// Commands for the input thread
#[allow(dead_code)]
enum InputCommand {
    SendPaste { reply: mpsc::Sender<Result<()>> },
    SendPasteCtrlShiftV { reply: mpsc::Sender<Result<()>> },
    SendPasteShiftInsert { reply: mpsc::Sender<Result<()>> },
    Shutdown,
}

/// Global input thread handle
static INPUT_THREAD: OnceLock<Mutex<Option<InputThread>>> = OnceLock::new();

struct InputThread {
    tx: mpsc::Sender<InputCommand>,
    #[allow(dead_code)]
    handle: JoinHandle<()>,
}

fn get_input_thread() -> &'static Mutex<Option<InputThread>> {
    INPUT_THREAD.get_or_init(|| {
        let thread = spawn_input_thread();
        Mutex::new(Some(thread))
    })
}

fn spawn_input_thread() -> InputThread {
    let (tx, rx) = mpsc::channel::<InputCommand>();

    let handle = thread::spawn(move || {
        let settings = Settings::default();
        let mut enigo = match Enigo::new(&settings) {
            Ok(e) => e,
            Err(e) => {
                log::error!("Failed to create Enigo instance: {}", e);
                return;
            }
        };

        for cmd in rx {
            match cmd {
                InputCommand::SendPaste { reply } => {
                    let result = send_paste_impl(&mut enigo);
                    let _ = reply.send(result);
                }
                InputCommand::SendPasteCtrlShiftV { reply } => {
                    let result = send_paste_ctrl_shift_v_impl(&mut enigo);
                    let _ = reply.send(result);
                }
                InputCommand::SendPasteShiftInsert { reply } => {
                    let result = send_paste_shift_insert_impl(&mut enigo);
                    let _ = reply.send(result);
                }
                InputCommand::Shutdown => break,
            }
        }
    });

    InputThread { tx, handle }
}

fn send_paste_impl(enigo: &mut Enigo) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        // macOS: Use Key::Other(9) for 'v' keycode - more stable than Key::Unicode
        let v_key = Key::Other(9); // macOS virtual keycode for 'v'

        enigo.key(Key::Meta, enigo::Direction::Press)?;
        sleep(Duration::from_millis(20));
        enigo.key(v_key, enigo::Direction::Click)?;
        sleep(Duration::from_millis(100));
        enigo.key(Key::Meta, enigo::Direction::Release)?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        enigo.key(Key::Control, enigo::Direction::Press)?;
        sleep(Duration::from_millis(20));
        enigo.key(Key::Unicode('v'), enigo::Direction::Click)?;
        sleep(Duration::from_millis(100));
        enigo.key(Key::Control, enigo::Direction::Release)?;
    }

    log::info!("Paste command sent");
    Ok(())
}

fn send_paste_ctrl_shift_v_impl(enigo: &mut Enigo) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let v_key = Key::Other(9); // macOS virtual keycode for 'v'

        enigo.key(Key::Meta, enigo::Direction::Press)?;
        enigo.key(Key::Shift, enigo::Direction::Press)?;
        sleep(Duration::from_millis(20));
        enigo.key(v_key, enigo::Direction::Click)?;
        sleep(Duration::from_millis(100));
        enigo.key(Key::Shift, enigo::Direction::Release)?;
        enigo.key(Key::Meta, enigo::Direction::Release)?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        enigo.key(Key::Control, enigo::Direction::Press)?;
        enigo.key(Key::Shift, enigo::Direction::Press)?;
        sleep(Duration::from_millis(20));
        enigo.key(Key::Unicode('v'), enigo::Direction::Click)?;
        sleep(Duration::from_millis(100));
        enigo.key(Key::Shift, enigo::Direction::Release)?;
        enigo.key(Key::Control, enigo::Direction::Release)?;
    }

    Ok(())
}

fn send_paste_shift_insert_impl(_enigo: &mut Enigo) -> Result<()> {
    // Shift+Insert is not widely supported in modern systems
    // Fall back to regular paste
    log::warn!("Shift+Insert not supported, use send_paste instead");
    Ok(())
}

/// Dummy state for compatibility (no longer used for actual state)
pub struct EnigoState;

impl EnigoState {
    pub fn new() -> Self {
        // Initialize the input thread when EnigoState is created
        let _ = get_input_thread();
        Self
    }
}

impl Default for EnigoState {
    fn default() -> Self {
        Self::new()
    }
}

/// Send paste command (Cmd+V on macOS, Ctrl+V on others)
pub fn send_paste(_enigo_state: &EnigoState) -> Result<()> {
    let guard = get_input_thread().lock().map_err(|e| anyhow!("{}", e))?;
    let thread = guard.as_ref().ok_or_else(|| anyhow!("Input thread not available"))?;

    let (reply_tx, reply_rx) = mpsc::channel();
    thread
        .tx
        .send(InputCommand::SendPaste { reply: reply_tx })
        .map_err(|e| anyhow!("Failed to send command: {}", e))?;

    reply_rx
        .recv()
        .map_err(|e| anyhow!("Failed to receive reply: {}", e))?
}

/// Send Ctrl+Shift+V paste command (for terminals)
#[allow(dead_code)]
pub fn send_paste_ctrl_shift_v(_enigo_state: &EnigoState) -> Result<()> {
    let guard = get_input_thread().lock().map_err(|e| anyhow!("{}", e))?;
    let thread = guard.as_ref().ok_or_else(|| anyhow!("Input thread not available"))?;

    let (reply_tx, reply_rx) = mpsc::channel();
    thread
        .tx
        .send(InputCommand::SendPasteCtrlShiftV { reply: reply_tx })
        .map_err(|e| anyhow!("Failed to send command: {}", e))?;

    reply_rx
        .recv()
        .map_err(|e| anyhow!("Failed to receive reply: {}", e))?
}

/// Send Shift+Insert paste command (legacy method)
#[allow(dead_code)]
pub fn send_paste_shift_insert(_enigo_state: &EnigoState) -> Result<()> {
    let guard = get_input_thread().lock().map_err(|e| anyhow!("{}", e))?;
    let thread = guard.as_ref().ok_or_else(|| anyhow!("Input thread not available"))?;

    let (reply_tx, reply_rx) = mpsc::channel();
    thread
        .tx
        .send(InputCommand::SendPasteShiftInsert { reply: reply_tx })
        .map_err(|e| anyhow!("Failed to send command: {}", e))?;

    reply_rx
        .recv()
        .map_err(|e| anyhow!("Failed to receive reply: {}", e))?
}
