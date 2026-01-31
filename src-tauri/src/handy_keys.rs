//! Handy-keys based keyboard shortcut implementation
//!
//! This module provides an alternative to Tauri's global-shortcut plugin
//! using the handy-keys library for more control over keyboard events,
//! including support for the Fn key on macOS.

use handy_keys::{Hotkey, HotkeyId, HotkeyManager, HotkeyState, KeyboardListener};
use log::{debug, error, info, warn};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tauri::{AppHandle, Emitter, Manager};

/// Commands that can be sent to the hotkey manager thread
enum ManagerCommand {
    Register {
        hotkey_string: String,
        response: Sender<Result<(), String>>,
    },
    Unregister {
        response: Sender<Result<(), String>>,
    },
    Shutdown,
}

/// State for the handy-keys shortcut manager
pub struct HandyKeysState {
    /// Channel to send commands to the manager thread
    command_sender: Mutex<Sender<ManagerCommand>>,
    /// Handle to the manager thread
    thread_handle: Mutex<Option<JoinHandle<()>>>,
    /// Recording listener for UI key capture
    recording_listener: Mutex<Option<KeyboardListener>>,
    /// Flag indicating if we're in recording mode
    is_recording: AtomicBool,
    /// Flag to stop recording loop
    recording_running: Arc<AtomicBool>,
}

/// Key event sent to frontend during recording mode
#[derive(Debug, Clone, Serialize)]
pub struct FrontendKeyEvent {
    /// Currently pressed modifier keys
    pub modifiers: Vec<String>,
    /// The key that was pressed (if any)
    pub key: Option<String>,
    /// Whether this is a key down event
    pub is_key_down: bool,
    /// The full hotkey string (e.g., "fn" or "command+space")
    pub hotkey_string: String,
}

impl HandyKeysState {
    /// Create a new HandyKeysState
    pub fn new(app: AppHandle) -> Result<Self, String> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerCommand>();

        // Start the manager thread
        let app_clone = app.clone();
        let thread_handle = thread::spawn(move || {
            Self::manager_thread(cmd_rx, app_clone);
        });

        Ok(Self {
            command_sender: Mutex::new(cmd_tx),
            thread_handle: Mutex::new(Some(thread_handle)),
            recording_listener: Mutex::new(None),
            is_recording: AtomicBool::new(false),
            recording_running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The main manager thread - owns the HotkeyManager and processes commands
    fn manager_thread(cmd_rx: Receiver<ManagerCommand>, app: AppHandle) {
        info!("handy-keys manager thread started");

        // Create the HotkeyManager in this thread
        let manager = match HotkeyManager::new() {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to create HotkeyManager: {}", e);
                return;
            }
        };

        // Current registered hotkey
        let mut current_hotkey: Option<(HotkeyId, String)> = None;

        loop {
            // Check for hotkey events (non-blocking)
            while let Some(event) = manager.try_recv() {
                if let Some((id, ref hotkey_string)) = current_hotkey {
                    if event.id == id {
                        debug!(
                            "handy-keys event: hotkey={}, state={:?}",
                            hotkey_string, event.state
                        );
                        let is_pressed = event.state == HotkeyState::Pressed;
                        handle_shortcut_event(&app, is_pressed);
                    }
                }
            }

            // Check for commands (non-blocking with timeout)
            match cmd_rx.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(cmd) => match cmd {
                    ManagerCommand::Register {
                        hotkey_string,
                        response,
                    } => {
                        // Unregister existing hotkey first
                        if let Some((id, _)) = current_hotkey.take() {
                            if let Err(e) = manager.unregister(id) {
                                warn!("Failed to unregister old hotkey: {}", e);
                            }
                        }

                        // Register new hotkey
                        let result = hotkey_string
                            .parse::<Hotkey>()
                            .map_err(|e| format!("Failed to parse hotkey '{}': {}", hotkey_string, e))
                            .and_then(|hotkey| {
                                manager
                                    .register(hotkey)
                                    .map_err(|e| format!("Failed to register hotkey: {}", e))
                                    .map(|id| {
                                        current_hotkey = Some((id, hotkey_string.clone()));
                                        info!("Registered handy-keys hotkey: {}", hotkey_string);
                                    })
                            });
                        let _ = response.send(result);
                    }
                    ManagerCommand::Unregister { response } => {
                        if let Some((id, hotkey_string)) = current_hotkey.take() {
                            let result = manager
                                .unregister(id)
                                .map_err(|e| format!("Failed to unregister hotkey: {}", e));
                            if result.is_ok() {
                                info!("Unregistered handy-keys hotkey: {}", hotkey_string);
                            }
                            let _ = response.send(result);
                        } else {
                            let _ = response.send(Ok(()));
                        }
                    }
                    ManagerCommand::Shutdown => {
                        info!("handy-keys manager thread shutting down");
                        break;
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // No command, continue
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    info!("Command channel disconnected, shutting down");
                    break;
                }
            }
        }

        info!("handy-keys manager thread stopped");
    }

    /// Register a hotkey
    pub fn register(&self, hotkey: &str) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.command_sender
            .lock()
            .map_err(|_| "Failed to lock command_sender")?
            .send(ManagerCommand::Register {
                hotkey_string: hotkey.to_string(),
                response: tx,
            })
            .map_err(|_| "Failed to send register command")?;

        rx.recv()
            .map_err(|_| "Failed to receive register response")?
    }

    /// Unregister the current hotkey
    pub fn unregister(&self) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.command_sender
            .lock()
            .map_err(|_| "Failed to lock command_sender")?
            .send(ManagerCommand::Unregister { response: tx })
            .map_err(|_| "Failed to send unregister command")?;

        rx.recv()
            .map_err(|_| "Failed to receive unregister response")?
    }

    /// Start recording mode for capturing key events
    pub fn start_recording(&self, app: &AppHandle) -> Result<(), String> {
        if self.is_recording.load(Ordering::SeqCst) {
            return Err("Already recording".into());
        }

        // Create a new keyboard listener for recording
        let listener = KeyboardListener::new()
            .map_err(|e| format!("Failed to create keyboard listener: {}", e))?;

        {
            let mut recording = self
                .recording_listener
                .lock()
                .map_err(|_| "Failed to lock recording_listener")?;
            *recording = Some(listener);
        }

        self.is_recording.store(true, Ordering::SeqCst);
        self.recording_running.store(true, Ordering::SeqCst);

        // Start a thread to emit key events to the frontend
        let app_clone = app.clone();
        let recording_running = Arc::clone(&self.recording_running);
        thread::spawn(move || {
            Self::recording_loop(app_clone, recording_running);
        });

        debug!("Started handy-keys recording mode");
        Ok(())
    }

    /// Recording loop - emits key events to frontend during recording
    fn recording_loop(app: AppHandle, running: Arc<AtomicBool>) {
        while running.load(Ordering::SeqCst) {
            let event = {
                let state = match app.try_state::<HandyKeysState>() {
                    Some(s) => s,
                    None => break,
                };
                let listener = state.recording_listener.lock().ok();
                listener.as_ref().and_then(|l| l.as_ref()?.try_recv())
            };

            if let Some(key_event) = event {
                // Convert to frontend-friendly format
                let frontend_event = FrontendKeyEvent {
                    modifiers: modifiers_to_strings(key_event.modifiers),
                    key: key_event.key.map(|k| k.to_string().to_lowercase()),
                    is_key_down: key_event.is_key_down,
                    hotkey_string: key_event
                        .as_hotkey()
                        .map(|h| h.to_handy_string())
                        .unwrap_or_default(),
                };

                // Emit to frontend
                if let Err(e) = app.emit("handy-keys-event", &frontend_event) {
                    error!("Failed to emit key event: {}", e);
                }
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        debug!("Recording loop ended");
    }

    /// Stop recording mode
    pub fn stop_recording(&self) -> Result<(), String> {
        self.is_recording.store(false, Ordering::SeqCst);
        self.recording_running.store(false, Ordering::SeqCst);

        {
            let mut recording = self
                .recording_listener
                .lock()
                .map_err(|_| "Failed to lock recording_listener")?;
            *recording = None;
        }

        debug!("Stopped handy-keys recording mode");
        Ok(())
    }
}

impl Drop for HandyKeysState {
    fn drop(&mut self) {
        // Signal recording to stop
        self.recording_running.store(false, Ordering::SeqCst);
        self.is_recording.store(false, Ordering::SeqCst);

        // Send shutdown command
        if let Ok(sender) = self.command_sender.lock() {
            let _ = sender.send(ManagerCommand::Shutdown);
        }

        // Wait for the manager thread to finish
        if let Ok(mut handle) = self.thread_handle.lock() {
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }
    }
}

/// Convert handy-keys Modifiers to a list of strings
fn modifiers_to_strings(modifiers: handy_keys::Modifiers) -> Vec<String> {
    let mut result = Vec::new();

    if modifiers.contains(handy_keys::Modifiers::CTRL) {
        result.push("ctrl".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::OPT) {
        result.push("option".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::SHIFT) {
        result.push("shift".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::CMD) {
        result.push("command".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::FN) {
        result.push("fn".to_string());
    }

    result
}

/// Handle shortcut press/release event
fn handle_shortcut_event(app: &AppHandle, is_pressed: bool) {
    if is_pressed {
        crate::hotkey::trigger_hotkey_pressed(app);
    } else {
        crate::hotkey::trigger_hotkey_released(app);
    }
}

/// Validate a hotkey string
/// Rejects single alphanumeric keys without modifiers to prevent accidental key capture
fn validate_hotkey(hotkey: &str) -> Result<(), String> {
    let hotkey_lower = hotkey.to_lowercase();
    let parts: Vec<&str> = hotkey_lower.split('+').collect();

    // Check if it's a single key without modifiers
    if parts.len() == 1 {
        let key = parts[0].trim();

        // Allow function keys (f1-f24)
        if key.starts_with('f') && key[1..].parse::<u32>().map(|n| n >= 1 && n <= 24).unwrap_or(false) {
            return Ok(());
        }

        // Allow special standalone keys
        let allowed_standalone = ["fn", "printscreen", "scrolllock", "pause", "insert"];
        if allowed_standalone.contains(&key) {
            return Ok(());
        }

        // Reject single alphanumeric characters
        if key.len() == 1 && key.chars().next().map(|c| c.is_alphanumeric()).unwrap_or(false) {
            return Err(format!(
                "Single key '{}' is not allowed as a hotkey. Please use a modifier (Cmd, Ctrl, Option, Shift) or a function key.",
                key.to_uppercase()
            ));
        }

        // Reject common keys that would interfere with normal typing
        let disallowed_standalone = [
            "space", "return", "enter", "tab", "escape", "backspace", "delete",
            "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
        ];
        if disallowed_standalone.contains(&key) {
            return Err(format!(
                "Key '{}' alone is not allowed as a hotkey. Please add a modifier (Cmd, Ctrl, Option, Shift).",
                key
            ));
        }
    }

    Ok(())
}

/// Initialize handy-keys state
pub fn init(app: &AppHandle) -> Result<(), String> {
    let state = HandyKeysState::new(app.clone())?;
    app.manage(state);
    info!("handy-keys state initialized");
    Ok(())
}

/// Register a hotkey using handy-keys
pub fn register_hotkey(app: &AppHandle, hotkey: &str) -> Result<(), String> {
    // Validate the hotkey first
    validate_hotkey(hotkey)?;

    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;
    state.register(hotkey)
}

/// Unregister the current hotkey
pub fn unregister_hotkey(app: &AppHandle) -> Result<(), String> {
    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;
    state.unregister()
}

/// Start recording mode
pub fn start_recording(app: &AppHandle) -> Result<(), String> {
    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;
    state.start_recording(app)
}

/// Stop recording mode
pub fn stop_recording(app: &AppHandle) -> Result<(), String> {
    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;
    state.stop_recording()
}
