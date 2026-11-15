//! Short UI sounds for hotkey dictation.
//!
//! The WAVs are embedded in the binary and played with `NSSound` on the main
//! thread, so playing one never blocks the caller.

use tauri::AppHandle;

#[derive(Clone, Copy)]
pub enum Sound {
    /// The hotkey was pressed and the recording popup appears.
    DictationStart,
    /// The transcript is ready (and pasted, with auto-insert on).
    DictationDone,
}

impl Sound {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn wav(self) -> &'static [u8] {
        match self {
            Sound::DictationStart => include_bytes!("../sounds/dictation-start.wav"),
            Sound::DictationDone => include_bytes!("../sounds/dictation-done.wav"),
        }
    }
}

pub fn play(app: &AppHandle, sound: Sound) {
    #[cfg(target_os = "macos")]
    if let Err(e) = app.run_on_main_thread(move || macos::play(sound)) {
        log::warn!("Failed to play sound: {}", e);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (app, sound);
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
mod macos {
    use super::Sound;
    use cocoa::base::{id, nil};
    use std::cell::RefCell;

    thread_local! {
        // One NSSound per sound, created on first use and kept for the app's
        // lifetime: a released NSSound stops playing. Only touched on the main thread.
        static SOUNDS: RefCell<[id; 2]> = const { RefCell::new([nil, nil]) };
    }

    pub fn play(sound: Sound) {
        SOUNDS.with(|sounds| {
            let slot = &mut sounds.borrow_mut()[sound as usize];
            unsafe {
                if *slot == nil {
                    *slot = load(sound);
                    if *slot == nil {
                        log::warn!("Failed to load sound");
                        return;
                    }
                }
                // Restart if it is still playing from the last time.
                let _: bool = msg_send![*slot, stop];
                let _: bool = msg_send![*slot, play];
            }
        });
    }

    /// A retained NSSound for `sound`, or nil if the WAV can't be decoded.
    unsafe fn load(sound: Sound) -> id {
        let wav = sound.wav();
        let data: id = msg_send![class!(NSData),
            dataWithBytes: wav.as_ptr()
            length: wav.len()];
        let ns_sound: id = msg_send![class!(NSSound), alloc];
        msg_send![ns_sound, initWithData: data]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn sounds_load_and_play() {
            for sound in [Sound::DictationStart, Sound::DictationDone] {
                unsafe {
                    let ns_sound = load(sound);
                    assert!(ns_sound != nil, "WAV did not decode");
                    let started: bool = msg_send![ns_sound, play];
                    assert!(started, "NSSound refused to play");
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
    }
}
