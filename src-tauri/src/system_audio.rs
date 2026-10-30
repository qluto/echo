//! System audio capture (what the Mac is playing — e.g. the other side of a
//! Zoom / Google Meet call) via a Core Audio process tap (macOS 14.2+).
//!
//! ```text
//! CATapDescription (global stereo tap, Echo excluded)
//!   → AudioHardwareCreateProcessTap
//!   → private aggregate device [default output device + tap]
//!   → IO proc → FrameAccumulator → 16 kHz mono 512-sample frames
//! ```
//!
//! Needs the "System Audio Recording" permission (`NSAudioCaptureUsageDescription`);
//! macOS asks on first use. When it is denied the tap still runs but delivers
//! silence, so callers should watch the level rather than rely on an error.

#![allow(deprecated)] // cocoa / objc 0.2, as in lib.rs

use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use cocoa::base::{id, nil};
use cocoa::foundation::{NSAutoreleasePool, NSString};
use objc::runtime::{Class, YES};

use crate::audio_capture::FrameAccumulator;

type AudioObjectID = u32;
type OSStatus = i32;

#[repr(C)]
struct AudioObjectPropertyAddress {
    selector: u32,
    scope: u32,
    element: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

type AudioDeviceIOProc = extern "C" fn(
    device: AudioObjectID,
    now: *const c_void,
    input: *const AudioBufferList,
    input_time: *const c_void,
    output: *mut AudioBufferList,
    output_time: *const c_void,
    client_data: *mut c_void,
) -> OSStatus;
type AudioDeviceIOProcID = *mut c_void;

#[link(name = "CoreAudio", kind = "framework")]
extern "C" {
    fn AudioObjectGetPropertyData(
        object: AudioObjectID,
        address: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        data_size: *mut u32,
        data: *mut c_void,
    ) -> OSStatus;
    fn AudioHardwareCreateProcessTap(description: id, out_tap: *mut AudioObjectID) -> OSStatus;
    fn AudioHardwareDestroyProcessTap(tap: AudioObjectID) -> OSStatus;
    fn AudioHardwareCreateAggregateDevice(description: id, out_device: *mut AudioObjectID) -> OSStatus;
    fn AudioHardwareDestroyAggregateDevice(device: AudioObjectID) -> OSStatus;
    fn AudioDeviceCreateIOProcID(
        device: AudioObjectID,
        proc_: AudioDeviceIOProc,
        client_data: *mut c_void,
        out_proc_id: *mut AudioDeviceIOProcID,
    ) -> OSStatus;
    fn AudioDeviceDestroyIOProcID(device: AudioObjectID, proc_id: AudioDeviceIOProcID) -> OSStatus;
    fn AudioDeviceStart(device: AudioObjectID, proc_id: AudioDeviceIOProcID) -> OSStatus;
    fn AudioDeviceStop(device: AudioObjectID, proc_id: AudioDeviceIOProcID) -> OSStatus;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
}

const fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

const SYSTEM_OBJECT: AudioObjectID = 1;
const SCOPE_GLOBAL: u32 = fourcc(b"glob");
const ELEMENT_MAIN: u32 = 0;
const PROP_TRANSLATE_PID: u32 = fourcc(b"id2p");
const PROP_DEFAULT_OUTPUT_DEVICE: u32 = fourcc(b"dOut");
const PROP_DEVICE_UID: u32 = fourcc(b"uid ");
const PROP_TAP_FORMAT: u32 = fourcc(b"tfmt");
const FORMAT_LINEAR_PCM: u32 = fourcc(b"lpcm");
const FORMAT_FLAG_IS_FLOAT: u32 = 1;

fn check(status: OSStatus, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let bytes = status.to_be_bytes();
    if bytes.iter().all(|b| b.is_ascii_graphic()) {
        bail!("{what} failed: '{}'", String::from_utf8_lossy(&bytes));
    }
    bail!("{what} failed: OSStatus {status}");
}

/// Read a fixed-size property of `object`.
unsafe fn get_property<T: Default>(
    object: AudioObjectID,
    selector: u32,
    qualifier: Option<&i32>,
    what: &str,
) -> Result<T> {
    let address = AudioObjectPropertyAddress { selector, scope: SCOPE_GLOBAL, element: ELEMENT_MAIN };
    let mut value = T::default();
    let mut size = std::mem::size_of::<T>() as u32;
    let (q_size, q_ptr) = match qualifier {
        Some(q) => (std::mem::size_of::<i32>() as u32, q as *const i32 as *const c_void),
        None => (0, ptr::null()),
    };
    check(
        AudioObjectGetPropertyData(object, &address, q_size, q_ptr, &mut size, &mut value as *mut T as *mut c_void),
        what,
    )?;
    Ok(value)
}

/// UID of the default output device, as an autoreleased `NSString`.
unsafe fn default_output_uid() -> Result<id> {
    let device: AudioObjectID = get_property(SYSTEM_OBJECT, PROP_DEFAULT_OUTPUT_DEVICE, None, "default output device")?;
    // The UID is returned as a +1 CFStringRef.
    let uid: usize = get_property(device, PROP_DEVICE_UID, None, "output device UID")?;
    if uid == 0 {
        bail!("Default output device has no UID");
    }
    let uid = uid as id;
    let copy: id = msg_send![uid, copy];
    CFRelease(uid as *const c_void);
    Ok(msg_send![copy, autorelease])
}

unsafe fn ns_string(s: &str) -> id {
    let s = NSString::alloc(nil).init_str(s);
    msg_send![s, autorelease]
}

unsafe fn ns_number(n: i32) -> id {
    msg_send![class!(NSNumber), numberWithInt: n]
}

unsafe fn ns_dictionary(entries: &[(&str, id)]) -> id {
    let dict: id = msg_send![class!(NSMutableDictionary), dictionary];
    for (key, value) in entries {
        let _: () = msg_send![dict, setObject: *value forKey: ns_string(key)];
    }
    dict
}

unsafe fn ns_array(items: &[id]) -> id {
    let array: id = msg_send![class!(NSMutableArray), array];
    for item in items {
        let _: () = msg_send![array, addObject: *item];
    }
    array
}

/// State the IO proc reads on Core Audio's real-time thread.
struct CallbackState {
    accumulator: Mutex<FrameAccumulator>,
    mono: Mutex<Vec<f32>>,
}

extern "C" fn io_proc(
    _device: AudioObjectID,
    _now: *const c_void,
    input: *const AudioBufferList,
    _input_time: *const c_void,
    _output: *mut AudioBufferList,
    _output_time: *const c_void,
    client_data: *mut c_void,
) -> OSStatus {
    if input.is_null() || client_data.is_null() {
        return 0;
    }
    let state = unsafe { &*(client_data as *const CallbackState) };
    let (Ok(mut mono), Ok(mut acc)) = (state.mono.try_lock(), state.accumulator.lock()) else {
        return 0;
    };
    unsafe {
        let list = &*input;
        let buffers = std::slice::from_raw_parts(list.buffers.as_ptr(), list.number_buffers as usize);
        downmix(buffers, &mut mono);
    }
    acc.push_samples(&mono);
    0
}

/// Mix float32 buffers (one interleaved buffer, or one buffer per channel)
/// down to mono into `out`.
unsafe fn downmix(buffers: &[AudioBuffer], out: &mut Vec<f32>) {
    out.clear();
    let channels_total: usize = buffers.iter().map(|b| b.number_channels.max(1) as usize).sum();
    for (bi, buffer) in buffers.iter().enumerate() {
        if buffer.data.is_null() {
            continue;
        }
        let ch = buffer.number_channels.max(1) as usize;
        let samples = std::slice::from_raw_parts(
            buffer.data as *const f32,
            buffer.data_byte_size as usize / std::mem::size_of::<f32>(),
        );
        let frames = samples.len() / ch;
        if bi == 0 {
            out.resize(frames, 0.0);
        }
        for (i, frame) in samples.chunks_exact(ch).take(out.len()).enumerate() {
            out[i] += frame.iter().sum::<f32>() / channels_total as f32;
        }
    }
}

/// A running system-audio tap. Frames stop when this is dropped.
pub struct SystemAudioCapture {
    tap: AudioObjectID,
    aggregate: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    state: *mut CallbackState,
}

// The raw pointers are only touched at start/stop (and by Core Audio's IO thread).
unsafe impl Send for SystemAudioCapture {}

impl SystemAudioCapture {
    /// Start tapping everything the Mac plays (except Echo itself). Frames are
    /// 16 kHz mono, 512 samples each, like `StreamingCapture`.
    pub fn start(frame_sender: crossbeam_channel::Sender<Vec<f32>>) -> Result<Self> {
        unsafe {
            let pool = NSAutoreleasePool::new(nil);
            let result = Self::start_inner(frame_sender);
            let _: () = msg_send![pool, drain];
            result
        }
    }

    unsafe fn start_inner(frame_sender: crossbeam_channel::Sender<Vec<f32>>) -> Result<Self> {
        let tap_class = Class::get("CATapDescription")
            .ok_or_else(|| anyhow!("System audio capture needs macOS 14.2 or later"))?;

        // Leave Echo's own output out of the tap.
        let pid = std::process::id() as i32;
        let excluded = match get_property::<AudioObjectID>(SYSTEM_OBJECT, PROP_TRANSLATE_PID, Some(&pid), "pid → process") {
            Ok(obj) if obj != 0 => vec![ns_number(obj as i32)],
            _ => Vec::new(),
        };
        let description: id = msg_send![tap_class, alloc];
        let description: id = msg_send![description, initStereoGlobalTapButExcludeProcesses: ns_array(&excluded)];
        if description == nil {
            bail!("Failed to create the tap description");
        }
        let _: () = msg_send![description, setPrivate: YES];
        let _: () = msg_send![description, setMono: YES];
        let _: () = msg_send![description, autorelease];

        let mut tap: AudioObjectID = 0;
        check(AudioHardwareCreateProcessTap(description, &mut tap), "AudioHardwareCreateProcessTap")?;
        // From here on, `Drop` tears down whatever has been set up so far.
        let mut capture = Self { tap, aggregate: 0, proc_id: ptr::null_mut(), state: ptr::null_mut() };

        let format: AudioStreamBasicDescription = get_property(tap, PROP_TAP_FORMAT, None, "tap format")?;
        log::info!("System audio tap format: {format:?}");
        if format.format_id != FORMAT_LINEAR_PCM
            || format.format_flags & FORMAT_FLAG_IS_FLOAT == 0
            || format.bits_per_channel != 32
        {
            bail!("Unsupported system audio format: {format:?}");
        }

        let tap_uuid: id = msg_send![description, UUID];
        let tap_uid: id = msg_send![tap_uuid, UUIDString];
        let output_uid = default_output_uid()?;
        let aggregate_uid = ns_string(&format!("io.qluto.echo.meeting-tap.{}", uuid::Uuid::new_v4()));
        let tap_entry = ns_dictionary(&[("uid", tap_uid), ("drift", ns_number(1))]);
        let sub_device = ns_dictionary(&[("uid", output_uid)]);
        let aggregate_desc = ns_dictionary(&[
            ("name", ns_string("Echo Meeting Tap")),
            ("uid", aggregate_uid),
            ("master", output_uid),
            ("private", ns_number(1)),
            ("stacked", ns_number(0)),
            ("tapautostart", ns_number(1)),
            ("subdevices", ns_array(&[sub_device])),
            ("taps", ns_array(&[tap_entry])),
        ]);
        check(
            AudioHardwareCreateAggregateDevice(aggregate_desc, &mut capture.aggregate),
            "AudioHardwareCreateAggregateDevice",
        )?;

        let accumulator = FrameAccumulator::new(format.sample_rate.round() as u32, 1, frame_sender)?;
        capture.state = Box::into_raw(Box::new(CallbackState {
            accumulator: Mutex::new(accumulator),
            mono: Mutex::new(Vec::with_capacity(4096)),
        }));
        check(
            AudioDeviceCreateIOProcID(capture.aggregate, io_proc, capture.state as *mut c_void, &mut capture.proc_id),
            "AudioDeviceCreateIOProcID",
        )?;
        check(AudioDeviceStart(capture.aggregate, capture.proc_id), "AudioDeviceStart")?;
        log::info!("System audio capture started ({} Hz)", format.sample_rate);
        Ok(capture)
    }
}

impl Drop for SystemAudioCapture {
    fn drop(&mut self) {
        unsafe {
            if !self.proc_id.is_null() {
                let _ = AudioDeviceStop(self.aggregate, self.proc_id);
                let _ = AudioDeviceDestroyIOProcID(self.aggregate, self.proc_id);
            }
            if self.aggregate != 0 {
                let _ = AudioHardwareDestroyAggregateDevice(self.aggregate);
            }
            if self.tap != 0 {
                let _ = AudioHardwareDestroyProcessTap(self.tap);
            }
            if !self.state.is_null() {
                drop(Box::from_raw(self.state));
            }
        }
        log::info!("System audio capture stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Captures 3 s of system audio while `afplay` plays a system sound
    /// (needs the System Audio Recording permission for the terminal):
    /// `cargo test system_audio_captures_playback -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn system_audio_captures_playback() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let capture = SystemAudioCapture::start(tx).expect("start tap");
        let mut player = std::process::Command::new("afplay")
            .arg("/System/Library/Sounds/Glass.aiff")
            .spawn()
            .unwrap();
        let t = Instant::now();
        let mut frames = 0usize;
        let mut peak = 0f32;
        while t.elapsed() < Duration::from_secs(3) {
            if let Ok(frame) = rx.recv_timeout(Duration::from_millis(100)) {
                assert_eq!(frame.len(), 512);
                frames += 1;
                peak = frame.iter().fold(peak, |p, s| p.max(s.abs()));
            }
        }
        let _ = player.wait();
        drop(capture);
        println!("{frames} frames ({:.2}s), peak {peak:.3}", frames as f64 * 512.0 / 16000.0);
        assert!(frames > 60, "no audio frames from the tap");
        assert!(peak > 0.01, "tap delivered silence (permission denied?)");
    }
}
