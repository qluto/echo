//! Audio capture module using cpal and hound
//!
//! This module handles audio recording from the default input device
//! and saves the audio to WAV files.
//! Uses a dedicated thread to handle the non-Send Stream type.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, SupportedStreamConfig};
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use crate::AudioDevice;

/// Commands for the audio thread
#[allow(dead_code)]
enum AudioCommand {
    StartRecording {
        device_name: Option<String>,
        reply: mpsc::Sender<Result<String>>,
    },
    StopRecording {
        reply: mpsc::Sender<Result<()>>,
    },
    Shutdown,
}

/// Global audio thread handle
static AUDIO_THREAD: OnceLock<Mutex<Option<AudioThread>>> = OnceLock::new();

struct AudioThread {
    tx: mpsc::Sender<AudioCommand>,
    #[allow(dead_code)]
    handle: JoinHandle<()>,
}

fn get_audio_thread() -> &'static Mutex<Option<AudioThread>> {
    AUDIO_THREAD.get_or_init(|| {
        let thread = spawn_audio_thread();
        Mutex::new(Some(thread))
    })
}

fn spawn_audio_thread() -> AudioThread {
    let (tx, rx) = mpsc::channel::<AudioCommand>();

    let handle = thread::spawn(move || {
        let mut current_recording: Option<RecordingState> = None;

        for cmd in rx {
            match cmd {
                AudioCommand::StartRecording { device_name, reply } => {
                    let result = start_recording_impl(device_name, &mut current_recording);
                    let _ = reply.send(result);
                }
                AudioCommand::StopRecording { reply } => {
                    let result = stop_recording_impl(&mut current_recording);
                    let _ = reply.send(result);
                }
                AudioCommand::Shutdown => break,
            }
        }
    });

    AudioThread { tx, handle }
}

struct RecordingState {
    _stream: Stream,
    writer: Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>,
    file_path: PathBuf,
}

fn start_recording_impl(
    device_name: Option<String>,
    current_recording: &mut Option<RecordingState>,
) -> Result<String> {
    if current_recording.is_some() {
        return Err(anyhow!("Already recording"));
    }

    let host = cpal::default_host();

    let device: Device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("Device not found: {}", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device"))?
    };

    // Use device's default configuration
    let supported_config: SupportedStreamConfig = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels();

    log::info!(
        "Using audio config: {} Hz, {} channels, format: {:?}",
        sample_rate,
        channels,
        supported_config.sample_format()
    );

    // Create temp file
    let temp_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("echo");
    std::fs::create_dir_all(&temp_dir)?;

    let file_name = format!("{}.wav", uuid::Uuid::new_v4());
    let file_path = temp_dir.join(&file_name);

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: HoundSampleFormat::Int,
    };

    let writer = WavWriter::create(&file_path, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_clone = Arc::clone(&writer);

    let config = supported_config.config();

    let stream = match supported_config.sample_format() {
        SampleFormat::I16 => build_stream::<i16>(&device, &config, writer_clone)?,
        SampleFormat::F32 => build_stream::<f32>(&device, &config, writer_clone)?,
        sample_format => return Err(anyhow!("Unsupported sample format: {:?}", sample_format)),
    };

    stream.play()?;

    let path_str = file_path.to_string_lossy().to_string();
    *current_recording = Some(RecordingState {
        _stream: stream,
        writer,
        file_path,
    });

    log::info!("Recording started: {}", path_str);
    Ok(path_str)
}

fn stop_recording_impl(current_recording: &mut Option<RecordingState>) -> Result<()> {
    if let Some(state) = current_recording.take() {
        // Drop the stream first to stop recording
        drop(state._stream);

        // Finalize the writer
        if let Ok(mut guard) = state.writer.lock() {
            if let Some(w) = guard.take() {
                w.finalize()?;
            }
        }

        log::info!("Recording stopped: {:?}", state.file_path);
    }

    Ok(())
}

fn build_stream<T>(
    device: &Device,
    config: &cpal::StreamConfig,
    writer: Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>,
) -> Result<Stream>
where
    T: cpal::Sample + cpal::SizedSample,
    i16: cpal::FromSample<T>,
{
    let err_fn = |err| log::error!("Audio stream error: {}", err);

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if let Ok(mut guard) = writer.lock() {
                if let Some(ref mut w) = *guard {
                    for &sample in data {
                        let sample_i16: i16 = cpal::Sample::from_sample(sample);
                        if w.write_sample(sample_i16).is_err() {
                            log::error!("Failed to write sample");
                        }
                    }
                }
            }
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

/// Start recording audio to a temp file
pub fn start_recording(device_name: Option<String>) -> Result<String> {
    let guard = get_audio_thread().lock().map_err(|e| anyhow!("{}", e))?;
    let thread = guard.as_ref().ok_or_else(|| anyhow!("Audio thread not available"))?;

    let (reply_tx, reply_rx) = mpsc::channel();
    thread
        .tx
        .send(AudioCommand::StartRecording {
            device_name,
            reply: reply_tx,
        })
        .map_err(|e| anyhow!("Failed to send command: {}", e))?;

    reply_rx
        .recv()
        .map_err(|e| anyhow!("Failed to receive reply: {}", e))?
}

/// Stop recording and finalize the WAV file
pub fn stop_recording() -> Result<()> {
    let guard = get_audio_thread().lock().map_err(|e| anyhow!("{}", e))?;
    let thread = guard.as_ref().ok_or_else(|| anyhow!("Audio thread not available"))?;

    let (reply_tx, reply_rx) = mpsc::channel();
    thread
        .tx
        .send(AudioCommand::StopRecording { reply: reply_tx })
        .map_err(|e| anyhow!("Failed to send command: {}", e))?;

    reply_rx
        .recv()
        .map_err(|e| anyhow!("Failed to receive reply: {}", e))?
}

/// Get list of available audio input devices
pub fn get_audio_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let default_device = host.default_input_device();
    let default_name = default_device.and_then(|d| d.name().ok());

    let devices: Vec<AudioDevice> = host
        .input_devices()?
        .filter_map(|device| {
            device.name().ok().map(|name| {
                let is_default = default_name.as_ref().map_or(false, |d| d == &name);
                AudioDevice { name, is_default }
            })
        })
        .collect();

    Ok(devices)
}
