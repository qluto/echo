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

// ---------------------------------------------------------------------------
// Streaming capture for continuous listening (16kHz mono, 512-sample frames)
// ---------------------------------------------------------------------------

use crate::vad::VAD_SAMPLE_RATE;

const STREAM_FRAME_SIZE: usize = 512;

/// Streaming audio capture that sends 16kHz mono f32 frames via crossbeam channel.
///
/// The cpal input stream runs on its own thread. Audio is captured at the device's
/// native rate and channels, then converted to 16kHz mono f32 in the callback.
pub struct StreamingCapture {
    /// Handle to stop the streaming thread.
    stop_tx: mpsc::Sender<()>,
    /// Thread handle for cleanup.
    handle: Option<JoinHandle<()>>,
}

impl StreamingCapture {
    /// Start streaming capture. Returns a receiver that yields 512-sample f32 frames.
    ///
    /// Audio is captured from the specified device (or default) and converted to
    /// 16kHz mono. Frames are sent as `Vec<f32>` with exactly 512 samples each.
    pub fn start(
        device_name: Option<String>,
        frame_sender: crossbeam_channel::Sender<Vec<f32>>,
    ) -> Result<Self> {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let handle = thread::spawn(move || {
            if let Err(e) = streaming_thread(device_name, frame_sender, stop_rx) {
                log::error!("Streaming capture thread error: {}", e);
            }
        });

        Ok(Self {
            stop_tx,
            handle: Some(handle),
        })
    }

    /// Stop the streaming capture and wait for the thread to finish.
    pub fn stop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for StreamingCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

fn streaming_thread(
    device_name: Option<String>,
    frame_sender: crossbeam_channel::Sender<Vec<f32>>,
    stop_rx: mpsc::Receiver<()>,
) -> Result<()> {
    let host = cpal::default_host();

    let device: Device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("Device not found: {}", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device"))?
    };

    let supported_config = device.default_input_config()?;
    let native_rate = supported_config.sample_rate().0;
    let native_channels = supported_config.channels() as usize;

    log::info!(
        "Streaming capture: native {}Hz {}ch, target {}Hz mono",
        native_rate,
        native_channels,
        VAD_SAMPLE_RATE
    );

    // Accumulator to collect samples and emit 512-sample frames
    let accumulator = Arc::new(Mutex::new(FrameAccumulator::new(
        native_rate,
        native_channels,
        frame_sender,
    )));
    let acc_clone = Arc::clone(&accumulator);

    let config = supported_config.config();

    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => {
            let err_fn = |err| log::error!("Streaming audio error: {}", err);
            device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut acc) = acc_clone.lock() {
                        acc.push_samples(data);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let err_fn = |err| log::error!("Streaming audio error: {}", err);
            let acc_clone2 = Arc::clone(&accumulator);
            device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut acc) = acc_clone2.lock() {
                        // Convert i16 to f32
                        let f32_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        acc.push_samples(&f32_data);
                    }
                },
                err_fn,
                None,
            )?
        }
        format => return Err(anyhow!("Unsupported sample format: {:?}", format)),
    };

    stream.play()?;
    log::info!("Streaming capture started");

    // Wait for stop signal
    let _ = stop_rx.recv();

    drop(stream);
    log::info!("Streaming capture stopped");
    Ok(())
}

/// Accumulates incoming audio samples, converts to 16kHz mono, and emits
/// 512-sample frames via the channel.
struct FrameAccumulator {
    native_rate: u32,
    native_channels: usize,
    buffer: Vec<f32>,
    frame_sender: crossbeam_channel::Sender<Vec<f32>>,
    /// Fractional sample position for resampling
    resample_pos: f64,
}

impl FrameAccumulator {
    fn new(
        native_rate: u32,
        native_channels: usize,
        frame_sender: crossbeam_channel::Sender<Vec<f32>>,
    ) -> Self {
        Self {
            native_rate,
            native_channels,
            buffer: Vec::with_capacity(STREAM_FRAME_SIZE * 4),
            frame_sender,
            resample_pos: 0.0,
        }
    }

    fn push_samples(&mut self, data: &[f32]) {
        // Convert to mono first
        let mono: Vec<f32> = if self.native_channels == 1 {
            data.to_vec()
        } else {
            data.chunks(self.native_channels)
                .map(|frame| frame.iter().sum::<f32>() / self.native_channels as f32)
                .collect()
        };

        // Resample to 16kHz if needed
        if self.native_rate == VAD_SAMPLE_RATE {
            self.buffer.extend_from_slice(&mono);
        } else {
            let ratio = self.native_rate as f64 / VAD_SAMPLE_RATE as f64;
            while self.resample_pos < mono.len() as f64 {
                let idx = self.resample_pos as usize;
                let frac = self.resample_pos - idx as f64;

                // Linear interpolation
                let sample = if idx + 1 < mono.len() {
                    mono[idx] * (1.0 - frac as f32) + mono[idx + 1] * frac as f32
                } else {
                    mono[idx]
                };
                self.buffer.push(sample);
                self.resample_pos += ratio;
            }
            self.resample_pos -= mono.len() as f64;
        }

        // Emit complete 512-sample frames
        while self.buffer.len() >= STREAM_FRAME_SIZE {
            let frame: Vec<f32> = self.buffer.drain(..STREAM_FRAME_SIZE).collect();
            if self.frame_sender.send(frame).is_err() {
                // Receiver dropped, stop accumulating
                return;
            }
        }
    }
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
