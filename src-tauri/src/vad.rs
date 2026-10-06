//! Voice Activity Detection using Silero VAD v5 via ONNX Runtime.
//!
//! Wraps the `voice_activity_detector` crate which embeds the Silero VAD model
//! and handles ONNX inference internally. Processes 32ms audio frames (512 samples
//! at 16kHz) and returns speech probability.

use anyhow::{Context, Result};
use voice_activity_detector::VoiceActivityDetector;

/// Sample rate for VAD processing (must be 16kHz for 512-sample frames).
pub const VAD_SAMPLE_RATE: u32 = 16000;

/// Frame size in samples (512 samples = 32ms at 16kHz).
pub const VAD_FRAME_SIZE: usize = 512;

/// Result of processing a single audio frame through VAD.
#[derive(Debug, Clone, Copy)]
pub enum VadEvent {
    /// No speech detected.
    Silence,
    /// Speech detected with given probability (0.0 - 1.0).
    Speech {
        #[allow(dead_code)]
        probability: f32,
    },
}

/// Voice Activity Detection processor wrapping Silero VAD v5.
///
/// Maintains internal LSTM state across frames for temporal continuity.
/// Feed sequential 512-sample (32ms) frames at 16kHz mono.
pub struct VadProcessor {
    detector: VoiceActivityDetector,
    threshold: f32,
}

impl VadProcessor {
    /// Create a new VAD processor.
    ///
    /// `threshold` is the speech probability threshold (typically 0.5).
    /// Values above the threshold are classified as speech.
    pub fn new(threshold: f32) -> Result<Self> {
        let detector = VoiceActivityDetector::builder()
            .sample_rate(VAD_SAMPLE_RATE)
            .chunk_size(VAD_FRAME_SIZE)
            .build()
            .context("Failed to initialize Silero VAD")?;

        Ok(Self {
            detector,
            threshold,
        })
    }

    /// Process a single 512-sample audio frame.
    ///
    /// `audio` must contain exactly 512 f32 samples in the range [-1.0, 1.0].
    /// Returns `VadEvent::Speech` if probability exceeds the threshold.
    pub fn process_frame(&mut self, audio: &[f32]) -> VadEvent {
        debug_assert_eq!(
            audio.len(),
            VAD_FRAME_SIZE,
            "VAD frame must be exactly {} samples",
            VAD_FRAME_SIZE
        );

        let probability = self.detector.predict(audio.iter().copied());

        if probability > self.threshold {
            VadEvent::Speech { probability }
        } else {
            VadEvent::Silence
        }
    }

    /// Get the raw speech probability for a frame without threshold classification.
    #[allow(dead_code)]
    pub fn predict(&mut self, audio: &[f32]) -> f32 {
        self.detector.predict(audio.iter().copied())
    }

    /// Reset internal LSTM state. Call between separate audio streams / segments.
    pub fn reset(&mut self) {
        // Recreate the detector to reset LSTM state.
        // voice_activity_detector doesn't expose a reset method directly.
        if let Ok(new_detector) = VoiceActivityDetector::builder()
            .sample_rate(VAD_SAMPLE_RATE)
            .chunk_size(VAD_FRAME_SIZE)
            .build()
        {
            self.detector = new_detector;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = VadProcessor::new(0.5);
        assert!(vad.is_ok(), "VAD processor should initialize successfully");
    }

    #[test]
    fn test_silence_detection() {
        let mut vad = VadProcessor::new(0.5).unwrap();

        // Feed silence (zeros) — should detect no speech
        let silence = vec![0.0f32; VAD_FRAME_SIZE];
        let event = vad.process_frame(&silence);
        assert!(
            matches!(event, VadEvent::Silence),
            "Silence should be detected for zero audio"
        );
    }

    #[test]
    fn test_predict_returns_low_for_silence() {
        let mut vad = VadProcessor::new(0.5).unwrap();

        let silence = vec![0.0f32; VAD_FRAME_SIZE];
        let prob = vad.predict(&silence);
        assert!(
            prob < 0.5,
            "Speech probability for silence should be below threshold, got {}",
            prob
        );
    }

    #[test]
    fn test_reset() {
        let mut vad = VadProcessor::new(0.5).unwrap();

        // Process some frames then reset
        let silence = vec![0.0f32; VAD_FRAME_SIZE];
        vad.process_frame(&silence);
        vad.process_frame(&silence);
        vad.reset();

        // Should still work after reset
        let event = vad.process_frame(&silence);
        assert!(matches!(event, VadEvent::Silence));
    }
}
