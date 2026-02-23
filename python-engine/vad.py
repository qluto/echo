"""
Voice Activity Detector using Silero VAD.

Wraps the Silero VAD v5 model for pre-transcription speech detection.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Silero VAD wrapper for pre-transcription speech detection"""

    def __init__(self, threshold: float = 0.3, min_speech_duration_ms: int = 250):
        """
        Initialize VAD with configurable parameters.

        Args:
            threshold: Voice probability threshold (0.0-1.0). Default 0.3 (same as Handy reference).
            min_speech_duration_ms: Minimum speech duration to consider as valid speech.
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self._model = None
        self._loaded = False

    def load(self) -> dict:
        """Load the Silero VAD model"""
        if self._loaded:
            return {"success": True, "already_loaded": True}

        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
            self._loaded = True
            logger.info("Silero VAD model loaded")
            return {"success": True}
        except ImportError as e:
            logger.error(f"silero-vad not installed: {e}")
            return {"success": False, "error": f"silero-vad not installed: {e}"}
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            return {"success": False, "error": str(e)}

    def has_speech(self, audio_path: str) -> tuple[bool, dict]:
        """
        Check if audio file contains speech.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (has_speech: bool, info: dict)
            info contains: speech_duration_ms, total_duration_ms, speech_ratio
        """
        if not self._loaded or self._model is None:
            # VAD not loaded, assume speech is present (fallback behavior)
            logger.warning("VAD not loaded, assuming speech is present")
            return True, {"vad_skipped": True}

        try:
            import torch
            import librosa
            from silero_vad import get_speech_timestamps

            # Read audio using librosa and resample to 16kHz
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            wav_tensor = torch.from_numpy(wav).float()

            # Get speech timestamps
            timestamps = get_speech_timestamps(
                wav_tensor,
                self._model,
                sampling_rate=16000,
                threshold=self.threshold
            )

            # Calculate durations (samples / 16 = ms at 16kHz)
            total_duration_ms = len(wav) / 16.0
            speech_duration_ms = sum(
                (t['end'] - t['start']) / 16.0 for t in timestamps
            )
            speech_ratio = speech_duration_ms / total_duration_ms if total_duration_ms > 0 else 0

            has_speech = speech_duration_ms >= self.min_speech_duration_ms

            info = {
                "speech_duration_ms": round(speech_duration_ms, 2),
                "total_duration_ms": round(total_duration_ms, 2),
                "speech_ratio": round(speech_ratio, 4),
                "speech_segments": len(timestamps),
            }

            logger.info(f"VAD result: has_speech={has_speech}, info={info}")
            return has_speech, info

        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            # On error, assume speech is present (fallback behavior)
            return True, {"vad_error": str(e)}

    def get_status(self) -> dict:
        """Get VAD status"""
        return {
            "loaded": self._loaded,
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
        }
