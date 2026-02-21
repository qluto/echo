"""Streaming VAD recorder using sounddevice + Silero VAD."""

import logging
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment ready for transcription."""

    audio_path: Path
    duration_seconds: float
    started_at: datetime


class StreamingRecorder:
    """Real-time speech segment detection using sounddevice + Silero VAD.

    Records audio from microphone, runs VAD frame-by-frame, and emits
    complete speech segments to a queue for async processing.
    """

    SAMPLE_RATE = 16000
    FRAME_SIZE = 512  # 32ms at 16kHz

    def __init__(
        self,
        vad_model,
        speech_threshold: float = 0.5,
        silence_duration_sec: float = 1.5,
        max_segment_sec: int = 60,
        pre_buffer_sec: float = 0.3,
        device: Optional[int] = None,
    ):
        self._vad = vad_model
        self._speech_threshold = speech_threshold
        self._silence_frames = int(silence_duration_sec * self.SAMPLE_RATE / self.FRAME_SIZE)
        self._max_frames = int(max_segment_sec * self.SAMPLE_RATE / self.FRAME_SIZE)
        self._pre_buffer_frames = int(pre_buffer_sec * self.SAMPLE_RATE / self.FRAME_SIZE)
        self._device = device

        self._segment_queue: Optional[queue.Queue] = None
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._tmp_dir: Optional[Path] = None

        # VAD state
        self._is_speaking = False
        self._silence_count = 0
        self._speech_frames: list[np.ndarray] = []
        self._pre_buffer: list[np.ndarray] = []
        self._speech_frame_count = 0
        self._speech_start_time: Optional[datetime] = None

        # Thread safety: sounddevice callback runs in a separate thread
        self._lock = threading.Lock()

    def start(self, segment_queue: queue.Queue):
        """Start recording from microphone. Detected segments go to segment_queue."""
        self._segment_queue = segment_queue
        self._running = True
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="echo_vad_"))
        self._vad.reset_states()

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.FRAME_SIZE,
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Streaming recorder started (device=%s)", self._device)

    def stop(self):
        """Stop recording and clean up."""
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Flush any remaining speech
        with self._lock:
            if self._is_speaking and self._speech_frames:
                self._finalize_segment()

        # Clean up temp dir (segments already consumed should be deleted by listener)
        if self._tmp_dir and self._tmp_dir.exists():
            for f in self._tmp_dir.iterdir():
                f.unlink(missing_ok=True)
            self._tmp_dir.rmdir()

        logger.info("Streaming recorder stopped")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called by sounddevice for each audio frame."""
        if status:
            logger.warning("Audio callback status: %s", status)

        if not self._running:
            return

        audio = indata[:, 0]  # mono
        chunk = torch.from_numpy(audio.copy()).float()

        try:
            speech_prob = self._vad(chunk, self.SAMPLE_RATE).item()
        except Exception as e:
            logger.error("VAD inference error: %s", e)
            return

        with self._lock:
            if not self._is_speaking:
                # Maintain pre-buffer (rolling window of recent frames)
                self._pre_buffer.append(audio.copy())
                if len(self._pre_buffer) > self._pre_buffer_frames:
                    self._pre_buffer.pop(0)

                if speech_prob > self._speech_threshold:
                    # Speech started
                    self._is_speaking = True
                    self._silence_count = 0
                    self._speech_frame_count = 0
                    self._speech_start_time = datetime.now()

                    # Include pre-buffer to avoid cutting off speech onset
                    self._speech_frames = list(self._pre_buffer)
                    self._pre_buffer.clear()
                    self._speech_frames.append(audio.copy())
                    self._speech_frame_count += 1

                    logger.debug("Speech started (prob=%.3f)", speech_prob)
            else:
                # Currently speaking
                self._speech_frames.append(audio.copy())
                self._speech_frame_count += 1

                if speech_prob < self._speech_threshold:
                    self._silence_count += 1
                else:
                    self._silence_count = 0

                # Check if segment should end
                if self._silence_count >= self._silence_frames:
                    logger.debug(
                        "Speech ended after %.1fs silence",
                        self._silence_count * self.FRAME_SIZE / self.SAMPLE_RATE,
                    )
                    self._finalize_segment()
                elif self._speech_frame_count >= self._max_frames:
                    logger.info("Max segment duration reached, forcing split")
                    self._finalize_segment()

    def _finalize_segment(self):
        """Save accumulated speech frames as WAV and enqueue."""
        if not self._speech_frames:
            self._reset_state()
            return

        audio_data = np.concatenate(self._speech_frames)
        duration = len(audio_data) / self.SAMPLE_RATE

        # Skip very short segments (< 0.5s of actual speech)
        if duration < 0.5:
            logger.debug("Skipping short segment (%.2fs)", duration)
            self._reset_state()
            return

        # Save to temp WAV file
        filename = f"segment_{int(time.time() * 1000)}.wav"
        filepath = self._tmp_dir / filename
        sf.write(str(filepath), audio_data, self.SAMPLE_RATE)

        segment = SpeechSegment(
            audio_path=filepath,
            duration_seconds=duration,
            started_at=self._speech_start_time or datetime.now(),
        )

        try:
            self._segment_queue.put_nowait(segment)
            logger.info(
                "Segment queued: %.1fs at %s",
                duration,
                segment.started_at.strftime("%H:%M:%S"),
            )
        except queue.Full:
            logger.warning("Segment queue full, dropping segment (%.1fs)", duration)
            filepath.unlink(missing_ok=True)

        self._reset_state()

    def _reset_state(self):
        """Reset VAD state for next segment."""
        self._is_speaking = False
        self._silence_count = 0
        self._speech_frames.clear()
        self._speech_frame_count = 0
        self._speech_start_time = None
        self._vad.reset_states()
