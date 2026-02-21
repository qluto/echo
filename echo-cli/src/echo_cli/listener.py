"""Continuous listening pipeline: recorder → queue → transcription worker → database."""

import json
import logging
import queue
import threading
from typing import Optional

from echo_cli.database import TranscriptionDatabase, TranscriptionEntry
from echo_cli.recorder import SpeechSegment, StreamingRecorder

logger = logging.getLogger(__name__)


class ContinuousListener:
    """Orchestrates continuous listening with async transcription.

    Architecture:
        [sounddevice callback]  →  Queue<SpeechSegment>  →  [transcription worker thread]
          runs VAD per frame                                   dequeues segments
          detects speech                                       calls ASR engine
          enqueues segments                                    saves to database
    """

    def __init__(
        self,
        engine,
        vad_model,
        database: TranscriptionDatabase,
        language: Optional[str] = None,
        silence_sec: float = 1.5,
        max_segment_sec: int = 60,
        device: Optional[int] = None,
    ):
        self._engine = engine
        self._vad = vad_model
        self._db = database
        self._language = language
        self._silence_sec = silence_sec
        self._max_segment_sec = max_segment_sec
        self._device = device

        self._segment_queue: queue.Queue[SpeechSegment] = queue.Queue(maxsize=10)
        self._recorder: Optional[StreamingRecorder] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._segment_count = 0

    def run(self):
        """Start continuous listening. Blocks until Ctrl+C."""
        self._stop_event.clear()
        self._segment_count = 0

        # Start transcription worker thread
        self._worker_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="transcription-worker",
        )
        self._worker_thread.start()

        # Start streaming recorder
        self._recorder = StreamingRecorder(
            vad_model=self._vad,
            silence_duration_sec=self._silence_sec,
            max_segment_sec=self._max_segment_sec,
            device=self._device,
        )
        self._recorder.start(self._segment_queue)

        lang_str = self._language or "auto"
        print(f"Listening... (language={lang_str}, silence={self._silence_sec}s)")
        print("Press Ctrl+C to stop.\n")

        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        """Graceful shutdown: stop recorder, drain queue, stop worker."""
        print("\nStopping...")

        # Stop recorder first (no more segments will be produced)
        if self._recorder:
            self._recorder.stop()

        # Signal worker to stop after draining queue
        self._stop_event.set()

        # Wait for worker to finish processing remaining segments
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=30)

        print(f"Done. {self._segment_count} segment(s) transcribed.")

    def _transcription_worker(self):
        """Worker thread: dequeues speech segments and transcribes them."""
        while not self._stop_event.is_set() or not self._segment_queue.empty():
            try:
                segment = self._segment_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self._process_segment(segment)
            except Exception as e:
                logger.error("Failed to process segment: %s", e)
            finally:
                # Always clean up temp file
                segment.audio_path.unlink(missing_ok=True)
                self._segment_queue.task_done()

    def _process_segment(self, segment: SpeechSegment):
        """Transcribe a single speech segment and save to database."""
        logger.info(
            "Processing segment: %.1fs from %s",
            segment.duration_seconds,
            segment.started_at.strftime("%H:%M:%S"),
        )

        # Suppress mlx-audio's stdout output during transcription
        import os

        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        os.dup2(devnull, 1)
        try:
            result = self._engine.transcribe(str(segment.audio_path), self._language)
        finally:
            os.dup2(old_stdout, 1)
            os.close(devnull)
            os.close(old_stdout)

        if not result.get("success"):
            logger.error("Transcription failed: %s", result.get("error"))
            return

        text = result.get("text", "").strip()
        if not text:
            logger.info("Empty transcription, skipping")
            return

        segments = result.get("segments", [])
        language = result.get("language", "")
        model_name = getattr(self._engine, "model_name", None)

        entry_id = self._db.insert(
            TranscriptionEntry(
                text=text,
                duration_seconds=segment.duration_seconds,
                language=language,
                model_name=model_name,
                segments_json=json.dumps(segments, ensure_ascii=False) if segments else None,
            )
        )

        self._segment_count += 1
        timestamp = segment.started_at.strftime("%H:%M:%S")
        print(f"  [{timestamp}] ({segment.duration_seconds:.1f}s) {text}")
