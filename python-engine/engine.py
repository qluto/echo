#!/usr/bin/env python3
"""
Echo ASR Engine

MLX-Audio based speech recognition engine for the Echo application.
Supports both daemon mode (JSON-RPC over stdin/stdout) and single-shot mode.

Usage:
    python engine.py daemon           # Run in daemon mode
    python engine.py single <path>    # Transcribe a single file
"""

import sys
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

# Configure logging to stderr (stdout is reserved for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stderr
)
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


class ASREngine:
    """MLX-Audio based ASR Engine supporting Whisper and Qwen3-ASR"""

    # Available models - Whisper models
    WHISPER_MODELS = [
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/whisper-large-v3",
        "mlx-community/whisper-medium",
        "mlx-community/whisper-small",
        "mlx-community/whisper-base",
        "mlx-community/whisper-tiny",
    ]

    # Qwen3-ASR models
    QWEN3_ASR_MODELS = [
        "mlx-community/Qwen3-ASR-1.7B-8bit",
        "mlx-community/Qwen3-ASR-0.6B-8bit",
    ]

    # All available models
    AVAILABLE_MODELS = QWEN3_ASR_MODELS + WHISPER_MODELS

    @staticmethod
    def is_qwen3_model(model_name: str) -> bool:
        """Check if a model is a Qwen3-ASR model"""
        return "Qwen3-ASR" in model_name

    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self._model_loaded = False
        self._loading = False
        self._load_error: Optional[str] = None
        self._model = None
        self._generate_fn = None
        self._model_type: Optional[str] = None  # "whisper" or "qwen3"
        self._vad = VoiceActivityDetector()  # VAD for speech detection

    def get_status(self) -> dict:
        """Get current engine status"""
        return {
            "model_name": self.model_name,
            "loaded": self._model_loaded,
            "loading": self._loading,
            "error": self._load_error,
            "available_models": self.AVAILABLE_MODELS,
            "vad": self._vad.get_status(),
        }

    def load_vad(self) -> dict:
        """Load the VAD model"""
        return self._vad.load()

    def set_model(self, model_name: str) -> dict:
        """Set a new model (requires reload)"""
        if model_name not in self.AVAILABLE_MODELS:
            return {
                "success": False,
                "error": f"Unknown model: {model_name}. Available: {self.AVAILABLE_MODELS}"
            }

        # Unload current model
        self._model = None
        self._model_loaded = False
        self._load_error = None
        self._generate_fn = None
        self._model_type = None
        self.model_name = model_name
        logger.info(f"Model set to: {model_name}")

        return {"success": True, "model_name": model_name}

    def load_model(self) -> dict:
        """Load the ASR model"""
        if self._model_loaded:
            return {"success": True, "model_name": self.model_name, "already_loaded": True}

        if self._loading:
            return {"success": False, "error": "Model is already loading"}

        self._loading = True
        self._load_error = None

        try:
            logger.info(f"Loading model: {self.model_name}")

            if self.is_qwen3_model(self.model_name):
                # Load Qwen3-ASR model using the new API
                from mlx_audio.stt import load as load_qwen3

                self._model = load_qwen3(self.model_name)
                self._model_type = "qwen3"
                self._generate_fn = None  # Qwen3 uses model.generate() directly
                logger.info(f"Qwen3-ASR model loaded: {self.model_name}")
            else:
                # Load Whisper model using the original API
                from mlx_audio.stt.utils import load_model
                from mlx_audio.stt.generate import generate_transcription
                from transformers import WhisperProcessor

                # Load MLX model weights
                self._model = load_model(self.model_name)

                # MLX models don't include processor files, so we load from the original OpenAI model
                # Map MLX model names to their OpenAI equivalents for processor
                processor_model = self.model_name.replace("mlx-community/", "openai/")
                logger.info(f"Loading processor from {processor_model}")
                processor = WhisperProcessor.from_pretrained(processor_model)
                self._model._processor = processor

                self._generate_fn = generate_transcription
                self._model_type = "whisper"
                logger.info(f"Whisper model loaded: {self.model_name}")

            self._model_loaded = True
            self._loading = False
            return {"success": True, "model_name": self.model_name}

        except ImportError as e:
            self._loading = False
            self._load_error = f"MLX-Audio is not installed: {e}"
            logger.error(self._load_error)
            return {"success": False, "error": self._load_error}

        except Exception as e:
            self._loading = False
            self._load_error = f"Failed to load model: {e}"
            logger.error(self._load_error)
            return {"success": False, "error": self._load_error}

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Transcribe an audio file"""
        logger.info(f"Transcribe called with language={language}")

        if not self._model_loaded:
            self.load_model()

        path = Path(audio_path)
        if not path.exists():
            return {
                "success": False,
                "text": "",
                "segments": [],
                "language": language or "auto",
                "error": f"File not found: {audio_path}"
            }

        # VAD check - skip transcription if no speech detected
        if self._vad._loaded:
            has_speech, vad_info = self._vad.has_speech(audio_path)
            if not has_speech:
                logger.info(f"No speech detected, skipping transcription: {vad_info}")
                return {
                    "success": True,
                    "text": "",
                    "segments": [],
                    "language": language or "auto",
                    "no_speech": True,
                    "vad_info": vad_info
                }

        try:
            logger.info(f"Transcribing with {self._model_type}: {audio_path}")
            effective_language = language if language and language != "auto" else None

            if self._model_type == "qwen3":
                # Use Qwen3-ASR API
                result = self._transcribe_qwen3(audio_path, effective_language)
            else:
                # Use Whisper API
                result = self._transcribe_whisper(audio_path, effective_language)

            # Use requested language if specified, otherwise use detected
            output_language = language if language else result.get("detected_language", "auto")
            logger.info(f"Requested language: {language}, Detected language: {result.get('detected_language')}, Output language: {output_language}")
            logger.info(f"Transcription complete: {len(result['text'])} chars, output language: {output_language}")

            return {
                "success": True,
                "text": result["text"],
                "segments": result["segments"],
                "language": output_language
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "text": "",
                "segments": [],
                "language": language or "auto",
                "error": str(e)
            }

    def _transcribe_qwen3(self, audio_path: str, language: Optional[str]) -> dict:
        """Transcribe using Qwen3-ASR model"""
        logger.info(f"Calling Qwen3-ASR generate with language={language}")

        # Qwen3-ASR uses model.generate() directly
        # Language should be the full language name (e.g., "English", "Japanese")
        # Note: Qwen3-ASR doesn't accept None for language - it defaults to "English"
        lang_map = {
            "en": "English",
            "ja": "Japanese",
            "zh": "Chinese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "th": "Thai",
            "vi": "Vietnamese",
        }

        # Build kwargs - only include language if specified (otherwise use model default)
        kwargs = {}
        if language:
            qwen_language = lang_map.get(language, language)
            # Capitalize if it's a full language name that wasn't in our map
            if qwen_language and qwen_language == language and len(language) > 2:
                qwen_language = language.capitalize()
            kwargs["language"] = qwen_language
            logger.info(f"Using language: {qwen_language}")

        result = self._model.generate(audio_path, **kwargs)

        # Parse result
        text = result.text.strip() if hasattr(result, 'text') else ""
        # Qwen3-ASR may return None for language, default to "auto"
        detected_language = result.language if hasattr(result, 'language') and result.language else "auto"

        # Convert segments to our format
        segments = []
        if hasattr(result, 'segments') and result.segments:
            for seg in result.segments:
                if hasattr(seg, 'start') and hasattr(seg, 'end') and hasattr(seg, 'text'):
                    segments.append({
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip()
                    })
                elif isinstance(seg, dict):
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "text": seg.get("text", "").strip()
                    })

        return {
            "text": text,
            "segments": segments,
            "detected_language": detected_language
        }

    def _transcribe_whisper(self, audio_path: str, language: Optional[str]) -> dict:
        """Transcribe using Whisper model"""
        logger.info(f"Calling Whisper generate_transcription with language={language}")

        # Use temp directory to avoid creating transcript files that trigger Tauri hot-reload
        temp_output = tempfile.mktemp(prefix="echo_transcript_")
        result = self._generate_fn(
            self._model,
            audio_path,
            language=language,
            output_path=temp_output,
        )
        # Clean up temp file
        for ext in ['.txt', '.vtt', '.srt', '.json']:
            try:
                os.remove(temp_output + ext)
            except OSError:
                pass

        # Parse result - STTOutput object with text, segments, language attributes
        text = result.text.strip() if hasattr(result, 'text') else ""
        detected_language = result.language if hasattr(result, 'language') else None

        # Convert segments to our format
        segments = []
        if hasattr(result, 'segments') and result.segments:
            for seg in result.segments:
                if isinstance(seg, dict):
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "text": seg.get("text", "").strip()
                    })

        return {
            "text": text,
            "segments": segments,
            "detected_language": detected_language
        }


def run_daemon(engine: ASREngine):
    """Run the engine in daemon mode (JSON-RPC over stdin/stdout)"""
    logger.info("Starting daemon mode (model not loaded yet)")

    # Signal ready to parent process (engine ready, but model not loaded)
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            command = request.get("command", "")
            request_id = request.get("id", 0)

            if command == "ping":
                response = {
                    "id": request_id,
                    "result": {"pong": True}
                }

            elif command == "get_status":
                response = {
                    "id": request_id,
                    "result": engine.get_status()
                }

            elif command == "load_model":
                result = engine.load_model()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "set_model":
                model_name = request.get("model_name", "")
                if not model_name:
                    response = {
                        "id": request_id,
                        "error": "Missing model_name parameter"
                    }
                else:
                    result = engine.set_model(model_name)
                    response = {
                        "id": request_id,
                        "result": result
                    }

            elif command == "load_vad":
                result = engine.load_vad()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "transcribe":
                audio_path = request.get("audio_path", "")
                language = request.get("language")
                logger.info(f"Received transcribe command: audio_path={audio_path}, language={language}")

                if not audio_path:
                    response = {
                        "id": request_id,
                        "error": "Missing audio_path parameter"
                    }
                elif not engine._model_loaded:
                    response = {
                        "id": request_id,
                        "error": "Model not loaded. Call load_model first."
                    }
                else:
                    result = engine.transcribe(audio_path, language)
                    response = {
                        "id": request_id,
                        "result": result
                    }

            elif command == "quit":
                logger.info("Received quit command")
                response = {
                    "id": request_id,
                    "result": {"quit": True}
                }
                print(json.dumps(response), flush=True)
                break

            else:
                response = {
                    "id": request_id,
                    "error": f"Unknown command: {command}"
                }

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            print(json.dumps({"error": str(e)}), flush=True)


def run_single(engine: ASREngine, audio_path: str, language: Optional[str] = None):
    """Run a single transcription"""
    result = engine.transcribe(audio_path, language)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    # Initialize engine (model will be loaded lazily)
    engine = ASREngine()

    if mode == "daemon":
        # In daemon mode, model is loaded on demand via load_model command
        run_daemon(engine)
    elif mode == "single":
        if len(sys.argv) < 3:
            print("Usage: python engine.py single <audio_path> [language]")
            sys.exit(1)
        audio_path = sys.argv[2]
        language = sys.argv[3] if len(sys.argv) > 3 else None
        # For single mode, load model immediately
        result = engine.load_model()
        if not result.get("success"):
            logger.error(result.get("error", "Unknown error"))
            sys.exit(1)
        run_single(engine, audio_path, language)
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
