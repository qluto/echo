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


class ASREngine:
    """MLX-Audio based ASR Engine using Whisper"""

    # Available models
    AVAILABLE_MODELS = [
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/whisper-large-v3",
        "mlx-community/whisper-medium",
        "mlx-community/whisper-small",
        "mlx-community/whisper-base",
        "mlx-community/whisper-tiny",
    ]

    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self._model_loaded = False
        self._loading = False
        self._load_error: Optional[str] = None
        self._model = None
        self._generate_fn = None

    def get_status(self) -> dict:
        """Get current engine status"""
        return {
            "model_name": self.model_name,
            "loaded": self._model_loaded,
            "loading": self._loading,
            "error": self._load_error,
            "available_models": self.AVAILABLE_MODELS,
        }

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
            from mlx_audio.stt.utils import load_model
            from mlx_audio.stt.generate import generate_transcription
            from transformers import WhisperProcessor

            logger.info(f"Loading model: {self.model_name}")

            # Load MLX model weights
            self._model = load_model(self.model_name)

            # MLX models don't include processor files, so we load from the original OpenAI model
            # Map MLX model names to their OpenAI equivalents for processor
            processor_model = self.model_name.replace("mlx-community/", "openai/")
            logger.info(f"Loading processor from {processor_model}")
            processor = WhisperProcessor.from_pretrained(processor_model)
            self._model._processor = processor

            self._generate_fn = generate_transcription
            self._model_loaded = True
            self._loading = False
            logger.info(f"MLX-Audio model loaded: {self.model_name}")
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

        try:
            logger.info(f"Transcribing: {audio_path}")

            # Use MLX-Audio transcription
            # Use temp directory to avoid creating transcript files that trigger Tauri hot-reload
            temp_output = tempfile.mktemp(prefix="echo_transcript_")
            effective_language = language if language and language != "auto" else None
            logger.info(f"Calling generate_transcription with language={effective_language}")
            result = self._generate_fn(
                self._model,
                audio_path,
                language=effective_language,
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
            detected_language = result.language if hasattr(result, 'language') else "auto"
            # Use requested language if specified, otherwise use detected
            output_language = language if language else detected_language
            logger.info(f"Requested language: {language}, Detected language: {detected_language}, Output language: {output_language}")

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

            logger.info(f"Transcription complete: {len(text)} chars, output language: {output_language}")

            return {
                "success": True,
                "text": text,
                "segments": segments,
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
