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

    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self._model_loaded = False
        self._model = None
        self._generate_fn = None

    def load_model(self) -> bool:
        """Load the ASR model"""
        if self._model_loaded:
            return True

        try:
            from mlx_audio.stt import load_model
            from mlx_audio.stt.generate import generate_transcription
            from transformers import WhisperProcessor

            logger.info(f"Loading model: {self.model_name}")

            # Load MLX model weights
            self._model = load_model(self.model_name)

            # MLX models don't include processor files, so we load from the original OpenAI model
            logger.info("Loading processor from openai/whisper-large-v3-turbo")
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
            self._model._processor = processor

            self._generate_fn = generate_transcription
            self._model_loaded = True
            logger.info(f"MLX-Audio model loaded: {self.model_name}")
            return True

        except ImportError as e:
            logger.error(f"MLX-Audio is not installed: {e}")
            logger.error("Please install mlx-audio: pip install mlx-audio")
            raise RuntimeError(
                "MLX-Audio is required but not installed. "
                "Install with: pip install mlx-audio"
            ) from e

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load ASR model: {e}") from e

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Transcribe an audio file"""
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
            result = self._generate_fn(
                self._model,
                audio_path,
                language=language if language and language != "auto" else None,
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
            detected_language = result.language if hasattr(result, 'language') else (language or "auto")

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

            logger.info(f"Transcription complete: {len(text)} chars, language: {detected_language}")

            return {
                "success": True,
                "text": text,
                "segments": segments,
                "language": detected_language
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
    logger.info("Starting daemon mode")

    # Signal ready to parent process
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

            elif command == "transcribe":
                audio_path = request.get("audio_path", "")
                language = request.get("language")

                if not audio_path:
                    response = {
                        "id": request_id,
                        "error": "Missing audio_path parameter"
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

    # Initialize engine
    engine = ASREngine()

    # Load model (will raise error if mlx_audio is not installed)
    try:
        engine.load_model()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    if mode == "daemon":
        run_daemon(engine)
    elif mode == "single":
        if len(sys.argv) < 3:
            print("Usage: python engine.py single <audio_path> [language]")
            sys.exit(1)
        audio_path = sys.argv[2]
        language = sys.argv[3] if len(sys.argv) > 3 else None
        run_single(engine, audio_path, language)
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
