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
from typing import Optional

# PyInstaller bundle support: Set METAL_PATH for MLX metallib
# Must be done BEFORE importing mlx
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as PyInstaller bundle
    bundle_dir = sys._MEIPASS
    metal_path = os.path.join(bundle_dir, 'mlx', 'lib')
    if os.path.exists(metal_path):
        os.environ['METAL_PATH'] = metal_path
        # Also ensure the dylib can be found
        os.environ['DYLD_LIBRARY_PATH'] = metal_path + ':' + os.environ.get('DYLD_LIBRARY_PATH', '')

# Configure logging to stderr (stdout is reserved for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

from asr import ASREngine


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

            elif command == "is_model_cached":
                model_name = request.get("model_name")
                result = engine.is_model_cached(model_name)
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "warmup_model":
                result = engine.warmup_model()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "warmup_vad":
                result = engine.warmup_vad()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "load_postprocess_model":
                result = engine._postprocessor.load_model()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "unload_postprocess_model":
                result = engine._postprocessor.unload_model()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "is_postprocess_model_cached":
                model_name = request.get("model_name")
                result = engine._postprocessor.is_model_cached(model_name)
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "set_postprocess_model":
                model_name = request.get("model_name", "")
                if not model_name:
                    response = {
                        "id": request_id,
                        "error": "Missing model_name parameter"
                    }
                else:
                    result = engine._postprocessor.set_model(model_name)
                    response = {
                        "id": request_id,
                        "result": result
                    }

            elif command == "get_postprocess_status":
                result = engine._postprocessor.get_status()
                response = {
                    "id": request_id,
                    "result": result
                }

            elif command == "postprocess_text":
                text = request.get("text", "")
                app_name = request.get("app_name")
                app_bundle_id = request.get("app_bundle_id")
                dictionary = request.get("dictionary")
                custom_prompt = request.get("custom_prompt")

                if not text:
                    response = {
                        "id": request_id,
                        "result": {"success": True, "processed_text": "", "processing_time_ms": 0}
                    }
                elif not engine._postprocessor._loaded:
                    response = {
                        "id": request_id,
                        "error": "Post-processor model not loaded. Call load_postprocess_model first."
                    }
                else:
                    result = engine._postprocessor.process(
                        text=text,
                        app_name=app_name,
                        app_bundle_id=app_bundle_id,
                        dictionary=dictionary,
                        custom_prompt=custom_prompt,
                    )
                    response = {
                        "id": request_id,
                        "result": result
                    }

            elif command == "summarize_transcriptions":
                texts = request.get("texts", [])
                language_hint = request.get("language_hint")
                custom_prompt = request.get("custom_prompt")

                if not texts:
                    response = {
                        "id": request_id,
                        "result": {"success": True, "summary": "", "processing_time_ms": 0}
                    }
                elif not engine._postprocessor._loaded:
                    response = {
                        "id": request_id,
                        "error": "Post-processor model not loaded. Call load_postprocess_model first."
                    }
                else:
                    result = engine._postprocessor.summarize(
                        texts=texts,
                        language_hint=language_hint,
                        custom_prompt=custom_prompt,
                    )
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
    # PyInstaller multiprocessing support
    # When torch/transformers spawn subprocesses, they may call this executable
    # with Python flags like -B. We need to handle this gracefully.
    import multiprocessing
    multiprocessing.freeze_support()

    # Filter out Python interpreter flags that might be passed in PyInstaller bundles
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

    if len(args) < 1:
        # Check if we're being called as a subprocess (no args but has Python flags)
        if any(arg.startswith('-') for arg in sys.argv[1:]):
            # Silently exit - this is a multiprocessing child process
            sys.exit(0)
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    mode = args[0]

    # Initialize engine (model will be loaded lazily)
    engine = ASREngine()

    if mode == "daemon":
        # In daemon mode, model is loaded on demand via load_model command
        run_daemon(engine)
    elif mode == "single":
        if len(args) < 2:
            print("Usage: python engine.py single <audio_path> [language]", file=sys.stderr)
            sys.exit(1)
        audio_path = args[1]
        language = args[2] if len(args) > 2 else None
        # For single mode, load model immediately
        result = engine.load_model()
        if not result.get("success"):
            logger.error(result.get("error", "Unknown error"))
            sys.exit(1)
        run_single(engine, audio_path, language)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
