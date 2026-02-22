"""
ASR Engine using MLX-Audio.

Supports Whisper and Qwen3-ASR model families for speech-to-text
transcription on Apple Silicon.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from postprocessor import PostProcessor
from vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


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

    def __init__(self, model_name: str = "mlx-community/Qwen3-ASR-0.6B-8bit"):
        self.model_name = model_name
        self._model_loaded = False
        self._loading = False
        self._load_error: Optional[str] = None
        self._model = None
        self._generate_fn = None
        self._model_type: Optional[str] = None  # "whisper" or "qwen3"
        self._vad = VoiceActivityDetector()  # VAD for speech detection
        self._postprocessor = PostProcessor()  # LLM post-processor

    def get_status(self) -> dict:
        """Get current engine status"""
        return {
            "model_name": self.model_name,
            "loaded": self._model_loaded,
            "loading": self._loading,
            "error": self._load_error,
            "available_models": self.AVAILABLE_MODELS,
            "vad": self._vad.get_status(),
            "postprocessor": self._postprocessor.get_status(),
        }

    def is_model_cached(self, model_name: Optional[str] = None) -> dict:
        """Check if a model is already cached locally.

        Returns:
            dict with 'cached' (bool) and 'model_name' (str)
        """
        target_model = model_name or self.model_name
        try:
            from huggingface_hub import try_to_load_from_cache

            # For MLX models, check if the config.json exists in cache
            result = try_to_load_from_cache(target_model, "config.json")
            is_cached = result is not None

            logger.info(f"Model cache check: {target_model} -> cached={is_cached}")
            return {"cached": is_cached, "model_name": target_model}
        except Exception as e:
            logger.warning(f"Failed to check cache for {target_model}: {e}")
            # If we can't check, assume not cached to be safe
            return {"cached": False, "model_name": target_model}

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

    def _generate_warmup_audio(self) -> str:
        """Generate a short silent audio file for warmup.

        Returns:
            Path to the temporary audio file.
        """
        import numpy as np
        import soundfile as sf

        # Generate 0.5 seconds of silence at 16kHz (standard ASR sample rate)
        sample_rate = 16000
        duration = 0.5
        samples = int(sample_rate * duration)
        # Use very small random noise instead of pure silence
        # This ensures audio processing paths are exercised
        audio = np.random.randn(samples).astype(np.float32) * 0.001

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='echo_warmup_')
        os.close(fd)
        sf.write(temp_path, audio, sample_rate)

        return temp_path

    def warmup_model(self) -> dict:
        """Warm up the model by running inference on dummy audio.

        This triggers JIT compilation of Metal kernels, making the first
        real transcription much faster.

        Returns:
            dict with success status and warmup_time_ms
        """
        if not self._model_loaded:
            return {"success": False, "error": "Model not loaded"}

        temp_path = None
        try:
            logger.info("Starting model warmup...")
            start_time = time.time()

            # Generate warmup audio
            temp_path = self._generate_warmup_audio()

            # Run inference to trigger JIT compilation
            if self._model_type == "qwen3":
                # Qwen3-ASR warmup
                self._model.generate(temp_path)
            else:
                # Whisper warmup
                self._generate_fn(
                    self._model,
                    temp_path,
                    language=None,
                    output_path=None,
                )

            warmup_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Model warmup complete in {warmup_time_ms:.0f}ms")

            return {
                "success": True,
                "warmup_time_ms": warmup_time_ms
            }

        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")
            return {"success": False, "error": str(e)}

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def warmup_vad(self) -> dict:
        """Warm up the VAD model with dummy audio.

        Returns:
            dict with success status and warmup_time_ms
        """
        if not self._vad._loaded:
            return {"success": False, "error": "VAD not loaded"}

        temp_path = None
        try:
            logger.info("Starting VAD warmup...")
            start_time = time.time()

            # Generate warmup audio
            temp_path = self._generate_warmup_audio()

            # Run VAD to trigger JIT compilation
            self._vad.has_speech(temp_path)

            warmup_time_ms = (time.time() - start_time) * 1000
            logger.info(f"VAD warmup complete in {warmup_time_ms:.0f}ms")

            return {
                "success": True,
                "warmup_time_ms": warmup_time_ms
            }

        except Exception as e:
            logger.warning(f"VAD warmup failed (non-critical): {e}")
            return {"success": False, "error": str(e)}

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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
