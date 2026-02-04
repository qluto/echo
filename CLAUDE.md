# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Echo is an offline voice input desktop application optimized for Apple Silicon. It uses Tauri 2.x for the desktop framework with a React frontend and Rust backend, combined with a Python sidecar process running MLX-Audio for speech recognition.

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Backend**: Tauri 2.x, Rust
- **ASR Engine**: Python 3.11+, MLX-Audio, Whisper/Qwen3-ASR (runs as sidecar process)
- **Post-Processing**: MLX LLM (Qwen3-1.7B-4bit) for cleaning up transcription results
- **Target Platform**: macOS 14.0+ on Apple Silicon (M1/M2/M3/M4)

## Common Commands

```bash
# Development
npm run tauri:dev         # Start development server with Tauri

# Build
npm run build             # Build frontend only
npm run tauri:build       # Build full desktop app

# Frontend only
npm run dev               # Start Vite dev server (no Tauri)

# Python engine binary build (PyInstaller)
cd python-engine
./build.sh                # Creates mlx-asr-engine-aarch64-apple-darwin in src-tauri/binaries/

# Python engine (manual testing - requires venv)
cd python-engine
source venv/bin/activate
python engine.py single <audio_path> [language]  # Single transcription
python engine.py daemon                           # JSON-RPC mode

# Release (creates tag and triggers GitHub Actions)
git tag -a v0.x.x -m "Release v0.x.x - description"
git push origin v0.x.x
```

## Architecture

### Data Flow

1. **Hotkey pressed** → `hotkey.rs` receives event, detects active app via `active_app.rs`
2. **Start recording** → `audio_capture.rs` captures audio to temp file
3. **Hotkey released** → `hotkey.rs` stops recording, triggers transcription
4. **Transcription** → `transcription.rs` sends JSON-RPC to Python sidecar with app context
5. **ASR Engine** → `engine.py` uses MLX-Audio (Whisper or Qwen3-ASR) for speech-to-text
6. **Post-Processing** (optional) → LLM cleans up filler words, self-corrections using custom prompt
7. **Result** → Emitted via Tauri events to React frontend
8. **Auto-insert** → Text inserted via clipboard + simulated paste

### Key Patterns

- **Python Sidecar (PyInstaller Bundled)**: The ASR engine runs as a standalone PyInstaller binary (`mlx-asr-engine-*`), communicating via JSON-RPC over stdin/stdout. Managed by `transcription.rs`. The binary is built by `python-engine/build.sh` and placed in `src-tauri/binaries/`. Requires Python 3.11 (not 3.13 - PyInstaller compatibility issues).
- **Lazy Model Loading**: Models are loaded on first use, not at startup. Status tracked via `ModelStatus`. Post-processor LLM auto-loads on app startup if enabled.
- **Global Hotkey**: Press-and-hold recording via tauri-plugin-global-shortcut. Logic in `hotkey.rs` handles both pressed and released states.
- **Dual Transcription Paths**: Transcription can be triggered via hotkey (handled entirely in Rust) or manual stop (via frontend hook).
- **Model Switching**: Users can switch between Whisper and Qwen3-ASR models at runtime via Settings panel.
- **Context-Aware Post-Processing**: Active app detection (via NSWorkspace API in `active_app.rs`) provides context for LLM-based cleanup. Customizable system prompts in Advanced Settings.

### ASR Model Support

Two model families are supported with different MLX-Audio APIs:

**Whisper models** (OpenAI):
- Uses `mlx_audio.stt.utils.load_model` + `generate_transcription`
- Requires separate `WhisperProcessor` from transformers
- Language: ISO codes ("en", "ja", etc.) or None for auto-detect

**Qwen3-ASR models** (Alibaba):
- Uses `mlx_audio.stt.load` + `model.generate()`
- Self-contained, no separate processor needed
- Language: Full names ("English", "Japanese") - does NOT accept None

### Post-Processing Pipeline

Optional LLM-based cleanup using Qwen3-1.7B-4bit:

**Default behavior** (SYSTEM_PROMPT in `engine.py`):
- Removes filler words (um, uh, like, あの, えーと, etc.)
- Handles self-corrections ("3時に、いや4時に" → "4時に")
- Applies user dictionary replacements
- Context-aware formatting based on active app (e.g., polite language for email clients)

**Customization**:
- Users can edit system prompt via Advanced Settings in UI
- Custom prompts stored in `PostProcessSettings.custom_prompt`
- Active app info (bundle ID, app name) passed to LLM for context-aware output

**Performance**:
- Model auto-loads on app startup if post-processing enabled
- Processing time: ~100-300ms on Apple Silicon
- Uses `/no_think` mode for faster generation

### Important Files

- `src-tauri/src/lib.rs` - Tauri commands, app state (Settings, RecordingState, ASREngine, PostProcessSettings)
- `src-tauri/src/hotkey.rs` - Hotkey press/release handling, coordinates recording + transcription
- `src-tauri/src/transcription.rs` - JSON-RPC communication with Python engine
- `src-tauri/src/active_app.rs` - Active application detection via NSWorkspace API
- `src/hooks/useTranscription.ts` - React hook for transcription state management
- `src/lib/tauri.ts` - TypeScript bindings for Tauri commands
- `python-engine/engine.py` - MLX-Audio wrapper (Whisper/Qwen3-ASR) + PostProcessor class for LLM cleanup

## Configuration

### Settings (stored in app state)

- `hotkey`: Recording shortcut (default: "CommandOrControl+Shift+Space")
- `language`: Recognition language ("auto" or ISO code like "ja", "en")
- `auto_insert`: Auto-paste transcription to active app (boolean)
- `device_name`: Audio input device selection
- `postprocess`: Post-processing settings object
  - `enabled`: Enable LLM-based cleanup (removes fillers, self-corrections)
  - `dictionary`: Custom word replacements (e.g., "GPT" → "GPT-4")
  - `custom_prompt`: Optional custom system prompt (null = use default)

### Available Models

Qwen3-ASR (recommended for accuracy):
- `mlx-community/Qwen3-ASR-0.6B-8bit` (default)
- `mlx-community/Qwen3-ASR-1.7B-8bit`

Whisper:
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/whisper-large-v3`
- `mlx-community/whisper-medium`
- `mlx-community/whisper-small`
- `mlx-community/whisper-base`
- `mlx-community/whisper-tiny`

### macOS Permissions

Microphone access requires `NSMicrophoneUsageDescription` in `src-tauri/Info.plist`.

### Model Cache Location

Models are cached in the app-managed directory:
- **macOS**: `~/Library/Caches/io.qluto.echo/huggingface/`

This ensures models are removed when the app is uninstalled. The following environment variables are set when starting the Python engine:
- `HF_HOME`: Hugging Face Hub cache (Whisper, Qwen3-ASR models)
- `TORCH_HOME`: PyTorch Hub cache (Silero VAD model)

## Build & Release Process

### Python Engine Binary Build

The Python ASR engine is bundled using PyInstaller:
- Script: `python-engine/build.sh`
- Requirements: Python 3.11 (ARM native) - **NOT 3.13** (PyInstaller compatibility issues)
- Output: `src-tauri/binaries/mlx-asr-engine-aarch64-apple-darwin`
- Local: Uses/creates venv automatically
- CI: Uses system Python (GitHub Actions on macos-14)

Key dependencies bundled:
- MLX framework (mlx.metallib, libmlx.dylib)
- mlx-audio for speech recognition
- Whisper/Qwen3-ASR model loaders

### GitHub Actions Release Workflow

Triggered by: `git push origin v*.*.*` tags

Workflow steps (`.github/workflows/release.yml`):
1. **Build Python Binary**: Runs `python-engine/build.sh` on macos-14 (Apple Silicon)
2. **Version Update**: Syncs version in `package.json`, `tauri.conf.json`, `Cargo.toml`
3. **Code Signing**: Imports Apple Developer certificate, signs app
4. **Notarization**: Submits to Apple for notarization
5. **Release Upload**: Uploads signed DMG and app bundle to GitHub Release

Required secrets:
- `APPLE_CERTIFICATE` / `APPLE_CERTIFICATE_PASSWORD`
- `APPLE_SIGNING_IDENTITY`
- `APPLE_ID` / `APPLE_PASSWORD` / `APPLE_TEAM_ID`

## Development Notes

- When modifying transcription flow, check both `hotkey.rs` (for hotkey-triggered) and `useTranscription.ts` (for manual)
- Python engine logs to stderr (stdout reserved for JSON-RPC)
- Temporary audio files are created in system temp directory and cleaned up after transcription
- When adding new ASR models, update: `engine.py` (AVAILABLE_MODELS, load/transcribe logic), `SettingsPanel.tsx` (MODEL_SIZES, MODEL_ORDER), `App.tsx` (MODEL_SIZES)
- Qwen3-ASR returns `null` for language field - Python must provide default value for Rust JSON parsing
- Python binary must be rebuilt after `engine.py` changes: `cd python-engine && ./build.sh`
- Post-processor uses Qwen3-1.7B-4bit with `/no_think` mode for fast cleanup (~100-300ms)
- Custom prompts are stored as `null` when they match the default to simplify version upgrades
- Active app detection happens on hotkey press, providing context for both ASR and post-processing
