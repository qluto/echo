# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Echo is an offline voice input desktop application optimized for Apple Silicon. It uses Tauri 2.x for the desktop framework with a React frontend and Rust backend, combined with a Python sidecar process running MLX-Audio for speech recognition.

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Backend**: Tauri 2.x, Rust
- **ASR Engine**: Python 3.11+, MLX-Audio, Whisper/Qwen3-ASR (runs as sidecar process)
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

# Python engine (manual testing)
cd python-engine
source venv/bin/activate
python engine.py single <audio_path> [language]  # Single transcription
python engine.py daemon                           # JSON-RPC mode
```

## Architecture

### Data Flow

1. **Hotkey pressed** → `hotkey.rs` receives event
2. **Start recording** → `audio_capture.rs` captures audio to temp file
3. **Hotkey released** → `hotkey.rs` stops recording, triggers transcription
4. **Transcription** → `transcription.rs` sends JSON-RPC to Python sidecar
5. **ASR Engine** → `engine.py` uses MLX-Audio (Whisper or Qwen3-ASR) for speech-to-text
6. **Result** → Emitted via Tauri events to React frontend
7. **Auto-insert** → Text inserted via clipboard + simulated paste

### Key Patterns

- **Python Sidecar**: The ASR engine runs as a separate Python process, communicating via JSON-RPC over stdin/stdout. Managed by `transcription.rs`.
- **Lazy Model Loading**: Models are loaded on first use, not at startup. Status tracked via `ModelStatus`.
- **Global Hotkey**: Press-and-hold recording via tauri-plugin-global-shortcut. Logic in `hotkey.rs` handles both pressed and released states.
- **Dual Transcription Paths**: Transcription can be triggered via hotkey (handled entirely in Rust) or manual stop (via frontend hook).
- **Model Switching**: Users can switch between Whisper and Qwen3-ASR models at runtime via Settings panel.

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

### Important Files

- `src-tauri/src/lib.rs` - Tauri commands, app state (Settings, RecordingState, ASREngine)
- `src-tauri/src/hotkey.rs` - Hotkey press/release handling, coordinates recording + transcription
- `src-tauri/src/transcription.rs` - JSON-RPC communication with Python engine
- `src/hooks/useTranscription.ts` - React hook for transcription state management
- `src/lib/tauri.ts` - TypeScript bindings for Tauri commands
- `python-engine/engine.py` - MLX-Audio wrapper supporting both Whisper and Qwen3-ASR

## Configuration

### Settings (stored in app state)

- `hotkey`: Recording shortcut (default: "CommandOrControl+Shift+Space")
- `language`: Recognition language ("auto" or ISO code like "ja", "en")
- `auto_insert`: Auto-paste transcription to active app (boolean)
- `device_name`: Audio input device selection

### Available Models

Qwen3-ASR (recommended for accuracy):
- `mlx-community/Qwen3-ASR-1.7B-8bit` (default)
- `mlx-community/Qwen3-ASR-0.6B-8bit`

Whisper:
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/whisper-large-v3`
- `mlx-community/whisper-medium`
- `mlx-community/whisper-small`
- `mlx-community/whisper-base`
- `mlx-community/whisper-tiny`

### macOS Permissions

Microphone access requires `NSMicrophoneUsageDescription` in `src-tauri/Info.plist`.

## Development Notes

- When modifying transcription flow, check both `hotkey.rs` (for hotkey-triggered) and `useTranscription.ts` (for manual)
- Python engine logs to stderr (stdout reserved for JSON-RPC)
- Temporary audio files are created in system temp directory and cleaned up after transcription
- When adding new ASR models, update: `engine.py` (AVAILABLE_MODELS, load/transcribe logic), `SettingsPanel.tsx` (MODEL_SIZES, MODEL_ORDER), `App.tsx` (MODEL_SIZES)
- Qwen3-ASR returns `null` for language field - Python must provide default value for Rust JSON parsing
