# CLAUDE.md - Echo Voice Input App

This file provides guidance for Claude Code when working on this repository.

## Project Overview

Echo is an offline voice input desktop application optimized for Apple Silicon. It uses Tauri 2.x for the desktop framework with a React frontend and Rust backend, combined with a Python sidecar process running MLX-Audio + Whisper for speech recognition.

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Backend**: Tauri 2.x, Rust
- **ASR Engine**: Python 3.11+, MLX-Audio, Whisper (runs as sidecar process)
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

## Project Structure

```
lisbon/
├── src/                    # React frontend
│   ├── components/         # UI components
│   ├── hooks/              # React hooks (useTranscription.ts)
│   ├── lib/                # Tauri API bindings (tauri.ts)
│   └── App.tsx             # Main app component
├── src-tauri/              # Rust backend
│   ├── src/
│   │   ├── lib.rs          # Main Tauri commands and app state
│   │   ├── hotkey.rs       # Global hotkey handling
│   │   ├── audio_capture.rs # Audio recording
│   │   ├── transcription.rs # ASR engine communication
│   │   ├── clipboard.rs    # Clipboard operations
│   │   └── input.rs        # Keyboard input simulation
│   ├── tauri.conf.json     # Tauri configuration
│   └── Info.plist          # macOS permissions (microphone)
└── python-engine/          # Python ASR engine
    ├── engine.py           # Main engine script
    ├── requirements.txt    # Python dependencies
    └── venv/               # Python virtual environment
```

## Architecture

### Data Flow

1. **Hotkey pressed** → `hotkey.rs` receives event
2. **Start recording** → `audio_capture.rs` captures audio to temp file
3. **Hotkey released** → `hotkey.rs` stops recording, triggers transcription
4. **Transcription** → `transcription.rs` sends JSON-RPC to Python sidecar
5. **ASR Engine** → `engine.py` uses MLX-Audio/Whisper for speech-to-text
6. **Result** → Emitted via Tauri events to React frontend
7. **Auto-insert** → Text inserted via clipboard + simulated paste

### Key Patterns

- **Python Sidecar**: The ASR engine runs as a separate Python process, communicating via JSON-RPC over stdin/stdout. Managed by `transcription.rs`.
- **Lazy Model Loading**: The Whisper model is loaded on first use, not at startup. Status tracked via `ModelStatus`.
- **Global Hotkey**: Press-and-hold recording via tauri-plugin-global-shortcut. Logic in `hotkey.rs` handles both pressed and released states.
- **Dual Transcription Paths**: Transcription can be triggered via hotkey (handled entirely in Rust) or manual stop (via frontend hook).

### Important Files

- `src-tauri/src/lib.rs` - Tauri commands, app state (Settings, RecordingState, ASREngine)
- `src-tauri/src/hotkey.rs` - Hotkey press/release handling, coordinates recording + transcription
- `src-tauri/src/transcription.rs` - JSON-RPC communication with Python engine
- `src/hooks/useTranscription.ts` - React hook for transcription state management
- `src/lib/tauri.ts` - TypeScript bindings for Tauri commands
- `python-engine/engine.py` - MLX-Audio Whisper wrapper, daemon mode implementation

## Configuration

### Settings (stored in app state)

- `hotkey`: Recording shortcut (default: "CommandOrControl+Shift+Space")
- `language`: Recognition language ("auto" or ISO code like "ja", "en")
- `auto_insert`: Auto-paste transcription to active app (boolean)
- `device_name`: Audio input device selection

### macOS Permissions

Microphone access requires `NSMicrophoneUsageDescription` in `src-tauri/Info.plist`.

## Development Notes

- When modifying transcription flow, check both `hotkey.rs` (for hotkey-triggered) and `useTranscription.ts` (for manual)
- Python engine logs to stderr (stdout reserved for JSON-RPC)
- Temporary audio files are created in system temp directory and cleaned up after transcription
- The ASR engine supports multiple Whisper model sizes (tiny to large-v3-turbo)
