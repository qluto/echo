# Echo - Voice Input Application

Offline voice input desktop application optimized for Apple Silicon

## Features

- **High-Accuracy Speech Recognition**: Powered by MLX-Audio with Whisper/Qwen3-ASR models
- **Model Switching**: Switch between Whisper and Qwen3-ASR models in Settings
- **LLM-Based Post-Processing**: Optional cleanup of filler words and self-corrections using on-device LLM
- **Context-Aware Formatting**: Detects active application for context-appropriate output
- **Customizable Prompts**: Edit post-processing behavior via Advanced Settings
- **Always-On Transcription**: Continuous listening mode with automatic speech detection (Silero VAD)
- **Global Hotkey**: System-wide keyboard shortcut for instant recording
- **Real-time Transcription**: Immediate transcription after recording ends
- **Auto Text Insertion**: Automatically paste transcription into active applications
- **Transcription History**: SQLite-backed history with full-text search for always-on mode
- **Fully Offline**: No internet connection required - all processing happens locally

## System Requirements

- **OS**: macOS 14.0 (Sonoma) or later
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **Memory**: 8GB+ recommended
- **Storage**: 2GB+ free space (varies by model size)

## Supported Models

### Qwen3-ASR (Recommended)
- `Qwen3-ASR-0.6B-8bit` - Lightweight, faster inference (default)
- `Qwen3-ASR-1.7B-8bit` - High accuracy, 52 languages supported

### Whisper (OpenAI)
- `whisper-large-v3-turbo` - Balanced performance
- `whisper-large-v3` - Highest accuracy
- `whisper-medium` / `small` / `base` / `tiny` - Lightweight models

## Installation

Download the latest release from the [Releases](https://github.com/qluto/echo/releases) page:

1. Download the `.dmg` file
2. Open the DMG and drag Echo to your Applications folder
3. Launch Echo from Applications
4. Grant microphone permissions when prompted

The app is self-contained - no additional Python installation required!

## Usage

1. Launch the app
2. Press and hold `Cmd+Shift+Space` to start recording
3. Speak your message
4. Release the key when finished
5. Transcription appears automatically
6. Text is auto-inserted into the active application (if enabled in Settings)

### Always-On Transcription

Echo also supports continuous listening mode, which runs in the background and automatically detects speech:

1. Click the "Start Listening" button in the app
2. Echo continuously monitors the microphone using Silero VAD (Voice Activity Detection)
3. When speech is detected, it automatically records the segment
4. After a silence pause (1.5s), the segment is transcribed and saved to history
5. Click "Stop Listening" to end the session

Transcription history is stored in a local SQLite database and can be searched via the History panel.

## Settings

Customize Echo via the Settings panel:

- **ASR Model**: Choose between Qwen3-ASR or Whisper models
- **Hotkey**: Customize the recording keyboard shortcut
- **Recognition Language**: Auto-detect or manually specify language
- **Input Device**: Select your preferred microphone
- **Auto Insert**: Enable/disable automatic paste after transcription
- **Post-Processing**: Enable LLM-based cleanup to remove filler words and self-corrections
- **Advanced Settings**: Customize the post-processing prompt for specialized use cases

## Development Setup

### Prerequisites

- Node.js 20+
- Rust 1.83+
- Python 3.11 (ARM native) - **Required for building the ASR engine**

### Installation

```bash
# Install Node.js dependencies
npm install

# Install Rust dependencies (handled automatically by Tauri)
cd src-tauri && cargo build

# Build Python ASR engine binary (required for development)
cd python-engine
./build.sh  # Creates venv automatically and builds binary
```

### Development Server

```bash
npm run tauri:dev
```

### Building

```bash
# Build frontend only
npm run build

# Build full Tauri application
npm run tauri:build

# Rebuild Python engine binary (after engine.py changes)
cd python-engine && ./build.sh
```

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Backend**: Tauri 2.x, Rust
- **Speech Recognition**: MLX-Audio, Whisper, Qwen3-ASR (bundled with PyInstaller)
- **Post-Processing**: MLX LLM (Qwen3-1.7B-4bit) for transcription cleanup
- **Platform**: macOS 14.0+ on Apple Silicon

## Architecture

Echo uses a multi-process architecture:

1. **Tauri App (Rust)**: Main application, hotkey handling, audio capture, VAD, active app detection
2. **React Frontend**: User interface, settings management, transcription history
3. **Python ASR Engine (Sidecar)**: Standalone PyInstaller binary running MLX-Audio for speech recognition
4. **JSON-RPC Communication**: Rust backend communicates with Python engine via stdin/stdout
5. **LLM Post-Processor**: Optional on-device cleanup using Qwen3-1.7B-4bit with context awareness
6. **Continuous Pipeline (Rust)**: Streaming audio → Silero VAD (ONNX) → segment detection → ASR → SQLite

The ASR engine is lazily loaded - models download on first use and remain cached locally. The post-processor LLM auto-loads on startup if enabled.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details
