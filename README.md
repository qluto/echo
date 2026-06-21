# Echo - Voice Input Application

Offline voice input desktop application optimized for Apple Silicon

## Features

- **High-Accuracy Speech Recognition**: Whisper, Parakeet TDT (JA), and Cohere Transcribe — all running **fully in-process in Rust** (no Python sidecar)
- **Model Switching**: Switch between Whisper, Parakeet-JA, and (opt-in) Cohere models in Settings
- **LLM-Based Post-Processing**: Optional on-device cleanup of filler words and self-corrections (Qwen3, in-process via MLX)
- **Context-Aware Formatting**: Detects active application for context-appropriate output
- **Customizable Prompts**: Edit post-processing behavior via Advanced Settings
- **Always-On Transcription**: Continuous listening mode with automatic speech detection (Silero VAD)
- **Global Hotkey**: System-wide keyboard shortcut for instant recording
- **Real-time Transcription**: Immediate transcription after recording ends
- **Auto Text Insertion**: Automatically paste transcription into active applications
- **Transcription History**: SQLite-backed history with full-text search for always-on mode
- **Fully Offline**: No internet connection required — all processing happens locally

## System Requirements

- **OS**: macOS 14.0 (Sonoma) or later
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **Memory**: 8GB+ recommended
- **Storage**: 2GB+ free space (varies by model size)

## Supported Models

All ASR models run in-process in Rust — there is no Python dependency at runtime.

### Whisper (default)
Runs on whisper.cpp (Metal) via `whisper-rs`.
- `whisper-large-v3-turbo` — balanced performance (**default**)
- `whisper-large-v3` — highest accuracy
- `whisper-medium` / `small` / `base` / `tiny` — lightweight models

### Parakeet TDT (Japanese)
NVIDIA FastConformer + TDT transducer, ported to Apple MLX via `mlx-rs`. Japanese-specialized (CER 6.4% on JSUT), ~160 ms inference, non-gated.
- `parakeet-tdt_ctc-0.6b-ja` — 0.6B

### Cohere Transcribe (gated, opt-in)
Cohere Labs 2B, ported to Apple MLX via `mlx-rs`. 14 languages including JA/ZH/KO; strong on clean read-speech. Requires HuggingFace authentication and license acceptance — see [Gated Model Access](#gated-model-access).
- `cohere-transcribe-03-2026` — 2B (BF16, ~4GB download)

> **Note:** Qwen3-ASR is not currently available — it has not yet been ported to the in-process Rust engine and is intentionally omitted to avoid a non-functional option.

## Installation

Download the latest release from the [Releases](https://github.com/qluto/echo/releases) page:

1. Download the `.dmg` file
2. Open the DMG and drag Echo to your Applications folder
3. Launch Echo from Applications
4. Grant microphone permissions when prompted

The app is fully self-contained — no Python or other runtime installation required.

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

- **ASR Model**: Choose between Whisper, Parakeet-JA, or (when enabled) Cohere models
- **Hotkey**: Customize the recording keyboard shortcut
- **Recognition Language**: Auto-detect or manually specify language
- **Input Device**: Select your preferred microphone
- **Auto Insert**: Enable/disable automatic paste after transcription
- **Post-Processing**: Enable LLM-based cleanup to remove filler words and self-corrections
- **Advanced Settings**: Customize the post-processing prompt, and configure gated model access

### Gated Model Access

Some models (currently Cohere Transcribe) require HuggingFace authentication and license acceptance. Enable them via **Settings → Gated Models (Advanced)**:

1. Accept the model license on the upstream HF page (e.g. `huggingface.co/CohereLabs/cohere-transcribe-03-2026`).
2. Create a read-scoped HF token at `huggingface.co/settings/tokens`.
3. Enter the token into Echo's Advanced settings and enable gated access.

The token is stored locally in `settings.json` and used only to download the gated checkpoint. When gated access is disabled, gated models are hidden from the picker.

## Development Setup

### Prerequisites

- Node.js 20+
- Rust 1.83+
- Xcode 16+ with the Metal toolchain installed:
  ```bash
  xcodebuild -downloadComponent MetalToolchain   # ~700 MB, one-time
  ```
  Required because both `mlx-rs` (Apple MLX + Metal kernels) and `whisper-rs` (whisper.cpp + ggml + Metal) compile native code from source.

No Python is required — the ASR and post-processing engines are compiled into the app binary via the `rust-asr` crate.

### Installation

```bash
# Install Node.js dependencies
npm install

# Build the Rust backend + in-process ASR engines (handled by Tauri)
cd src-tauri && cargo build
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

# Signed local build (preserves Accessibility permissions across updates)
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
npm run tauri:build:signed
```

### Standalone ASR CLI / parity harness

The in-process engines can be exercised directly via the `rust-asr` crate:

```bash
cd rust-asr && cargo build --release
./target/release/rust-asr run <audio_16k.wav> <lang>   # Whisper
./target/release/rust-asr pk-run <audio.wav>           # Parakeet-JA
./target/release/rust-asr pp "<text>" [qwen3-model]    # Post-processing
```

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Backend**: Tauri 2.x, Rust
- **ASR Engines (in-process)**: Whisper via `whisper-rs` (whisper.cpp/Metal); Parakeet-JA + Cohere via `mlx-rs` (Apple MLX)
- **VAD**: Silero VAD v5 via ONNX Runtime, rubato for resampling
- **Post-Processing**: Qwen3 (4-bit, default 4B) ported to `mlx-rs`, in-process
- **Database**: SQLite with FTS5 full-text search (`rusqlite`)
- **Platform**: macOS 14.0+ on Apple Silicon

## Architecture

Echo runs entirely in a single process — speech recognition and post-processing
are native Rust, with no Python sidecar.

1. **Tauri App (Rust)**: Main application, hotkey handling, audio capture, VAD, active app detection
2. **React Frontend**: User interface, settings management, transcription history
3. **In-Process ASR Engines (`rust-asr`)**: Whisper (whisper.cpp/Metal), Parakeet-JA and Cohere (Apple MLX). `ASREngine` in `transcription.rs` owns the loaded engines and dispatches by the active model id.
4. **In-Process Post-Processor**: Optional on-device cleanup using Qwen3 (full-Rust MLX port), context-aware via active-app detection
5. **Continuous Pipeline (Rust)**: Streaming audio → Silero VAD (ONNX) → segment detection → ASR → SQLite

Both Metal backends (whisper.cpp and MLX) coexist in-process and are serialized
by the `ASREngine` mutex. Models are lazily loaded — they download on first use
and remain cached locally; the post-processor LLM auto-loads on startup if enabled.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details
