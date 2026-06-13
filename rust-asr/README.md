# rust-asr — full-Rust Cohere Transcribe (POC)

A native Rust port of `CohereLabs/cohere-transcribe-03-2026` (2.06B-param
FastConformer + Transformer AED) running on Apple MLX via [`mlx-rs`], to replace
Echo's Python/MLX sidecar and eliminate Python-interpreter + heavy-import
startup cost.

## Why

The Python sidecar pays a large fixed startup tax: interpreter init + importing
`mlx` / `mlx_audio` (and, in production, unpacking the PyInstaller bundle) before
it can transcribe. The model inference itself is already fast on Metal. Doing
everything in Rust keeps the Metal acceleration (same MLX kernels) while dropping
the Python startup entirely.

## Result

Same warm cache, same machine, same 6.8s English clip, process-spawn → result:

| Engine | Wall-clock | Output |
| --- | --- | --- |
| Python sidecar (venv) | **3.14 s** | identical |
| `rust-asr` (full MLX) | **0.73 s** | identical |

~4.3× faster cold start — and that is against the *lightweight* venv, not the
production PyInstaller binary (which is heavier still). A resident daemon would
amortize the model load and pay only ~0.5–0.7 s per transcription.

### Numerical parity (validated against the Python reference, all in f32)

| Stage | max abs diff | note |
| --- | --- | --- |
| Audio frontend (log-mel) | 5e-4 | BF16 filterbank rounding |
| FastConformer encoder | 7e-3 | f32-vs-f32, op-ordering only |
| Decoder (same encoder in) | **exact token match** | EN 26/26, JA 25/25 ids |
| End-to-end EN | **exact text** | — |
| End-to-end JA | matches / slightly better | encoder-precision-sensitive homophones; Rust picks the correct kanji (天気, 音声認識) |

The decoder is byte-exact: feeding the Python f32 encoder output into the Rust
decoder reproduces Python's token ids exactly for both EN and JA. End-to-end JA
differs only because a ~7e-3 encoder numerical difference flips a few
homophone/kanji-vs-kana greedy choices — not a bug.

## Architecture (ported 1:1 from `mlx_audio.stt.models.cohere_asr`)

- `audio.rs` — preemphasis → centered STFT (n_fft 512, hann-400→512, hop 160) →
  power → mel filterbank (loaded from checkpoint) → log → per-feature normalize.
- `encoder.rs` — ConvSubsampling (Conv2d incl. depthwise/grouped, with the
  **time-masking between conv stages** that NeMo applies — omitting it leaks
  padding into boundary frames), RelPositionMultiHeadAttention (manual rel-shift
  + fused SDPA for the bias), ConformerConvolution (pointwise/depthwise Conv1d +
  BatchNorm eval + SiLU), 48 Conformer layers, encoder→decoder projection.
- `decoder.rs` — 8-layer Transformer AED: causal self-attn with KV cache,
  cross-attn cached once from the encoder, ReLU FFN, log-softmax head, greedy loop.
- `tokenizer.rs` — HF `tokenizers` (pure Rust) loading the model's `tokenizer.json`.
- `weights.rs` — safetensors loader (`Array::load_safetensors`, lazy mmap).

## Run

```bash
cargo build --release
./target/release/rust-asr run /path/to/audio_16k_mono.wav ja
```

Other subcommands (parity harness): `smoke`, `audio <gt_dir>`,
`encoder <gt_dir>`, `dbg <gt_dir>`, `decode <gt_dir> <lang>`,
`transcribe <gt_dir> <lang>`.

The model path defaults to Echo's app cache; override with `COHERE_SNAPSHOT`.

## Build requirements

- `mlx-rs` with `features = ["metal", "accelerate"]` (Metal is **off** by default).
- `mlx-sys` compiles MLX C++ + Metal kernels from source via cmake. On Xcode 16/26
  this needs the **Metal Toolchain** component:
  `xcodebuild -downloadComponent MetalToolchain` (~700 MB, one-time).

## Integration into Echo (done)

`rust-asr` is consumed by the Tauri app as an **in-process library**
(`rust-asr = { path = "../rust-asr" }` in `src-tauri/Cargo.toml`). `ASREngine`
in `src-tauri/src/transcription.rs` embeds a `CohereEngine` and dispatches
internally: when the active model is Cohere, `load_model` / `transcribe` /
`warmup` / status run in-process (full Rust, no subprocess); every other model,
plus VAD / post-processing / summarization, still uses the Python sidecar. All
call sites (hotkey, commands, always-on pipeline) are unchanged because the
dispatch lives inside `ASREngine`. `CohereEngine::load` downloads/locates the
gated checkpoint in the shared HF hub cache via `hf-hub`, honoring the HF token.

Covered by an integration test (`cohere_routes_in_process` in `transcription.rs`,
skips if the model isn't cached). CI ensures the Metal toolchain before building.

Remaining for production: f32→bf16 weights (~2× lower memory, faster load);
resampling for non-16 kHz input; long-audio chunking
(`split_audio_chunks_energy`, >30 s); `no_speech`/VAD gating on the hotkey path.
