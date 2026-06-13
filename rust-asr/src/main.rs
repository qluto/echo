//! Full-Rust Cohere Transcribe (cohere-transcribe-03-2026) POC.
//!
//! Replaces the Python/MLX sidecar with a native Rust binary linking Apple MLX
//! directly, eliminating Python-interpreter + heavy-import startup cost.
//!
//! Subcommands:
//!   smoke               load checkpoint, run a Metal matmul (toolchain check)
//!   audio  <gt_dir>     run the audio frontend on <gt_dir>/waveform.npy and
//!                       compare against <gt_dir>/mel.npy

use anyhow::{anyhow, Result};
use mlx_rs::ops::{abs, max, mean};
use mlx_rs::{Array, Dtype};
use rust_asr::weights::Weights;
use rust_asr::{audio, decoder, encoder, tokenizer};

fn model_snapshot() -> String {
    std::env::var("COHERE_SNAPSHOT").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!(
            "{home}/Library/Caches/io.qluto.echo/huggingface/hub/\
             models--CohereLabs--cohere-transcribe-03-2026/snapshots/\
             32d9e4ba6271d78168c095c2f90bc173eaad97d2"
        )
    })
}

fn safetensors_path() -> String {
    format!("{}/model.safetensors", model_snapshot())
}

/// max |a-b| and mean |a-b| between two f32 arrays.
fn diff(a: &Array, b: &Array) -> Result<(f32, f32)> {
    let d = abs(&(a - b))?;
    let mx = max(&d, None)?.item::<f32>();
    let mn = mean(&d, None)?.item::<f32>();
    Ok((mx, mn))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("smoke");

    match cmd {
        "smoke" => smoke(),
        "audio" => {
            let gt = args.get(2).ok_or_else(|| anyhow!("usage: audio <gt_dir>"))?;
            validate_audio(gt)
        }
        "encoder" => {
            let gt = args.get(2).ok_or_else(|| anyhow!("usage: encoder <gt_dir>"))?;
            validate_encoder(gt)
        }
        "dbg" => {
            let gt = args.get(2).ok_or_else(|| anyhow!("usage: dbg <gt_dir>"))?;
            debug_stages(gt)
        }
        "decode" => {
            // Decode directly from a saved encoder output, isolating the decoder
            // from encoder numerics. usage: decode <gt_dir> <lang> [enc_npy]
            let gt = args.get(2).ok_or_else(|| anyhow!("usage: decode <gt_dir> <lang> [enc]"))?;
            let lang = args.get(3).map(String::as_str).unwrap_or("en");
            let enc_file = args.get(4).map(String::as_str).unwrap_or("encoder_f32.npy");
            decode_from_encoder(gt, lang, enc_file)
        }
        "transcribe" => {
            let gt = args
                .get(2)
                .ok_or_else(|| anyhow!("usage: transcribe <gt_dir> <lang>"))?;
            let lang = args.get(3).map(String::as_str).unwrap_or("en");
            transcribe_from_gt(gt, lang)
        }
        "run" => {
            let wav = args.get(2).ok_or_else(|| anyhow!("usage: run <wav> <lang>"))?;
            let lang = args.get(3).map(String::as_str).unwrap_or("en");
            run_wav(wav, lang)
        }
        "whisper" => {
            // CLI check for the Whisper engine: whisper <wav> <lang> [model_id]
            let wav = args.get(2).ok_or_else(|| anyhow!("usage: whisper <wav> <lang> [model]"))?;
            let lang = args.get(3).map(String::as_str).unwrap_or("auto");
            let model = args
                .get(4)
                .map(String::as_str)
                .unwrap_or("mlx-community/whisper-large-v3-turbo");
            let home = std::env::var("HOME").unwrap_or_default();
            let hub = format!("{home}/Library/Caches/io.qluto.echo/huggingface/hub");
            let t = std::time::Instant::now();
            let eng = rust_asr::WhisperEngine::load(std::path::Path::new(&hub), model)?;
            println!("whisper loaded in {:?}", t.elapsed());
            let t = std::time::Instant::now();
            let out = eng.transcribe_wav(wav, lang)?;
            println!("text: {:?} (lang={}) in {:?}", out.text, out.language, t.elapsed());
            Ok(())
        }
        "engine" => {
            // Exercise the high-level CohereEngine API (hub-cache resolution +
            // load + transcribe), the same path the Tauri app uses.
            let wav = args.get(2).ok_or_else(|| anyhow!("usage: engine <wav> <lang>"))?;
            let lang = args.get(3).map(String::as_str).unwrap_or("en");
            let home = std::env::var("HOME").unwrap_or_default();
            let hub = format!("{home}/Library/Caches/io.qluto.echo/huggingface/hub");
            let t = std::time::Instant::now();
            let eng = rust_asr::CohereEngine::load(std::path::Path::new(&hub), None)?;
            println!("engine loaded in {:?}", t.elapsed());
            let t = std::time::Instant::now();
            let out = eng.transcribe_wav(wav, lang)?;
            println!("text: {:?} (lang={}) in {:?}", out.text, out.language, t.elapsed());
            Ok(())
        }
        other => Err(anyhow!("unknown subcommand: {other}")),
    }
}

fn smoke() -> Result<()> {
    println!("default device: {:?}", mlx_rs::Device::default());
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
    println!("matmul: {:?}", a.matmul(&b)?);

    let t = std::time::Instant::now();
    let w = Weights::load(&safetensors_path())?;
    println!("loaded weights in {:?}", t.elapsed());
    for key in [
        "encoder.pre_encode.out.weight",
        "preprocessor.featurizer.fb",
        "preprocessor.featurizer.window",
    ] {
        let a = w.raw(key)?;
        println!("  {key}: shape={:?} dtype={:?}", a.shape(), a.dtype());
    }
    Ok(())
}

fn validate_audio(gt_dir: &str) -> Result<()> {
    let w = Weights::load(&safetensors_path())?;

    let waveform = Array::load_numpy(format!("{gt_dir}/waveform.npy"))
        .map_err(|e| anyhow!("load waveform.npy: {e}"))?
        .as_dtype(Dtype::Float32)?;
    println!("waveform: shape={:?}", waveform.shape());

    let t = std::time::Instant::now();
    let (feats, seq_len) = audio::extract_features(&w, &waveform)?;
    feats.eval()?;
    println!(
        "rust mel: shape={:?} seq_len={} ({:?})",
        feats.shape(),
        seq_len,
        t.elapsed()
    );

    let gt = Array::load_numpy(format!("{gt_dir}/mel.npy"))
        .map_err(|e| anyhow!("load mel.npy: {e}"))?
        .as_dtype(Dtype::Float32)?;
    println!("gt   mel: shape={:?}", gt.shape());

    let (mx, mn) = diff(&feats, &gt)?;
    println!("mel diff: max={mx:.4e} mean={mn:.4e}");
    if mx < 5e-2 {
        println!("AUDIO PARITY: PASS (max abs diff {mx:.4e})");
    } else {
        println!("AUDIO PARITY: FAIL (max abs diff {mx:.4e})");
    }
    Ok(())
}

fn validate_encoder(gt_dir: &str) -> Result<()> {
    let w = Weights::load(&safetensors_path())?;
    let t = std::time::Instant::now();
    let enc = encoder::Encoder::load(&w)?;
    println!("encoder loaded in {:?}", t.elapsed());

    let mel = Array::load_numpy(format!("{gt_dir}/mel.npy"))
        .map_err(|e| anyhow!("load mel.npy: {e}"))?
        .as_dtype(Dtype::Float32)?;

    let t = std::time::Instant::now();
    let seq_len = mel.shape()[1] - 1;
    let out = enc.forward(&mel, seq_len)?;
    out.eval()?;
    println!("rust encoder out: shape={:?} ({:?})", out.shape(), t.elapsed());

    let gt = Array::load_numpy(format!("{gt_dir}/encoder.npy"))
        .map_err(|e| anyhow!("load encoder.npy: {e}"))?
        .as_dtype(Dtype::Float32)?;
    println!("gt   encoder out: shape={:?}", gt.shape());

    let (mx, mn) = diff(&out, &gt)?;
    println!("encoder diff: max={mx:.4e} mean={mn:.4e}");
    if mx < 5e-1 {
        println!("ENCODER PARITY: PASS (max abs diff {mx:.4e})");
    } else {
        println!("ENCODER PARITY: FAIL (max abs diff {mx:.4e})");
    }
    Ok(())
}

fn debug_stages(gt_dir: &str) -> Result<()> {
    let w = Weights::load(&safetensors_path())?;
    let enc = encoder::Encoder::load(&w)?;
    let mel = Array::load_numpy(format!("{gt_dir}/mel.npy"))
        .map_err(|e| anyhow!("load mel.npy: {e}"))?
        .as_dtype(Dtype::Float32)?;
    let seq_len = mel.shape()[1] - 1;
    let (sub, posemb, l0) = enc.debug_stages(&mel, seq_len)?;
    for (name, arr) in [("sub", &sub), ("posemb", &posemb), ("layer0", &l0)] {
        let gt = Array::load_numpy(format!("{gt_dir}/{name}.npy"))
            .map_err(|e| anyhow!("load {name}.npy: {e}"))?
            .as_dtype(Dtype::Float32)?;
        let (mx, mn) = diff(arr, &gt)?;
        println!(
            "{name}: rust={:?} gt={:?} diff max={mx:.4e} mean={mn:.4e}",
            arr.shape(),
            gt.shape()
        );
    }
    Ok(())
}

/// Load a 16kHz mono WAV as f32 samples in [-1, 1].
fn load_wav(path: &str) -> Result<Array> {
    let mut reader = hound::WavReader::open(path).map_err(|e| anyhow!("open wav: {e}"))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| anyhow!("read samples: {e}"))?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow!("read samples: {e}"))?,
    };
    // Downmix to mono if needed.
    let mono: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        raw.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()
    } else {
        raw
    };
    if spec.sample_rate != 16000 {
        eprintln!(
            "warning: wav sample_rate={} (expected 16000); resampling not implemented",
            spec.sample_rate
        );
    }
    Ok(Array::from_slice(&mono, &[mono.len() as i32]))
}

fn run_wav(wav: &str, lang: &str) -> Result<()> {
    let t_proc = std::time::Instant::now();
    let w = Weights::load(&safetensors_path())?;
    let enc_model = encoder::Encoder::load(&w)?;
    let dec_model = decoder::Decoder::load(&w)?;
    let tok = tokenizer::Tok::load(&format!("{}/tokenizer.json", model_snapshot()))?;
    let t_loaded = t_proc.elapsed();

    let waveform = load_wav(wav)?;
    let t_infer = std::time::Instant::now();
    let (mel, seq_len) = audio::extract_features(&w, &waveform)?;
    let enc = enc_model.forward(&mel, seq_len)?;
    let prompt = tok.build_prompt(lang, true)?;
    let ids = dec_model.generate(&enc, &prompt, 256)?;
    let text = tok.decode(&ids)?;
    let infer = t_infer.elapsed();

    println!("text: {text:?}");
    println!(
        "timing: load={:?} infer={:?} total={:?}",
        t_loaded,
        infer,
        t_proc.elapsed()
    );
    Ok(())
}

fn decode_from_encoder(gt_dir: &str, lang: &str, enc_file: &str) -> Result<()> {
    let w = Weights::load(&safetensors_path())?;
    let dec = decoder::Decoder::load(&w)?;
    let tok = tokenizer::Tok::load(&format!("{}/tokenizer.json", model_snapshot()))?;

    let enc = Array::load_numpy(format!("{gt_dir}/{enc_file}"))
        .map_err(|e| anyhow!("load {enc_file}: {e}"))?
        .as_dtype(Dtype::Float32)?;
    let prompt = tok.build_prompt(lang, true)?;
    let ids = dec.generate(&enc, &prompt, 256)?;
    let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    println!("RUST ids ({}): {ids_i32:?}", ids.len());
    println!("RUST text: {:?}", tok.decode(&ids)?);

    if let Ok(py) = Array::load_numpy(format!("{gt_dir}/py_ids.npy")) {
        let py_ids: Vec<i32> = py.as_slice::<i32>().to_vec();
        println!("PY   ids ({}): {py_ids:?}", py_ids.len());
        println!("IDS MATCH: {}", if py_ids == ids_i32 { "EXACT" } else { "DIFFERENT" });
    }
    Ok(())
}

fn transcribe_from_gt(gt_dir: &str, lang: &str) -> Result<()> {
    let t_all = std::time::Instant::now();
    let w = Weights::load(&safetensors_path())?;
    let enc_model = encoder::Encoder::load(&w)?;
    let dec_model = decoder::Decoder::load(&w)?;
    let tok = tokenizer::Tok::load(&format!("{}/tokenizer.json", model_snapshot()))?;
    println!("models+tokenizer loaded in {:?}", t_all.elapsed());

    // Use the audio frontend on the raw waveform (full pipeline minus wav I/O).
    let waveform = Array::load_numpy(format!("{gt_dir}/waveform.npy"))
        .map_err(|e| anyhow!("load waveform.npy: {e}"))?
        .as_dtype(Dtype::Float32)?;

    let t = std::time::Instant::now();
    let (mel, seq_len) = audio::extract_features(&w, &waveform)?;
    let enc = enc_model.forward(&mel, seq_len)?;
    let prompt = tok.build_prompt(lang, true)?;
    let ids = dec_model.generate(&enc, &prompt, 256)?;
    let text = tok.decode(&ids)?;
    println!("inference in {:?}", t.elapsed());
    println!("prompt tokens: {prompt:?}");
    println!("generated {} tokens", ids.len());
    println!("RUST TEXT: {text:?}");

    // Compare against the Python ground-truth text from meta.json.
    if let Ok(meta) = std::fs::read_to_string(format!("{gt_dir}/meta.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&meta) {
            if let Some(gt) = v.get("text").and_then(|t| t.as_str()) {
                println!("PY   TEXT: {gt:?}");
                println!(
                    "TEXT MATCH: {}",
                    if gt == text { "EXACT" } else { "DIFFERENT" }
                );
            }
        }
    }
    Ok(())
}
