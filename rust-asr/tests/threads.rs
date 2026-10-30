//! Echo loads models on one thread and runs them from others (hotkey,
//! voice-memo, meeting threads), serialized by its engine mutex. MLX streams
//! are thread-affine, so this must keep working across threads.
//!
//! Needs the cached models: `cargo test --release --test threads -- --ignored --test-threads=1`

use std::path::PathBuf;
use std::thread;

fn hub() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Caches/io.qluto.echo/huggingface/hub")
}

#[test]
#[ignore]
fn post_processor_loaded_on_one_thread_runs_on_others() {
    let mut pp = thread::spawn(|| rust_asr::PostProcessor::load(&hub(), "mlx-community/gemma-4-e4b-it-4bit").unwrap())
        .join()
        .unwrap();
    for _ in 0..2 {
        let out;
        (pp, out) = thread::spawn(move || {
            let out = pp.chat("Reply with one word.", "こんにちは", 8).unwrap();
            (pp, out)
        })
        .join()
        .unwrap();
        assert!(!out.trim().is_empty());
    }
}

#[test]
#[ignore]
fn parakeet_loaded_on_one_thread_runs_on_others() {
    let mut engine = thread::spawn(|| rust_asr::ParakeetEngine::load(&hub()).unwrap()).join().unwrap();
    for _ in 0..2 {
        engine = thread::spawn(move || {
            let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
            engine.transcribe_samples(&samples, "ja").unwrap();
            engine
        })
        .join()
        .unwrap();
    }
}

fn speech() -> Vec<f32> {
    let path = std::env::var("ECHO_TEST_WAV").expect("set ECHO_TEST_WAV to a 16 kHz mono speech WAV");
    rust_asr::read_wav_16k_mono(&path).unwrap()[..16000 * 6].to_vec()
}

/// ASR keeps running while an LLM generates on another thread: each chunk
/// waits at most one prefill chunk or decode step, and neither result changes.
/// `ECHO_TEST_LLM` picks the LLM (default Gemma 4 E4B).
/// `ECHO_TEST_WAV=<speech.wav> cargo test --release --test threads asr_runs_during -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore]
fn asr_runs_during_llm_generation() {
    use std::time::{Duration, Instant};
    // As the app does (ASREngine::start): keep MLX's buffer cache small.
    rust_asr::set_cache_limit(512 * 1024 * 1024);
    let llm = std::env::var("ECHO_TEST_LLM").unwrap_or_else(|_| "mlx-community/gemma-4-e4b-it-4bit".into());
    let pp = std::sync::Arc::new(rust_asr::PostProcessor::load(&hub(), &llm).unwrap());
    let parakeet = rust_asr::ParakeetEngine::load(&hub()).unwrap();
    let whisper = rust_asr::WhisperEngine::load(&hub(), "mlx-community/whisper-large-v3-turbo").unwrap();
    let audio = speech();
    let (active, _) = rust_asr::mlx_memory();
    println!("MLX active after loading: {:.1} GB (whisper.cpp not included)", active as f64 / 1e9);
    // A meeting round's prompt is ~2,000 tokens; make the prefill that long.
    let context = "会議の書き起こしの一行です。".repeat(120);
    let (system, user) = ("Write a detailed 30-item checklist for a product launch.", format!("{context}\nJapanese, one line per item."));
    let user = user.as_str();

    // Alone first (after a warm-up, which compiles the Metal kernels):
    // reference outputs and latencies.
    whisper.warmup().unwrap();
    parakeet.warmup().unwrap();
    let t = Instant::now();
    let reference_llm = pp.chat(system, user, 400).unwrap();
    let alone_llm = t.elapsed();
    let t = Instant::now();
    let reference_pk = parakeet.transcribe_samples(&audio, "ja").unwrap().text;
    let alone_pk = t.elapsed();
    let t = Instant::now();
    let reference_wh = whisper.transcribe_samples(&audio, "ja").unwrap().text;
    let alone_wh = t.elapsed();

    let llm = {
        let pp = std::sync::Arc::clone(&pp);
        let user = user.to_string();
        thread::spawn(move || {
            let t = Instant::now();
            let out = pp.chat(system, &user, 400).unwrap();
            (out, t.elapsed())
        })
    };
    thread::sleep(Duration::from_millis(100)); // ASR runs through prefill and decode
    let mut worst = (Duration::ZERO, Duration::ZERO);
    let mut rounds = 0;
    while !llm.is_finished() {
        let t = Instant::now();
        assert_eq!(parakeet.transcribe_samples(&audio, "ja").unwrap().text, reference_pk);
        worst.0 = worst.0.max(t.elapsed());
        let t = Instant::now();
        assert_eq!(whisper.transcribe_samples(&audio, "ja").unwrap().text, reference_wh);
        worst.1 = worst.1.max(t.elapsed());
        rounds += 1;
        // A meeting's two streams deliver a chunk every few seconds.
        thread::sleep(Duration::from_secs(3));
    }
    let (out, llm_time) = llm.join().unwrap();
    let (active, cache) = rust_asr::mlx_memory();
    println!("MLX active after generating: {:.1} GB, cache {:.1} GB", active as f64 / 1e9, cache as f64 / 1e9);
    println!(
        "{rounds} ASR rounds (every 3 s) during a {:.1}s generation ({:.1}s alone)\nparakeet: alone {:?}, worst during LLM {:?}\nwhisper: alone {:?}, worst during LLM {:?}",
        llm_time.as_secs_f64(), alone_llm.as_secs_f64(), alone_pk, worst.0, alone_wh, worst.1
    );
    assert!(rounds >= 2, "the generation finished before ASR could interleave");
    assert_eq!(out, reference_llm);
    // At worst a chunk waits out one prefill chunk (512 tokens: ~1–2 s for
    // Qwen3.6-27B) or decode step — never the whole generation.
    let bound = Duration::from_millis(2500);
    assert!(worst.0 < alone_pk + bound, "parakeet waited on the LLM");
    assert!(worst.1 < alone_wh + bound, "whisper waited on the LLM");
}
