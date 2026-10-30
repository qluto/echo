//! The one lock every MLX / whisper.cpp call goes through, and the MLX
//! streams shared by every thread.
//!
//! Echo runs the engines from many threads: hotkey dictation, voice memos and
//! the meeting companion's ASR worker and notes thread. Two rules hold:
//!
//! - GPU work is never encoded from two threads at once. MLX does no locking
//!   on the shared streams below, and concurrent Metal command encoding by
//!   MLX and whisper.cpp is unsafe. [`lock`] serializes it.
//! - Since MLX 0.32 a stream is thread-affine (its command encoder lives on
//!   the thread that created it), and arrays — lazily loaded weights included —
//!   remember the stream they were built on. So a model loaded on one thread
//!   fails on another with "There is no Stream(cpu, 0) in current thread".
//!   [`lock`] first points the calling thread's default streams at one
//!   process-wide pair made with `mlx_stream_new_thread_unsafe`, which MLX
//!   registers globally — safe exactly because of the first rule.
//!
//! The lock is meant to be held briefly: an ASR call holds it for one chunk,
//! an LLM generation only per prefill chunk / decode step (see
//! `llm::decode`), so transcription keeps going while an LLM writes minutes.
//! For that it has to be fair: `std::sync::Mutex` (os_unfair_lock on macOS)
//! lets the decode loop re-take it step after step while ASR starves, so it
//! is a FIFO ticket lock.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::{Condvar, Mutex, OnceLock};

use mlx_sys::{
    mlx_device_free, mlx_device_new_type, mlx_device_type, mlx_device_type__MLX_CPU,
    mlx_device_type__MLX_GPU, mlx_set_default_stream, mlx_stream, mlx_stream_new_thread_unsafe,
};

struct SharedStreams {
    gpu: mlx_stream,
    cpu: mlx_stream,
}

// The handles are only used under `GPU`.
unsafe impl Send for SharedStreams {}
unsafe impl Sync for SharedStreams {}

static STREAMS: OnceLock<SharedStreams> = OnceLock::new();

/// FIFO ticket lock: (next ticket to hand out, ticket being served).
static TICKETS: Mutex<(u64, u64)> = Mutex::new((0, 0));
static TURN: Condvar = Condvar::new();

thread_local! {
    static STREAMS_INSTALLED: Cell<bool> = const { Cell::new(false) };
    /// Nesting depth of `lock()` on this thread; the lock is held while > 0.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn acquire() {
    // Poisoning can't leave the counters inconsistent: they are only bumped.
    let mut t = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    let mine = t.0;
    t.0 += 1;
    while t.1 != mine {
        t = TURN.wait(t).unwrap_or_else(|e| e.into_inner());
    }
}

fn release() {
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
    TURN.notify_all();
}

fn new_stream(kind: mlx_device_type) -> mlx_stream {
    unsafe {
        let device = mlx_device_new_type(kind, 0);
        let stream = mlx_stream_new_thread_unsafe(device);
        mlx_device_free(device);
        stream
    }
}

fn use_shared_streams() {
    if STREAMS_INSTALLED.get() {
        return;
    }
    let streams = STREAMS.get_or_init(|| SharedStreams {
        gpu: new_stream(mlx_device_type__MLX_GPU),
        cpu: new_stream(mlx_device_type__MLX_CPU),
    });
    unsafe {
        mlx_set_default_stream(streams.gpu);
        mlx_set_default_stream(streams.cpu);
    }
    STREAMS_INSTALLED.set(true);
}

/// Held while this thread may issue GPU work. Reentrant on one thread.
pub(crate) struct GpuGuard {
    // Must be dropped on the thread that took it.
    _not_send: PhantomData<*const ()>,
}

/// Take the GPU, after every thread that asked before (each waits for at most
/// the others' current step or chunk).
pub(crate) fn lock() -> GpuGuard {
    if DEPTH.get() == 0 {
        acquire();
        use_shared_streams();
    }
    DEPTH.set(DEPTH.get() + 1);
    GpuGuard { _not_send: PhantomData }
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        DEPTH.set(DEPTH.get() - 1);
        if DEPTH.get() == 0 {
            release();
        }
    }
}
