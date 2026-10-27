//! Greedy decoding loop shared by the LLM ports.
//!
//! Mirrors mlx-lm's `generate_step`, so greedy output matches mlx-lm token
//! for token: all prompt tokens but the last are prefilled in chunks (so a
//! long transcript doesn't build one huge attention graph), the last one runs
//! as the first single-token step, and the argmax is taken over log-probs.
//! Decoding is pipelined — the graph for token n+1 is built and queued with
//! `async_eval` before token n is read back to the CPU, so the GPU never idles
//! while Rust assembles the next step.
//!
//! Greedy decoding can fall into a loop that repeats the same passage until
//! `max_tokens`; that is cut off as soon as it is detected.

use anyhow::Result;
use mlx_rs::ops::indexing::argmax_axis;
use mlx_rs::ops::{logsumexp_axis, reshape};
use mlx_rs::transforms::async_eval;
use mlx_rs::Array;
use std::time::Instant;

/// Prompt tokens fed per prefill forward pass (mlx-lm's default).
const PREFILL_CHUNK: usize = 2048;

/// A loop is the output ending in this many back-to-back copies of a block…
const LOOP_REPEATS: usize = 3;
/// …spanning at least this many tokens in total (so short legitimate
/// repetition like "…" or a repeated word is left alone).
const LOOP_MIN_TOKENS: usize = 48;
/// Longest repeating block looked for, in tokens.
const LOOP_MAX_PERIOD: usize = 200;

/// Runs greedy generation. `forward(ids, offset)` feeds token ids `[T]`
/// starting at position `offset` (updating the model's caches) and returns
/// the logits of the last position, `[1, vocab]`. Returns the generated ids,
/// excluding the prompt and the stop token.
pub fn greedy(
    prompt_ids: &[i32],
    max_tokens: usize,
    stop_ids: &[i32],
    mut forward: impl FnMut(&Array, i32) -> Result<Array>,
) -> Result<Vec<i32>> {
    let Some((&last, head)) = prompt_ids.split_last() else {
        return Ok(Vec::new());
    };
    let start = Instant::now();
    let mut pos = 0;
    for chunk in head.chunks(PREFILL_CHUNK) {
        let l = forward(&Array::from_slice(chunk, &[chunk.len() as i32]), pos)?;
        pos += chunk.len() as i32;
        // Materialize each chunk's caches before queuing the next one.
        async_eval([&l])?;
    }

    let mut step = |token: &Array, pos: i32| -> Result<Array> {
        let logits = forward(token, pos)?;
        let logprobs = &logits - &logsumexp_axis(&logits, -1, true)?;
        let next = argmax_axis(&logprobs, 1, false)?;
        async_eval([&next])?;
        Ok(next)
    };

    let mut next = step(&Array::from_slice(&[last], &[1]), pos)?;
    pos += 1;
    let mut out = Vec::new();
    let mut first_token_at = None;
    while out.len() < max_tokens {
        // Queue the following step on the still-lazy token before syncing.
        let following = if out.len() + 1 < max_tokens {
            Some(step(&reshape(&next, &[1])?, pos)?)
        } else {
            None
        };
        let tok = next.item_cast::<u32>() as i32;
        first_token_at.get_or_insert_with(Instant::now);
        if stop_ids.contains(&tok) {
            break;
        }
        out.push(tok);
        pos += 1;
        if let Some(period) = repeating_tail(&out) {
            log::warn!("LLM output fell into a repetition loop (period {period} tokens); stopping");
            out.truncate(out.len() - (LOOP_REPEATS - 1) * period);
            break;
        }
        match following {
            Some(f) => next = f,
            None => break,
        }
    }
    let first = first_token_at.unwrap_or_else(Instant::now);
    let decode = first.elapsed().as_secs_f64();
    log::info!(
        "LLM: {} prompt tokens, first token after {:.2}s; {} tokens generated at {:.1} tok/s{}",
        prompt_ids.len(),
        (first - start).as_secs_f64(),
        out.len(),
        out.len().saturating_sub(1) as f64 / decode.max(1e-9),
        if out.len() >= max_tokens { " (hit max_tokens)" } else { "" }
    );
    Ok(out)
}

/// If `out` ends in `LOOP_REPEATS` back-to-back copies of one block spanning
/// at least `LOOP_MIN_TOKENS`, returns the block length.
fn repeating_tail(out: &[i32]) -> Option<usize> {
    (1..=LOOP_MAX_PERIOD).find(|&p| {
        let span = p * LOOP_REPEATS;
        span >= LOOP_MIN_TOKENS && span <= out.len() && {
            let tail = &out[out.len() - span..];
            tail.chunks(p).all(|c| c == &tail[..p])
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_block_repeated_three_times() {
        let block: Vec<i32> = (0..20).collect();
        let mut out = vec![99, 98];
        for _ in 0..3 {
            out.extend(&block);
        }
        assert_eq!(repeating_tail(&out), Some(20));
    }

    #[test]
    fn ignores_short_or_twice_repeated_text() {
        let block: Vec<i32> = (0..20).collect();
        let twice = [block.clone(), block].concat();
        assert_eq!(repeating_tail(&twice), None);
        // "……" style runs below LOOP_MIN_TOKENS.
        assert_eq!(repeating_tail(&[7; 10]), None);
        assert_eq!(repeating_tail(&[7; 48]), Some(16));
    }
}
