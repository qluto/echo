//! In-process LLM post-processing (Qwen3.5 / Qwen3) — the full-Rust
//! replacement for the Python `PostProcessor`. Cleans up ASR text (filler
//! removal, self-correction, dictionary, app-context formatting) and
//! summarizes transcription history.

pub mod gated_delta;
pub mod qwen3;
pub mod qwen3_5;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

use qwen3::{Config, Qwen3};
use qwen3_5::Qwen3_5;
use tokenizers::Tokenizer;

/// System prompt for cleanup — kept byte-identical to the Python PostProcessor.
pub const SYSTEM_PROMPT: &str = "/no_think
You are an assistant that cleans up speech recognition results while preserving the speaker's intended meaning.

## Your Task
Remove verbal noise while keeping the speaker's message intact:

1. **Remove filler words** - These add no meaning:
   - English: um, uh, like, you know, well, so, I mean, kind of, sort of, basically, actually, literally, right?, anyway
   - Japanese: ええと, えーと, あの, まあ, なんか, その, うーん, ちょっと, やっぱ

2. **Handle self-corrections** - When someone corrects themselves mid-sentence, keep only their final intent:
   - \"I'll be there at 3, no 4 o'clock\" → \"I'll be there at 4 o'clock\"
   - \"Send it to Tom, I mean Jerry\" → \"Send it to Jerry\"
   - \"The meeting is on Monday, wait, Tuesday\" → \"The meeting is on Tuesday\"
   - \"AですあやっぱりBです\" → \"Bです\"
   - \"3時に、いや4時に行きます\" → \"4時に行きます\"

3. **Apply user dictionary** - Replace terms as specified

4. **Format for target app** (if specified):
   - Email: Use polite business language
   - Notion/Markdown: Format lists as Markdown

## Output
Output ONLY the cleaned text. No explanations.";

pub const SUMMARIZE_PROMPT: &str = "You are an assistant that creates concise summaries of speech transcriptions.

## Input
You will receive a chronological list of speech transcription segments with timestamps.

## Your Task
1. Identify the main topics and key points discussed
2. Create a well-organized summary that captures the essential information
3. Group related topics together
4. Preserve important details: names, numbers, dates, decisions, action items
5. Output the summary in the same language as the input transcriptions

## Output Format
Write a clear, structured summary. Use bullet points for distinct topics.
Do NOT include timestamps in the summary unless they are semantically important (e.g., \"meeting at 3pm\").";

/// The loaded LLM, by architecture (picked from config.json's `model_type`).
enum Model {
    Qwen3(Qwen3),
    Qwen3_5(Qwen3_5),
}

pub struct PostProcessor {
    model: Model,
    tokenizer: Tokenizer,
    /// Token id of `</think>`, to cut the reasoning trace off generated ids.
    think_end_id: Option<i32>,
}

impl PostProcessor {
    pub fn load(hub_cache_dir: &Path, model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::ApiBuilder;
        let api = ApiBuilder::new()
            .with_cache_dir(hub_cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|e| anyhow!("hf-hub init: {e}"))?;
        let repo = api.model(model_id.to_string());
        // Small non-LFS files (config.json, tokenizer.json) come through HF's
        // relative resolve-cache redirect, which hf-hub 0.3.2 can't follow;
        // fetch them directly. The LFS weights below still use hf-hub.
        let config_path = crate::hf::fetch_small_file(hub_cache_dir, model_id, "config.json", None)
            .map_err(|e| anyhow!("config: {e}"))?;
        let tok_path = crate::hf::fetch_small_file(hub_cache_dir, model_id, "tokenizer.json", None)
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let st = repo
            .get("model.safetensors")
            .map_err(|e| anyhow!("weights: {e}"))?;

        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("tokenizer load: {e}"))?;
        let token_id = |t: &str| tokenizer.token_to_id(t).map(|id| id as i32);
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let weights = crate::weights::Weights::load(st.to_str().ok_or_else(|| anyhow!("path"))?)?;
        let model = if config.get("model_type").and_then(|v| v.as_str()) == Some("qwen3_5") {
            // config.json's eos is <|endoftext|>; a chat turn ends at <|im_end|>.
            let im_end = token_id("<|im_end|>").ok_or_else(|| anyhow!("no <|im_end|> token"))?;
            Model::Qwen3_5(Qwen3_5::load(&weights, parse_config_3_5(&config, im_end)?)?)
        } else {
            Model::Qwen3(Qwen3::load(&weights, parse_config(&config)?)?)
        };
        let think_end_id = token_id("</think>");
        log::info!("Post-processor loaded: {model_id}");
        Ok(Self {
            model,
            tokenizer,
            think_end_id,
        })
    }

    /// Clean up ASR text. Mirrors the Python PostProcessor.process().
    pub fn process(
        &self,
        text: &str,
        app_name: Option<&str>,
        app_bundle_id: Option<&str>,
        dictionary: Option<&HashMap<String, String>>,
        custom_prompt: Option<&str>,
    ) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }
        let system = custom_prompt.unwrap_or(SYSTEM_PROMPT);
        let user = build_user_message(text, app_name, app_bundle_id, dictionary);
        // No-think mode (matches Python enable_thinking=False) for fast cleanup.
        let prompt = chat_prompt(system, &user, false);
        let max_tokens = text.chars().count() + 100;
        let out = self.run(&prompt, max_tokens)?;
        Ok(out)
    }

    /// Summarize transcription entries (single pass). `entries` = (created_at, text).
    pub fn summarize(
        &self,
        entries: &[(String, String)],
        language_hint: Option<&str>,
        custom_prompt: Option<&str>,
    ) -> Result<String> {
        if entries.is_empty() {
            return Ok(String::new());
        }
        let system = custom_prompt.unwrap_or(SUMMARIZE_PROMPT);
        let mut user = entries
            .iter()
            .map(|(ts, t)| format!("[{ts}] {t}"))
            .collect::<Vec<_>>()
            .join("\n");
        if let Some(lang) = language_hint {
            user.push_str(&format!("\n\n(Primary language: {lang})"));
        }
        // Qwen3 summarizes in thinking mode (matches Python summarize).
        // Qwen3.5 doesn't: under greedy decoding its reasoning trace falls
        // into a repetition loop and never reaches the answer.
        let thinking = matches!(self.model, Model::Qwen3(_));
        let prompt = chat_prompt(system, &user, thinking);
        self.run(&prompt, 2048)
    }

    /// Free-form completion (no thinking): system + user message → assistant
    /// text. Used for meeting-minutes generation.
    pub fn chat(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        self.run(&chat_prompt(system, user, false), max_tokens)
    }

    fn run(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let gen = match &self.model {
            Model::Qwen3(m) => m.generate(&ids, max_tokens)?,
            Model::Qwen3_5(m) => m.generate(&ids, max_tokens)?,
        };
        // Keep only what follows the reasoning trace, if there is one.
        let answer = match self.think_end_id.and_then(|id| gen.iter().rposition(|&t| t == id)) {
            Some(end) => &gen[end + 1..],
            None => &gen[..],
        };
        let answer: Vec<u32> = answer.iter().map(|&i| i as u32).collect();
        let text = self
            .tokenizer
            .decode(&answer, true)
            .map_err(|e| anyhow!("decode: {e}"))?;
        Ok(strip_think(&text).trim().to_string())
    }

    pub fn warmup(&self) -> Result<()> {
        let _ = self.process("テスト", None, None, None, None)?;
        Ok(())
    }
}

/// Qwen chat template (system + user). When `thinking` is false this matches
/// `apply_chat_template(enable_thinking=False)`, which primes the assistant turn
/// with an empty `<think></think>` block so the model skips its reasoning trace.
fn chat_prompt(system: &str, user: &str, thinking: bool) -> String {
    let head = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    if thinking {
        head
    } else {
        format!("{head}<think>\n\n</think>\n\n")
    }
}

fn build_user_message(
    text: &str,
    app_name: Option<&str>,
    app_bundle_id: Option<&str>,
    dictionary: Option<&HashMap<String, String>>,
) -> String {
    let mut parts = vec![format!("Speech recognition text: {text}")];
    if app_name.is_some() || app_bundle_id.is_some() {
        let mut app_info = app_name.unwrap_or("").to_string();
        if let Some(bid) = app_bundle_id {
            if app_info.is_empty() {
                app_info = bid.to_string();
            } else {
                app_info = format!("{app_info} ({bid})");
            }
        }
        parts.push(format!("Target app: {app_info}"));
    }
    if let Some(dict) = dictionary {
        if !dict.is_empty() {
            let s = dict
                .iter()
                .map(|(k, v)| format!("\"{k}\"→\"{v}\""))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!("User dictionary: {s}"));
        }
    }
    parts.join("\n")
}

/// Drop a leading <think>…</think> block (Qwen3 thinking trace), if present.
fn strip_think(s: &str) -> String {
    if let Some(end) = s.find("</think>") {
        s[end + "</think>".len()..].to_string()
    } else {
        s.to_string()
    }
}

fn parse_config(v: &serde_json::Value) -> Result<Config> {
    let g = |k: &str| v.get(k).and_then(|x| x.as_i64());
    let gf = |k: &str| v.get(k).and_then(|x| x.as_f64());
    Ok(Config {
        hidden_size: g("hidden_size").ok_or_else(|| anyhow!("hidden_size"))? as i32,
        n_layers: g("num_hidden_layers").ok_or_else(|| anyhow!("num_hidden_layers"))? as usize,
        n_heads: g("num_attention_heads").ok_or_else(|| anyhow!("num_attention_heads"))? as i32,
        n_kv_heads: g("num_key_value_heads").ok_or_else(|| anyhow!("num_key_value_heads"))? as i32,
        head_dim: g("head_dim").unwrap_or(128) as i32,
        rope_theta: gf("rope_theta").unwrap_or(1_000_000.0) as f32,
        rms_eps: gf("rms_norm_eps").unwrap_or(1e-6) as f32,
        eos_token_id: g("eos_token_id").unwrap_or(151645) as i32,
    })
}

fn parse_config_3_5(v: &serde_json::Value, im_end_id: i32) -> Result<qwen3_5::Config> {
    let quant = v.get("quantization").ok_or_else(|| anyhow!("quantization"))?;
    let bits = quant.get("bits").and_then(|x| x.as_i64());
    let group_size = quant.get("group_size").and_then(|x| x.as_i64());
    if (bits, group_size) != (Some(qwen3::BITS as i64), Some(qwen3::GROUP_SIZE as i64)) {
        return Err(anyhow!("unsupported quantization: {quant}"));
    }

    let t = v.get("text_config").ok_or_else(|| anyhow!("text_config"))?;
    let g = |k: &str| {
        t.get(k)
            .and_then(|x| x.as_i64())
            .ok_or_else(|| anyhow!("text_config.{k}"))
    };
    let rope = t.get("rope_parameters");
    let rope_f = |k: &str, default: f64| {
        rope.and_then(|r| r.get(k)).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let n_heads = g("num_attention_heads")? as i32;
    let hidden_size = g("hidden_size")? as i32;
    let head_dim = g("head_dim").map(|d| d as i32).unwrap_or(hidden_size / n_heads);
    let mut stop_token_ids = vec![im_end_id];
    if let Ok(eos) = g("eos_token_id") {
        stop_token_ids.push(eos as i32);
    }
    Ok(qwen3_5::Config {
        hidden_size,
        n_layers: g("num_hidden_layers")? as usize,
        n_heads,
        n_kv_heads: g("num_key_value_heads")? as i32,
        head_dim,
        rope_dims: (head_dim as f64 * rope_f("partial_rotary_factor", 0.25)) as i32,
        rope_theta: rope_f("rope_theta", 100_000.0) as f32,
        rms_eps: t.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32,
        full_attention_interval: g("full_attention_interval").unwrap_or(4) as usize,
        linear_k_heads: g("linear_num_key_heads")? as i32,
        linear_v_heads: g("linear_num_value_heads")? as i32,
        linear_k_dim: g("linear_key_head_dim")? as i32,
        linear_v_dim: g("linear_value_head_dim")? as i32,
        conv_kernel: g("linear_conv_kernel_dim")? as i32,
        stop_token_ids,
    })
}
