//! In-process LLM post-processing (Qwen3) — the full-Rust replacement for the
//! Python `PostProcessor`. Cleans up ASR text (filler removal, self-correction,
//! dictionary, app-context formatting) and summarizes transcription history.

pub mod qwen3;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

use qwen3::{Config, Qwen3};
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

pub struct PostProcessor {
    model: Qwen3,
    tokenizer: Tokenizer,
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

        let cfg = parse_config(&config_path)?;
        let weights = crate::weights::Weights::load(st.to_str().ok_or_else(|| anyhow!("path"))?)?;
        let model = Qwen3::load(&weights, cfg)?;
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("tokenizer load: {e}"))?;
        log::info!("Post-processor (Qwen3) loaded: {model_id}");
        Ok(Self { model, tokenizer })
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
        // Thinking mode (matches Python summarize) for summary quality.
        let prompt = chat_prompt(system, &user, true);
        self.run(&prompt, 2048)
    }

    fn run(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let gen = self.model.generate(&ids, max_tokens)?;
        let gen_u32: Vec<u32> = gen.iter().map(|&i| i as u32).collect();
        let text = self
            .tokenizer
            .decode(&gen_u32, true)
            .map_err(|e| anyhow!("decode: {e}"))?;
        Ok(strip_think(&text).trim().to_string())
    }

    pub fn warmup(&self) -> Result<()> {
        let _ = self.process("テスト", None, None, None, None)?;
        Ok(())
    }
}

/// Qwen3 chat template (system + user). When `thinking` is false this matches
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

fn parse_config(path: &Path) -> Result<Config> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
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
