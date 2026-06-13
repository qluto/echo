//! Tokenizer wrapper over the model's tokenizer.json (HF `tokenizers`, pure
//! Rust). Mirrors CohereAsrTokenizer.build_prompt_tokens + decode.

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub const EOS_ID: i32 = 3; // matches generation_config.eos_token_id

pub struct Tok {
    inner: Tokenizer,
}

impl Tok {
    pub fn load(path: &str) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| anyhow!("tokenizer load: {e}"))?;
        Ok(Self { inner })
    }

    fn id(&self, piece: &str) -> Result<i32> {
        self.inner
            .token_to_id(piece)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("unknown special token: {piece}"))
    }

    /// Prompt prefix per CohereAsrTokenizer.build_prompt_tokens.
    pub fn build_prompt(&self, lang: &str, punctuation: bool) -> Result<Vec<i32>> {
        let lang_tok = format!("<|{lang}|>");
        let pnc = if punctuation { "<|pnc|>" } else { "<|nopnc|>" };
        let pieces = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            &lang_tok,
            &lang_tok,
            pnc,
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ];
        pieces.iter().map(|p| self.id(p)).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}
