//! Minimal SentencePiece `tokenizer.model` reader: the piece table only, for
//! decoding ids to text (the `tokenizers` crate does not read `.model` files).
//!
//! `ModelProto.pieces` is field 1 (repeated message); each `SentencePiece` has
//! `piece` (1, string), `score` (2, float) and `type` (3, enum; NORMAL when
//! absent). Haas-ja's tokenizer uses byte fallback, so pieces of type BYTE
//! (`<0xE3>`) are raw bytes to be joined into UTF-8.

use anyhow::{anyhow, bail, Result};

const TYPE_NORMAL: u64 = 1;
const TYPE_UNKNOWN: u64 = 2;
const TYPE_CONTROL: u64 = 3;
const TYPE_BYTE: u64 = 6;

#[derive(Debug, Clone, PartialEq)]
enum Piece {
    Text(String),
    Byte(u8),
    Skip, // control / unknown: not part of the decoded text
}

pub struct SentencePiece {
    pieces: Vec<Piece>,
}

impl SentencePiece {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let buf = std::fs::read(path).map_err(|e| anyhow!("read {path:?}: {e}"))?;
        Self::parse(&buf)
    }

    pub fn parse(buf: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut r = Reader { buf, pos: 0 };
        while !r.done() {
            let (field, wire) = r.key()?;
            if field == 1 && wire == 2 {
                pieces.push(parse_piece(r.bytes()?)?);
            } else {
                r.skip(wire)?;
            }
        }
        if pieces.is_empty() {
            bail!("no pieces in SentencePiece model");
        }
        Ok(Self { pieces })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Decode ids as `SentencePieceProcessor.decode` does for this model:
    /// byte pieces joined into UTF-8, `▁` as a space, control/unknown dropped,
    /// surrounding whitespace trimmed.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match usize::try_from(id).ok().and_then(|i| self.pieces.get(i)) {
                Some(Piece::Text(s)) => bytes.extend_from_slice(s.as_bytes()),
                Some(Piece::Byte(b)) => bytes.push(*b),
                Some(Piece::Skip) | None => {}
            }
        }
        String::from_utf8_lossy(&bytes)
            .replace('\u{2581}', " ")
            .trim()
            .to_string()
    }
}

fn parse_piece(buf: &[u8]) -> Result<Piece> {
    let mut r = Reader { buf, pos: 0 };
    let mut text = String::new();
    let mut ty = TYPE_NORMAL;
    while !r.done() {
        match r.key()? {
            (1, 2) => text = String::from_utf8(r.bytes()?.to_vec())?,
            (3, 0) => ty = r.varint()?,
            (_, wire) => r.skip(wire)?,
        }
    }
    Ok(match ty {
        TYPE_UNKNOWN | TYPE_CONTROL => Piece::Skip,
        TYPE_BYTE => {
            let hex = text
                .strip_prefix("<0x")
                .and_then(|s| s.strip_suffix('>'))
                .ok_or_else(|| anyhow!("bad byte piece {text:?}"))?;
            Piece::Byte(u8::from_str_radix(hex, 16)?)
        }
        _ => Piece::Text(text),
    })
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or_else(|| anyhow!("truncated varint"))?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("varint too long")
    }

    fn key(&mut self) -> Result<(u64, u64)> {
        let k = self.varint()?;
        Ok((k >> 3, k & 7))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| anyhow!("truncated field"))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        self.take(n)
    }

    fn skip(&mut self, wire: u64) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.take(4)?;
            }
            _ => bail!("unsupported wire type {wire}"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(text: &str, ty: Option<u8>) -> Vec<u8> {
        let mut p = vec![0x0a, text.len() as u8];
        p.extend_from_slice(text.as_bytes());
        p.extend_from_slice(&[0x15, 0, 0, 0, 0]); // score (fixed32)
        if let Some(t) = ty {
            p.extend_from_slice(&[0x18, t]);
        }
        let mut out = vec![0x0a, p.len() as u8];
        out.extend(p);
        out
    }

    #[test]
    fn decodes_text_bytes_and_skips_control() {
        let mut m = Vec::new();
        m.extend(piece("<unk>", Some(2)));
        m.extend(piece("<s>", Some(3)));
        m.extend(piece("\u{2581}今日", None));
        m.extend(piece("<0xE3>", Some(6)));
        m.extend(piece("<0x81>", Some(6)));
        m.extend(piece("<0xAF>", Some(6)));
        m.extend(piece("晴れ", Some(1)));
        // An unrelated top-level field (trainer_spec) is skipped.
        m.extend_from_slice(&[0x12, 0x02, 0x08, 0x01]);
        let sp = SentencePiece::parse(&m).unwrap();
        assert_eq!(sp.len(), 7);
        assert_eq!(sp.decode(&[1, 2, 3, 4, 5, 6, 0, 99]), "今日は晴れ");
    }
}
