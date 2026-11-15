//! SQLite database layer: the `memos` table (meeting recordings and older
//! voice memos).
//!
//! Databases from earlier versions also hold the always-on history
//! (`transcriptions` and its FTS index). Nothing reads it any more; it is left
//! in place rather than dropped, so that no one's data disappears.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Memos: one recording each (a meeting, or a voice memo from an earlier
/// version) with its transcript and minutes (see `memo.rs`, `meeting.rs`).
const MEMO_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    duration_seconds REAL,
    audio_path TEXT NOT NULL,
    status TEXT NOT NULL,
    transcript TEXT,
    segments_json TEXT,
    minutes TEXT,
    language TEXT,
    model_name TEXT,
    error TEXT
);
";

/// Lifecycle of a memo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoStatus {
    Recording,
    Transcribing,
    Summarizing,
    Done,
    Error,
    /// The app exited while recording or processing; the audio captured so
    /// far is still on disk and can be re-processed.
    Interrupted,
}

impl MemoStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Recording => "recording",
            Self::Transcribing => "transcribing",
            Self::Summarizing => "summarizing",
            Self::Done => "done",
            Self::Error => "error",
            Self::Interrupted => "interrupted",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "recording" => Self::Recording,
            "transcribing" => Self::Transcribing,
            "summarizing" => Self::Summarizing,
            "done" => Self::Done,
            "interrupted" => Self::Interrupted,
            _ => Self::Error,
        }
    }

    /// Transcription or minutes generation is in flight.
    pub fn is_processing(self) -> bool {
        matches!(self, Self::Transcribing | Self::Summarizing)
    }
}

/// A memo with its full transcript and minutes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memo {
    pub id: i64,
    pub created_at: String,
    pub duration_seconds: Option<f64>,
    pub audio_path: String,
    pub status: MemoStatus,
    /// Timestamped transcript, one `[MM:SS] text` line per chunk.
    pub transcript: Option<String>,
    pub segments_json: Option<String>,
    pub minutes: Option<String>,
    pub language: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

/// List row for a memo: metadata plus a short preview, without the
/// (potentially large) transcript and minutes bodies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoListItem {
    pub id: i64,
    pub created_at: String,
    pub duration_seconds: Option<f64>,
    pub status: MemoStatus,
    pub preview: Option<String>,
    pub error: Option<String>,
}

pub struct TranscriptionDb {
    conn: Connection,
}

impl TranscriptionDb {
    /// Open (or create) the database at the given path and initialize schema.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create DB directory: {:?}", parent))?;
        }

        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open database: {:?}", path))?;

        conn.execute_batch(MEMO_SCHEMA)
            .context("Failed to initialize memo schema")?;

        Ok(Self { conn })
    }

    // ===== Memos =====

    /// Create a memo row for a recording that is starting now. The audio file
    /// path is `<audio_dir>/memo-<local timestamp>-<id>.wav`.
    pub fn insert_memo(&self, audio_dir: &Path) -> Result<Memo> {
        let id: i64 = self.conn.query_row(
            "INSERT INTO memos (audio_path, status) VALUES ('', ?1) RETURNING id",
            params![MemoStatus::Recording.as_str()],
            |row| row.get(0),
        )?;
        // The id keeps two memos started within the same second apart.
        self.conn.execute(
            "UPDATE memos
             SET audio_path = ?2 || '/memo-' || strftime('%Y%m%d-%H%M%S', created_at) || '-' || id || '.wav'
             WHERE id = ?1",
            params![id, audio_dir.to_string_lossy()],
        )?;
        self.get_memo(id)?.context("memo vanished after insert")
    }

    pub fn get_memo(&self, id: i64) -> Result<Option<Memo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, duration_seconds, audio_path, status, transcript,
                    segments_json, minutes, language, model_name, error
             FROM memos WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Memo {
                id: row.get(0)?,
                created_at: row.get(1)?,
                duration_seconds: row.get(2)?,
                audio_path: row.get(3)?,
                status: MemoStatus::parse(&row.get::<_, String>(4)?),
                transcript: row.get(5)?,
                segments_json: row.get(6)?,
                minutes: row.get(7)?,
                language: row.get(8)?,
                model_name: row.get(9)?,
                error: row.get(10)?,
            })
        })?;
        Ok(rows.next().transpose()?)
    }

    /// All memos, newest first. The preview is the head of the minutes, or of
    /// the transcript while minutes don't exist yet.
    pub fn list_memos(&self) -> Result<Vec<MemoListItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, duration_seconds, status,
                    substr(COALESCE(minutes, transcript), 1, 200), error
             FROM memos ORDER BY created_at DESC, id DESC",
        )?;
        let items = stmt
            .query_map([], |row| {
                Ok(MemoListItem {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    duration_seconds: row.get(2)?,
                    status: MemoStatus::parse(&row.get::<_, String>(3)?),
                    preview: row.get(4)?,
                    error: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(items)
    }

    /// Set a memo's status; `error` is stored alongside (None clears it).
    pub fn set_memo_status(&self, id: i64, status: MemoStatus, error: Option<&str>) -> Result<()> {
        self.conn.execute(
            "UPDATE memos SET status = ?2, error = ?3 WHERE id = ?1",
            params![id, status.as_str(), error],
        )?;
        Ok(())
    }

    pub fn set_memo_duration(&self, id: i64, duration_seconds: f64) -> Result<()> {
        self.conn.execute(
            "UPDATE memos SET duration_seconds = ?2 WHERE id = ?1",
            params![id, duration_seconds],
        )?;
        Ok(())
    }

    pub fn set_memo_transcript(
        &self,
        id: i64,
        transcript: &str,
        segments_json: Option<&str>,
        language: Option<&str>,
        model_name: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE memos SET transcript = ?2, segments_json = ?3, language = ?4, model_name = ?5
             WHERE id = ?1",
            params![id, transcript, segments_json, language, model_name],
        )?;
        Ok(())
    }

    pub fn set_memo_minutes(&self, id: i64, minutes: Option<&str>) -> Result<()> {
        self.conn.execute(
            "UPDATE memos SET minutes = ?2 WHERE id = ?1",
            params![id, minutes],
        )?;
        Ok(())
    }

    /// Move a memo's start back by `seconds` (a meeting that begins with audio
    /// from before it was started). Returns the new `created_at`.
    pub fn backdate_memo(&self, id: i64, seconds: u64) -> Result<String> {
        Ok(self.conn.query_row(
            "UPDATE memos SET created_at = datetime(created_at, ?2) WHERE id = ?1 RETURNING created_at",
            params![id, format!("-{seconds} seconds")],
            |row| row.get(0),
        )?)
    }

    /// Delete a memo row. Returns its audio path so the caller can remove the file.
    pub fn delete_memo(&self, id: i64) -> Result<Option<String>> {
        let path = self.get_memo(id)?.map(|m| m.audio_path);
        self.conn.execute("DELETE FROM memos WHERE id = ?1", params![id])?;
        Ok(path)
    }

    /// Mark memos left mid-recording or mid-processing by a previous run as
    /// interrupted. Call once at startup, before anything can be in flight.
    pub fn mark_interrupted_memos(&self) -> Result<u32> {
        let rows = self.conn.execute(
            "UPDATE memos SET status = 'interrupted'
             WHERE status IN ('recording', 'transcribing', 'summarizing')",
            [],
        )?;
        Ok(rows as u32)
    }

    /// Whether a memo is recording or being processed (what an exit would interrupt).
    pub fn has_memo_in_progress(&self) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM memos
                           WHERE status IN ('recording', 'transcribing', 'summarizing'))",
            [],
            |row| row.get(0),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (TranscriptionDb, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = TranscriptionDb::open(&db_path).unwrap();
        (db, dir)
    }

    #[test]
    fn test_memo_lifecycle() {
        let (db, _dir) = temp_db();

        let memo = db.insert_memo(Path::new("/tmp")).unwrap();
        assert_eq!(memo.status, MemoStatus::Recording);
        assert!(memo.audio_path.starts_with("/tmp/memo-") && memo.audio_path.ends_with(".wav"));
        assert!(!memo.created_at.is_empty());

        db.set_memo_duration(memo.id, 12.5).unwrap();
        db.set_memo_transcript(memo.id, "[00:00] こんにちは", Some("[]"), Some("ja"), Some("m"))
            .unwrap();
        db.set_memo_status(memo.id, MemoStatus::Error, Some("boom")).unwrap();
        let list = db.list_memos().unwrap();
        assert_eq!(list[0].preview.as_deref(), Some("[00:00] こんにちは"));
        assert_eq!(list[0].error.as_deref(), Some("boom"));

        db.set_memo_minutes(memo.id, Some("## 概要")).unwrap();
        db.set_memo_status(memo.id, MemoStatus::Done, None).unwrap();
        let got = db.get_memo(memo.id).unwrap().unwrap();
        assert_eq!(got.status, MemoStatus::Done);
        assert_eq!(got.duration_seconds, Some(12.5));
        assert_eq!(got.minutes.as_deref(), Some("## 概要"));
        assert!(got.error.is_none());
        assert_eq!(db.list_memos().unwrap()[0].preview.as_deref(), Some("## 概要"));

        assert_eq!(db.delete_memo(memo.id).unwrap(), Some(memo.audio_path));
        assert!(db.get_memo(memo.id).unwrap().is_none());
    }

    #[test]
    fn test_mark_interrupted_memos() {
        let (db, _dir) = temp_db();

        let recording = db.insert_memo(Path::new("/tmp")).unwrap();
        let summarizing = db.insert_memo(Path::new("/tmp")).unwrap();
        assert_ne!(recording.audio_path, summarizing.audio_path);
        db.set_memo_status(summarizing.id, MemoStatus::Summarizing, None).unwrap();
        let done = db.insert_memo(Path::new("/tmp")).unwrap();
        db.set_memo_status(done.id, MemoStatus::Done, None).unwrap();

        assert_eq!(db.mark_interrupted_memos().unwrap(), 2);
        let status = |id| db.get_memo(id).unwrap().unwrap().status;
        assert_eq!(status(recording.id), MemoStatus::Interrupted);
        assert_eq!(status(summarizing.id), MemoStatus::Interrupted);
        assert_eq!(status(done.id), MemoStatus::Done);
    }
}
