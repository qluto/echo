//! SQLite database layer for transcription history with FTS5 full-text search.
//!
//! Schema is compatible with echo-cli's Python database implementation.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single transcription entry stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionEntry {
    pub id: Option<i64>,
    pub created_at: String,
    pub duration_seconds: Option<f64>,
    pub text: String,
    pub raw_text: Option<String>,
    pub language: Option<String>,
    pub model_name: Option<String>,
    pub segments_json: Option<String>,
}

/// A page of transcription entries with pagination metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryPage {
    pub entries: Vec<TranscriptionEntry>,
    pub total_count: u32,
    pub has_more: bool,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    duration_seconds REAL,
    text TEXT NOT NULL,
    raw_text TEXT,
    language TEXT,
    model_name TEXT,
    segments_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at
    ON transcriptions(created_at);
";

const FTS_SCHEMA: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts
    USING fts5(text, content=transcriptions, content_rowid=id, tokenize='trigram');

CREATE TRIGGER IF NOT EXISTS transcriptions_ai AFTER INSERT ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS transcriptions_ad AFTER DELETE ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(transcriptions_fts, rowid, text)
        VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS transcriptions_au AFTER UPDATE ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(transcriptions_fts, rowid, text)
        VALUES('delete', old.id, old.text);
    INSERT INTO transcriptions_fts(rowid, text) VALUES (new.id, new.text);
END;
";

pub struct TranscriptionDb {
    conn: Connection,
    path: PathBuf,
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

        conn.execute_batch(SCHEMA)
            .context("Failed to initialize schema")?;
        conn.execute_batch(FTS_SCHEMA)
            .context("Failed to initialize FTS schema")?;

        Ok(Self {
            conn,
            path: path.to_path_buf(),
        })
    }

    /// Insert a new transcription entry. Returns the new row ID.
    pub fn insert(&self, entry: &TranscriptionEntry) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO transcriptions (duration_seconds, text, raw_text, language, model_name, segments_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                entry.duration_seconds,
                entry.text,
                entry.raw_text,
                entry.language,
                entry.model_name,
                entry.segments_json,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get paginated history ordered by created_at DESC.
    pub fn get_all(&self, limit: u32, offset: u32) -> Result<HistoryPage> {
        let total = self.count()?;

        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, duration_seconds, text, raw_text, language, model_name, segments_json
             FROM transcriptions ORDER BY created_at DESC LIMIT ?1 OFFSET ?2",
        )?;

        let entries = stmt
            .query_map(params![limit, offset], Self::row_to_entry)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(HistoryPage {
            entries,
            total_count: total,
            has_more: (offset + limit) < total,
        })
    }

    /// Full-text search. Uses FTS5 trigram for queries with 3+ chars,
    /// falls back to LIKE for shorter queries (trigram minimum is 3 characters).
    pub fn search(&self, query: &str, limit: u32, offset: u32) -> Result<HistoryPage> {
        let char_count = query.chars().count();

        if char_count >= 3 {
            // FTS5 trigram search (fast, indexed)
            let total: u32 = self.conn.query_row(
                "SELECT COUNT(*) FROM transcriptions_fts WHERE text MATCH ?1",
                params![query],
                |row| row.get(0),
            )?;

            let mut stmt = self.conn.prepare(
                "SELECT t.id, t.created_at, t.duration_seconds, t.text, t.raw_text, t.language, t.model_name, t.segments_json
                 FROM transcriptions t
                 JOIN transcriptions_fts fts ON t.id = fts.rowid
                 WHERE fts.text MATCH ?1
                 ORDER BY t.created_at DESC LIMIT ?2 OFFSET ?3",
            )?;

            let entries = stmt
                .query_map(params![query, limit, offset], Self::row_to_entry)?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(HistoryPage {
                entries,
                total_count: total,
                has_more: (offset + limit) < total,
            })
        } else {
            // LIKE fallback for short queries (1-2 chars)
            let like_pattern = format!("%{}%", query);

            let total: u32 = self.conn.query_row(
                "SELECT COUNT(*) FROM transcriptions WHERE text LIKE ?1",
                params![like_pattern],
                |row| row.get(0),
            )?;

            let mut stmt = self.conn.prepare(
                "SELECT id, created_at, duration_seconds, text, raw_text, language, model_name, segments_json
                 FROM transcriptions WHERE text LIKE ?1
                 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
            )?;

            let entries = stmt
                .query_map(params![like_pattern, limit, offset], Self::row_to_entry)?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(HistoryPage {
                entries,
                total_count: total,
                has_more: (offset + limit) < total,
            })
        }
    }

    /// Count total transcription entries.
    pub fn count(&self) -> Result<u32> {
        let count: u32 = self
            .conn
            .query_row("SELECT COUNT(*) FROM transcriptions", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Delete a single entry by ID. Returns true if a row was deleted.
    pub fn delete(&self, entry_id: i64) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM transcriptions WHERE id = ?1", params![entry_id])?;
        Ok(rows > 0)
    }

    /// Delete all entries. Returns the number of deleted rows.
    pub fn delete_all(&self) -> Result<u32> {
        let rows = self.conn.execute("DELETE FROM transcriptions", [])?;
        Ok(rows as u32)
    }

    /// Get the database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn row_to_entry(row: &rusqlite::Row) -> rusqlite::Result<TranscriptionEntry> {
        Ok(TranscriptionEntry {
            id: row.get(0)?,
            created_at: row.get(1)?,
            duration_seconds: row.get(2)?,
            text: row.get(3)?,
            raw_text: row.get(4)?,
            language: row.get(5)?,
            model_name: row.get(6)?,
            segments_json: row.get(7)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_db() -> (TranscriptionDb, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = TranscriptionDb::open(&db_path).unwrap();
        (db, dir)
    }

    fn sample_entry(text: &str) -> TranscriptionEntry {
        TranscriptionEntry {
            id: None,
            created_at: String::new(),
            duration_seconds: Some(2.5),
            text: text.to_string(),
            raw_text: None,
            language: Some("Japanese".to_string()),
            model_name: Some("mlx-community/Qwen3-ASR-0.6B-8bit".to_string()),
            segments_json: None,
        }
    }

    #[test]
    fn test_insert_and_get() {
        let (db, _dir) = temp_db();

        let id = db.insert(&sample_entry("テスト書き起こし")).unwrap();
        assert!(id > 0);

        let page = db.get_all(10, 0).unwrap();
        assert_eq!(page.total_count, 1);
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].text, "テスト書き起こし");
        assert_eq!(page.entries[0].id, Some(id));
        assert!(!page.entries[0].created_at.is_empty());
    }

    #[test]
    fn test_pagination() {
        let (db, _dir) = temp_db();

        for i in 0..5 {
            db.insert(&sample_entry(&format!("Entry {}", i))).unwrap();
        }

        let page1 = db.get_all(2, 0).unwrap();
        assert_eq!(page1.total_count, 5);
        assert_eq!(page1.entries.len(), 2);
        assert!(page1.has_more);

        let page2 = db.get_all(2, 2).unwrap();
        assert_eq!(page2.entries.len(), 2);
        assert!(page2.has_more);

        let page3 = db.get_all(2, 4).unwrap();
        assert_eq!(page3.entries.len(), 1);
        assert!(!page3.has_more);
    }

    #[test]
    fn test_fts5_search() {
        let (db, _dir) = temp_db();

        db.insert(&sample_entry("今日は天気がいいですね")).unwrap();
        db.insert(&sample_entry("明日の会議は10時からです")).unwrap();
        db.insert(&sample_entry("天気予報によると雨です")).unwrap();

        // 3+ chars: uses FTS5 trigram
        let page = db.search("天気予報", 10, 0).unwrap();
        assert_eq!(page.total_count, 1);

        // 2 chars: falls back to LIKE
        let page = db.search("天気", 10, 0).unwrap();
        assert_eq!(page.total_count, 2);
        assert_eq!(page.entries.len(), 2);

        // English substring (3+ chars via FTS5 trigram)
        db.insert(&sample_entry("hello world")).unwrap();
        let page = db.search("hello", 10, 0).unwrap();
        assert_eq!(page.total_count, 1);
    }

    #[test]
    fn test_delete() {
        let (db, _dir) = temp_db();

        let id = db.insert(&sample_entry("削除テスト")).unwrap();
        assert_eq!(db.count().unwrap(), 1);

        assert!(db.delete(id).unwrap());
        assert_eq!(db.count().unwrap(), 0);

        assert!(!db.delete(id).unwrap());
    }

    #[test]
    fn test_delete_all() {
        let (db, _dir) = temp_db();

        for i in 0..3 {
            db.insert(&sample_entry(&format!("Entry {}", i))).unwrap();
        }

        let deleted = db.delete_all().unwrap();
        assert_eq!(deleted, 3);
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn test_fts5_sync_on_delete() {
        let (db, _dir) = temp_db();

        let id = db.insert(&sample_entry("検索用テキスト")).unwrap();

        // 3+ chars uses FTS5 trigram
        let page = db.search("検索用", 10, 0).unwrap();
        assert_eq!(page.total_count, 1);

        db.delete(id).unwrap();

        let page = db.search("検索用", 10, 0).unwrap();
        assert_eq!(page.total_count, 0);

        // Also verify LIKE fallback works after delete
        let id2 = db.insert(&sample_entry("テスト文字列")).unwrap();
        let page = db.search("テス", 10, 0).unwrap();
        assert_eq!(page.total_count, 1);
        db.delete(id2).unwrap();
        let page = db.search("テス", 10, 0).unwrap();
        assert_eq!(page.total_count, 0);
    }
}
