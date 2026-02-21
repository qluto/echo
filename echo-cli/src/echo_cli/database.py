"""SQLite database layer for transcription history with FTS5 full-text search."""

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptionEntry:
    text: str
    created_at: str = ""
    duration_seconds: Optional[float] = None
    raw_text: Optional[str] = None
    language: Optional[str] = None
    model_name: Optional[str] = None
    segments_json: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["segments_json"]:
            d["segments"] = json.loads(d["segments_json"])
            del d["segments_json"]
        else:
            d["segments"] = []
            del d["segments_json"]
        return d


@dataclass
class HistoryPage:
    entries: list[TranscriptionEntry]
    total_count: int
    has_more: bool


_SCHEMA = """
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
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts
    USING fts5(text, content=transcriptions, content_rowid=id);

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
"""


def _default_db_path() -> Path:
    """Return default DB path following macOS conventions."""
    base = Path.home() / "Library" / "Application Support" / "io.qluto.echo"
    base.mkdir(parents=True, exist_ok=True)
    return base / "transcriptions.db"


class TranscriptionDatabase:
    def __init__(self, db_path: Optional[Path] = None):
        self._path = db_path or _default_db_path()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_FTS_SCHEMA)

    def insert(self, entry: TranscriptionEntry) -> int:
        cur = self._conn.execute(
            """INSERT INTO transcriptions
               (duration_seconds, text, raw_text, language, model_name, segments_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entry.duration_seconds,
                entry.text,
                entry.raw_text,
                entry.language,
                entry.model_name,
                entry.segments_json,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_all(self, limit: int = 20, offset: int = 0) -> HistoryPage:
        total = self.count()
        rows = self._conn.execute(
            """SELECT * FROM transcriptions
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        return HistoryPage(
            entries=entries,
            total_count=total,
            has_more=(offset + limit) < total,
        )

    def search(self, query: str, limit: int = 20, offset: int = 0) -> HistoryPage:
        count_row = self._conn.execute(
            """SELECT COUNT(*) FROM transcriptions_fts WHERE text MATCH ?""",
            (query,),
        ).fetchone()
        total = count_row[0]

        rows = self._conn.execute(
            """SELECT t.* FROM transcriptions t
               JOIN transcriptions_fts fts ON t.id = fts.rowid
               WHERE fts.text MATCH ?
               ORDER BY t.created_at DESC LIMIT ? OFFSET ?""",
            (query, limit, offset),
        ).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        return HistoryPage(
            entries=entries,
            total_count=total,
            has_more=(offset + limit) < total,
        )

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()
        return row[0]

    def delete(self, entry_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM transcriptions WHERE id = ?", (entry_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_all(self) -> int:
        cur = self._conn.execute("DELETE FROM transcriptions")
        self._conn.commit()
        return cur.rowcount

    def export_all(self, fmt: str = "json") -> str:
        rows = self._conn.execute(
            "SELECT * FROM transcriptions ORDER BY created_at ASC"
        ).fetchall()
        entries = [self._row_to_entry(r) for r in rows]

        if fmt == "json":
            return json.dumps(
                [e.to_dict() for e in entries], ensure_ascii=False, indent=2
            )
        else:
            lines = []
            for e in entries:
                lines.append(f"[{e.created_at}] ({e.duration_seconds or 0:.1f}s) {e.text}")
            return "\n".join(lines)

    def close(self):
        self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> TranscriptionEntry:
        return TranscriptionEntry(
            id=row["id"],
            created_at=row["created_at"],
            duration_seconds=row["duration_seconds"],
            text=row["text"],
            raw_text=row["raw_text"],
            language=row["language"],
            model_name=row["model_name"],
            segments_json=row["segments_json"],
        )
