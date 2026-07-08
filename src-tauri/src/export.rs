//! Serialize transcription history for file export (JSON / CSV / Markdown).

use crate::database::TranscriptionEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Json,
    Csv,
    Markdown,
}

impl ExportFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
            ExportFormat::Markdown => "md",
        }
    }

    pub fn filter_name(&self) -> &'static str {
        match self {
            ExportFormat::Json => "JSON",
            ExportFormat::Csv => "CSV",
            ExportFormat::Markdown => "Markdown",
        }
    }
}

/// Render entries (expected in chronological order) into the given format.
pub fn format_entries(format: ExportFormat, entries: &[TranscriptionEntry]) -> Result<String, String> {
    match format {
        ExportFormat::Json => serde_json::to_string_pretty(entries).map_err(|e| e.to_string()),
        ExportFormat::Csv => Ok(to_csv(entries)),
        ExportFormat::Markdown => Ok(to_markdown(entries)),
    }
}

fn to_csv(entries: &[TranscriptionEntry]) -> String {
    let mut out = String::from("id,created_at,duration_seconds,text,raw_text,language,model_name\n");
    for e in entries {
        let id = e.id.map(|v| v.to_string()).unwrap_or_default();
        let duration = e.duration_seconds.map(|v| v.to_string()).unwrap_or_default();
        out.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            id,
            csv_field(&e.created_at),
            duration,
            csv_field(&e.text),
            csv_field(e.raw_text.as_deref().unwrap_or("")),
            csv_field(e.language.as_deref().unwrap_or("")),
            csv_field(e.model_name.as_deref().unwrap_or("")),
        ));
    }
    out
}

/// Quote a CSV field unconditionally, doubling embedded quotes (RFC 4180).
fn csv_field(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn to_markdown(entries: &[TranscriptionEntry]) -> String {
    let mut out = String::from("# Echo Transcripts\n");
    let mut current_date = "";
    for e in entries {
        // created_at is "YYYY-MM-DD HH:MM:SS" (SQLite datetime, localtime)
        let (date, time) = match e.created_at.split_once(' ') {
            Some((d, t)) => (d, &t[..t.len().min(5)]),
            None => (e.created_at.as_str(), ""),
        };
        if date != current_date {
            out.push_str(&format!("\n## {}\n\n", date));
            current_date = date;
        }
        let text = e.text.replace('\n', " ");
        if time.is_empty() {
            out.push_str(&format!("- {}\n", text));
        } else {
            out.push_str(&format!("- **{}** {}\n", time, text));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: i64, created_at: &str, text: &str) -> TranscriptionEntry {
        TranscriptionEntry {
            id: Some(id),
            created_at: created_at.to_string(),
            duration_seconds: Some(2.5),
            text: text.to_string(),
            raw_text: None,
            language: Some("ja".to_string()),
            model_name: Some("mlx-community/whisper-large-v3-turbo".to_string()),
            segments_json: None,
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let entries = vec![entry(1, "2026-07-08 10:00:00", "こんにちは")];
        let json = format_entries(ExportFormat::Json, &entries).unwrap();
        let parsed: Vec<TranscriptionEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "こんにちは");
    }

    #[test]
    fn test_csv_escaping() {
        let entries = vec![entry(1, "2026-07-08 10:00:00", "He said \"hi\",\nthen left")];
        let csv = format_entries(ExportFormat::Csv, &entries).unwrap();
        let mut lines = csv.lines();
        assert_eq!(
            lines.next().unwrap(),
            "id,created_at,duration_seconds,text,raw_text,language,model_name"
        );
        // Embedded quotes doubled, comma and newline preserved inside quotes
        assert!(csv.contains("\"He said \"\"hi\"\",\nthen left\""));
    }

    #[test]
    fn test_csv_empty_optionals() {
        let mut e = entry(1, "2026-07-08 10:00:00", "text");
        e.duration_seconds = None;
        e.raw_text = None;
        e.language = None;
        e.model_name = None;
        let csv = format_entries(ExportFormat::Csv, &[e]).unwrap();
        assert!(csv.contains("1,\"2026-07-08 10:00:00\",,\"text\",\"\",\"\",\"\""));
    }

    #[test]
    fn test_markdown_groups_by_date() {
        let entries = vec![
            entry(1, "2026-07-07 09:15:00", "first day"),
            entry(2, "2026-07-08 10:00:00", "second\nday"),
            entry(3, "2026-07-08 11:30:00", "same day"),
        ];
        let md = format_entries(ExportFormat::Markdown, &entries).unwrap();
        assert_eq!(md.matches("## 2026-07-07").count(), 1);
        assert_eq!(md.matches("## 2026-07-08").count(), 1);
        assert!(md.contains("- **09:15** first day"));
        // Newlines inside a transcript are flattened for the list item
        assert!(md.contains("- **10:00** second day"));
    }

    #[test]
    fn test_format_deserializes_from_lowercase() {
        let f: ExportFormat = serde_json::from_str("\"markdown\"").unwrap();
        assert_eq!(f, ExportFormat::Markdown);
        assert_eq!(f.extension(), "md");
    }

    #[test]
    fn test_empty_history() {
        assert_eq!(format_entries(ExportFormat::Json, &[]).unwrap(), "[]");
        let csv = format_entries(ExportFormat::Csv, &[]).unwrap();
        assert_eq!(csv.lines().count(), 1); // header only
    }
}
