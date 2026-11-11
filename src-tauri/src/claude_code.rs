//! Minutes and meeting notes through the user's local Claude Code CLI
//! (`claude -p`), an opt-in alternative to the in-process notes LLM.
//!
//! This sends transcripts off the device (to Anthropic, under the user's own
//! Claude Code login), so it is off by default. Each call is a one-shot,
//! tool-less completion: no tools, MCP servers, hooks, CLAUDE.md or settings
//! files are loaded, and the session is not saved. The transcript goes in on
//! stdin, never on the command line (where `ps` would show it), and is never
//! logged.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

/// Model aliases offered in Settings; Claude Code resolves each to the
/// latest model of that family.
pub const MODELS: [&str; 4] = ["sonnet", "opus", "haiku", "fable"];
pub const DEFAULT_MODEL: &str = "sonnet";

/// A long memo's minutes with a large model can take minutes; a stuck CLI
/// (e.g. waiting on a login) must not hold a meeting's notes forever.
const CALL_TIMEOUT: Duration = Duration::from_secs(600);
const VERSION_TIMEOUT: Duration = Duration::from_secs(15);

/// One `claude -p` configuration.
#[derive(Debug, Clone)]
pub struct ClaudeCode {
    pub model: String,
}

impl ClaudeCode {
    pub fn new(model: &str) -> Result<Self> {
        if !MODELS.contains(&model) {
            bail!("Unknown Claude Code model: {model}");
        }
        Ok(Self { model: model.to_string() })
    }

    /// One system + user completion. The answer's text, or an error with the
    /// CLI's message (not logged in, no access to the model, offline, …).
    pub fn chat(&self, system: &str, user: &str) -> Result<String> {
        let binary = find_binary()?;
        let t = Instant::now();
        let mut cmd = command(&binary);
        cmd.args(["-p", "--output-format", "json", "--model", &self.model])
            .args(["--system-prompt", system])
            // A plain completion: no tools, MCP, hooks, plugins, CLAUDE.md or
            // settings files, and nothing written to the session history.
            .args(["--tools", "", "--strict-mcp-config", "--safe-mode", "--setting-sources", ""])
            .args(["--no-session-persistence", "--disable-slash-commands"]);
        let out = run(cmd, user.as_bytes(), CALL_TIMEOUT)?;
        let text = parse_result(&out.stdout).map_err(|e| match out.stderr.trim() {
            "" => e,
            stderr => e.context(stderr.to_string()),
        })?;
        log::info!(
            "Claude Code ({}) answered in {:.1}s ({} chars in, {} out)",
            self.model,
            t.elapsed().as_secs_f64(),
            user.chars().count(),
            text.chars().count()
        );
        Ok(text)
    }
}

/// The result text of `claude -p --output-format json`.
fn parse_result(stdout: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(stdout.trim())
        .with_context(|| format!("Unexpected output from Claude Code: {}", truncate(stdout, 300)))?;
    let result = value.get("result").and_then(|r| r.as_str()).unwrap_or_default();
    if value.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false) {
        bail!("Claude Code failed: {}", if result.is_empty() { "unknown error" } else { result });
    }
    Ok(result.to_string())
}

/// Where the CLI is and its version, for Settings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CliInfo {
    pub path: String,
    pub version: String,
}

pub fn cli_info() -> Result<CliInfo> {
    let binary = find_binary()?;
    let mut cmd = command(&binary);
    cmd.arg("--version");
    let out = run(cmd, b"", VERSION_TIMEOUT)?;
    if !out.success {
        bail!("`claude --version` failed: {}", out.stderr.trim());
    }
    Ok(CliInfo { path: binary.display().to_string(), version: out.stdout.trim().to_string() })
}

/// `claude` with an environment that works from a GUI app: its own directory
/// on PATH (an npm-installed CLI needs the `node` next to it), not nested in a
/// Claude Code session (when Echo itself was started from one), and run from
/// the temp dir so no project's files or settings come into play.
fn command(binary: &Path) -> Command {
    let mut cmd = Command::new(binary);
    let path = std::env::var_os("PATH").unwrap_or_default();
    let dirs = binary.parent().map(Path::to_path_buf).into_iter().chain(std::env::split_paths(&path));
    if let Ok(path) = std::env::join_paths(dirs) {
        cmd.env("PATH", path);
    }
    cmd.env_remove("CLAUDECODE").current_dir(std::env::temp_dir());
    cmd
}

/// The `claude` executable. A Finder-launched app gets a minimal PATH, so
/// look in the installers' usual places, then ask the login shell. Found
/// once per run; a CLI installed later is picked up after a restart.
fn find_binary() -> Result<PathBuf> {
    static FOUND: OnceLock<Option<PathBuf>> = OnceLock::new();
    FOUND
        .get_or_init(|| {
            let found = locate();
            match &found {
                Some(p) => log::info!("Claude Code CLI: {}", p.display()),
                None => log::warn!("Claude Code CLI not found"),
            }
            found
        })
        .clone()
        .ok_or_else(|| anyhow!("Claude Code (`claude`) was not found. Install it and sign in with `claude` in a terminal, then restart Echo."))
}

fn locate() -> Option<PathBuf> {
    let home = dirs::home_dir();
    let candidates = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).map(|d| d.join("claude")).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .chain(home.iter().flat_map(|h| [h.join(".local/bin/claude"), h.join(".claude/local/claude")]))
        .chain(["/opt/homebrew/bin/claude", "/usr/local/bin/claude"].map(PathBuf::from));
    for candidate in candidates {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // nvm and similar set PATH up in the shell's rc files only.
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    let mut cmd = Command::new(shell);
    cmd.args(["-ilc", "command -v claude"]);
    let out = run(cmd, b"", Duration::from_secs(10)).ok()?;
    let path = PathBuf::from(out.stdout.lines().map(str::trim).rfind(|l| l.starts_with('/'))?);
    path.is_file().then_some(path)
}

struct Output {
    success: bool,
    stdout: String,
    stderr: String,
}

/// Run `cmd` with `input` on stdin; killed after `timeout`.
fn run(mut cmd: Command, input: &[u8], timeout: Duration) -> Result<Output> {
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to start {:?}", cmd.get_program()))?;
    // Feed stdin and drain both pipes on their own threads, so neither side
    // blocks on a full pipe.
    let mut stdin = child.stdin.take().expect("piped stdin");
    let input = input.to_vec();
    let writer = thread::spawn(move || stdin.write_all(&input));
    let reader = |mut pipe: Box<dyn Read + Send>| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        })
    };
    let stdout = reader(Box::new(child.stdout.take().expect("piped stdout")));
    let stderr = reader(Box::new(child.stderr.take().expect("piped stderr")));

    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("Claude Code did not answer within {} s", timeout.as_secs());
        }
        thread::sleep(Duration::from_millis(100));
    };
    let _ = writer.join();
    Ok(Output {
        success: status.success(),
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_success_and_error_results() {
        let ok = r#"{"type":"result","subtype":"success","is_error":false,"result":"- 来週リリース"}"#;
        assert_eq!(parse_result(ok).unwrap(), "- 来週リリース");
        let err = r#"{"type":"result","is_error":true,"result":"There's an issue with the selected model"}"#;
        assert!(parse_result(err).unwrap_err().to_string().contains("selected model"));
        assert!(parse_result("Not logged in").is_err());
    }

    #[test]
    fn rejects_unknown_models() {
        assert!(ClaudeCode::new("sonnet").is_ok());
        assert!(ClaudeCode::new("gpt-5").is_err());
    }

    /// Calls the real CLI (needs it installed and logged in):
    /// `cargo test claude_code_round_trip -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn claude_code_round_trip() {
        println!("{:?}", cli_info().unwrap());
        let answer = ClaudeCode::new("haiku").unwrap().chat("Reply with the word OK only.", "ping").unwrap();
        assert!(answer.contains("OK"), "{answer}");
    }
}
