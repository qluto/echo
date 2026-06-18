//! Minimal HuggingFace Hub fetch for small (non-LFS) files.
//!
//! `hf-hub` 0.3.2 follows redirects by re-parsing the `Location` header as an
//! *absolute* URL. HF now serves small non-LFS files (config.json,
//! tokenizer.json, …) through a `resolve-cache` endpoint that answers with a
//! **307 and a relative `Location`** (e.g. `/api/resolve-cache/models/...`),
//! so hf-hub 0.3.2 dies with `RelativeUrlWithoutBase: relative URL without a
//! base`. Large LFS weights still get an absolute 302 to the CDN, so those keep
//! working through hf-hub.
//!
//! This helper fetches small files itself, following redirects while resolving
//! relative `Location` headers against the current URL (and only sending the
//! auth token to huggingface.co, never to a CDN). Files are cached under
//! `<hub_cache_dir>/echo-direct/<repo>/<file>`.

use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};

const HF_HOST: &str = "https://huggingface.co/";
const MAX_REDIRECTS: usize = 10;

/// Download `file` from `repo` (main revision) into the cache and return its
/// path. Returns the cached copy if already present. `token` (when set) is sent
/// as a bearer token to huggingface.co for gated repos. Use only for small
/// non-LFS files — large LFS weights should still go through hf-hub.
pub fn fetch_small_file(
    hub_cache_dir: &Path,
    repo: &str,
    file: &str,
    token: Option<&str>,
) -> Result<PathBuf> {
    let dest_dir = hub_cache_dir
        .join("echo-direct")
        .join(repo.replace('/', "--"));
    let dest = dest_dir.join(file);
    if fs::metadata(&dest).map(|m| m.is_file() && m.len() > 0).unwrap_or(false) {
        return Ok(dest);
    }
    fs::create_dir_all(&dest_dir).map_err(|e| anyhow!("create cache dir: {e}"))?;

    let agent = ureq::AgentBuilder::new().redirects(0).build();
    let mut url = format!("{HF_HOST}{repo}/resolve/main/{file}");

    let resp = loop_follow(&agent, &mut url, token)?;
    let mut reader = resp.into_reader();
    let tmp = dest_dir.join(format!("{file}.download"));
    let mut out = fs::File::create(&tmp).map_err(|e| anyhow!("create temp file: {e}"))?;
    std::io::copy(&mut reader, &mut out).map_err(|e| anyhow!("write {file}: {e}"))?;
    out.sync_all().ok();
    drop(out);
    fs::rename(&tmp, &dest).map_err(|e| anyhow!("finalize {file}: {e}"))?;
    Ok(dest)
}

/// Issue GETs, manually following redirects (resolving relative `Location`s),
/// until a non-redirect response is returned. Bearer token is only attached
/// while the URL points at huggingface.co.
fn loop_follow(agent: &ureq::Agent, url: &mut String, token: Option<&str>) -> Result<ureq::Response> {
    for _ in 0..MAX_REDIRECTS {
        let mut req = agent.get(url);
        if let Some(tok) = token.filter(|t| !t.is_empty()) {
            if url.starts_with(HF_HOST) {
                req = req.set("Authorization", &format!("Bearer {tok}"));
            }
        }
        let resp = match req.call() {
            Ok(r) => r,
            // ureq surfaces 4xx/5xx as Err(Status); keep the original message.
            Err(ureq::Error::Status(code, r)) => {
                return Err(anyhow!(
                    "HTTP {code} for {url}: {}",
                    r.into_string().unwrap_or_default()
                ));
            }
            Err(e) => return Err(anyhow!("request {url}: {e}")),
        };
        let status = resp.status();
        if (300..400).contains(&status) {
            let loc = resp
                .header("location")
                .ok_or_else(|| anyhow!("redirect {status} without Location for {url}"))?;
            *url = join_url(url, loc)?;
            continue;
        }
        return Ok(resp);
    }
    Err(anyhow!("too many redirects fetching {url}"))
}

/// Resolve a (possibly relative) `Location` against the request URL. HF's
/// resolve-cache redirect is an absolute path (`/api/...`) on the same host.
fn join_url(base: &str, loc: &str) -> Result<String> {
    if loc.starts_with("http://") || loc.starts_with("https://") {
        Ok(loc.to_string())
    } else if let Some(rest) = loc.strip_prefix('/') {
        let scheme_end = base.find("://").ok_or_else(|| anyhow!("bad base URL: {base}"))? + 3;
        let host_end = base[scheme_end..]
            .find('/')
            .map(|i| scheme_end + i)
            .unwrap_or(base.len());
        Ok(format!("{}/{}", &base[..host_end], rest))
    } else {
        Err(anyhow!("unexpected relative Location '{loc}' for {base}"))
    }
}
