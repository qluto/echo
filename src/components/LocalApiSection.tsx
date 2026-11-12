import { useEffect, useState } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { getLocalApi, setLocalApi, LocalApiStatus } from "../lib/tauri";

/**
 * Local transcription API: other apps on this Mac can send audio to Echo's
 * loaded ASR model over HTTP (127.0.0.1 only, Bearer token). It shares the
 * engine with dictation, taking turns chunk by chunk.
 */
export function LocalApiSection() {
  const [status, setStatus] = useState<LocalApiStatus | null>(null);
  const [port, setPort] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    getLocalApi()
      .then((s) => {
        setStatus(s);
        setPort(String(s.port));
      })
      .catch((e) => console.error("Failed to get local API status:", e));
  }, []);

  const update = async (enabled: boolean, nextPort: number, regenerate = false) => {
    setError(null);
    try {
      const next = await setLocalApi(enabled, nextPort, regenerate);
      setStatus(next);
      setPort(String(next.port));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      if (status) setPort(String(status.port));
    }
  };

  const commitPort = () => {
    if (!status) return;
    const n = Number(port);
    if (!Number.isInteger(n) || n < 1024 || n > 65535) {
      setError("Port must be a number from 1024 to 65535");
      setPort(String(status.port));
      return;
    }
    if (n !== status.port) update(status.enabled, n);
  };

  const copy = async (label: string, text: string) => {
    await writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 1500);
  };

  if (!status) return null;
  const url = `http://127.0.0.1:${status.port}/v1/audio/transcriptions`;
  const curl = `curl ${url} \\\n  -H "Authorization: Bearer ${status.token}" \\\n  -F file=@audio.wav`;
  const problem = error ?? (status.enabled && !status.running ? status.error ?? "Not running" : null);

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center gap-2.5">
        <div
          className="w-6 h-6 rounded-md flex items-center justify-center"
          style={{ backgroundColor: "rgba(124, 154, 146, 0.12)" }}
        >
          <svg
            className="w-3.5 h-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--glow-idle)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M8 9l-4 3 4 3m8-6l4 3-4 3M14 5l-4 14" />
          </svg>
        </div>
        <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Local Transcription API
        </span>
      </div>

      <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Enable API (Advanced)
          </span>
          <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            {status.enabled && status.running ? `Listening on 127.0.0.1:${status.port}` : "Apps on this Mac only"}
          </span>
        </div>
        <button
          onClick={() => update(!status.enabled, status.port)}
          className="relative w-10 h-5 rounded-full transition-colors flex-shrink-0"
          style={{ backgroundColor: status.enabled ? "var(--glow-idle)" : "var(--border-subtle)" }}
          aria-label="Toggle the local transcription API"
        >
          <div
            className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform"
            style={{ transform: status.enabled ? "translateX(22px)" : "translateX(2px)" }}
          />
        </button>
      </div>

      {status.enabled && (
        <>
          <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
            <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
              Port
            </span>
            <input
              value={port}
              onChange={(e) => setPort(e.target.value.replace(/\D/g, ""))}
              onBlur={commitPort}
              onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
              inputMode="numeric"
              className="w-[100px] bg-transparent font-mono text-xs text-right focus:outline-none"
              style={{ color: "var(--text-primary)" }}
            />
          </div>

          <div className="px-4 py-3 rounded-xl bg-surface border border-subtle flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                API Token
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => copy("token", status.token)}
                  className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  {copied === "token" ? "Copied" : "Copy"}
                </button>
                <button
                  onClick={() => update(true, status.port, true)}
                  className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Regenerate
                </button>
              </div>
            </div>
            <span className="font-mono text-xs break-all" style={{ color: "var(--text-primary)" }}>
              {status.token}
            </span>
          </div>

          <div className="px-4 py-3 rounded-xl bg-surface border border-subtle flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Example
              </span>
              <button
                onClick={() => copy("curl", curl)}
                className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                style={{ color: "var(--text-tertiary)" }}
              >
                {copied === "curl" ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="font-mono text-xs whitespace-pre-wrap break-all" style={{ color: "var(--text-primary)" }}>
              {curl}
            </pre>
          </div>
        </>
      )}

      {problem && (
        <div
          className="px-4 py-2 rounded-lg text-xs"
          style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
        >
          {problem}
        </div>
      )}

      <div
        className="px-4 py-2 rounded-lg text-xs"
        style={{ backgroundColor: "rgba(124, 154, 146, 0.08)", color: "var(--text-tertiary)" }}
      >
        OpenAI-compatible, using the selected speech model: files to{" "}
        <span className="font-mono">POST /v1/audio/transcriptions</span> (WAV, m4a, mp3, aac, flac, caf), and live
        PCM over WebSocket to <span className="font-mono">/v1/realtime</span> with drafts as you speak. Requests take
        turns with dictation.
      </div>
    </section>
  );
}
