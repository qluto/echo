import { useEffect, useState } from "react";
import { formatClock } from "../lib/format";
import { getMeetingStatus, onMeetingState, openMeetingWindow, MeetingStatus } from "../lib/tauri";

const STATUS_POLL_MS = 1000;

/** Main-window entry to the meeting companion; the meeting itself runs in its own window. */
export function MeetingCard() {
  const [status, setStatus] = useState<MeetingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const active = status?.phase === "running" || status?.phase === "finalizing";

  // Pick up the state on mount and whenever the phase changes
  useEffect(() => {
    const load = () =>
      getMeetingStatus()
        .then(setStatus)
        .catch((e) => console.error("Failed to get meeting status:", e));
    load();
    const unlisten = onMeetingState(load);
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  // Slow poll for the elapsed time while a meeting is running
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => {
      getMeetingStatus()
        .then(setStatus)
        .catch(() => {});
    }, STATUS_POLL_MS);
    return () => clearInterval(timer);
  }, [active]);

  const handleOpen = async () => {
    setError(null);
    try {
      await openMeetingWindow();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div
        className="flex items-center gap-3 rounded-xl bg-surface px-3 py-2.5 border shadow-sm"
        style={{ borderColor: active ? "var(--glow-recording)" : "#E8E4DF" }}
      >
        <div className="flex flex-col min-w-0 flex-1">
          <span className="text-[13px] font-semibold leading-tight truncate" style={{ color: "var(--text-primary)" }}>
            Meeting
          </span>
          {active ? (
            <span className="font-mono text-[11px] leading-tight mt-0.5" style={{ color: "var(--glow-recording)" }}>
              {formatClock(status.elapsed_seconds)}
              <span className="font-body ml-1.5">{status.phase === "finalizing" ? "まとめ中" : "記録中"}</span>
            </span>
          ) : (
            <span
              className="text-[10px] leading-tight truncate"
              style={{ color: "var(--text-tertiary)" }}
              title="Transcribe both sides of a call and keep decisions, actions and open questions up to date"
            >
              Live notes for calls
            </span>
          )}
        </div>

        <button
          onClick={handleOpen}
          className="px-3 h-7 rounded-full text-[11px] font-medium border shrink-0 transition-colors hover:bg-surface-elevated"
          style={{ borderColor: "#E0DDD8", color: "var(--text-secondary)" }}
        >
          開く
        </button>
      </div>

      {error && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
