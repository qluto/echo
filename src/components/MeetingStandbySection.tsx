import { useEffect, useState } from "react";
import {
  getMeetingStandby,
  onMeetingStandby,
  setMeetingStandbySeconds,
  MeetingStandbyStatus,
} from "../lib/tauri";
import { formatMinutesJa } from "../lib/format";

/** 1–4 min in 30 s steps (the backend takes any 10 s step in 60–240 s). */
const CHOICES = [60, 90, 120, 150, 180, 210, 240];
const DEFAULT_SECONDS = 180;

/**
 * Automatic meeting detection (Advanced): how long a conversation must go on
 * before meeting standby starts a meeting. The other conditions (share of
 * talk, the user taking part) scale with it.
 */
export function MeetingStandbySection() {
  const [status, setStatus] = useState<MeetingStandbyStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMeetingStandby()
      .then(setStatus)
      .catch((e) => console.error("Failed to get meeting standby:", e));
    const unlisten = onMeetingStandby(setStatus);
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  if (!status) return null;
  const seconds = status.detect_seconds;
  const choices = CHOICES.includes(seconds) ? CHOICES : [...CHOICES, seconds].sort((a, b) => a - b);

  const change = async (next: number) => {
    setError(null);
    try {
      setStatus(await setMeetingStandbySeconds(next));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

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
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7v5l3 2" />
          </svg>
        </div>
        <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Meeting Detection
        </span>
      </div>

      <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
        <div className="flex flex-col min-w-0">
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Detect after (Advanced)
          </span>
          <span className="text-xs truncate" style={{ color: "var(--text-tertiary)" }}>
            {formatMinutesJa(seconds)}続く会話で会議を開始
            {seconds !== DEFAULT_SECONDS && ` · 既定 ${formatMinutesJa(DEFAULT_SECONDS)}`}
          </span>
        </div>
        <select
          value={seconds}
          onChange={(e) => change(Number(e.target.value))}
          className="h-8 px-2 rounded-lg bg-surface border border-subtle text-xs focus:outline-none"
          style={{ color: "var(--text-primary)" }}
          aria-label="Conversation length before a meeting starts"
        >
          {choices.map((s) => (
            <option key={s} value={s}>
              {formatMinutesJa(s)}
            </option>
          ))}
        </select>
      </div>

      <span className="text-xs px-1" style={{ color: "var(--text-tertiary)" }}>
        短くすると早く記録が始まりますが、短い立ち話や通話でも会議として記録されやすくなります。
        会話の割合や自分の発話量の条件も、この長さに比例します。待機中でもすぐ反映されます。
      </span>

      {error && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
        >
          {error}
        </div>
      )}
    </section>
  );
}
