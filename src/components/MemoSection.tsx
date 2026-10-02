import { useState } from "react";
import { useMemos } from "../hooks/useMemos";
import { MemoModal, statusLabel } from "./MemoModal";
import { formatClock } from "../lib/format";
import { MemoListItem } from "../lib/tauri";

/** Rows shown before the list is expanded. */
const COLLAPSED_COUNT = 3;
const LEVEL_BARS = 5;

/** "2026-10-02 14:30:05" → "10/02 14:30" */
const formatCreatedAt = (createdAt: string) => {
  const [date, time] = createdAt.split(" ");
  if (!date || !time) return createdAt;
  return `${date.slice(5).replace("-", "/")} ${time.slice(0, 5)}`;
};

export function MemoSection() {
  const { memos, recording, progress, isBusy, error, start, stop, remove, refresh } = useMemos();
  const [openId, setOpenId] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(false);

  const isRecording = recording !== null;
  // The memo being recorded is represented by the card itself.
  const finished = memos.filter((m) => m.id !== recording?.memo_id);
  const visible = expanded ? finished : finished.slice(0, COLLAPSED_COUNT);

  const statusColor = (m: MemoListItem) =>
    m.status === "error" || m.status === "interrupted"
      ? "var(--glow-recording)"
      : "var(--glow-processing)";

  return (
    <div className="flex flex-col gap-2">
      {/* Recorder card */}
      <div
        className="flex items-center gap-3 rounded-xl bg-surface px-3 py-2.5 border shadow-sm"
        style={{ borderColor: isRecording ? "var(--glow-recording)" : "#E8E4DF" }}
      >
        <div className="flex flex-col min-w-0 flex-1">
          <span className="text-[13px] font-semibold leading-tight truncate" style={{ color: "var(--text-primary)" }}>
            Voice Memo
          </span>
          {isRecording ? (
            <div className="flex items-center gap-2 mt-0.5">
              <span className="font-mono text-[11px] leading-tight" style={{ color: "var(--glow-recording)" }}>
                {formatClock(recording.elapsed_seconds)}
              </span>
              <div className="flex items-end gap-[2px] h-3" aria-hidden>
                {Array.from({ length: LEVEL_BARS }, (_, i) => (
                  <div
                    key={i}
                    className="w-[3px] rounded-full transition-all duration-100"
                    style={{
                      height: `${25 + (i + 1) * 15}%`,
                      backgroundColor:
                        recording.level * LEVEL_BARS > i ? "var(--glow-recording)" : "var(--border-subtle)",
                    }}
                  />
                ))}
              </div>
            </div>
          ) : (
            <span
              className="text-[10px] leading-tight truncate"
              style={{ color: "var(--text-tertiary)" }}
              title="Record a meeting, then get a transcript and minutes when you stop"
            >
              Record, then get minutes
            </span>
          )}
        </div>

        <button
          onClick={isRecording ? stop : start}
          disabled={isBusy}
          title={isRecording ? "録音を終了して書き起こす" : "録音を開始"}
          aria-label={isRecording ? "録音を終了して書き起こす" : "録音を開始"}
          className="w-8 h-8 rounded-full flex items-center justify-center border shrink-0 transition-colors hover:bg-surface-elevated disabled:opacity-50"
          style={{ borderColor: isRecording ? "var(--glow-recording)" : "#E0DDD8" }}
        >
          <div
            className={`transition-all duration-150 ${isRecording ? "w-3 h-3 rounded-[3px]" : "w-3.5 h-3.5 rounded-full"}`}
            style={{ backgroundColor: "var(--glow-recording)" }}
          />
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

      {/* Memo list */}
      {finished.length > 0 && (
        <div className="rounded-[10px] bg-surface border shadow-sm overflow-hidden" style={{ borderColor: "#E8E4DF" }}>
          <div className={expanded ? "max-h-52 overflow-y-auto" : ""}>
            {visible.map((m) => (
              <button
                key={m.id}
                onClick={() => setOpenId(m.id)}
                className="w-full text-left px-3.5 py-2 hover:bg-surface-elevated transition-colors"
                style={{ borderBottom: "0.5px solid #F0EFEC" }}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                    {formatCreatedAt(m.created_at)}
                  </span>
                  {m.duration_seconds != null && (
                    <span className="font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                      {formatClock(m.duration_seconds)}
                    </span>
                  )}
                  {m.status !== "done" && (
                    <span className="text-[10px] font-medium ml-auto" style={{ color: statusColor(m) }}>
                      {statusLabel(m.status, progress[m.id])}
                    </span>
                  )}
                </div>
                {m.preview && (
                  <p className="text-xs leading-relaxed truncate" style={{ color: "var(--text-secondary)" }}>
                    {m.preview.replace(/^#+\s*/gm, "").replace(/\s+/g, " ")}
                  </p>
                )}
              </button>
            ))}
          </div>
          {finished.length > COLLAPSED_COUNT && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="w-full py-1 text-[10px] hover:bg-surface-elevated transition-colors"
              style={{ color: "var(--text-tertiary)" }}
            >
              {expanded ? "閉じる" : `すべて表示 (${finished.length})`}
            </button>
          )}
        </div>
      )}

      {openId != null && (
        <MemoModal
          memoId={openId}
          progress={progress[openId]}
          onClose={() => setOpenId(null)}
          onDelete={async () => {
            await remove(openId);
            setOpenId(null);
          }}
          onChanged={refresh}
        />
      )}
    </div>
  );
}
