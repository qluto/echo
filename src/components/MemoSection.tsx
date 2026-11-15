import { useState } from "react";
import { useMemos } from "../hooks/useMemos";
import { MemoModal, statusLabel } from "./MemoModal";
import { formatClock } from "../lib/format";
import { MemoListItem } from "../lib/tauri";

/** Rows shown before the list is expanded. */
const COLLAPSED_COUNT = 3;

/** "2026-10-02 14:30:05" → "10/02 14:30" */
const formatCreatedAt = (createdAt: string) => {
  const [date, time] = createdAt.split(" ");
  if (!date || !time) return createdAt;
  return `${date.slice(5).replace("-", "/")} ${time.slice(0, 5)}`;
};

/** The stored meetings (and voice memos from earlier versions). */
export function MemoSection() {
  const { memos, progress, error, remove, refresh } = useMemos();
  const [openId, setOpenId] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(false);

  // The meeting in progress is shown by the Meeting card.
  const finished = memos.filter((m) => m.status !== "recording");
  const visible = expanded ? finished : finished.slice(0, COLLAPSED_COUNT);

  const statusColor = (m: MemoListItem) =>
    m.status === "error" || m.status === "interrupted"
      ? "var(--glow-recording)"
      : "var(--glow-processing)";

  return (
    <div className="flex flex-col gap-2">
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
