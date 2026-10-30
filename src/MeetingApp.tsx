import { useEffect, useRef, useState } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { useMeeting, isMeetingActive } from "./hooks/useMeeting";
import { formatClock } from "./lib/format";
import { MeetingItem, MeetingItemChange, MeetingItemKind, MeetingSource, MeetingStatus } from "./lib/tauri";

/** Distance from the bottom (px) within which the transcript keeps following new lines. */
const STICK_THRESHOLD = 40;

const SECTIONS: { kind: MeetingItemKind; title: string; empty: string }[] = [
  { kind: "topic", title: "主に話されたこと", empty: "まだ話題はありません" },
  { kind: "decision", title: "決定事項", empty: "まだ決まったことはありません" },
  { kind: "action", title: "ネクストアクション", empty: "まだアクションはありません" },
  { kind: "open", title: "残論点", empty: "未解決の論点はありません" },
];

const SPEAKERS: Record<MeetingSource, { label: string; color: string; soft: string }> = {
  me: { label: "自分", color: "var(--glow-idle)", soft: "var(--glow-idle-soft)" },
  others: { label: "相手", color: "#7A8CA8", soft: "rgba(122, 140, 168, 0.22)" },
};

const CHANGE_STYLES: Record<MeetingItemChange, { label: string; color: string; tint: string }> = {
  added: { label: "新規", color: "var(--glow-success)", tint: "rgba(124, 144, 112, 0.12)" },
  updated: { label: "更新", color: "#B5854F", tint: "rgba(212, 165, 116, 0.16)" },
  deleted: { label: "削除", color: "var(--text-tertiary)", tint: "transparent" },
};

/** The lists as Markdown; items deleted in the latest round are left out. */
const itemsToMarkdown = (items: MeetingItem[]) =>
  SECTIONS.map(({ kind, title }) => {
    const live = items.filter((i) => i.kind === kind && i.change !== "deleted");
    const body = live.length > 0 ? live.map((i) => `- ${i.text}`).join("\n") : "なし";
    return `## ${title}\n${body}`;
  }).join("\n\n");

function LevelMeter({ label, level, color }: { label: string; level: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5" title={`${label}の入力レベル`}>
      <span className="text-[10px]" style={{ color: "var(--text-tertiary)" }}>
        {label}
      </span>
      <div className="w-14 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: "var(--border-subtle)" }}>
        <div
          className="h-full rounded-full transition-all duration-100"
          style={{ width: `${Math.min(1, Math.max(0, level)) * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function statusText(phase: string | undefined, status: MeetingStatus | null) {
  switch (phase) {
    case "running":
      if (status?.updating) return "議事メモを更新中…";
      if (status?.next_update_in != null) return `次の更新まで ${formatClock(status.next_update_in)}`;
      return "記録中";
    case "finalizing":
      return "まとめ中…";
    case "done":
      return "完了";
    case "error":
      return "エラーで停止しました";
    default:
      return "待機中";
  }
}

function ItemRow({ item }: { item: MeetingItem }) {
  const style = item.change ? CHANGE_STYLES[item.change] : null;
  const deleted = item.change === "deleted";
  return (
    <li
      className="flex items-start gap-2 px-2.5 py-1.5 rounded-lg transition-colors duration-500"
      style={{ backgroundColor: style?.tint ?? "transparent", opacity: deleted ? 0.55 : 1 }}
    >
      <span className="mt-[7px] w-1 h-1 rounded-full shrink-0" style={{ backgroundColor: "var(--text-tertiary)" }} />
      <span
        className="text-xs leading-relaxed flex-1 min-w-0"
        style={{ color: "var(--text-primary)", textDecoration: deleted ? "line-through" : undefined }}
      >
        {item.text}
      </span>
      {style && (
        <span
          className="text-[9px] font-semibold px-1.5 py-[1px] rounded-full border shrink-0 mt-[2px]"
          style={{ color: style.color, borderColor: style.color }}
        >
          {style.label}
        </span>
      )}
    </li>
  );
}

export default function MeetingApp() {
  const { meeting, lines, items, round, status, isBusy, error, start, stop } = useMeeting();
  const [copied, setCopied] = useState(false);

  const phase = meeting?.phase;
  const active = isMeetingActive(phase);
  const running = phase === "running";

  // Follow new transcript lines unless the user has scrolled up
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_THRESHOLD;
    if (stickRef.current) setShowJump(false);
  };
  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stickRef.current = true;
    setShowJump(false);
  };
  useEffect(() => {
    if (stickRef.current) scrollToBottom();
    else setShowJump(true);
  }, [lines.length]);

  const handleCopy = async () => {
    try {
      await writeText(itemsToMarkdown(items));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  };

  const banner = error ?? meeting?.error ?? null;
  const actionButton =
    "px-2.5 py-1 text-[11px] rounded-lg border border-subtle hover:bg-surface-elevated transition-colors disabled:opacity-40";

  return (
    <div className="h-screen flex flex-col bg-surface-muted select-none">
      {/* Top bar */}
      <div className="flex items-center gap-4 px-5 py-3 bg-surface border-b border-subtle shrink-0">
        <button
          onClick={running ? stop : start}
          disabled={isBusy || phase === "finalizing"}
          title={running ? "会議を終了してまとめる" : "会議の記録を開始"}
          className="flex items-center gap-2 pl-2.5 pr-3.5 h-8 rounded-full border shrink-0 transition-colors hover:bg-surface-elevated disabled:opacity-50"
          style={{ borderColor: running ? "var(--glow-recording)" : "#E0DDD8" }}
        >
          <div
            className={`transition-all duration-150 ${running ? "w-2.5 h-2.5 rounded-[2px]" : "w-3 h-3 rounded-full"}`}
            style={{ backgroundColor: "var(--glow-recording)" }}
          />
          <span className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
            {running ? "終了" : "会議を開始"}
          </span>
        </button>

        <span
          className="font-mono text-sm tabular-nums"
          style={{ color: running ? "var(--glow-recording)" : "var(--text-secondary)" }}
        >
          {formatClock(status?.elapsed_seconds ?? meeting?.duration_seconds ?? 0)}
        </span>

        {active && (
          <div className="flex items-center gap-3">
            <LevelMeter label="自分" level={status?.mic_level ?? 0} color={SPEAKERS.me.color} />
            <LevelMeter label="相手" level={status?.system_level ?? 0} color={SPEAKERS.others.color} />
          </div>
        )}

        <div className="ml-auto flex items-center gap-2 min-w-0">
          {(status?.updating || phase === "finalizing") && active && (
            <div
              className="w-3 h-3 border-[1.5px] rounded-full animate-spin shrink-0"
              style={{ borderColor: "var(--border-subtle)", borderTopColor: "var(--glow-processing)" }}
            />
          )}
          <span className="text-[11px] truncate" style={{ color: "var(--text-tertiary)" }}>
            {statusText(phase, status)}
          </span>
        </div>
      </div>

      {/* Banners */}
      {(banner || meeting?.warning) && (
        <div className="flex flex-col gap-1.5 px-5 pt-3 shrink-0">
          {banner && (
            <div
              className="px-3 py-2 rounded-lg text-xs"
              style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
            >
              {banner}
            </div>
          )}
          {meeting?.warning && (
            <div
              className="px-3 py-2 rounded-lg text-xs"
              style={{ backgroundColor: "var(--glow-processing-soft)", color: "#8A6438" }}
            >
              {meeting.warning}
            </div>
          )}
        </div>
      )}

      {phase === "done" && (
        <div
          className="mx-5 mt-3 px-3 py-2 rounded-lg text-xs flex items-center gap-3 shrink-0"
          style={{ backgroundColor: "var(--glow-success-soft)", color: "var(--text-primary)" }}
        >
          <span className="flex-1">ボイスメモに保存しました。メイン画面のメモ一覧から開けます。</span>
          <button onClick={handleCopy} className={`${actionButton} bg-surface`} style={{ color: "var(--text-secondary)" }}>
            {copied ? "コピーしました" : "Markdown でコピー"}
          </button>
        </div>
      )}

      {/* Body */}
      <div className="flex-1 min-h-0 flex gap-3 p-5 pt-3">
        {/* Live transcript */}
        <section className="relative flex-[1.15] min-w-0 flex flex-col rounded-xl bg-surface border border-subtle shadow-sm overflow-hidden">
          <div className="px-4 py-2 border-b border-subtle shrink-0">
            <h2 className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
              書き起こし
            </h2>
          </div>
          <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-4 py-2 select-text">
            {lines.length === 0 ? (
              <p className="text-xs leading-relaxed py-8 text-center" style={{ color: "var(--text-tertiary)" }}>
                {active
                  ? "話し始めると、ここに書き起こしが表示されます"
                  : "「会議を開始」で、マイク（自分）とシステム音声（相手）の書き起こしを始めます"}
              </p>
            ) : (
              lines.map((line) => {
                const speaker = SPEAKERS[line.source];
                return (
                  <div key={line.id} className="flex items-start gap-2 py-1">
                    <span className="font-mono text-[10px] mt-[3px] shrink-0" style={{ color: "var(--text-tertiary)" }}>
                      {formatClock(line.start)}
                    </span>
                    <span
                      className="text-[10px] font-semibold px-1.5 py-[1px] rounded-full shrink-0 mt-[1px]"
                      style={{ color: speaker.color, backgroundColor: speaker.soft }}
                    >
                      {speaker.label}
                    </span>
                    <span className="text-xs leading-relaxed min-w-0" style={{ color: "var(--text-primary)" }}>
                      {line.text}
                    </span>
                  </div>
                );
              })
            )}
          </div>
          {showJump && (
            <button
              onClick={scrollToBottom}
              className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 text-[11px] rounded-full bg-surface border border-subtle shadow-md hover:bg-surface-elevated transition-colors"
              style={{ color: "var(--text-secondary)" }}
            >
              最新へ ↓
            </button>
          )}
        </section>

        {/* Topics / decisions / actions / open questions */}
        <section className="flex-1 min-w-0 flex flex-col rounded-xl bg-surface border border-subtle shadow-sm overflow-hidden">
          <div className="flex items-center px-4 py-2 border-b border-subtle shrink-0">
            <h2 className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
              議事メモ
            </h2>
            {round > 0 && (
              <span className="text-[10px] ml-2" style={{ color: "var(--text-tertiary)" }}>
                {round} 回更新
              </span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto px-2 py-2 flex flex-col gap-3 select-text">
            {SECTIONS.map(({ kind, title, empty }) => {
              const sectionItems = items.filter((i) => i.kind === kind);
              const count = sectionItems.filter((i) => i.change !== "deleted").length;
              return (
                <div key={kind}>
                  <div className="flex items-baseline gap-1.5 px-2.5 pb-1">
                    <h3 className="text-[11px] font-semibold" style={{ color: "var(--text-secondary)" }}>
                      {title}
                    </h3>
                    <span className="font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                      {count}
                    </span>
                  </div>
                  {sectionItems.length === 0 ? (
                    <p className="text-[11px] px-2.5 py-1" style={{ color: "var(--text-tertiary)" }}>
                      {empty}
                    </p>
                  ) : (
                    <ul className="flex flex-col gap-0.5">
                      {sectionItems.map((item) => (
                        <ItemRow key={item.id} item={item} />
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
