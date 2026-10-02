import { useCallback, useEffect, useState } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import {
  getMemo,
  onMemoProgress,
  reprocessMemo,
  revealMemoAudio,
  Memo,
  MemoStatus,
} from "../lib/tauri";
import { formatClock } from "../lib/format";

type Tab = "minutes" | "transcript";

const STATUS_LABELS: Record<MemoStatus, string> = {
  recording: "録音中",
  transcribing: "書き起こし中",
  summarizing: "議事録を作成中",
  done: "",
  error: "エラー",
  interrupted: "中断",
};

export function statusLabel(status: MemoStatus, progress?: number) {
  const label = STATUS_LABELS[status];
  return progress != null && status !== "done" ? `${label} ${Math.round(progress * 100)}%` : label;
}

interface MemoModalProps {
  memoId: number;
  /** Stage progress (0–1) while the memo is being processed. */
  progress?: number;
  onClose: () => void;
  onDelete: () => void;
  /** Called after an action that changes the memo's list row. */
  onChanged: () => void;
}

/** Minutes are Markdown; headings and bold are the only syntax the model uses. */
function MinutesText({ text }: { text: string }) {
  return (
    <div className="text-xs leading-relaxed" style={{ color: "var(--text-primary)" }}>
      {text.split("\n").map((line, i) => {
        const heading = line.match(/^#{1,6}\s+(.*)$/);
        return heading ? (
          <p key={i} className="font-semibold text-[13px] mt-3 first:mt-0 mb-1">
            {heading[1]}
          </p>
        ) : (
          <p key={i} className="whitespace-pre-wrap min-h-[0.5rem]">
            {line.split(/\*\*(.+?)\*\*/g).map((part, j) =>
              j % 2 === 1 ? <strong key={j}>{part}</strong> : part
            )}
          </p>
        );
      })}
    </div>
  );
}

export function MemoModal({ memoId, progress, onClose, onDelete, onChanged }: MemoModalProps) {
  const [memo, setMemo] = useState<Memo | null>(null);
  const [tab, setTab] = useState<Tab>("minutes");
  const [copied, setCopied] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setMemo(await getMemo(memoId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [memoId]);

  useEffect(() => {
    load();
  }, [load]);

  // Reload on stage changes of this memo (transcript ready, minutes ready, …)
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    onMemoProgress((event) => {
      if (event.id === memoId && event.progress == null) load();
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, [memoId, load]);

  const reprocess = async (retranscribe: boolean) => {
    setError(null);
    try {
      await reprocessMemo(memoId, retranscribe);
      await load();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const isProcessing = memo?.status === "transcribing" || memo?.status === "summarizing";
  const body = tab === "minutes" ? memo?.minutes : memo?.transcript;

  const handleCopy = async () => {
    if (!body) return;
    try {
      await writeText(body);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  };

  const actionButton = "px-2.5 py-1 text-[11px] rounded-lg border border-subtle hover:bg-surface-elevated transition-colors disabled:opacity-40";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0"
        style={{ backgroundColor: "rgba(0, 0, 0, 0.3)", backdropFilter: "blur(4px)" }}
        onClick={onClose}
      />

      <div
        className="relative w-full max-w-md h-[80vh] flex flex-col rounded-2xl border border-subtle shadow-2xl overflow-hidden"
        style={{ backgroundColor: "var(--surface)" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-subtle flex-shrink-0">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Voice Memo
            </h2>
            <p className="text-[10px] mt-0.5 font-mono truncate" style={{ color: "var(--text-tertiary)" }}>
              {memo?.created_at.slice(0, 16)}
              {memo?.duration_seconds != null && ` / ${formatClock(memo.duration_seconds)}`}
            </p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-surface-elevated transition-colors" aria-label="閉じる">
            <svg className="w-3.5 h-3.5" fill="none" stroke="var(--text-tertiary)" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 px-5 pt-2.5 flex-shrink-0">
          {(["minutes", "transcript"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-2.5 py-1 text-xs rounded-lg transition-colors"
              style={{
                color: tab === t ? "var(--text-primary)" : "var(--text-tertiary)",
                backgroundColor: tab === t ? "var(--surface-muted)" : "transparent",
                fontWeight: tab === t ? 600 : 400,
              }}
            >
              {t === "minutes" ? "議事録" : "書き起こし"}
            </button>
          ))}
          <button
            onClick={handleCopy}
            disabled={!body}
            className={`${actionButton} ml-auto`}
            style={{ color: "var(--text-secondary)" }}
          >
            {copied ? "コピーしました" : "コピー"}
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-3 min-h-0 select-text">
          {(error || memo?.error) && (
            <div
              className="px-3 py-2 mb-3 rounded-lg text-xs"
              style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
            >
              {error || memo?.error}
            </div>
          )}

          {body ? (
            tab === "minutes" ? (
              <MinutesText text={body} />
            ) : (
              <div className="text-xs leading-relaxed whitespace-pre-wrap" style={{ color: "var(--text-primary)" }}>
                {body}
              </div>
            )
          ) : memo && isProcessing ? (
            <div className="flex flex-col items-center gap-3 py-10">
              <div
                className="w-4 h-4 border-2 rounded-full animate-spin"
                style={{ borderColor: "var(--border-subtle)", borderTopColor: "var(--glow-processing)" }}
              />
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                {statusLabel(memo.status, progress)}
              </span>
            </div>
          ) : memo ? (
            <p className="text-xs text-center py-10" style={{ color: "var(--text-tertiary)" }}>
              {memo.status === "interrupted"
                ? "録音または処理が中断されました。「音声から再処理」で書き起こせます。"
                : memo.status === "done" && !memo.transcript
                ? "音声が検出されませんでした。"
                : tab === "minutes"
                ? "議事録はまだありません。"
                : "書き起こしはまだありません。"}
            </p>
          ) : null}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1.5 px-5 py-2.5 border-t border-subtle flex-shrink-0 flex-wrap">
          <button
            onClick={() => reprocess(false)}
            disabled={!memo?.transcript || isProcessing}
            className={actionButton}
            style={{ color: "var(--text-secondary)" }}
            title="書き起こしはそのままに、議事録だけ作り直します"
          >
            議事録を再生成
          </button>
          <button
            onClick={() => reprocess(true)}
            disabled={!memo || isProcessing}
            className={actionButton}
            style={{ color: "var(--text-secondary)" }}
            title="現在の認識モデルで音声から書き起こし直します"
          >
            音声から再処理
          </button>
          <button
            onClick={() => revealMemoAudio(memoId).catch((e) => setError(String(e)))}
            className={actionButton}
            style={{ color: "var(--text-secondary)" }}
          >
            音声ファイル
          </button>
          <button
            onClick={() => (confirmingDelete ? onDelete() : setConfirmingDelete(true))}
            onBlur={() => setConfirmingDelete(false)}
            disabled={isProcessing}
            className={`${actionButton} ml-auto`}
            style={{ color: "var(--glow-recording)" }}
          >
            {confirmingDelete ? "本当に削除" : "削除"}
          </button>
        </div>
      </div>
    </div>
  );
}
