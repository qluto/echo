import { useState } from "react";

interface SummaryModalProps {
  summary: string;
  entryCount: number;
  processingTime: number | null;
  windowMinutes: number;
  onClose: () => void;
}

export function SummaryModal({
  summary,
  entryCount,
  processingTime,
  windowMinutes,
  onClose,
}: SummaryModalProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: ignore clipboard errors
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0"
        style={{ backgroundColor: "rgba(0, 0, 0, 0.3)", backdropFilter: "blur(4px)" }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className="relative w-full max-w-md max-h-[70vh] flex flex-col rounded-2xl border border-subtle shadow-2xl overflow-hidden"
        style={{ backgroundColor: "var(--surface)" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-subtle flex-shrink-0">
          <div>
            <h2
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Summary
              <span
                className="ml-2 font-normal text-xs"
                style={{ color: "var(--text-tertiary)" }}
              >
                (last {windowMinutes} min)
              </span>
            </h2>
            <p
              className="text-[10px] mt-0.5 font-mono"
              style={{ color: "var(--text-tertiary)" }}
            >
              {entryCount} entries
              {processingTime != null && ` / ${(processingTime / 1000).toFixed(1)}s`}
            </p>
          </div>

          <div className="flex items-center gap-1.5">
            {/* Copy button */}
            <button
              onClick={handleCopy}
              className="px-2.5 py-1 text-xs rounded-lg border border-subtle hover:bg-surface-elevated transition-colors"
              style={{ color: "var(--text-secondary)" }}
            >
              {copied ? "Copied!" : "Copy"}
            </button>

            {/* Close button */}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-surface-elevated transition-colors"
            >
              <svg
                className="w-3.5 h-3.5"
                fill="none"
                stroke="var(--text-tertiary)"
                strokeWidth={2}
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4 min-h-0">
          {summary ? (
            <div
              className="text-xs leading-relaxed whitespace-pre-wrap"
              style={{ color: "var(--text-primary)" }}
            >
              {summary}
            </div>
          ) : (
            <p
              className="text-xs text-center py-8"
              style={{ color: "var(--text-tertiary)" }}
            >
              No transcriptions in the last {windowMinutes} minutes to summarize.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
