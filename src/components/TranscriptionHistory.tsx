import { useEffect, useRef, useCallback, useState } from "react";
import { useTranscriptionHistory } from "../hooks/useTranscriptionHistory";
import { useContinuousListening } from "../hooks/useContinuousListening";

const TIME_OPTIONS = [
  { label: "5 min", minutes: 5 },
  { label: "15 min", minutes: 15 },
  { label: "30 min", minutes: 30 },
  { label: "1 hour", minutes: 60 },
  { label: "3 hours", minutes: 180 },
];

interface TranscriptionHistoryProps {
  onSummarize?: (minutes: number) => void;
  isSummarizing?: boolean;
}

export function TranscriptionHistory({ onSummarize, isSummarizing }: TranscriptionHistoryProps) {
  const {
    entries,
    totalCount,
    hasMore,
    isLoading,
    searchQuery,
    error,
    loadHistory,
    loadMore,
    search,
    clearSearch,
    remove,
    clearAll,
    refresh,
  } = useTranscriptionHistory();

  const { recentEntries } = useContinuousListening();

  const scrollRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [showTimeMenu, setShowTimeMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Load initial history
  useEffect(() => {
    loadHistory();
  }, []);

  // Refresh when new transcriptions arrive
  useEffect(() => {
    if (recentEntries.length > 0) {
      refresh();
    }
  }, [recentEntries.length]);

  // Close menu on outside click
  useEffect(() => {
    if (!showTimeMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowTimeMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showTimeMenu]);

  // Debounced search
  const handleSearchInput = useCallback(
    (value: string) => {
      if (searchTimerRef.current) {
        clearTimeout(searchTimerRef.current);
      }
      searchTimerRef.current = setTimeout(() => {
        search(value);
      }, 300);
    },
    [search]
  );

  // Infinite scroll
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el || isLoading || !hasMore) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    if (scrollHeight - scrollTop - clientHeight < 100) {
      loadMore();
    }
  }, [isLoading, hasMore, loadMore]);

  const formatTime = (dateStr: string) => {
    if (!dateStr) return "";
    try {
      // created_at is in "YYYY-MM-DD HH:MM:SS" format (localtime)
      const parts = dateStr.split(" ");
      if (parts.length >= 2) {
        return parts[1].slice(0, 5); // HH:MM
      }
      return dateStr;
    } catch {
      return dateStr;
    }
  };

  const formatDuration = (seconds: number | null) => {
    if (seconds == null) return "";
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      {/* Header with search */}
      <div className="flex items-center justify-between gap-2">
        <span
          className="text-xs font-medium flex-shrink-0"
          style={{ color: "var(--text-tertiary)" }}
        >
          History
          {totalCount > 0 && (
            <span className="ml-1 opacity-60">({totalCount})</span>
          )}
        </span>

        {/* Search input */}
        <div className="relative flex-1 max-w-[180px]">
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search..."
            defaultValue={searchQuery}
            onChange={(e) => handleSearchInput(e.target.value)}
            className="w-full h-6 pl-6 pr-2 text-xs rounded-md bg-surface border border-subtle outline-none focus:border-[var(--glow-idle)] transition-colors"
            style={{ color: "var(--text-primary)" }}
          />
          <svg
            className="absolute left-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5"
            fill="none"
            stroke="var(--text-tertiary)"
            strokeWidth={1.5}
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
            />
          </svg>
          {searchQuery && (
            <button
              onClick={() => {
                if (searchInputRef.current) {
                  searchInputRef.current.value = "";
                }
                clearSearch();
              }}
              className="absolute right-1.5 top-1/2 -translate-y-1/2"
            >
              <svg
                className="w-3 h-3"
                fill="none"
                stroke="var(--text-tertiary)"
                strokeWidth={2}
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {totalCount > 0 && onSummarize && (
          <div className="relative" ref={menuRef}>
            <button
              onClick={() => {
                if (!isSummarizing) setShowTimeMenu((v) => !v);
              }}
              disabled={isSummarizing}
              className="text-xs flex-shrink-0 px-2 py-0.5 rounded-md transition-colors border border-subtle hover:bg-surface-elevated disabled:opacity-50"
              style={{ color: "var(--text-secondary)" }}
              title="Summarize recent transcriptions"
            >
              {isSummarizing ? (
                <span className="flex items-center gap-1">
                  <span
                    className="inline-block w-2.5 h-2.5 border border-current rounded-full animate-spin"
                    style={{ borderTopColor: "transparent" }}
                  />
                  Summarizing
                </span>
              ) : (
                "Summarize"
              )}
            </button>

            {showTimeMenu && (
              <div
                className="absolute right-0 top-full mt-1 py-1 rounded-lg border border-subtle shadow-lg z-10 min-w-[100px]"
                style={{ backgroundColor: "var(--surface)" }}
              >
                {TIME_OPTIONS.map((opt) => (
                  <button
                    key={opt.minutes}
                    onClick={() => {
                      setShowTimeMenu(false);
                      onSummarize(opt.minutes);
                    }}
                    className="w-full px-3 py-1.5 text-xs text-left hover:bg-surface-elevated transition-colors"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {totalCount > 0 && (
          <button
            onClick={clearAll}
            className="text-xs flex-shrink-0 opacity-50 hover:opacity-100 transition-opacity"
            style={{ color: "var(--text-tertiary)" }}
            title="Clear all history"
          >
            Clear
          </button>
        )}
      </div>

      {/* Entry list */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto rounded-xl bg-surface border border-subtle min-h-0"
      >
        {error && (
          <div className="p-3 text-xs" style={{ color: "var(--glow-recording)" }}>
            {error}
          </div>
        )}

        {entries.length === 0 && !isLoading && !error && (
          <div className="flex items-center justify-center h-full p-4">
            <p className="text-xs text-center" style={{ color: "var(--text-tertiary)" }}>
              {searchQuery ? "No results found" : "No transcriptions yet"}
            </p>
          </div>
        )}

        {entries.map((entry) => (
          <div
            key={entry.id}
            className="group px-3 py-2.5 border-b border-subtle last:border-b-0 hover:bg-surface-elevated transition-colors"
          >
            <div className="flex items-start gap-2">
              <div className="flex-1 min-w-0">
                {/* Metadata */}
                <div className="flex items-center gap-2 mb-0.5">
                  <span
                    className="font-mono text-[10px]"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {formatTime(entry.created_at)}
                  </span>
                  {entry.duration_seconds != null && (
                    <span
                      className="font-mono text-[10px]"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      {formatDuration(entry.duration_seconds)}
                    </span>
                  )}
                </div>

                {/* Text preview */}
                <p
                  className="text-xs leading-relaxed line-clamp-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  {entry.text}
                </p>
              </div>

              {/* Delete button */}
              <button
                onClick={() => remove(entry.id)}
                className="opacity-0 group-hover:opacity-50 hover:!opacity-100 transition-opacity flex-shrink-0 mt-0.5"
                title="Delete"
              >
                <svg
                  className="w-3.5 h-3.5"
                  fill="none"
                  stroke="var(--text-tertiary)"
                  strokeWidth={1.5}
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center justify-center p-3">
            <div
              className="w-3 h-3 border-2 rounded-full animate-spin"
              style={{
                borderColor: "var(--border-subtle)",
                borderTopColor: "var(--glow-idle)",
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
