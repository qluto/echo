import { useState, useEffect, useCallback } from "react";
import { listMemos, deleteMemo, onMemoProgress, MemoListItem } from "../lib/tauri";

/** Stored meeting recordings (and voice memos from earlier versions). */
export function useMemos() {
  const [memos, setMemos] = useState<MemoListItem[]>([]);
  /** Stage progress (0–1) per memo id, while known. */
  const [progress, setProgress] = useState<Record<number, number>>({});
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setMemos(await listMemos());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Follow background processing
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    onMemoProgress((event) => {
      if (event.progress != null) {
        const value = event.progress;
        setProgress((prev) => ({ ...prev, [event.id]: value }));
        setMemos((prev) =>
          prev.map((m) => (m.id === event.id ? { ...m, status: event.status } : m))
        );
      } else {
        // Stage change: drop stale progress and pick up the new preview/error
        setProgress((prev) => {
          const { [event.id]: _, ...rest } = prev;
          return rest;
        });
        refresh();
      }
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, [refresh]);

  const remove = useCallback(
    async (id: number) => {
      try {
        await deleteMemo(id);
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [refresh]
  );

  return { memos, progress, error, remove, refresh };
}
