import { useState, useEffect, useCallback } from "react";
import {
  startMemoRecording,
  stopMemoRecording,
  getMemoRecordingStatus,
  listMemos,
  deleteMemo,
  onMemoProgress,
  MemoListItem,
  MemoRecordingStatus,
} from "../lib/tauri";

const RECORDING_POLL_MS = 150;

export function useMemos() {
  const [memos, setMemos] = useState<MemoListItem[]>([]);
  /** Live status of the in-progress recording; null when not recording. */
  const [recording, setRecording] = useState<MemoRecordingStatus | null>(null);
  /** Stage progress (0–1) per memo id, while known. */
  const [progress, setProgress] = useState<Record<number, number>>({});
  /** True while a start/stop request is in flight. */
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setMemos(await listMemos());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Initial state (a recording may already be running if the window reloaded)
  useEffect(() => {
    refresh();
    getMemoRecordingStatus()
      .then(setRecording)
      .catch((e) => console.error("Failed to get memo recording status:", e));
  }, [refresh]);

  // Poll elapsed time + input level while recording
  const isRecording = recording !== null;
  useEffect(() => {
    if (!isRecording) return;
    const timer = setInterval(() => {
      getMemoRecordingStatus()
        .then(setRecording)
        .catch(() => {});
    }, RECORDING_POLL_MS);
    return () => clearInterval(timer);
  }, [isRecording]);

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

  const start = useCallback(async () => {
    setIsBusy(true);
    setError(null);
    try {
      const memo = await startMemoRecording();
      setRecording({ memo_id: memo.id, elapsed_seconds: 0, level: 0 });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsBusy(false);
    }
  }, [refresh]);

  const stop = useCallback(async () => {
    setIsBusy(true);
    setError(null);
    try {
      await stopMemoRecording();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecording(null);
      setIsBusy(false);
      await refresh();
    }
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

  return { memos, recording, progress, isBusy, error, start, stop, remove, refresh };
}
