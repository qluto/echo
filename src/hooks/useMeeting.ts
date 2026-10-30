import { useState, useEffect, useCallback, useRef } from "react";
import {
  startMeeting,
  stopMeeting,
  getMeetingSnapshot,
  getMeetingStatus,
  onMeetingLine,
  onMeetingItems,
  onMeetingState,
  MeetingItem,
  MeetingLine,
  MeetingPhase,
  MeetingSnapshot,
  MeetingStatus,
} from "../lib/tauri";

const STATUS_POLL_MS = 200;

/** Meeting-level fields of the snapshot; lines and items are tracked separately. */
type MeetingMeta = Omit<MeetingSnapshot, "lines" | "items" | "round">;

export const isMeetingActive = (phase: MeetingPhase | undefined) =>
  phase === "running" || phase === "finalizing";

/** Appends lines not seen yet (events and snapshots can overlap), kept in id order. */
const mergeLines = (prev: MeetingLine[], incoming: MeetingLine[]) => {
  const seen = new Set(prev.map((l) => l.id));
  const fresh = incoming.filter((l) => !seen.has(l.id));
  if (fresh.length === 0) return prev;
  return [...prev, ...fresh].sort((a, b) => a.id - b.id);
};

const errorMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function useMeeting() {
  /** null until a meeting has been started in this app session. */
  const [meta, setMeta] = useState<MeetingMeta | null>(null);
  const [lines, setLines] = useState<MeetingLine[]>([]);
  const [items, setItems] = useState<{ items: MeetingItem[]; round: number }>({ items: [], round: 0 });
  /** Last polled status; kept after the meeting ends so the final elapsed time stays visible. */
  const [status, setStatus] = useState<MeetingStatus | null>(null);
  /** True while a start/stop request is in flight. */
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** memo_id of the meeting whose lines/items are currently held. */
  const memoIdRef = useRef<number | null>(null);

  const applySnapshot = useCallback((snap: MeetingSnapshot) => {
    const { lines: snapLines, items: snapItems, round, ...rest } = snap;
    if (memoIdRef.current !== null && memoIdRef.current !== rest.memo_id) {
      // A different meeting: drop what belonged to the previous one
      setLines(snapLines);
      setItems({ items: snapItems, round });
    } else {
      setLines((l) => mergeLines(l, snapLines));
      // Never let a stale snapshot overwrite a newer `meeting-items` event
      setItems((cur) => (round >= cur.round ? { items: snapItems, round } : cur));
    }
    memoIdRef.current = rest.memo_id;
    setMeta(rest);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const snap = await getMeetingSnapshot();
      if (snap) applySnapshot(snap);
    } catch (e) {
      console.error("Failed to get meeting snapshot:", e);
    }
  }, [applySnapshot]);

  // Subscribe first, then load the snapshot, so nothing emitted in between is lost
  useEffect(() => {
    const unlistens: Promise<() => void>[] = [
      onMeetingLine(({ memo_id, ...line }) => {
        if (memo_id === memoIdRef.current) setLines((prev) => mergeLines(prev, [line]));
      }),
      onMeetingItems(({ memo_id, items, round }) => {
        if (memo_id === memoIdRef.current)
          setItems((cur) => (round >= cur.round ? { items, round } : cur));
      }),
      onMeetingState((event) => {
        setMeta((prev) =>
          prev && prev.memo_id === event.memo_id
            ? { ...prev, phase: event.phase, error: event.error, warning: event.warning }
            : prev
        );
        // Picks up a meeting started elsewhere and the final items/lines after a phase change
        refresh();
      }),
    ];
    refresh();
    return () => {
      unlistens.forEach((p) => p.then((fn) => fn()));
    };
  }, [refresh]);

  // Poll elapsed time, levels and update timing while the meeting is active
  const active = isMeetingActive(meta?.phase);
  useEffect(() => {
    if (!active) return;
    const poll = () =>
      getMeetingStatus()
        .then((s) => {
          if (s) setStatus(s);
        })
        .catch(() => {});
    poll();
    const timer = setInterval(poll, STATUS_POLL_MS);
    return () => clearInterval(timer);
  }, [active]);

  const start = useCallback(async () => {
    setIsBusy(true);
    setError(null);
    try {
      // Events are matched to the meeting by memo_id; until start returns, none match
      setLines([]);
      setItems({ items: [], round: 0 });
      setStatus(null);
      setMeta(null);
      memoIdRef.current = null;
      applySnapshot(await startMeeting());
    } catch (e) {
      setError(errorMessage(e));
      await refresh();
    } finally {
      setIsBusy(false);
    }
  }, [applySnapshot, refresh]);

  const stop = useCallback(async () => {
    setIsBusy(true);
    setError(null);
    try {
      applySnapshot(await stopMeeting());
    } catch (e) {
      setError(errorMessage(e));
      await refresh();
    } finally {
      setIsBusy(false);
    }
  }, [applySnapshot, refresh]);

  return {
    meeting: meta,
    lines,
    items: items.items,
    round: items.round,
    status,
    isBusy,
    error,
    start,
    stop,
  };
}
