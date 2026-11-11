import { useState, useEffect, useCallback } from "react";
import {
  startContinuousListening,
  onContinuousListeningResumed,
  stopContinuousListening,
  getContinuousListeningStatus,
  onContinuousTranscription,
  onContinuousActivity,
  ContinuousListeningStatus,
  ContinuousTranscriptionEvent,
} from "../lib/tauri";

interface UseContinuousListeningReturn {
  isListening: boolean;
  isSpeechDetected: boolean;
  segmentCount: number;
  /** System audio is captured along with the mic. */
  systemAudio: boolean;
  /** Why system audio isn't captured, if it couldn't start. */
  warning: string | null;
  recentEntries: ContinuousTranscriptionEvent[];
  error: string | null;
  startListening: () => Promise<void>;
  stopListening: () => Promise<void>;
  toggleListening: () => Promise<void>;
}

const MAX_RECENT_ENTRIES = 50;

export function useContinuousListening(): UseContinuousListeningReturn {
  const [isListening, setIsListening] = useState(false);
  const [isSpeechDetected, setIsSpeechDetected] = useState(false);
  const [segmentCount, setSegmentCount] = useState(0);
  const [systemAudio, setSystemAudio] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [recentEntries, setRecentEntries] = useState<ContinuousTranscriptionEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  const applyStatus = useCallback((status: ContinuousListeningStatus) => {
    setIsListening(status.is_listening);
    setSegmentCount(status.segment_count);
    setSystemAudio(status.system_audio);
    setWarning(status.warning);
  }, []);

  // Check initial status
  useEffect(() => {
    getContinuousListeningStatus()
      .then(applyStatus)
      .catch((e) => console.error("Failed to get listening status:", e));
  }, [applyStatus]);

  // Always-on turned back on by the backend after an update restart
  useEffect(() => {
    const unlisten = onContinuousListeningResumed(applyStatus);
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [applyStatus]);

  // Subscribe to continuous transcription events
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    onContinuousTranscription((event) => {
      setRecentEntries((prev) => {
        const next = [event, ...prev];
        return next.slice(0, MAX_RECENT_ENTRIES);
      });
      setSegmentCount((c) => c + 1);
    }).then((fn) => {
      unlisten = fn;
    });

    return () => {
      unlisten?.();
    };
  }, []);

  // Subscribe to sound activity changes
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    onContinuousActivity((event) => {
      setIsSpeechDetected(event.active);
    }).then((fn) => {
      unlisten = fn;
    });

    return () => {
      unlisten?.();
    };
  }, []);

  const startListening = useCallback(async () => {
    try {
      setError(null);
      applyStatus(await startContinuousListening());
      setIsSpeechDetected(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [applyStatus]);

  const stopListening = useCallback(async () => {
    try {
      setError(null);
      const count = await stopContinuousListening();
      setIsListening(false);
      setIsSpeechDetected(false);
      setSegmentCount(count);
      setSystemAudio(false);
      setWarning(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, []);

  const toggleListening = useCallback(async () => {
    if (isListening) {
      await stopListening();
    } else {
      await startListening();
    }
  }, [isListening, startListening, stopListening]);

  return {
    isListening,
    isSpeechDetected,
    segmentCount,
    systemAudio,
    warning,
    recentEntries,
    error,
    startListening,
    stopListening,
    toggleListening,
  };
}
