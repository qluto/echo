import { useState, useEffect, useCallback } from "react";
import {
  startContinuousListening,
  stopContinuousListening,
  getContinuousListeningStatus,
  onContinuousTranscription,
  onContinuousVadState,
  ContinuousTranscriptionEvent,
} from "../lib/tauri";

interface UseContinuousListeningReturn {
  isListening: boolean;
  isSpeechDetected: boolean;
  segmentCount: number;
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
  const [recentEntries, setRecentEntries] = useState<ContinuousTranscriptionEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Check initial status
  useEffect(() => {
    getContinuousListeningStatus()
      .then((status) => {
        setIsListening(status.is_listening);
        setSegmentCount(status.segment_count);
      })
      .catch((e) => console.error("Failed to get listening status:", e));
  }, []);

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

  // Subscribe to VAD speech state changes
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    onContinuousVadState((event) => {
      setIsSpeechDetected(event.is_speech);
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
      await startContinuousListening();
      setIsListening(true);
      setIsSpeechDetected(false);
      setSegmentCount(0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, []);

  const stopListening = useCallback(async () => {
    try {
      setError(null);
      const count = await stopContinuousListening();
      setIsListening(false);
      setIsSpeechDetected(false);
      setSegmentCount(count);
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
    recentEntries,
    error,
    startListening,
    stopListening,
    toggleListening,
  };
}
