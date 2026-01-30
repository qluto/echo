import { useState, useEffect, useCallback } from "react";
import {
  startRecording,
  stopRecording,
  transcribe,
  insertText,
  onRecordingStateChange,
  onTranscriptionComplete,
  onError,
  RecordingState,
  TranscriptionResult,
} from "../lib/tauri";

interface UseTranscriptionReturn {
  state: RecordingState;
  result: TranscriptionResult | null;
  error: string | null;
  isRecording: boolean;
  isTranscribing: boolean;
  startRecord: () => Promise<void>;
  stopRecord: () => Promise<void>;
  clearResult: () => void;
  insertResult: () => Promise<void>;
}

export function useTranscription(): UseTranscriptionReturn {
  const [state, setState] = useState<RecordingState>("idle");
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const unlisteners: Array<() => void> = [];

    onRecordingStateChange((event) => {
      setState(event.state);
    }).then((unlisten) => unlisteners.push(unlisten));

    onTranscriptionComplete((event) => {
      if (event.error) {
        setError(event.error);
      } else if (event.result) {
        setResult(event.result);
        setError(null);
      }
    }).then((unlisten) => unlisteners.push(unlisten));

    onError((errorMessage) => {
      setError(errorMessage);
    }).then((unlisten) => unlisteners.push(unlisten));

    return () => {
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, []);

  const startRecord = useCallback(async () => {
    try {
      setError(null);
      await startRecording();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const stopRecord = useCallback(async () => {
    try {
      const audioPath = await stopRecording();
      setState("transcribing");
      const transcriptionResult = await transcribe(audioPath);
      setResult(transcriptionResult);
      setState("idle");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("idle");
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  const insertResult = useCallback(async () => {
    if (result?.text) {
      try {
        await insertText(result.text);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  }, [result]);

  return {
    state,
    result,
    error,
    isRecording: state === "recording",
    isTranscribing: state === "transcribing",
    startRecord,
    stopRecord,
    clearResult,
    insertResult,
  };
}
