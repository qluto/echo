import { useState, useCallback } from "react";
import { summarizeRecentTranscriptions } from "../lib/tauri";

export function useSummarize() {
  const [summary, setSummary] = useState<string | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [entryCount, setEntryCount] = useState(0);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const summarize = useCallback(async (minutes?: number) => {
    setIsSummarizing(true);
    setError(null);
    setSummary(null);
    try {
      const result = await summarizeRecentTranscriptions(minutes);
      if (result.success) {
        setSummary(result.summary);
        setEntryCount(result.entry_count);
        setProcessingTime(result.processing_time_ms);
      } else {
        setError(result.error || "Summarization failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSummarizing(false);
    }
  }, []);

  const dismiss = useCallback(() => {
    setSummary(null);
    setError(null);
  }, []);

  return { summary, isSummarizing, error, entryCount, processingTime, summarize, dismiss };
}
