import { useEffect, useState } from "react";

interface RecordingOverlayProps {
  isRecording: boolean;
  isTranscribing: boolean;
}

export function RecordingOverlay({
  isRecording,
  isTranscribing,
}: RecordingOverlayProps) {
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!isRecording) {
      setDuration(0);
      return;
    }

    const interval = setInterval(() => {
      setDuration((d) => d + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [isRecording]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  if (!isRecording && !isTranscribing) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-fade-in">
      <div className="bg-gray-800 rounded-2xl p-8 shadow-2xl flex flex-col items-center gap-4">
        {isRecording ? (
          <>
            <div className="relative">
              <div className="w-20 h-20 rounded-full bg-red-500 recording-indicator flex items-center justify-center">
                <svg
                  className="w-10 h-10 text-white"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
                </svg>
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-600 rounded-full animate-ping" />
            </div>
            <p className="text-xl font-semibold text-white">録音中...</p>
            <p className="text-3xl font-mono text-red-400">
              {formatDuration(duration)}
            </p>
            <p className="text-sm text-gray-400">
              ホットキーを離すと録音が終了します
            </p>
          </>
        ) : (
          <>
            <div className="w-20 h-20 rounded-full bg-primary-500 flex items-center justify-center">
              <svg
                className="w-10 h-10 text-white animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            </div>
            <p className="text-xl font-semibold text-white">文字起こし中...</p>
            <p className="text-sm text-gray-400">
              音声を解析しています
            </p>
          </>
        )}
      </div>
    </div>
  );
}
