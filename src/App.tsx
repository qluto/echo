import { useState, useEffect, useCallback } from "react";
import { RecordingOverlay } from "./components/RecordingOverlay";
import { TranscriptionDisplay } from "./components/TranscriptionDisplay";
import { SettingsPanel } from "./components/SettingsPanel";
import { useTranscription } from "./hooks/useTranscription";
import { pingAsrEngine, startAsrEngine, insertText } from "./lib/tauri";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";

function App() {
  const {
    state,
    result,
    error,
    isRecording,
    isTranscribing,
    clearResult,
  } = useTranscription();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [engineStatus, setEngineStatus] = useState<"checking" | "ready" | "error">("checking");

  useEffect(() => {
    checkEngineStatus();
  }, []);

  const checkEngineStatus = async () => {
    setEngineStatus("checking");
    try {
      const isReady = await pingAsrEngine();
      if (isReady) {
        setEngineStatus("ready");
      } else {
        await startAsrEngine();
        setEngineStatus("ready");
      }
    } catch {
      setEngineStatus("error");
    }
  };

  const handleCopy = useCallback(async () => {
    if (result?.text) {
      try {
        await writeText(result.text);
      } catch (e) {
        console.error("Failed to copy:", e);
      }
    }
  }, [result]);

  const handleInsert = useCallback(async () => {
    if (result?.text) {
      try {
        await insertText(result.text);
      } catch (e) {
        console.error("Failed to insert:", e);
      }
    }
  }, [result]);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Title bar */}
      <div className="titlebar h-8 bg-gray-800 flex items-center justify-between px-4 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4 text-primary-500"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
          </svg>
          <span className="text-sm font-medium text-white">Echo</span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              engineStatus === "ready"
                ? "bg-green-500"
                : engineStatus === "checking"
                ? "bg-yellow-500 animate-pulse"
                : "bg-red-500"
            }`}
            title={
              engineStatus === "ready"
                ? "エンジン準備完了"
                : engineStatus === "checking"
                ? "エンジン起動中..."
                : "エンジンエラー"
            }
          />
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 p-4 flex flex-col gap-4">
        {/* Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">状態:</span>
            <span
              className={`text-sm font-medium ${
                state === "idle"
                  ? "text-gray-300"
                  : state === "recording"
                  ? "text-red-400"
                  : "text-primary-400"
              }`}
            >
              {state === "idle"
                ? "待機中"
                : state === "recording"
                ? "録音中"
                : "文字起こし中"}
            </span>
          </div>
          <div className="text-xs text-gray-500">
            <kbd className="px-1.5 py-0.5 bg-gray-700 rounded">Cmd+Shift+Space</kbd> で録音
          </div>
        </div>

        {/* Transcription display */}
        <div className="flex-1">
          <TranscriptionDisplay
            result={result}
            error={error}
            onClear={clearResult}
            onInsert={handleInsert}
            onCopy={handleCopy}
          />
        </div>

        {/* Instructions */}
        <div className="bg-gray-800/50 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">使い方</h3>
          <ul className="text-sm text-gray-400 space-y-1">
            <li className="flex items-start gap-2">
              <span className="text-primary-500">1.</span>
              <span>ホットキーを押し続けて録音を開始</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-500">2.</span>
              <span>話し終わったらホットキーを離す</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-500">3.</span>
              <span>自動的に文字起こしが開始されます</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-500">4.</span>
              <span>「挿入」をクリックでアクティブなアプリに貼り付け</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Recording overlay */}
      <RecordingOverlay
        isRecording={isRecording}
        isTranscribing={isTranscribing}
      />

      {/* Settings panel */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
}

export default App;
