import { useState, useEffect, useCallback } from "react";
import { RecordingOverlay } from "./components/RecordingOverlay";
import { TranscriptionDisplay } from "./components/TranscriptionDisplay";
import { SettingsPanel } from "./components/SettingsPanel";
import { useTranscription } from "./hooks/useTranscription";
import {
  pingAsrEngine,
  startAsrEngine,
  insertText,
  getModelStatus,
  loadAsrModel,
} from "./lib/tauri";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";

type EngineStatus = "starting" | "ready" | "error";
type ModelLoadStatus = "not_loaded" | "loading" | "loaded" | "error";

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
  const [engineStatus, setEngineStatus] = useState<EngineStatus>("starting");
  const [modelStatus, setModelStatus] = useState<ModelLoadStatus>("not_loaded");
  const [modelName, setModelName] = useState<string>("");
  const [modelError, setModelError] = useState<string | null>(null);

  useEffect(() => {
    initializeEngine();
  }, []);

  const initializeEngine = async () => {
    setEngineStatus("starting");
    setModelStatus("not_loaded");

    try {
      // Start the engine process
      const isReady = await pingAsrEngine();
      if (!isReady) {
        console.log("ASR engine not ready, starting...");
        await startAsrEngine();
        console.log("ASR engine started successfully");
      }
      setEngineStatus("ready");

      // Get model status
      const status = await getModelStatus();
      setModelName(status.model_name);

      if (status.loaded) {
        setModelStatus("loaded");
      } else {
        // Start loading the model
        await loadModel();
      }
    } catch (e) {
      console.error("Failed to initialize engine:", e);
      setEngineStatus("error");
      setModelError(String(e));
    }
  };

  const loadModel = async () => {
    setModelStatus("loading");
    setModelError(null);

    try {
      const status = await loadAsrModel();
      setModelName(status.model_name);
      setModelStatus("loaded");
      console.log("Model loaded:", status.model_name);
    } catch (e) {
      console.error("Failed to load model:", e);
      setModelStatus("error");
      setModelError(String(e));
    }
  };

  const getStatusText = () => {
    if (engineStatus === "starting") return "エンジン起動中...";
    if (engineStatus === "error") return "エンジンエラー";
    if (modelStatus === "loading") return "モデル読み込み中...";
    if (modelStatus === "error") return "モデルエラー";
    if (modelStatus === "not_loaded") return "モデル未読込";
    return "準備完了";
  };

  const getStatusColor = () => {
    if (engineStatus === "error" || modelStatus === "error") return "bg-red-500";
    if (engineStatus === "starting" || modelStatus === "loading") return "bg-yellow-500 animate-pulse";
    if (modelStatus === "loaded") return "bg-green-500";
    return "bg-gray-500";
  };

  const isReady = engineStatus === "ready" && modelStatus === "loaded";

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
            className={`w-2 h-2 rounded-full ${getStatusColor()}`}
            title={getStatusText()}
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
        {/* Engine & Model Status */}
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
              <div className="flex flex-col">
                <span className="text-sm font-medium text-gray-200">{getStatusText()}</span>
                {modelName && (
                  <span className="text-xs text-gray-500">{modelName}</span>
                )}
              </div>
            </div>
            {modelStatus === "error" && modelError && (
              <button
                onClick={loadModel}
                className="px-2 py-1 text-xs bg-primary-600 hover:bg-primary-500 text-white rounded transition-colors"
              >
                再試行
              </button>
            )}
          </div>
          {modelError && (
            <div className="mt-2 text-xs text-red-400 bg-red-900/20 rounded p-2">
              {modelError}
            </div>
          )}
        </div>

        {/* Recording Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">状態:</span>
            <span
              className={`text-sm font-medium ${
                !isReady
                  ? "text-gray-500"
                  : state === "idle"
                  ? "text-gray-300"
                  : state === "recording"
                  ? "text-red-400"
                  : "text-primary-400"
              }`}
            >
              {!isReady
                ? "準備中..."
                : state === "idle"
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
