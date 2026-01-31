import { useState, useEffect, useCallback, useRef } from "react";
import { getCurrentWindow, currentMonitor, LogicalPosition } from "@tauri-apps/api/window";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { emit } from "@tauri-apps/api/event";
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
    result,
    error,
    isRecording,
    isTranscribing,
    recordingDuration,
  } = useTranscription();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [engineStatus, setEngineStatus] = useState<EngineStatus>("starting");
  const [modelStatus, setModelStatus] = useState<ModelLoadStatus>("not_loaded");
  const [modelName, setModelName] = useState<string>("Qwen3-ASR");
  const [modelSize] = useState<string>("1.7B");
  const [showSuccess, setShowSuccess] = useState(false);
  const floatWindowRef = useRef<WebviewWindow | null>(null);

  // Get float window reference
  useEffect(() => {
    const getFloatWindow = async () => {
      try {
        const floatWin = await WebviewWindow.getByLabel("float");
        if (floatWin) {
          floatWindowRef.current = floatWin;
          // Position at bottom center of screen
          await positionFloatWindow(floatWin);
        }
      } catch (e) {
        console.error("Failed to get float window:", e);
      }
    };
    getFloatWindow();
  }, []);

  // Position float window at bottom center
  const positionFloatWindow = async (floatWin: WebviewWindow) => {
    try {
      const monitor = await currentMonitor();
      if (monitor) {
        const screenWidth = monitor.size.width / monitor.scaleFactor;
        const screenHeight = monitor.size.height / monitor.scaleFactor;
        const windowWidth = 240;
        const windowHeight = 60;
        const x = Math.round((screenWidth - windowWidth) / 2);
        const y = Math.round(screenHeight - windowHeight - 80); // 80px from bottom
        await floatWin.setPosition(new LogicalPosition(x, y));
      }
    } catch (e) {
      console.error("Failed to position float window:", e);
    }
  };

  // Emit state to float window
  const emitFloatState = useCallback(
    async (state: "idle" | "recording" | "processing" | "success", duration: number) => {
      try {
        await emit("float-state", { state, duration });
      } catch (e) {
        console.error("Failed to emit float state:", e);
      }
    },
    []
  );

  // Update float window when state changes
  useEffect(() => {
    const state = showSuccess
      ? "success"
      : isRecording
      ? "recording"
      : isTranscribing
      ? "processing"
      : "idle";
    emitFloatState(state, recordingDuration);

    // Auto-hide success after delay
    if (showSuccess) {
      const timer = setTimeout(() => {
        setShowSuccess(false);
        emitFloatState("idle", 0);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [isRecording, isTranscribing, showSuccess, recordingDuration, emitFloatState]);

  // Window controls
  const handleClose = useCallback(async () => {
    const window = getCurrentWindow();
    await window.close();
  }, []);

  const handleMinimize = useCallback(async () => {
    const window = getCurrentWindow();
    await window.minimize();
  }, []);

  const handleMaximize = useCallback(async () => {
    const window = getCurrentWindow();
    const isMaximized = await window.isMaximized();
    if (isMaximized) {
      await window.unmaximize();
    } else {
      await window.maximize();
    }
  }, []);

  useEffect(() => {
    initializeEngine();
  }, []);

  const initializeEngine = async () => {
    setEngineStatus("starting");
    setModelStatus("not_loaded");

    try {
      const isReady = await pingAsrEngine();
      if (!isReady) {
        console.log("ASR engine not ready, starting...");
        await startAsrEngine();
        console.log("ASR engine started successfully");
      }
      setEngineStatus("ready");

      const status = await getModelStatus();
      setModelName(status.model_name || "Qwen3-ASR");

      if (status.loaded) {
        setModelStatus("loaded");
      } else {
        await loadModel();
      }
    } catch (e) {
      console.error("Failed to initialize engine:", e);
      setEngineStatus("error");
    }
  };

  const loadModel = async () => {
    setModelStatus("loading");
    try {
      const status = await loadAsrModel();
      setModelName(status.model_name || "Qwen3-ASR");
      setModelStatus("loaded");
    } catch (e) {
      console.error("Failed to load model:", e);
      setModelStatus("error");
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
        setShowSuccess(true);
      } catch (e) {
        console.error("Failed to insert:", e);
      }
    }
  }, [result]);

  // Auto-insert effect
  useEffect(() => {
    if (result?.text && !showSuccess) {
      handleInsert();
    }
  }, [result]);

  // Calculate duration from segments
  const getDuration = () => {
    if (!result?.segments || result.segments.length === 0) return null;
    const lastSegment = result.segments[result.segments.length - 1];
    return lastSegment.end;
  };

  return (
    <div className="h-screen bg-void flex flex-col border border-subtle rounded-lg overflow-hidden select-none">
      {/* Header */}
      <header
        data-tauri-drag-region
        className="h-[52px] flex items-center justify-between px-5 flex-shrink-0"
      >
        {/* Traffic Lights */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleClose}
            className="w-3 h-3 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all"
            aria-label="Close"
          />
          <button
            onClick={handleMinimize}
            className="w-3 h-3 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all"
            aria-label="Minimize"
          />
          <button
            onClick={handleMaximize}
            className="w-3 h-3 rounded-full bg-[#28C840] hover:brightness-90 transition-all"
            aria-label="Maximize"
          />
        </div>

        {/* Logo */}
        <div className="flex items-center gap-2.5" data-tauri-drag-region>
          <div
            className="w-7 h-7 rounded-[14px] flex items-center justify-center relative"
            style={{
              background:
                "radial-gradient(circle, var(--glow-idle) 0%, transparent 100%)",
              boxShadow: "0 0 16px var(--glow-idle-soft)",
            }}
          >
            <div
              className="w-3 h-3 rounded-full"
              style={{
                backgroundColor: "var(--glow-idle)",
                boxShadow: "0 0 8px 2px var(--glow-idle)",
              }}
            />
          </div>
          <span
            className="font-display text-[15px] font-bold tracking-[2px]"
            style={{ color: "var(--text-primary)" }}
          >
            echo
          </span>
        </div>

        {/* Settings Button */}
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="w-9 h-9 rounded-[10px] bg-surface flex items-center justify-center hover:bg-surface-elevated transition-colors"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="var(--text-secondary)"
            strokeWidth={1.5}
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"
            />
          </svg>
        </button>
      </header>

      {/* Content */}
      <main className="flex-1 p-5 flex flex-col gap-4 overflow-auto">
        {/* Hotkey Section */}
        <div
          className="flex items-center justify-between px-3.5 py-3 rounded-[10px] bg-surface border border-subtle"
        >
          <div className="flex items-center gap-2.5">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: "rgba(99, 102, 241, 0.08)" }}
            >
              <svg
                className="w-4 h-4"
                fill="var(--glow-idle)"
                viewBox="0 0 24 24"
              >
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
              </svg>
            </div>
            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
              Hold to record
            </span>
          </div>
          <div
            className="h-[26px] px-2.5 rounded-md flex items-center bg-surface-elevated border border-subtle"
          >
            <span
              className="font-display text-[11px] tracking-[0.5px]"
              style={{ color: "var(--text-primary)" }}
            >
              {"\u2318\u21E7Space"}
            </span>
          </div>
        </div>

        {/* Transcript Section */}
        <div className="flex flex-col gap-2.5 flex-1">
          {/* Header */}
          <div className="flex items-center justify-between">
            <span
              className="text-[11px] font-medium tracking-[0.5px]"
              style={{ color: "var(--text-tertiary)" }}
            >
              Last transcript
            </span>
            {result && (
              <span
                className="font-display text-[10px]"
                style={{ color: "var(--text-tertiary)" }}
              >
                {getDuration() ? `${getDuration()!.toFixed(1)}s` : ""}{" "}
                {result.language ? `\u00B7 ${result.language}` : ""}
              </span>
            )}
          </div>

          {/* Transcript Card */}
          <div
            className="flex-1 flex flex-col rounded-xl bg-surface border border-subtle p-4 gap-3"
          >
            {error ? (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-sm" style={{ color: "var(--glow-recording)" }}>
                  {error}
                </p>
              </div>
            ) : result?.text ? (
              <>
                <p
                  className="text-[13px] leading-relaxed flex-1"
                  style={{ color: "var(--text-primary)" }}
                >
                  {result.text}
                </p>
                <div className="flex items-center justify-end gap-2">
                  <button
                    onClick={handleCopy}
                    className="h-7 px-3 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors"
                    style={{
                      backgroundColor: "var(--surface-elevated)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    Copy
                  </button>
                  <button
                    onClick={handleInsert}
                    className="h-7 px-3 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors"
                    style={{
                      background:
                        "linear-gradient(180deg, var(--glow-idle) 0%, #4F46E5 100%)",
                      color: "white",
                    }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                      />
                    </svg>
                    Insert
                  </button>
                </div>
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center gap-3">
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: "var(--surface-elevated)" }}
                >
                  <svg
                    className="w-6 h-6"
                    fill="none"
                    stroke="var(--text-tertiary)"
                    strokeWidth={1.5}
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                    />
                  </svg>
                </div>
                <p
                  className="text-xs text-center"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Press and hold the hotkey to start recording
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        className="h-11 flex items-center justify-between px-5 border-t border-subtle flex-shrink-0"
      >
        {/* Model Info */}
        <div className="flex items-center gap-2">
          <div
            className="w-5 h-5 rounded-[5px] bg-surface flex items-center justify-center"
          >
            <svg
              className="w-[11px] h-[11px]"
              fill="var(--text-tertiary)"
              viewBox="0 0 24 24"
            >
              <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z" />
            </svg>
          </div>
          <span
            className="font-display text-[10px]"
            style={{ color: "var(--text-tertiary)" }}
          >
            {modelName}
          </span>
          <div
            className="h-[18px] px-1.5 rounded flex items-center"
            style={{ backgroundColor: "rgba(99, 102, 241, 0.12)" }}
          >
            <span
              className="font-display text-[9px] font-medium"
              style={{ color: "var(--glow-idle)" }}
            >
              {modelSize}
            </span>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center gap-1.5">
          <div
            className={`w-1.5 h-1.5 rounded-full ${
              modelStatus === "loaded"
                ? "glow-success"
                : modelStatus === "loading"
                ? "glow-processing animate-glow-pulse"
                : engineStatus === "error" || modelStatus === "error"
                ? "glow-recording"
                : ""
            }`}
            style={{
              backgroundColor:
                modelStatus === "loaded"
                  ? "var(--glow-success)"
                  : modelStatus === "loading"
                  ? "var(--glow-processing)"
                  : engineStatus === "error" || modelStatus === "error"
                  ? "var(--glow-recording)"
                  : "var(--text-tertiary)",
            }}
          />
          <span
            className="font-display text-[10px]"
            style={{
              color:
                modelStatus === "loaded"
                  ? "var(--glow-success)"
                  : modelStatus === "loading"
                  ? "var(--glow-processing)"
                  : engineStatus === "error" || modelStatus === "error"
                  ? "var(--glow-recording)"
                  : "var(--text-tertiary)",
            }}
          >
            {modelStatus === "loaded"
              ? "ready"
              : modelStatus === "loading"
              ? "loading"
              : engineStatus === "error" || modelStatus === "error"
              ? "error"
              : "idle"}
          </span>
        </div>
      </footer>

      {/* Settings Panel */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
}

export default App;
