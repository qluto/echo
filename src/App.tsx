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
  getSettings,
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
  const [modelName, setModelName] = useState<string>("mlx-community/whisper-large-v3-turbo");
  const [hotkey, setHotkey] = useState<string>("command+shift+space");

  // Model parameter counts
  const MODEL_SIZES: Record<string, string> = {
    // Qwen3-ASR models
    "mlx-community/Qwen3-ASR-1.7B-8bit": "1.7B",
    "mlx-community/Qwen3-ASR-0.6B-8bit": "0.6B",
    // Whisper models
    "mlx-community/whisper-large-v3-turbo": "Turbo",
    "mlx-community/whisper-large-v3": "1.5B",
    "mlx-community/whisper-medium": "769M",
    "mlx-community/whisper-small": "244M",
    "mlx-community/whisper-base": "74M",
    "mlx-community/whisper-tiny": "39M",
  };

  const getModelSize = (name: string): string => {
    return MODEL_SIZES[name] || "unknown";
  };

  const getModelFamily = (name: string): string => {
    if (name.includes("Qwen3-ASR")) return "Qwen3";
    if (name.includes("whisper")) return "Whisper";
    return "Unknown";
  };

  const getModelShortName = (name: string): string => {
    const family = getModelFamily(name);
    const size = getModelSize(name);
    return `${family} · ${size}`;
  };

  // Format hotkey for display
  const formatHotkey = (hk: string): string => {
    return hk
      // Remove fn when combined with function keys (fn+f12 -> f12)
      .replace(/\bfn\+?(f(?:[1-9]|1[0-9]|2[0-4]))\b/gi, "$1")
      .replace(/command/gi, "⌘")
      .replace(/ctrl/gi, "⌃")
      .replace(/control/gi, "⌃")
      .replace(/shift/gi, "⇧")
      .replace(/option/gi, "⌥")
      .replace(/alt/gi, "⌥")
      .replace(/\bfn\b/gi, "🌐")  // Fn key alone
      .replace(/return/gi, "↵")
      .replace(/space/gi, "␣")
      .replace(/escape/gi, "⎋")
      .replace(/backspace/gi, "⌫")
      .replace(/delete/gi, "⌦")
      .replace(/tab/gi, "⇥")
      // Function keys - uppercase for readability
      .replace(/\b(f[1-9]|f1[0-9]|f2[0-4])\b/gi, (match) => match.toUpperCase())
      .replace("CommandOrControl", "⌘")
      .replace(/\+/g, "");
  };

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

  // Load settings (hotkey)
  const loadHotkey = async () => {
    try {
      const settings = await getSettings();
      setHotkey(settings.hotkey);
    } catch (e) {
      console.error("Failed to load settings:", e);
    }
  };

  useEffect(() => {
    loadHotkey();
    initializeEngine();
  }, []);

  // Reload hotkey when settings panel closes
  useEffect(() => {
    if (!isSettingsOpen) {
      loadHotkey();
    }
  }, [isSettingsOpen]);

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
      setModelName(status.model_name || "mlx-community/whisper-large-v3-turbo");

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
      setModelName(status.model_name || "mlx-community/whisper-large-v3-turbo");
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

  // Auto-insert is now handled by the backend (hotkey.rs)
  // Frontend only shows success state when result arrives
  useEffect(() => {
    if (result?.text) {
      setShowSuccess(true);
    }
  }, [result]);

  // Calculate duration from segments
  const getDuration = () => {
    if (!result?.segments || result.segments.length === 0) return null;
    const lastSegment = result.segments[result.segments.length - 1];
    return lastSegment.end;
  };

  return (
    <div className="h-screen bg-surface-muted flex flex-col rounded-2xl overflow-hidden select-none card-shadow border border-subtle">
      {/* Header */}
      <header
        data-tauri-drag-region
        className="h-14 flex items-center justify-between px-5 flex-shrink-0"
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
          <div className="w-7 h-7 rounded-lg overflow-hidden" style={{ backgroundColor: "#7C9082" }}>
            <svg viewBox="0 0 28 28" className="w-full h-full">
              {/* Wave rings */}
              <ellipse cx="14" cy="14" rx="11" ry="11" fill="none" stroke="white" strokeOpacity="0.25" strokeWidth="1"/>
              <ellipse cx="14" cy="14" rx="8" ry="8" fill="none" stroke="white" strokeOpacity="0.44" strokeWidth="1"/>
              <ellipse cx="14" cy="14" rx="5" ry="5" fill="none" stroke="white" strokeOpacity="0.63" strokeWidth="1"/>
              {/* Core */}
              <ellipse cx="14" cy="14" rx="3" ry="3" fill="white"/>
            </svg>
          </div>
          <span
            className="font-display text-xl tracking-tight"
            style={{ color: "var(--text-primary)" }}
          >
            echo
          </span>
        </div>

        {/* Settings Button */}
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="w-9 h-9 rounded-xl bg-surface-muted flex items-center justify-center hover:bg-surface-elevated transition-colors border border-subtle"
        >
          <svg
            className="w-[18px] h-[18px]"
            fill="none"
            stroke="var(--text-secondary)"
            strokeWidth={1.5}
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
            />
          </svg>
        </button>
      </header>

      {/* Content */}
      <main className="flex-1 px-5 pb-5 flex flex-col gap-4 overflow-auto">
        {/* Hotkey Section */}
        <div className="flex items-center justify-center gap-2 py-2">
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Hold
          </span>
          <div className="flex items-center gap-1">
            <div className="h-7 px-2.5 rounded-lg flex items-center bg-surface-muted border border-subtle">
              <span
                className="font-mono text-xs font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                {formatHotkey(hotkey)}
              </span>
            </div>
          </div>
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            to record
          </span>
        </div>

        {/* Transcript Section */}
        <div className="flex flex-col gap-2 flex-1">
          {/* Header */}
          <span
            className="text-xs font-medium"
            style={{ color: "var(--text-tertiary)" }}
          >
            Last transcript
          </span>

          {/* Transcript Card */}
          <div className="flex-1 flex flex-col rounded-xl bg-surface border border-subtle p-4 gap-3">
            {error ? (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-sm" style={{ color: "var(--glow-recording)" }}>
                  {error}
                </p>
              </div>
            ) : result?.text ? (
              <>
                {/* Duration & Language */}
                <div className="flex items-center justify-between">
                  <span
                    className="font-mono text-xs"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {getDuration() ? `${getDuration()!.toFixed(1)}s` : ""}
                  </span>
                  {result.language && (
                    <span
                      className="font-mono text-xs capitalize"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      {result.language}
                    </span>
                  )}
                </div>

                {/* Transcript Text */}
                <p
                  className="text-[15px] leading-relaxed flex-1"
                  style={{ color: "var(--text-primary)" }}
                >
                  {result.text}
                </p>

                {/* Action Buttons */}
                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    onClick={handleCopy}
                    className="h-9 px-4 rounded-full text-sm font-medium flex items-center gap-2 transition-colors bg-surface border border-subtle hover:bg-surface-elevated"
                    style={{ color: "var(--text-primary)" }}
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={1.5}
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75"
                      />
                    </svg>
                    Copy
                  </button>
                  <button
                    onClick={handleInsert}
                    className="h-9 px-4 rounded-full text-sm font-medium flex items-center gap-2 transition-colors text-white"
                    style={{
                      backgroundColor: "var(--text-primary)",
                    }}
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={1.5}
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="m7.49 12-3.75 3.75m0 0 3.75 3.75m-3.75-3.75h16.5V4.499"
                      />
                    </svg>
                    Insert
                  </button>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <p
                  className="text-sm text-center"
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
      <footer className="h-11 flex items-center justify-between px-5 border-t border-subtle flex-shrink-0">
        {/* Model Info */}
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4"
            fill="var(--text-tertiary)"
            viewBox="0 0 24 24"
          >
            <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z" />
          </svg>
          <span
            className="font-mono text-xs"
            style={{ color: "var(--text-tertiary)" }}
          >
            {getModelShortName(modelName)}
          </span>
        </div>

        {/* Status */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              modelStatus === "loading" ? "animate-glow-pulse" : ""
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
            className="font-mono text-xs"
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
        onClose={async () => {
          setIsSettingsOpen(false);
          // Refresh model status after settings panel closes
          try {
            const status = await getModelStatus();
            if (status.model_name) {
              setModelName(status.model_name);
            }
            setModelStatus(status.loaded ? "loaded" : "not_loaded");
          } catch (e) {
            console.error("Failed to refresh model status:", e);
          }
        }}
      />
    </div>
  );
}

export default App;
