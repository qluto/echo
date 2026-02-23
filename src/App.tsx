import { useState, useEffect, useCallback, useRef } from "react";
import { getCurrentWindow, currentMonitor, LogicalPosition } from "@tauri-apps/api/window";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { emit, listen } from "@tauri-apps/api/event";
import { SettingsPanel } from "./components/SettingsPanel";
import { TranscriptionHistory } from "./components/TranscriptionHistory";
import { SummaryModal } from "./components/SummaryModal";
import { useTranscription } from "./hooks/useTranscription";
import { useContinuousListening } from "./hooks/useContinuousListening";
import { useSummarize } from "./hooks/useSummarize";
import {
  pingAsrEngine,
  startAsrEngine,
  getModelStatus,
  loadAsrModel,
  warmupAsrModel,
  isModelCached,
  getSettings,
  requestAccessibilityPermission,
  openAccessibilitySettings,
  restartApp,
  loadPostprocessModel,
} from "./lib/tauri";
import { getModelShortName } from "./lib/models";
import { formatHotkey } from "./lib/format";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";

// Detailed loading phase for user feedback
type LoadingPhase =
  | "idle"
  | "starting_engine"
  | "downloading_model"
  | "loading_model"
  | "loading_vad"
  | "warming_up"
  | "ready"
  | "error";

const LOADING_MESSAGES: Record<LoadingPhase, string> = {
  idle: "Initializing...",
  starting_engine: "Starting engine...",
  downloading_model: "Downloading model...",
  loading_model: "Loading model...",
  loading_vad: "Preparing...",
  warming_up: "Warming up...",
  ready: "Ready",
  error: "Error",
};

function App() {
  const {
    result,
    error,
    isRecording,
    isTranscribing,
    recordingDuration,
  } = useTranscription();

  const {
    isListening,
    isSpeechDetected,
    toggleListening,
    error: listeningError,
  } = useContinuousListening();

  const { summary, isSummarizing, error: summaryError, entryCount, processingTime, summarize, dismiss: dismissSummary } = useSummarize();
  const [summaryWindowMinutes, setSummaryWindowMinutes] = useState(30);

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<LoadingPhase>("idle");
  const [modelName, setModelName] = useState<string>("mlx-community/Qwen3-ASR-0.6B-8bit");
  const [hotkey, setHotkey] = useState<string>("command+shift+space");
  const [hotkeyError, setHotkeyError] = useState<string | null>(null);
  const [showRestartPrompt, setShowRestartPrompt] = useState(false);

  const [showSuccess, setShowSuccess] = useState(false);
  const floatWindowRef = useRef<WebviewWindow | null>(null);

  // Remove native loading screen after React has painted
  // Using useEffect (not useLayoutEffect) ensures the React loading overlay
  // is visible before we remove the native HTML loading screen
  useEffect(() => {
    const appLoading = document.getElementById("app-loading");
    if (appLoading) {
      // Add fade-out animation before removing
      appLoading.style.transition = "opacity 150ms ease-out";
      appLoading.style.opacity = "0";
      // Remove after animation completes
      const timer = setTimeout(() => {
        appLoading.remove();
      }, 150);
      return () => clearTimeout(timer);
    }
  }, []);

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
        const y = Math.round(screenHeight - windowHeight - 32); // 32px from bottom
        await floatWin.setPosition(new LogicalPosition(x, y));
      }
    } catch (e) {
      console.error("Failed to position float window:", e);
    }
  };

  // Emit state to float window
  type FloatState = "idle" | "recording" | "processing" | "success" | "ambient" | "ambient-active";
  const emitFloatState = useCallback(
    async (state: FloatState, duration: number, isListening: boolean) => {
      try {
        await emit("float-state", { state, duration, isListening });
      } catch (e) {
        console.error("Failed to emit float state:", e);
      }
    },
    []
  );

  // Update float window when state changes
  useEffect(() => {
    const state: FloatState = showSuccess
      ? "success"
      : isRecording
      ? "recording"
      : isTranscribing
      ? "processing"
      : isListening
      ? (isSpeechDetected ? "ambient-active" : "ambient")
      : "ambient";
    emitFloatState(state, recordingDuration, isListening);

    // Auto-hide success after delay
    if (showSuccess) {
      const timer = setTimeout(() => {
        setShowSuccess(false);
        emitFloatState(isListening ? "ambient" : "ambient", 0, isListening);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [isRecording, isTranscribing, showSuccess, recordingDuration, isListening, isSpeechDetected, emitFloatState]);

  // Listen for toggle-listening requests from float window hover panel
  useEffect(() => {
    const unlisten = listen("request-toggle-listening", () => {
      toggleListening();
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [toggleListening]);

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

  // Listen for hotkey initialization events
  useEffect(() => {
    let unlistenError: (() => void) | null = null;
    let unlistenRegistered: (() => void) | null = null;

    const setupListeners = async () => {
      unlistenError = await listen<{ error: string }>("hotkey-init-error", (event) => {
        console.error("Hotkey init error:", event.payload.error);
        setHotkeyError(event.payload.error);
      });

      unlistenRegistered = await listen<{ hotkey: string }>("hotkey-registered", (event) => {
        console.log("Hotkey registered:", event.payload.hotkey);
        setHotkeyError(null);
        setHotkey(event.payload.hotkey);
      });
    };

    setupListeners();

    return () => {
      unlistenError?.();
      unlistenRegistered?.();
    };
  }, []);

  // Reload hotkey when settings panel closes
  useEffect(() => {
    if (!isSettingsOpen) {
      loadHotkey();
    }
  }, [isSettingsOpen]);

  const initializeEngine = async () => {
    setLoadingPhase("starting_engine");

    try {
      const isReady = await pingAsrEngine();
      if (!isReady) {
        console.log("ASR engine not ready, starting...");
        await startAsrEngine();
        console.log("ASR engine started successfully");
      }

      const status = await getModelStatus();
      setModelName(status.model_name || "mlx-community/Qwen3-ASR-0.6B-8bit");

      if (status.loaded) {
        setLoadingPhase("ready");
      } else {
        await loadModel();
      }

      // Load post-processor model if enabled in settings
      const settings = await getSettings();
      if (settings.postprocess?.enabled) {
        console.log("Post-processing enabled, loading LLM model...");
        try {
          await loadPostprocessModel();
          console.log("Post-processor model loaded successfully");
        } catch (e) {
          // Non-critical - log but don't fail app initialization
          console.warn("Failed to load post-processor model:", e);
        }
      }
    } catch (e) {
      console.error("Failed to initialize engine:", e);
      setLoadingPhase("error");
    }
  };

  const loadModel = async () => {
    try {
      // Check if model is cached to show appropriate loading message
      const cacheStatus = await isModelCached();
      if (cacheStatus.cached) {
        setLoadingPhase("loading_model");
        console.log("Model is cached, loading from local storage...");
      } else {
        setLoadingPhase("downloading_model");
        console.log("Model not cached, downloading...");
      }

      const status = await loadAsrModel();
      setModelName(status.model_name || "mlx-community/Qwen3-ASR-0.6B-8bit");

      // Warmup the model to trigger JIT compilation
      setLoadingPhase("warming_up");
      try {
        await warmupAsrModel();
        console.log("Model warmup complete");
      } catch (e) {
        // Warmup failure is non-critical, just log it
        console.warn("Model warmup failed (non-critical):", e);
      }

      setLoadingPhase("ready");
    } catch (e) {
      console.error("Failed to load model:", e);
      setLoadingPhase("error");
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

  // Show loading overlay during initial startup
  // Only check loadingPhase - engineStatus check was causing early dismissal
  const isInitializing = loadingPhase !== "ready" && loadingPhase !== "error";

  return (
    <div className="h-screen bg-surface-muted flex flex-col rounded-2xl overflow-hidden select-none card-shadow border border-subtle">
      {/* Loading Overlay - shown during initial startup */}
      {isInitializing && (
        <div className="absolute inset-0 z-50 bg-surface-muted flex flex-col items-center justify-center gap-6 rounded-2xl">
          {/* Logo */}
          <div className="flex flex-col items-center gap-4">
            <div
              className="w-16 h-16 rounded-2xl overflow-hidden animate-pulse"
              style={{ backgroundColor: "#7C9082" }}
            >
              <svg viewBox="0 0 64 64" className="w-full h-full">
                <ellipse
                  cx="32"
                  cy="32"
                  rx="24"
                  ry="24"
                  fill="none"
                  stroke="white"
                  strokeOpacity="0.25"
                  strokeWidth="2"
                />
                <ellipse
                  cx="32"
                  cy="32"
                  rx="17"
                  ry="17"
                  fill="none"
                  stroke="white"
                  strokeOpacity="0.44"
                  strokeWidth="2"
                />
                <ellipse
                  cx="32"
                  cy="32"
                  rx="10"
                  ry="10"
                  fill="none"
                  stroke="white"
                  strokeOpacity="0.63"
                  strokeWidth="2"
                />
                <ellipse cx="32" cy="32" rx="5" ry="5" fill="white" />
              </svg>
            </div>
            <span
              className="font-display text-2xl tracking-tight"
              style={{ color: "var(--text-primary)" }}
            >
              echo
            </span>
          </div>

          {/* Loading Status */}
          <div className="flex flex-col items-center gap-3">
            <div className="flex items-center gap-3">
              <div
                className="w-4 h-4 border-2 rounded-full animate-spin"
                style={{
                  borderColor: "var(--border-subtle)",
                  borderTopColor: "var(--glow-idle)",
                }}
              />
              <span
                className="text-sm font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                {LOADING_MESSAGES[loadingPhase]}
              </span>
            </div>
          </div>

          {/* Hint text */}
          <p
            className="text-xs text-center max-w-[200px]"
            style={{ color: "var(--text-tertiary)" }}
          >
            {loadingPhase === "starting_engine"
              ? "Initializing speech recognition..."
              : loadingPhase === "downloading_model"
              ? "First-time setup: downloading ~600MB"
              : loadingPhase === "loading_model"
              ? "Loading from local cache..."
              : loadingPhase === "warming_up"
              ? "Preparing for faster transcription..."
              : "Almost ready..."}
          </p>
        </div>
      )}

      {/* Header */}
      <header
        data-tauri-drag-region
        className="h-[52px] flex items-center justify-between px-4 flex-shrink-0 border-b"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        {/* Traffic Lights */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleClose}
            className="w-2.5 h-2.5 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all"
            aria-label="Close"
          />
          <button
            onClick={handleMinimize}
            className="w-2.5 h-2.5 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all"
            aria-label="Minimize"
          />
          <button
            onClick={handleMaximize}
            className="w-2.5 h-2.5 rounded-full bg-[#28C840] hover:brightness-90 transition-all"
            aria-label="Maximize"
          />
        </div>

        {/* Logo */}
        <div className="flex items-center gap-2 pointer-events-none" data-tauri-drag-region>
          <div className="w-6 h-6 rounded-[7px] overflow-hidden" style={{ backgroundColor: "#7C9082" }}>
            <svg viewBox="0 0 24 24" className="w-full h-full">
              <ellipse cx="12" cy="12" rx="9" ry="9" fill="none" stroke="white" strokeOpacity="0.25" strokeWidth="1"/>
              <ellipse cx="12" cy="12" rx="6.5" ry="6.5" fill="none" stroke="white" strokeOpacity="0.44" strokeWidth="1"/>
              <ellipse cx="12" cy="12" rx="4" ry="4" fill="none" stroke="white" strokeOpacity="0.63" strokeWidth="1"/>
              <ellipse cx="12" cy="12" rx="2.5" ry="2.5" fill="white"/>
            </svg>
          </div>
          <span
            className="font-display text-base tracking-tight"
            style={{ color: "var(--text-primary)" }}
          >
            echo
          </span>
        </div>

        {/* Settings Button */}
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="w-8 h-8 rounded-2xl flex items-center justify-center hover:bg-surface-elevated transition-colors border"
          style={{ backgroundColor: "var(--surface-muted)", borderColor: "#E0DDD8" }}
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
      <main className="flex-1 px-5 py-5 flex flex-col gap-6 overflow-auto">
        {/* Mode Cards */}
        <div className="flex gap-3">
          {/* Quick Input Card */}
          <div className="flex-1 flex flex-col gap-3 rounded-xl bg-surface p-4 border shadow-sm" style={{ borderColor: "#E8E4DF" }}>
            <div className="flex items-center justify-between">
              <svg className="w-4 h-4" fill="none" stroke="var(--text-secondary)" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 7.5l3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0021 18V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v12a2.25 2.25 0 002.25 2.25z" />
              </svg>
              <div
                className="h-5 px-1.5 rounded flex items-center bg-surface-muted border border-subtle"
              >
                <span
                  className="font-mono text-[10px] font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  {formatHotkey(hotkey)}
                </span>
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[13px] font-semibold" style={{ color: "var(--text-primary)" }}>
                Quick Input
              </span>
              <span className="text-[11px] leading-snug" style={{ color: "var(--text-tertiary)" }}>
                Hold to transcribe and insert into active app
              </span>
            </div>
          </div>

          {/* Always-on Card */}
          <div className="flex-1 flex flex-col gap-3 rounded-xl bg-surface p-4 border shadow-sm" style={{ borderColor: "#E8E4DF" }}>
            <div className="flex items-center justify-between">
              <svg className="w-4 h-4" fill="none" stroke="var(--text-secondary)" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
              <button
                onClick={toggleListening}
                className="w-10 h-[22px] rounded-full flex items-center transition-all duration-200"
                style={{
                  backgroundColor: isListening ? "var(--glow-idle)" : "var(--border-subtle)",
                  padding: "2px",
                }}
              >
                <div
                  className="w-[18px] h-[18px] rounded-full bg-white transition-transform duration-200"
                  style={{
                    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.15)",
                    transform: isListening ? "translateX(18px)" : "translateX(0)",
                  }}
                />
              </button>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[13px] font-semibold" style={{ color: "var(--text-primary)" }}>
                Always-on
              </span>
              <span className="text-[11px] leading-snug" style={{ color: "var(--text-tertiary)" }}>
                Continuously transcribe and save to log
              </span>
            </div>
          </div>
        </div>

        {/* Hotkey Error */}
        {hotkeyError && (
          <div
            className="px-3 py-2.5 rounded-lg text-xs flex flex-col gap-2.5"
            style={{
              backgroundColor: "rgba(198, 125, 99, 0.15)",
              color: "var(--glow-recording)",
            }}
          >
            <div className="flex items-start gap-2">
              <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
              </svg>
              <span>
                {showRestartPrompt
                  ? "Permission granted? Restart to apply."
                  : "Accessibility permission required for hotkey"}
              </span>
            </div>
            <div className="flex gap-2">
              {!showRestartPrompt ? (
                <button
                  onClick={async () => {
                    await requestAccessibilityPermission();
                    await openAccessibilitySettings();
                    setShowRestartPrompt(true);
                  }}
                  className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
                  style={{ backgroundColor: "var(--glow-recording)", color: "white" }}
                >
                  Open System Settings
                </button>
              ) : (
                <>
                  <button
                    onClick={() => restartApp()}
                    className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
                    style={{ backgroundColor: "var(--glow-idle)", color: "white" }}
                  >
                    Restart App
                  </button>
                  <button
                    onClick={() => openAccessibilitySettings()}
                    className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors border"
                    style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
                  >
                    Open Settings
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {/* Listening Error */}
        {listeningError && (
          <div
            className="px-3 py-2 rounded-lg text-xs"
            style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
          >
            {listeningError}
          </div>
        )}

        {/* Last Input Section */}
        <div className="flex flex-col gap-2">
          <span
            className="text-[11px] font-semibold tracking-wide"
            style={{ color: "var(--text-tertiary)" }}
          >
            Last input
          </span>

          <div className="flex flex-col rounded-[14px] bg-surface p-3.5 gap-3 shadow-sm border" style={{ borderColor: "#E8E4DF" }}>
            {error ? (
              <p className="text-sm" style={{ color: "var(--glow-recording)" }}>{error}</p>
            ) : result?.text ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                    {getDuration() ? `${getDuration()!.toFixed(1)}s` : ""}
                  </span>
                  {result.language && (
                    <span
                      className="font-mono text-[10px] capitalize px-1.5 py-0.5 rounded-md"
                      style={{ color: "var(--text-tertiary)", backgroundColor: "var(--surface-muted)" }}
                    >
                      {result.language}
                    </span>
                  )}
                </div>
                <p className="text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
                  {result.text}
                </p>
                <div className="flex items-center justify-end">
                  <button
                    onClick={handleCopy}
                    className="h-8 px-3.5 rounded-full text-xs font-medium flex items-center gap-1.5 transition-colors border hover:bg-surface-elevated"
                    style={{ color: "var(--text-primary)", borderColor: "#E0DDD8" }}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.75" />
                    </svg>
                    Copy
                  </button>
                </div>
              </>
            ) : (
              <p className="text-xs text-center py-2" style={{ color: "var(--text-tertiary)" }}>
                Press and hold the hotkey to transcribe
              </p>
            )}
          </div>
        </div>

        {/* Transcription History */}
        <TranscriptionHistory
          onSummarize={(minutes) => {
            setSummaryWindowMinutes(minutes);
            summarize(minutes);
          }}
          isSummarizing={isSummarizing}
        />
      </main>

      {/* Footer */}
      <footer className="h-10 flex items-center justify-between px-5 border-t flex-shrink-0" style={{ borderColor: "var(--border-subtle)" }}>
        {/* Model Info */}
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--text-tertiary)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
            <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
            <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
            <path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
            <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
            <path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
            <path d="M19.938 10.5a4 4 0 0 1 .585.396" />
            <path d="M6 18a4 4 0 0 1-1.967-.516" />
            <path d="M19.967 17.484A4 4 0 0 1 18 18" />
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
              loadingPhase !== "ready" && loadingPhase !== "error" && loadingPhase !== "idle"
                ? "animate-glow-pulse"
                : ""
            }`}
            style={{
              backgroundColor:
                loadingPhase === "ready"
                  ? "var(--glow-success)"
                  : loadingPhase === "error"
                  ? "var(--glow-recording)"
                  : loadingPhase !== "idle"
                  ? "var(--glow-processing)"
                  : "var(--text-tertiary)",
            }}
          />
          <span
            className="font-mono text-xs"
            style={{
              color:
                loadingPhase === "ready"
                  ? "var(--glow-success)"
                  : loadingPhase === "error"
                  ? "var(--glow-recording)"
                  : loadingPhase !== "idle"
                  ? "var(--glow-processing)"
                  : "var(--text-tertiary)",
            }}
          >
            {LOADING_MESSAGES[loadingPhase]}
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
            setLoadingPhase(status.loaded ? "ready" : "loading_model");
          } catch (e) {
            console.error("Failed to refresh model status:", e);
          }
        }}
      />

      {/* Summary Modal */}
      {(summary !== null || summaryError) && (
        <SummaryModal
          summary={summary || summaryError || ""}
          entryCount={entryCount}
          processingTime={processingTime}
          windowMinutes={summaryWindowMinutes}
          onClose={dismissSummary}
        />
      )}
    </div>
  );
}

export default App;
