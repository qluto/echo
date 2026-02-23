import { useEffect, useState, useRef, useCallback } from "react";
import {
  getSettings,
  updateSettings,
  getAudioDevices,
  registerHotkey,
  getModelStatus,
  setAsrModel,
  loadAsrModelAsync,
  isModelCached,
  startHotkeyRecording,
  stopHotkeyRecording,
  onHandyKeysEvent,
  onModelLoadComplete,
  onModelLoadError,
  updatePostprocessSettings,
  loadPostprocessModel,
  isPostprocessModelCached,
  setPostprocessModel,
  getPostprocessModelStatus,
  clearTranscriptionHistory,
  AppSettings,
  AudioDevice,
  HandyKeysEvent,
  PostProcessSettings,
} from "../lib/tauri";
import { MODEL_ORDER, SUPPORTED_LANGUAGES, getModelDisplayName, getModelSize } from "../lib/models";
import { formatHotkey } from "../lib/format";
import { DEFAULT_POSTPROCESS_PROMPT, DEFAULT_SUMMARIZE_PROMPT } from "../lib/prompts";

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [settings, setSettings] = useState<AppSettings>({
    hotkey: "CommandOrControl+Shift+Space",
    language: "auto",
    auto_insert: true,
    device_name: null,
    model_name: null,
    postprocess: { enabled: false, dictionary: {} },
  });
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [modelName, setModelName] = useState<string>("mlx-community/Qwen3-ASR-0.6B-8bit");
  const [availableModels, setAvailableModels] = useState<string[]>(MODEL_ORDER);
  const [isLoading, setIsLoading] = useState(true);
  const [isModelChanging, setIsModelChanging] = useState(false);
  const [modelChangePhase, setModelChangePhase] = useState<"idle" | "switching" | "downloading" | "loading">("idle");
  const [hotkeyInput, setHotkeyInput] = useState("");
  const [isRecordingHotkey, setIsRecordingHotkey] = useState(false);
  const [currentKeys, setCurrentKeys] = useState("");
  const [hotkeyError, setHotkeyError] = useState<string | null>(null);
  const [isPostprocessLoading, setIsPostprocessLoading] = useState(false);
  const [postprocessLoadPhase, setPostprocessLoadPhase] = useState<"idle" | "checking" | "downloading" | "loading">("idle");
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [customPrompt, setCustomPrompt] = useState<string | null>(null);
  const [customSummaryPrompt, setCustomSummaryPrompt] = useState<string | null>(null);
  const [postprocessModelName, setPostprocessModelName] = useState<string>("mlx-community/Qwen3-4B-4bit");
  const [availablePostprocessModels, setAvailablePostprocessModels] = useState<string[]>([
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
  ]);
  const [isPostprocessModelChanging, setIsPostprocessModelChanging] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const currentKeysRef = useRef("");
  const unlistenRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen]);

  const loadSettings = async () => {
    setIsLoading(true);
    try {
      const [loadedSettings, loadedDevices, modelStatus, postprocessStatus] = await Promise.all([
        getSettings(),
        getAudioDevices(),
        getModelStatus(),
        getPostprocessModelStatus().catch(() => null),
      ]);
      setSettings(loadedSettings);
      setDevices(loadedDevices);
      setHotkeyInput(loadedSettings.hotkey);
      setCustomPrompt(loadedSettings.postprocess?.custom_prompt ?? null);
      setCustomSummaryPrompt(loadedSettings.postprocess?.custom_summary_prompt ?? null);
      if (modelStatus.model_name) {
        setModelName(modelStatus.model_name);
      }
      if (modelStatus.available_models && modelStatus.available_models.length > 0) {
        // Sort by our preferred order
        const sortedModels = [...modelStatus.available_models].sort((a, b) => {
          const aIndex = MODEL_ORDER.indexOf(a);
          const bIndex = MODEL_ORDER.indexOf(b);
          if (aIndex === -1 && bIndex === -1) return 0;
          if (aIndex === -1) return 1;
          if (bIndex === -1) return -1;
          return aIndex - bIndex;
        });
        setAvailableModels(sortedModels);
      }
      // Load post-process model status
      if (postprocessStatus) {
        if (postprocessStatus.model_name) {
          setPostprocessModelName(postprocessStatus.model_name);
        }
        if (postprocessStatus.available_models && postprocessStatus.available_models.length > 0) {
          setAvailablePostprocessModels(postprocessStatus.available_models);
        }
      }
      // Override with saved settings if available
      if (loadedSettings.postprocess?.model_name) {
        setPostprocessModelName(loadedSettings.postprocess.model_name);
      }
    } catch (e) {
      console.error("Failed to load settings:", e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = async (newModel: string) => {
    if (newModel === modelName || isModelChanging) return;

    setIsModelChanging(true);
    setModelChangePhase("switching");
    try {
      // Set the new model (this unloads the current one)
      await setAsrModel(newModel);
      setModelName(newModel);

      // Check if model is cached to show appropriate status
      const cacheStatus = await isModelCached(newModel);
      if (cacheStatus.cached) {
        setModelChangePhase("loading");
        console.log("Model is cached, loading from local storage...");
      } else {
        setModelChangePhase("downloading");
        console.log("Model not cached, downloading...");
      }

      // Start loading in background - completion handled by event listener
      await loadAsrModelAsync();
    } catch (e) {
      console.error("Failed to start model change:", e);
      setIsModelChanging(false);
      setModelChangePhase("idle");
    }
    // Note: Don't reset state here - event listener will handle completion
  };

  const handleSettingChange = async <K extends keyof AppSettings>(
    key: K,
    value: AppSettings[K]
  ) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    try {
      await updateSettings(newSettings);
    } catch (e) {
      console.error("Failed to save settings:", e);
    }
  };

  const handlePostprocessToggle = async () => {
    if (isPostprocessLoading) return;

    const newEnabled = !settings.postprocess.enabled;

    if (newEnabled) {
      // Turning ON - need to check/download model
      setIsPostprocessLoading(true);
      setPostprocessLoadPhase("checking");

      try {
        // Check if model is cached
        const cacheStatus = await isPostprocessModelCached();

        if (cacheStatus.loaded) {
          // Model is already loaded/cached
          setPostprocessLoadPhase("idle");
          setIsPostprocessLoading(false);
        } else {
          // Need to download/load model
          setPostprocessLoadPhase("downloading");

          // Start loading the model
          const result = await loadPostprocessModel();

          if (!result.loaded) {
            console.error("Failed to load postprocess model:", result.error);
            setIsPostprocessLoading(false);
            setPostprocessLoadPhase("idle");
            return;
          }
        }

        // Enable postprocessing
        const newPostprocess: PostProcessSettings = {
          ...settings.postprocess,
          enabled: true,
        };
        const newSettings = { ...settings, postprocess: newPostprocess };
        setSettings(newSettings);
        await updatePostprocessSettings(newPostprocess);
        setIsPostprocessLoading(false);
        setPostprocessLoadPhase("idle");
      } catch (e) {
        console.error("Failed to enable postprocessing:", e);
        setIsPostprocessLoading(false);
        setPostprocessLoadPhase("idle");
      }
    } else {
      // Turning OFF - just disable
      const newPostprocess: PostProcessSettings = {
        ...settings.postprocess,
        enabled: false,
      };
      const newSettings = { ...settings, postprocess: newPostprocess };
      setSettings(newSettings);
      try {
        await updatePostprocessSettings(newPostprocess);
      } catch (e) {
        console.error("Failed to disable postprocessing:", e);
      }
    }
  };

  const handlePostprocessModelChange = async (newModel: string) => {
    if (newModel === postprocessModelName || isPostprocessModelChanging) return;

    setIsPostprocessModelChanging(true);
    try {
      // Set the new model
      await setPostprocessModel(newModel);
      setPostprocessModelName(newModel);

      // If postprocessing is enabled, reload the model
      if (settings.postprocess.enabled) {
        setPostprocessLoadPhase("downloading");
        await loadPostprocessModel();
        setPostprocessLoadPhase("idle");
      }
    } catch (e) {
      console.error("Failed to change postprocess model:", e);
    } finally {
      setIsPostprocessModelChanging(false);
    }
  };

  // Listen for model load completion/error events
  useEffect(() => {
    let mounted = true;

    const setupListeners = async () => {
      const unlistenComplete = await onModelLoadComplete(() => {
        if (!mounted) return;
        console.log("Model load complete");
        setIsModelChanging(false);
        setModelChangePhase("idle");
      });

      const unlistenError = await onModelLoadError((event) => {
        if (!mounted) return;
        console.error("Model load error:", event.error);
        setIsModelChanging(false);
        setModelChangePhase("idle");
      });

      return () => {
        unlistenComplete();
        unlistenError();
      };
    };

    const cleanupPromise = setupListeners();

    return () => {
      mounted = false;
      cleanupPromise.then((cleanup) => cleanup?.());
    };
  }, []);

  // Cancel recording mode
  const cancelRecording = useCallback(async () => {
    if (!isRecordingHotkey) return;

    // Stop listening for backend events
    if (unlistenRef.current) {
      unlistenRef.current();
      unlistenRef.current = null;
    }

    // Stop backend recording
    await stopHotkeyRecording().catch(console.error);

    setIsRecordingHotkey(false);
    setCurrentKeys("");
    currentKeysRef.current = "";
  }, [isRecordingHotkey]);

  // Set up event listener for handy-keys events
  useEffect(() => {
    if (!isRecordingHotkey) return;

    let cleanup = false;

    const setupListener = async () => {
      const unlisten = await onHandyKeysEvent(async (event: HandyKeysEvent) => {
        if (cleanup) return;

        const { hotkey_string, is_key_down } = event;

        if (is_key_down && hotkey_string) {
          // Update both state (for display) and ref (for release handler)
          currentKeysRef.current = hotkey_string;
          setCurrentKeys(hotkey_string);
        } else if (!is_key_down && currentKeysRef.current) {
          // Key released - commit the shortcut
          const newHotkey = currentKeysRef.current;

          // Stop recording first
          if (unlistenRef.current) {
            unlistenRef.current();
            unlistenRef.current = null;
          }
          await stopHotkeyRecording().catch(console.error);
          setIsRecordingHotkey(false);
          setCurrentKeys("");
          currentKeysRef.current = "";

          try {
            setHotkeyError(null);
            await registerHotkey(newHotkey);
            await handleSettingChange("hotkey", newHotkey);
            setHotkeyInput(newHotkey);
          } catch (err) {
            console.error("Failed to register hotkey:", err);
            // Show error message
            const errorMsg = err instanceof Error ? err.message : String(err);
            setHotkeyError(errorMsg);
            // Revert to previous hotkey and re-register it
            setHotkeyInput(settings.hotkey);
            try {
              await registerHotkey(settings.hotkey);
            } catch {
              // Ignore re-registration error
            }
          }
        }
      });

      unlistenRef.current = unlisten;
    };

    setupListener();

    return () => {
      cleanup = true;
      if (unlistenRef.current) {
        unlistenRef.current();
        unlistenRef.current = null;
      }
      stopHotkeyRecording().catch(console.error);
    };
  }, [isRecordingHotkey, settings.hotkey]);

  // Handle click outside to cancel recording
  useEffect(() => {
    if (!isRecordingHotkey) return;

    const handleClickOutside = () => {
      cancelRecording();
    };

    // Delay adding the listener to avoid immediate trigger
    const timer = setTimeout(() => {
      window.addEventListener("click", handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timer);
      window.removeEventListener("click", handleClickOutside);
    };
  }, [isRecordingHotkey, cancelRecording]);

  // Start recording mode
  const startRecording = async () => {
    if (isRecordingHotkey) return;

    try {
      await startHotkeyRecording();
      setIsRecordingHotkey(true);
      setCurrentKeys("");
      currentKeysRef.current = "";
    } catch (err) {
      console.error("Failed to start recording:", err);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex rounded-2xl overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30 rounded-2xl"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className="relative ml-auto h-full w-[380px] bg-surface-muted flex flex-col overflow-hidden animate-float-in card-shadow rounded-2xl"
      >
        {/* Header */}
        <header
          data-tauri-drag-region
          className="h-[52px] flex items-center gap-3 px-5 flex-shrink-0"
        >
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-surface-elevated transition-colors"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="var(--text-secondary)"
              strokeWidth={2}
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18"
              />
            </svg>
          </button>
          <span
            className="font-display text-lg font-semibold pointer-events-none"
            style={{ color: "var(--text-primary)" }}
          >
            Settings
          </span>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto px-5 pb-5 flex flex-col gap-6">
          {isLoading ? (
            <div className="flex-1 flex items-center justify-center">
              <div
                className="w-6 h-6 border-2 rounded-full animate-spin"
                style={{
                  borderColor: "var(--border-subtle)",
                  borderTopColor: "var(--glow-idle)",
                }}
              />
            </div>
          ) : (
            <>
              {/* Model Section */}
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: "rgba(124, 144, 112, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--glow-idle)"
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
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Model
                  </span>
                </div>

                {/* ASR Model Field */}
                <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                  <span className="text-sm flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
                    ASR Model
                  </span>
                  <div className="flex items-center gap-2 w-[160px] justify-end">
                    {isModelChanging ? (
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 border-2 rounded-full animate-spin"
                          style={{
                            borderColor: "var(--border-subtle)",
                            borderTopColor: "var(--glow-idle)",
                          }}
                        />
                        <span
                          className="font-mono text-xs"
                          style={{ color: "var(--text-tertiary)" }}
                        >
                          {modelChangePhase === "switching"
                            ? "Switching..."
                            : modelChangePhase === "downloading"
                            ? "Downloading..."
                            : "Loading..."}
                        </span>
                      </div>
                    ) : (
                      <>
                        <select
                          value={modelName}
                          onChange={(e) => handleModelChange(e.target.value)}
                          className="bg-transparent font-mono text-xs appearance-none cursor-pointer focus:outline-none w-full"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {availableModels.map((model) => (
                            <option
                              key={model}
                              value={model}
                              className="bg-surface"
                            >
                              {getModelDisplayName(model)} {getModelSize(model)}
                            </option>
                          ))}
                        </select>
                        <svg
                          className="w-4 h-4 flex-shrink-0"
                          fill="none"
                          stroke="var(--text-tertiary)"
                          strokeWidth={2}
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M19.5 8.25l-7.5 7.5-7.5-7.5"
                          />
                        </svg>
                      </>
                    )}
                  </div>
                </div>

                {/* Language Field */}
                <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                  <span className="text-sm flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
                    Language
                  </span>
                  <div className="flex items-center gap-2 w-[160px] justify-end">
                    <select
                      value={settings.language}
                      onChange={(e) => handleSettingChange("language", e.target.value)}
                      className="bg-transparent font-mono text-xs appearance-none cursor-pointer focus:outline-none w-full"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {SUPPORTED_LANGUAGES.map((lang) => (
                        <option
                          key={lang.code}
                          value={lang.code}
                          className="bg-surface"
                        >
                          {lang.name}
                        </option>
                      ))}
                    </select>
                    <svg
                      className="w-4 h-4 flex-shrink-0"
                      fill="none"
                      stroke="var(--text-tertiary)"
                      strokeWidth={2}
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M19.5 8.25l-7.5 7.5-7.5-7.5"
                      />
                    </svg>
                  </div>
                </div>
              </section>

              {/* Input Section */}
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: "rgba(198, 125, 99, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="var(--glow-recording)"
                      viewBox="0 0 24 24"
                    >
                      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
                    </svg>
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Input
                  </span>
                </div>

                {/* Microphone Field */}
                <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                  <span className="text-sm flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
                    Microphone
                  </span>
                  <div className="flex items-center gap-2 w-[160px] justify-end">
                    <select
                      value={settings.device_name || ""}
                      onChange={(e) =>
                        handleSettingChange("device_name", e.target.value || null)
                      }
                      className="bg-transparent font-mono text-xs appearance-none cursor-pointer focus:outline-none w-full truncate"
                      style={{ color: "var(--text-primary)" }}
                    >
                      <option value="" className="bg-surface">
                        Default
                      </option>
                      {devices.map((device) => (
                        <option
                          key={device.name}
                          value={device.name}
                          className="bg-surface"
                        >
                          {device.name}
                        </option>
                      ))}
                    </select>
                    <svg
                      className="w-4 h-4 flex-shrink-0"
                      fill="none"
                      stroke="var(--text-tertiary)"
                      strokeWidth={2}
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M19.5 8.25l-7.5 7.5-7.5-7.5"
                      />
                    </svg>
                  </div>
                </div>

                {/* Hotkey Field */}
                <div className="flex flex-col gap-2">
                  <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                    <span className="text-sm flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
                      Hotkey
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!isRecordingHotkey) {
                          setHotkeyError(null);
                          startRecording();
                        }
                      }}
                      className={`h-7 px-3 rounded-lg flex items-center bg-surface-muted border transition-colors ${
                        isRecordingHotkey ? "border-glow-idle" : "border-subtle"
                      }`}
                      style={{
                        borderColor: isRecordingHotkey
                          ? "var(--glow-idle)"
                          : "var(--border-subtle)",
                      }}
                    >
                      <span
                        className="font-mono text-xs"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {isRecordingHotkey
                          ? currentKeys
                            ? formatHotkey(currentKeys)
                            : "Press keys..."
                          : formatHotkey(hotkeyInput)}
                      </span>
                    </button>
                  </div>
                  {hotkeyError && (
                    <div
                      className="px-4 py-2 rounded-lg text-xs"
                      style={{
                        backgroundColor: "rgba(198, 125, 99, 0.15)",
                        color: "var(--glow-recording)",
                      }}
                    >
                      {hotkeyError}
                    </div>
                  )}
                </div>
              </section>

              {/* Behavior Section */}
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: "rgba(124, 144, 112, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--glow-idle)"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />
                      <path d="M20 3v4" />
                      <path d="M22 5h-4" />
                      <path d="M4 17v2" />
                      <path d="M5 18H3" />
                    </svg>
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Behavior
                  </span>
                </div>

                {/* Auto-insert Toggle */}
                <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    Auto-insert text
                  </span>
                  <button
                    onClick={() =>
                      handleSettingChange("auto_insert", !settings.auto_insert)
                    }
                    className="w-12 h-7 rounded-full flex items-center transition-all duration-200"
                    style={{
                      backgroundColor: settings.auto_insert
                        ? "var(--glow-idle)"
                        : "var(--border-subtle)",
                      padding: "2px",
                    }}
                  >
                    <div
                      className="w-6 h-6 rounded-full bg-white transition-transform duration-200"
                      style={{
                        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.15)",
                        transform: settings.auto_insert ? "translateX(20px)" : "translateX(0)",
                      }}
                    />
                  </button>
                </div>
              </section>

              {/* Post-processing Section */}
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: "rgba(147, 112, 219, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="#9370DB"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M12 3v3m6.366-.366-2.12 2.12M21 12h-3m.366 6.366-2.12-2.12M12 21v-3m-6.366.366 2.12-2.12M3 12h3m-.366-6.366 2.12 2.12" />
                    </svg>
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Post-processing
                  </span>
                </div>

                {/* Post-process Toggle */}
                <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                      AI Enhancement
                    </span>
                    <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                      Filler removal, formatting
                    </span>
                  </div>
                  {isPostprocessLoading ? (
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 border-2 rounded-full animate-spin"
                        style={{
                          borderColor: "var(--border-subtle)",
                          borderTopColor: "#9370DB",
                        }}
                      />
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-tertiary)" }}
                      >
                        {postprocessLoadPhase === "checking"
                          ? "Checking..."
                          : postprocessLoadPhase === "downloading"
                          ? "Downloading..."
                          : "Loading..."}
                      </span>
                    </div>
                  ) : (
                    <button
                      onClick={handlePostprocessToggle}
                      className="w-12 h-7 rounded-full flex items-center transition-all duration-200"
                      style={{
                        backgroundColor: settings.postprocess.enabled
                          ? "#9370DB"
                          : "var(--border-subtle)",
                        padding: "2px",
                      }}
                    >
                      <div
                        className="w-6 h-6 rounded-full bg-white transition-transform duration-200"
                        style={{
                          boxShadow: "0 1px 3px rgba(0, 0, 0, 0.15)",
                          transform: settings.postprocess.enabled ? "translateX(20px)" : "translateX(0)",
                        }}
                      />
                    </button>
                  )}
                </div>

                {/* Model selector when enabled */}
                {settings.postprocess.enabled && (
                  <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                    <span className="text-sm flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
                      LLM Model
                    </span>
                    <div className="flex items-center gap-2 w-[140px] justify-end">
                      {isPostprocessModelChanging ? (
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 border-2 rounded-full animate-spin"
                            style={{
                              borderColor: "var(--border-subtle)",
                              borderTopColor: "#9370DB",
                            }}
                          />
                          <span
                            className="font-mono text-xs"
                            style={{ color: "var(--text-tertiary)" }}
                          >
                            Switching...
                          </span>
                        </div>
                      ) : (
                        <>
                          <select
                            value={postprocessModelName}
                            onChange={(e) => handlePostprocessModelChange(e.target.value)}
                            className="bg-transparent font-mono text-xs appearance-none cursor-pointer focus:outline-none w-full"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {availablePostprocessModels.map((model) => {
                              const displayName = model.includes("8B") ? "Qwen3 8B" : model.includes("4B") ? "Qwen3 4B" : "Qwen3 1.7B";
                              const memInfo = model.includes("8B") ? "~5GB" : model.includes("4B") ? "~2.5GB" : "~1.3GB";
                              return (
                                <option
                                  key={model}
                                  value={model}
                                  className="bg-surface"
                                >
                                  {displayName} ({memInfo})
                                </option>
                              );
                            })}
                          </select>
                          <svg
                            className="w-4 h-4 flex-shrink-0"
                            fill="none"
                            stroke="var(--text-tertiary)"
                            strokeWidth={2}
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M19.5 8.25l-7.5 7.5-7.5-7.5"
                            />
                          </svg>
                        </>
                      )}
                    </div>
                  </div>
                )}

                {/* Info text when enabled */}
                {settings.postprocess.enabled && (
                  <div
                    className="px-4 py-2 rounded-lg text-xs"
                    style={{
                      backgroundColor: "rgba(147, 112, 219, 0.1)",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    Removes fillers, applies corrections, and formats text based on the target app.
                  </div>
                )}

                {/* Advanced Section - Collapsible */}
                {settings.postprocess.enabled && (
                  <div className="flex flex-col gap-2">
                    <button
                      onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs transition-colors hover:bg-surface-elevated"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      <svg
                        className={`w-3 h-3 transition-transform ${isAdvancedOpen ? "rotate-90" : ""}`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
                          clipRule="evenodd"
                        />
                      </svg>
                      Advanced Settings
                    </button>

                    {isAdvancedOpen && (
                      <div className="flex flex-col gap-3 px-2">
                        <div
                          className="px-3 py-2 rounded-lg text-xs flex items-center gap-2"
                          style={{
                            backgroundColor: "rgba(212, 165, 116, 0.15)",
                            color: "var(--glow-processing)",
                          }}
                        >
                          <svg
                            className="w-4 h-4 flex-shrink-0"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
                              clipRule="evenodd"
                            />
                          </svg>
                          <span>
                            Modifying the prompt may cause unexpected behavior.
                          </span>
                        </div>

                        <div className="flex flex-col gap-2">
                          <div className="flex items-center justify-between">
                            <span
                              className="text-xs font-medium"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              Custom System Prompt
                            </span>
                            <button
                              onClick={() => {
                                setCustomPrompt(null);
                                const newPostprocess: PostProcessSettings = {
                                  ...settings.postprocess,
                                  custom_prompt: null,
                                };
                                const newSettings = { ...settings, postprocess: newPostprocess };
                                setSettings(newSettings);
                                updatePostprocessSettings(newPostprocess).catch(console.error);
                              }}
                              className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                              style={{ color: "var(--text-tertiary)" }}
                            >
                              Reset to Default
                            </button>
                          </div>
                          <textarea
                            value={customPrompt === null ? DEFAULT_POSTPROCESS_PROMPT : customPrompt}
                            onChange={(e) => {
                              const value = e.target.value;
                              // Always set to the actual value (including empty string)
                              setCustomPrompt(value);
                              // Save: use null only if it exactly matches default
                              const customValue = value === DEFAULT_POSTPROCESS_PROMPT ? null : value;
                              const newPostprocess: PostProcessSettings = {
                                ...settings.postprocess,
                                custom_prompt: customValue,
                              };
                              const newSettings = { ...settings, postprocess: newPostprocess };
                              setSettings(newSettings);
                              updatePostprocessSettings(newPostprocess).catch(console.error);
                            }}
                            className="w-full h-48 px-3 py-2 rounded-lg bg-surface border border-subtle text-xs font-mono resize-none focus:outline-none focus:border-glow-idle"
                            style={{ color: "var(--text-primary)" }}
                            placeholder={DEFAULT_POSTPROCESS_PROMPT}
                          />
                          <span
                            className="text-xs"
                            style={{ color: "var(--text-tertiary)" }}
                          >
                            {customPrompt !== null
                              ? "Using custom prompt"
                              : "Using default prompt"}
                          </span>
                        </div>

                        {/* Custom Summary Prompt */}
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center justify-between">
                            <span
                              className="text-xs font-medium"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              Summary Prompt
                            </span>
                            <button
                              onClick={() => {
                                setCustomSummaryPrompt(null);
                                const newPostprocess: PostProcessSettings = {
                                  ...settings.postprocess,
                                  custom_summary_prompt: null,
                                };
                                const newSettings = { ...settings, postprocess: newPostprocess };
                                setSettings(newSettings);
                                updatePostprocessSettings(newPostprocess).catch(console.error);
                              }}
                              className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                              style={{ color: "var(--text-tertiary)" }}
                            >
                              Reset to Default
                            </button>
                          </div>
                          <textarea
                            value={customSummaryPrompt === null ? DEFAULT_SUMMARIZE_PROMPT : customSummaryPrompt}
                            onChange={(e) => {
                              const value = e.target.value;
                              setCustomSummaryPrompt(value);
                              const customValue = value === DEFAULT_SUMMARIZE_PROMPT ? null : value;
                              const newPostprocess: PostProcessSettings = {
                                ...settings.postprocess,
                                custom_summary_prompt: customValue,
                              };
                              const newSettings = { ...settings, postprocess: newPostprocess };
                              setSettings(newSettings);
                              updatePostprocessSettings(newPostprocess).catch(console.error);
                            }}
                            className="w-full h-48 px-3 py-2 rounded-lg bg-surface border border-subtle text-xs font-mono resize-none focus:outline-none focus:border-glow-idle"
                            style={{ color: "var(--text-primary)" }}
                            placeholder={DEFAULT_SUMMARIZE_PROMPT}
                          />
                          <span
                            className="text-xs"
                            style={{ color: "var(--text-tertiary)" }}
                          >
                            {customSummaryPrompt !== null
                              ? "Using custom prompt"
                              : "Using default prompt"}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </section>

              {/* Data Section */}
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: "rgba(198, 125, 99, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--glow-recording)"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M3 6h18" />
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                      <line x1="10" y1="11" x2="10" y2="17" />
                      <line x1="14" y1="11" x2="14" y2="17" />
                    </svg>
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Data
                  </span>
                </div>

                <div className="flex flex-col gap-2">
                  {!showClearConfirm ? (
                    <button
                      onClick={() => setShowClearConfirm(true)}
                      className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between hover:bg-surface-elevated transition-colors"
                    >
                      <span className="text-sm" style={{ color: "var(--glow-recording)" }}>
                        Clear All History
                      </span>
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="var(--text-tertiary)"
                        strokeWidth={2}
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                      </svg>
                    </button>
                  ) : (
                    <div
                      className="px-4 py-3 rounded-xl border flex flex-col gap-3"
                      style={{
                        backgroundColor: "rgba(198, 125, 99, 0.08)",
                        borderColor: "rgba(198, 125, 99, 0.3)",
                      }}
                    >
                      <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                        This will permanently delete all transcription history. This action cannot be undone.
                      </p>
                      <div className="flex items-center gap-2 justify-end">
                        <button
                          onClick={() => setShowClearConfirm(false)}
                          className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors hover:bg-surface-elevated"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Cancel
                        </button>
                        <button
                          onClick={async () => {
                            try {
                              await clearTranscriptionHistory();
                              setShowClearConfirm(false);
                            } catch (e) {
                              console.error("Failed to clear history:", e);
                            }
                          }}
                          className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors text-white"
                          style={{ backgroundColor: "var(--glow-recording)" }}
                        >
                          Delete All
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
