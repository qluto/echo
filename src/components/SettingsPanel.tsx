import { useEffect, useState, useRef, useCallback } from "react";
import {
  getSettings,
  updateSettings,
  getAudioDevices,
  registerHotkey,
  getModelStatus,
  setAsrModel,
  loadAsrModel,
  startHotkeyRecording,
  stopHotkeyRecording,
  onHandyKeysEvent,
  AppSettings,
  AudioDevice,
  HandyKeysEvent,
} from "../lib/tauri";

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const SUPPORTED_LANGUAGES = [
  { code: "auto", name: "Auto-detect" },
  { code: "ja", name: "Japanese" },
  { code: "en", name: "English" },
  { code: "zh", name: "Chinese" },
  { code: "ko", name: "Korean" },
  { code: "de", name: "German" },
  { code: "fr", name: "French" },
  { code: "es", name: "Spanish" },
];

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

// Model display order (for UI)
const MODEL_ORDER = [
  "mlx-community/Qwen3-ASR-1.7B-8bit",
  "mlx-community/Qwen3-ASR-0.6B-8bit",
  "mlx-community/whisper-large-v3-turbo",
  "mlx-community/whisper-large-v3",
  "mlx-community/whisper-medium",
  "mlx-community/whisper-small",
  "mlx-community/whisper-base",
  "mlx-community/whisper-tiny",
];

const getModelDisplayName = (name: string): string => {
  const parts = name.split("/");
  const modelPart = parts[parts.length - 1];

  // Handle Qwen3-ASR models
  if (modelPart.includes("Qwen3-ASR")) {
    return modelPart
      .replace("-8bit", "")
      .replace("Qwen3-ASR-", "Qwen3-ASR ");
  }

  // Handle Whisper models
  return modelPart
    .replace("whisper-", "Whisper ")
    .split("-")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
};

const getModelSize = (name: string): string => {
  return MODEL_SIZES[name] || "unknown";
};

export function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [settings, setSettings] = useState<AppSettings>({
    hotkey: "CommandOrControl+Shift+Space",
    language: "auto",
    auto_insert: true,
    device_name: null,
    model_name: null,
  });
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [modelName, setModelName] = useState<string>("mlx-community/whisper-large-v3-turbo");
  const [availableModels, setAvailableModels] = useState<string[]>(MODEL_ORDER);
  const [isLoading, setIsLoading] = useState(true);
  const [isModelChanging, setIsModelChanging] = useState(false);
  const [hotkeyInput, setHotkeyInput] = useState("");
  const [isRecordingHotkey, setIsRecordingHotkey] = useState(false);
  const [currentKeys, setCurrentKeys] = useState("");
  const [hotkeyError, setHotkeyError] = useState<string | null>(null);
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
      const [loadedSettings, loadedDevices, modelStatus] = await Promise.all([
        getSettings(),
        getAudioDevices(),
        getModelStatus(),
      ]);
      setSettings(loadedSettings);
      setDevices(loadedDevices);
      setHotkeyInput(loadedSettings.hotkey);
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
    } catch (e) {
      console.error("Failed to load settings:", e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = async (newModel: string) => {
    if (newModel === modelName || isModelChanging) return;

    setIsModelChanging(true);
    try {
      // Set the new model (this unloads the current one)
      await setAsrModel(newModel);
      setModelName(newModel);

      // Load the new model
      await loadAsrModel();
    } catch (e) {
      console.error("Failed to change model:", e);
    } finally {
      setIsModelChanging(false);
    }
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

  const formatHotkey = (hotkey: string): string => {
    // handy-keys uses lowercase with + separator
    let formatted = hotkey
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
      // Legacy format support
      .replace("CommandOrControl", "⌘")
      .replace(/\+/g, "");
    return formatted;
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
                          Loading...
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

              {/* About Section */}
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: "rgba(212, 165, 116, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="var(--glow-processing)"
                      viewBox="0 0 24 24"
                    >
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
                    </svg>
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    About
                  </span>
                </div>

                <div className="rounded-xl bg-surface border border-subtle overflow-hidden">
                  <div className="h-12 px-4 flex items-center justify-between border-b border-subtle">
                    <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                      Version
                    </span>
                    <span
                      className="font-mono text-xs"
                      style={{ color: "var(--text-primary)" }}
                    >
                      0.1.0
                    </span>
                  </div>
                  <div className="h-12 px-4 flex items-center justify-between">
                    <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                      Build
                    </span>
                    <span
                      className="font-mono text-xs"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      2026.01.30
                    </span>
                  </div>
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
