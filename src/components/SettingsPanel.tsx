import { useEffect, useState } from "react";
import {
  getSettings,
  updateSettings,
  getAudioDevices,
  registerHotkey,
  getModelStatus,
  setAsrModel,
  loadAsrModel,
  AppSettings,
  AudioDevice,
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

  const handleHotkeyRecord = (e: React.KeyboardEvent) => {
    if (!isRecordingHotkey) return;

    e.preventDefault();
    const keys: string[] = [];

    if (e.metaKey || e.ctrlKey) keys.push("CommandOrControl");
    if (e.altKey) keys.push("Alt");
    if (e.shiftKey) keys.push("Shift");

    if (e.key && !["Control", "Alt", "Shift", "Meta"].includes(e.key)) {
      keys.push(e.key.length === 1 ? e.key.toUpperCase() : e.key);
    }

    if (keys.length > 1) {
      const newHotkey = keys.join("+");
      setHotkeyInput(newHotkey);
      setIsRecordingHotkey(false);
      registerHotkey(newHotkey).then(() => {
        handleSettingChange("hotkey", newHotkey);
      });
    }
  };

  const formatHotkey = (hotkey: string): string => {
    return hotkey
      .replace("CommandOrControl", "⌘")
      .replace("Shift", "⇧")
      .replace("Alt", "⌥")
      .replace(/\+/g, "");
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className="relative ml-auto h-full w-[380px] bg-surface-muted flex flex-col overflow-hidden animate-float-in card-shadow"
      >
        {/* Header */}
        <header className="h-[52px] flex items-center gap-3 px-5 flex-shrink-0">
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
            className="font-display text-lg font-semibold"
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
                    style={{ backgroundColor: "rgba(124, 144, 130, 0.12)" }}
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="var(--glow-idle)"
                      viewBox="0 0 24 24"
                    >
                      <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z" />
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
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    ASR Model
                  </span>
                  <div className="flex items-center gap-2">
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
                          className="bg-transparent font-mono text-xs text-right appearance-none cursor-pointer focus:outline-none max-w-[160px]"
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
                          className="w-4 h-4"
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
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    Language
                  </span>
                  <div className="flex items-center gap-2">
                    <select
                      value={settings.language}
                      onChange={(e) => handleSettingChange("language", e.target.value)}
                      className="bg-transparent font-mono text-xs text-right appearance-none cursor-pointer focus:outline-none"
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
                      className="w-4 h-4"
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
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    Microphone
                  </span>
                  <div className="flex items-center gap-2">
                    <select
                      value={settings.device_name || ""}
                      onChange={(e) =>
                        handleSettingChange("device_name", e.target.value || null)
                      }
                      className="bg-transparent font-mono text-xs text-right appearance-none cursor-pointer focus:outline-none max-w-[140px] truncate"
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
                      className="w-4 h-4"
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
                <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    Hotkey
                  </span>
                  <button
                    onClick={() => setIsRecordingHotkey(true)}
                    onKeyDown={handleHotkeyRecord}
                    onBlur={() => setIsRecordingHotkey(false)}
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
                      {isRecordingHotkey ? "Press keys..." : formatHotkey(hotkeyInput)}
                    </span>
                  </button>
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
                      fill="var(--glow-success)"
                      viewBox="0 0 24 24"
                    >
                      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
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
