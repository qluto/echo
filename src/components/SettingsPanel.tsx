import { useEffect, useState } from "react";
import {
  getSettings,
  updateSettings,
  getAudioDevices,
  registerHotkey,
  AppSettings,
  AudioDevice,
} from "../lib/tauri";

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const SUPPORTED_LANGUAGES = [
  { code: "auto", name: "自動検出" },
  { code: "ja", name: "日本語" },
  { code: "en", name: "English" },
  { code: "zh", name: "中文" },
  { code: "ko", name: "한국어" },
  { code: "de", name: "Deutsch" },
  { code: "fr", name: "Français" },
  { code: "es", name: "Español" },
];

export function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [settings, setSettings] = useState<AppSettings>({
    hotkey: "CommandOrControl+Shift+Space",
    language: "auto",
    auto_insert: true,
    device_name: null,
  });
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
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
      const [loadedSettings, loadedDevices] = await Promise.all([
        getSettings(),
        getAudioDevices(),
      ]);
      setSettings(loadedSettings);
      setDevices(loadedDevices);
      setHotkeyInput(loadedSettings.hotkey);
    } catch (e) {
      console.error("Failed to load settings:", e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await updateSettings(settings);
      if (hotkeyInput !== settings.hotkey) {
        await registerHotkey(hotkeyInput);
        setSettings((prev) => ({ ...prev, hotkey: hotkeyInput }));
      }
      onClose();
    } catch (e) {
      console.error("Failed to save settings:", e);
    } finally {
      setIsSaving(false);
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
      setHotkeyInput(keys.join("+"));
      setIsRecordingHotkey(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h2 className="text-lg font-semibold">設定</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        {isLoading ? (
          <div className="p-8 flex items-center justify-center">
            <svg
              className="w-8 h-8 text-primary-500 animate-spin"
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
        ) : (
          <div className="p-4 space-y-6">
            {/* Hotkey */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                録音ホットキー
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={hotkeyInput}
                  readOnly
                  onKeyDown={handleHotkeyRecord}
                  onFocus={() => setIsRecordingHotkey(true)}
                  onBlur={() => setIsRecordingHotkey(false)}
                  className={`flex-1 px-3 py-2 bg-gray-700 border rounded text-white focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    isRecordingHotkey
                      ? "border-primary-500"
                      : "border-gray-600"
                  }`}
                  placeholder="キーを押して設定..."
                />
                <button
                  onClick={() => setIsRecordingHotkey(true)}
                  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded transition-colors"
                >
                  変更
                </button>
              </div>
              {isRecordingHotkey && (
                <p className="text-xs text-primary-400 mt-1">
                  新しいホットキーを押してください...
                </p>
              )}
            </div>

            {/* Language */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                認識言語
              </label>
              <select
                value={settings.language}
                onChange={(e) =>
                  setSettings((prev) => ({ ...prev, language: e.target.value }))
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Audio Device */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                入力デバイス
              </label>
              <select
                value={settings.device_name || ""}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    device_name: e.target.value || null,
                  }))
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="">デフォルト</option>
                {devices.map((device) => (
                  <option key={device.name} value={device.name}>
                    {device.name} {device.is_default ? "(デフォルト)" : ""}
                  </option>
                ))}
              </select>
            </div>

            {/* Auto Insert */}
            <div className="flex items-center justify-between">
              <div>
                <label className="block text-sm font-medium text-gray-300">
                  自動挿入
                </label>
                <p className="text-xs text-gray-500">
                  文字起こし完了後に自動的にテキストを挿入
                </p>
              </div>
              <button
                onClick={() =>
                  setSettings((prev) => ({
                    ...prev,
                    auto_insert: !prev.auto_insert,
                  }))
                }
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.auto_insert ? "bg-primary-600" : "bg-gray-600"
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
                    settings.auto_insert ? "translate-x-6" : ""
                  }`}
                />
              </button>
            </div>
          </div>
        )}

        <div className="p-4 border-t border-gray-700 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            キャンセル
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-500 rounded transition-colors disabled:opacity-50"
          >
            {isSaving ? "保存中..." : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
}
