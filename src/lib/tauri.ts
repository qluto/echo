import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// Types for transcription
export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
}

export interface TranscriptionResult {
  success: boolean;
  text: string;
  segments: TranscriptionSegment[];
  language: string;
}

export interface AudioDevice {
  name: string;
  is_default: boolean;
}

export interface AppSettings {
  hotkey: string;
  language: string;
  auto_insert: boolean;
  device_name: string | null;
  model_name: string | null;
}

// Tauri commands
export async function startRecording(): Promise<void> {
  return invoke("start_recording");
}

export async function stopRecording(): Promise<string> {
  return invoke("stop_recording");
}

export async function transcribe(
  audioPath: string,
  language?: string
): Promise<TranscriptionResult> {
  return invoke("transcribe", { audioPath, language });
}

export async function insertText(text: string): Promise<void> {
  return invoke("insert_text", { text });
}

export async function getAudioDevices(): Promise<AudioDevice[]> {
  return invoke("get_audio_devices");
}

export async function setAudioDevice(deviceName: string): Promise<void> {
  return invoke("set_audio_device", { deviceName });
}

export async function getSettings(): Promise<AppSettings> {
  return invoke("get_settings");
}

export async function updateSettings(settings: Partial<AppSettings>): Promise<void> {
  return invoke("update_settings", { settings });
}

export async function registerHotkey(hotkey: string): Promise<void> {
  return invoke("register_global_hotkey", { hotkey });
}

export async function unregisterHotkey(): Promise<void> {
  return invoke("unregister_global_hotkey");
}

export async function startHotkeyRecording(): Promise<void> {
  return invoke("start_hotkey_recording");
}

export async function stopHotkeyRecording(): Promise<void> {
  return invoke("stop_hotkey_recording");
}

// Handy-keys event types
export interface HandyKeysEvent {
  modifiers: string[];
  key: string | null;
  is_key_down: boolean;
  hotkey_string: string;
}

export function onHandyKeysEvent(
  callback: (event: HandyKeysEvent) => void
): Promise<UnlistenFn> {
  return listen<HandyKeysEvent>("handy-keys-event", (event) => {
    callback(event.payload);
  });
}

export async function pingAsrEngine(): Promise<boolean> {
  return invoke("ping_asr_engine");
}

export async function startAsrEngine(): Promise<void> {
  return invoke("start_asr_engine");
}

export async function stopAsrEngine(): Promise<void> {
  return invoke("stop_asr_engine");
}

// Model status and management
export interface ModelStatus {
  model_name: string;
  loaded: boolean;
  loading: boolean;
  error: string | null;
  available_models: string[];
}

export async function getModelStatus(): Promise<ModelStatus> {
  return invoke("get_model_status");
}

export async function loadAsrModel(): Promise<ModelStatus> {
  return invoke("load_asr_model");
}

export async function setAsrModel(modelName: string): Promise<ModelStatus> {
  return invoke("set_asr_model", { modelName });
}

// Event listeners
export type RecordingState = "idle" | "recording" | "transcribing";

export interface RecordingStateEvent {
  state: RecordingState;
}

export interface TranscriptionEvent {
  result: TranscriptionResult | null;
  error: string | null;
}

export function onRecordingStateChange(
  callback: (event: RecordingStateEvent) => void
): Promise<UnlistenFn> {
  return listen<RecordingStateEvent>("recording-state-change", (event) => {
    callback(event.payload);
  });
}

export function onTranscriptionComplete(
  callback: (event: TranscriptionEvent) => void
): Promise<UnlistenFn> {
  return listen<TranscriptionEvent>("transcription-complete", (event) => {
    callback(event.payload);
  });
}

export function onError(
  callback: (error: string) => void
): Promise<UnlistenFn> {
  return listen<string>("error", (event) => {
    callback(event.payload);
  });
}

// Accessibility permissions
export async function checkAccessibilityPermission(): Promise<boolean> {
  return invoke("check_accessibility_permission");
}

export async function requestAccessibilityPermission(): Promise<boolean> {
  return invoke("request_accessibility_permission");
}

export async function openAccessibilitySettings(): Promise<void> {
  return invoke("open_accessibility_settings");
}

export async function restartApp(): Promise<void> {
  return invoke("restart_app");
}
