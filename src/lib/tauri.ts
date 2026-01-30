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

export async function pingAsrEngine(): Promise<boolean> {
  return invoke("ping_asr_engine");
}

export async function startAsrEngine(): Promise<void> {
  return invoke("start_asr_engine");
}

export async function stopAsrEngine(): Promise<void> {
  return invoke("stop_asr_engine");
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
