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
  no_speech?: boolean | null;
  /** Raw transcription before AI post-processing (present only when it changed). */
  raw_text?: string | null;
}

export interface AudioDevice {
  name: string;
  is_default: boolean;
}

export interface AppProfile {
  bundle_id: string;
  name: string;
  prompt: string;
  enabled: boolean;
}

export interface PostProcessSettings {
  enabled: boolean;
  dictionary: Record<string, string>;
  custom_prompt?: string | null;
  model_name?: string | null;
  app_profiles?: AppProfile[];
}

export interface GatedAccessSettings {
  enabled: boolean;
  hf_token: string | null;
}

export interface AppSettings {
  hotkey: string;
  language: string;
  auto_insert: boolean;
  device_name: string | null;
  model_name: string | null;
  postprocess: PostProcessSettings;
  gated_access: GatedAccessSettings;
  /** LLM for meeting notes and minutes (set via setNotesModel). */
  notes_model?: string | null;
  /** Minutes and meeting notes via Claude Code (set via setNotesClaude). */
  notes_claude?: NotesClaudeSettings;
}

/** Minutes and meeting notes through the local Claude Code CLI (`claude -p`). */
export interface NotesClaudeSettings {
  enabled: boolean;
  /** Claude Code model alias: "sonnet" | "opus" | "haiku" | "fable". */
  model: string;
}

export interface ClaudeCliInfo {
  path: string;
  version: string;
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

/**
 * Load ASR model asynchronously in background thread.
 * This function returns immediately and emits events when complete.
 * Listen for "model-load-complete" or "model-load-error" events.
 */
export async function loadAsrModelAsync(): Promise<void> {
  return invoke("load_asr_model_async");
}

export interface ModelLoadErrorEvent {
  error: string;
}

export function onModelLoadComplete(
  callback: (status: ModelStatus) => void
): Promise<UnlistenFn> {
  return listen<ModelStatus>("model-load-complete", (event) => {
    callback(event.payload);
  });
}

export function onModelLoadError(
  callback: (event: ModelLoadErrorEvent) => void
): Promise<UnlistenFn> {
  return listen<ModelLoadErrorEvent>("model-load-error", (event) => {
    callback(event.payload);
  });
}

// Warmup result
export interface WarmupResult {
  success: boolean;
  warmup_time_ms: number | null;
  error: string | null;
}

export async function warmupAsrModel(): Promise<WarmupResult> {
  return invoke("warmup_asr_model");
}

export async function setAsrModel(modelName: string): Promise<ModelStatus> {
  return invoke("set_asr_model", { modelName });
}

// Model cache status
export interface ModelCacheStatus {
  cached: boolean;
  model_name: string;
}

export async function isModelCached(modelName?: string): Promise<ModelCacheStatus> {
  return invoke("is_model_cached", { modelName: modelName ?? null });
}

// Event listeners
export type RecordingState = "idle" | "recording" | "transcribing";

export interface RecordingStateEvent {
  state: RecordingState;
}

export interface TranscriptionError {
  kind: string;
  message: string;
}

export interface TranscriptionEvent {
  result: TranscriptionResult | null;
  error: TranscriptionError | null;
}

// Structured status events ("transcription-status") from the hotkey path.
// no_speech is a normal outcome, not an error.
export type StatusKind =
  | "no_speech"
  | "asr_error"
  | "engine_busy"
  | "audio_error"
  | "permission_denied";

export interface TranscriptionStatusEvent {
  kind: StatusKind;
  source: "hotkey";
  message: string | null;
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

export function onTranscriptionStatus(
  callback: (event: TranscriptionStatusEvent) => void
): Promise<UnlistenFn> {
  return listen<TranscriptionStatusEvent>("transcription-status", (event) => {
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

// Auto-update (installed in the background; the app restarts when idle)

/** The installed version waiting for a restart, if any. */
export async function getUpdateReady(): Promise<string | null> {
  return invoke("update_ready");
}

/** Restart into the installed update now. */
export async function restartForUpdate(): Promise<void> {
  return invoke("restart_for_update");
}

/** This run is newer than the last one: the version it moved to, and from. */
export interface JustUpdated {
  version: string;
  previous: string;
}

/** The update this run is (newer than the last run), else null. */
export async function getJustUpdated(): Promise<JustUpdated | null> {
  return invoke("just_updated");
}

export type UpdateCheck =
  | { status: "up_to_date"; current: string }
  | { status: "installed"; version: string }
  | { status: "unsupported" };

/** Check for an update now; a newer version is downloaded and installed (restart to use it). */
export async function checkForUpdate(): Promise<UpdateCheck> {
  return invoke("check_for_update");
}

/** Ask the main window to show the changelog (from the update notice). */
export function onShowChangelog(callback: () => void): Promise<UnlistenFn> {
  return listen("show-changelog", () => callback());
}

// The popup under the menu bar icon (notice.rs)
export type NoticeAction = "open_meeting" | "restart_for_update" | "show_changelog";

export interface Notice {
  id: number;
  title: string;
  body: string | null;
  action: NoticeAction | null;
  timeout_ms: number;
}

export async function getNotice(): Promise<Notice | null> {
  return invoke("get_notice");
}

export async function dismissNotice(id: number): Promise<void> {
  return invoke("dismiss_notice", { id });
}

export async function noticeAction(action: NoticeAction): Promise<void> {
  return invoke("notice_action", { action });
}

export function onUpdateReady(callback: (version: string) => void): Promise<UnlistenFn> {
  return listen<string>("update-ready", (event) => callback(event.payload));
}

// Post-processing types and functions
export interface PostProcessModelStatus {
  model_name: string;
  loaded: boolean;
  loading: boolean;
  error: string | null;
  available_models: string[];
}

export interface PostProcessResult {
  success: boolean;
  processed_text: string;
  processing_time_ms: number | null;
  error: string | null;
}

export interface ActiveAppInfo {
  bundle_id: string | null;
  app_name: string | null;
}

export async function getPostprocessSettings(): Promise<PostProcessSettings> {
  return invoke("get_postprocess_settings");
}

export async function updatePostprocessSettings(postprocess: PostProcessSettings): Promise<void> {
  return invoke("update_postprocess_settings", { postprocess });
}

export async function loadPostprocessModel(): Promise<PostProcessModelStatus> {
  return invoke("load_postprocess_model");
}

export async function unloadPostprocessModel(): Promise<void> {
  return invoke("unload_postprocess_model");
}

export async function isPostprocessModelCached(): Promise<PostProcessModelStatus> {
  return invoke("is_postprocess_model_cached");
}

export async function postprocessText(
  text: string,
  appName?: string,
  appBundleId?: string
): Promise<PostProcessResult> {
  return invoke("postprocess_text", {
    text,
    appName: appName ?? null,
    appBundleId: appBundleId ?? null,
  });
}

export async function getFrontmostApp(): Promise<ActiveAppInfo> {
  return invoke("get_frontmost_app");
}

export async function setPostprocessModel(modelName: string): Promise<PostProcessModelStatus> {
  return invoke("set_postprocess_model", { modelName });
}

export async function getPostprocessModelStatus(): Promise<PostProcessModelStatus> {
  return invoke("get_postprocess_model_status");
}

// Notes model: the LLM for meeting notes and minutes, separate from
// dictation post-processing. In these statuses `loaded` means "downloaded".
export async function getNotesModelStatus(): Promise<PostProcessModelStatus> {
  return invoke("get_notes_model_status");
}

export async function setNotesModel(modelName: string): Promise<PostProcessModelStatus> {
  return invoke("set_notes_model", { modelName });
}

/** Download the notes model if it isn't yet. */
export async function prepareNotesModel(): Promise<PostProcessModelStatus> {
  return invoke("prepare_notes_model");
}

export async function getNotesClaude(): Promise<NotesClaudeSettings> {
  return invoke("get_notes_claude");
}

export async function setNotesClaude(notesClaude: NotesClaudeSettings): Promise<NotesClaudeSettings> {
  return invoke("set_notes_claude", { notesClaude });
}

/** The local transcription API (HTTP on 127.0.0.1, opt-in). */
export interface LocalApiStatus {
  enabled: boolean;
  port: number;
  token: string;
  running: boolean;
  /** Why it isn't running although enabled (e.g. the port is in use). */
  error: string | null;
}

export async function getLocalApi(): Promise<LocalApiStatus> {
  return invoke("get_local_api");
}

export async function setLocalApi(
  enabled: boolean,
  port: number,
  regenerateToken = false,
): Promise<LocalApiStatus> {
  return invoke("set_local_api", { enabled, port, regenerateToken });
}

/** Find the Claude Code CLI; rejects with the reason when it isn't usable. */
export async function checkClaudeCode(): Promise<ClaudeCliInfo> {
  return invoke("check_claude_code");
}

// Draft of the hotkey recording in progress (both empty = cleared).
// `committed` is final for this recording; `draft` is still tentative.
export interface HotkeyPartialEvent {
  committed: string;
  draft: string;
}

export function onHotkeyPartial(
  callback: (event: HotkeyPartialEvent) => void
): Promise<UnlistenFn> {
  return listen<HotkeyPartialEvent>("hotkey-partial", (event) => {
    callback(event.payload);
  });
}

// Memo (meeting recording) types and commands
export type MemoStatus =
  | "recording"
  | "transcribing"
  | "summarizing"
  | "done"
  | "error"
  | "interrupted";

export interface Memo {
  id: number;
  created_at: string;
  duration_seconds: number | null;
  audio_path: string;
  status: MemoStatus;
  /** Timestamped transcript, one `[MM:SS] text` line per chunk. */
  transcript: string | null;
  segments_json: string | null;
  minutes: string | null;
  language: string | null;
  model_name: string | null;
  error: string | null;
}

export interface MemoListItem {
  id: number;
  created_at: string;
  duration_seconds: number | null;
  status: MemoStatus;
  preview: string | null;
  error: string | null;
}

export interface MemoProgressEvent {
  id: number;
  status: MemoStatus;
  /** 0.0–1.0 within the current stage, when known. */
  progress: number | null;
}

export async function listMemos(): Promise<MemoListItem[]> {
  return invoke("list_memos");
}

export async function getMemo(id: number): Promise<Memo | null> {
  return invoke("get_memo", { id });
}

export async function deleteMemo(id: number): Promise<void> {
  return invoke("delete_memo", { id });
}

/** Regenerate the minutes, or (`retranscribe`) redo everything from the audio. */
export async function reprocessMemo(id: number, retranscribe: boolean): Promise<void> {
  return invoke("reprocess_memo", { id, retranscribe });
}

export async function revealMemoAudio(id: number): Promise<void> {
  return invoke("reveal_memo_audio", { id });
}

export function onMemoProgress(
  callback: (event: MemoProgressEvent) => void
): Promise<UnlistenFn> {
  return listen<MemoProgressEvent>("memo-progress", (event) => {
    callback(event.payload);
  });
}

// Meeting companion types and commands
export type MeetingSource = "me" | "others";

export interface MeetingLine {
  id: number;
  source: MeetingSource;
  /** Seconds since the meeting started. */
  start: number;
  end: number;
  text: string;
}

export type MeetingItemKind = "topic" | "decision" | "action" | "open";
export type MeetingItemChange = "added" | "updated" | "deleted";

export interface MeetingItem {
  id: string;
  kind: MeetingItemKind;
  text: string;
  /** What happened to the item in the latest round; null = unchanged.
   *  Items deleted in the latest round are still listed (as "deleted") until the next one. */
  change: MeetingItemChange | null;
  updated_round: number;
}

export type MeetingPhase = "running" | "finalizing" | "done" | "error";

export interface MeetingSnapshot {
  phase: MeetingPhase;
  memo_id: number;
  started_at: string;
  /** Elapsed time, frozen once the meeting is stopped. */
  duration_seconds: number;
  lines: MeetingLine[];
  items: MeetingItem[];
  round: number;
  warning: string | null;
  error: string | null;
  /** How the meeting started or ended by itself (meeting standby). */
  notice: string | null;
}

export interface MeetingStatus {
  phase: MeetingPhase;
  elapsed_seconds: number;
  /** Input levels, 0.0–1.0. */
  mic_level: number;
  system_level: number;
  /** True while an LLM update round is running. */
  updating: boolean;
  /** Seconds until the next update round, when scheduled. */
  next_update_in: number | null;
}

export interface MeetingLineEvent extends MeetingLine {
  memo_id: number;
}

export interface MeetingItemsEvent {
  memo_id: number;
  items: MeetingItem[];
  round: number;
}

export interface MeetingStateEvent {
  phase: MeetingPhase;
  memo_id: number;
  error: string | null;
  warning: string | null;
}

export async function startMeeting(): Promise<MeetingSnapshot> {
  return invoke("start_meeting");
}

/** Stops capture; the final update round runs before the phase becomes "done". */
export async function stopMeeting(): Promise<MeetingSnapshot> {
  return invoke("stop_meeting");
}

/** The current meeting, or the last finished one in this app session. */
export async function getMeetingSnapshot(): Promise<MeetingSnapshot | null> {
  return invoke("get_meeting_snapshot");
}

export async function getMeetingStatus(): Promise<MeetingStatus | null> {
  return invoke("get_meeting_status");
}

export async function openMeetingWindow(): Promise<void> {
  return invoke("open_meeting_window");
}

/** Meeting standby: listens for a conversation and starts a meeting by itself. */
export interface MeetingStandbyStatus {
  enabled: boolean;
  /** How long a conversation goes on before a meeting starts (seconds, 60–240). */
  detect_seconds: number;
  /** Listening now (not while a meeting runs). */
  listening: boolean;
  warning: string | null;
  error: string | null;
}

export async function getMeetingStandby(): Promise<MeetingStandbyStatus> {
  return invoke("get_meeting_standby");
}

/** Starting to listen finishes in the background; "meeting-standby" reports it. */
export async function setMeetingStandby(enabled: boolean): Promise<MeetingStandbyStatus> {
  return invoke("set_meeting_standby", { enabled });
}

/** How long a conversation must go on before standby starts a meeting (60–240 s, 10 s steps). */
export async function setMeetingStandbySeconds(seconds: number): Promise<MeetingStandbyStatus> {
  return invoke("set_meeting_standby_seconds", { seconds });
}

export function onMeetingStandby(callback: (status: MeetingStandbyStatus) => void): Promise<UnlistenFn> {
  return listen<MeetingStandbyStatus>("meeting-standby", (event) => {
    callback(event.payload);
  });
}

export function onMeetingLine(callback: (line: MeetingLineEvent) => void): Promise<UnlistenFn> {
  return listen<MeetingLineEvent>("meeting-line", (event) => {
    callback(event.payload);
  });
}

export function onMeetingItems(callback: (event: MeetingItemsEvent) => void): Promise<UnlistenFn> {
  return listen<MeetingItemsEvent>("meeting-items", (event) => {
    callback(event.payload);
  });
}

export function onMeetingState(callback: (event: MeetingStateEvent) => void): Promise<UnlistenFn> {
  return listen<MeetingStateEvent>("meeting-state", (event) => {
    callback(event.payload);
  });
}
