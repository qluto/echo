import { useEffect, useState } from "react";
import {
  getNotesModelStatus,
  setNotesModel,
  prepareNotesModel,
  PostProcessModelStatus,
} from "../lib/tauri";
import { POSTPROCESS_MODEL_SIZES, getPostprocessModelName } from "../lib/models";

/**
 * The LLM for voice-memo minutes and meeting notes, chosen separately from
 * the dictation post-processor: these favor quality over speed. A model that
 * isn't downloaded yet is fetched on selection, so a memo or meeting never
 * waits for the download.
 */
export function NotesModelSection() {
  const [status, setStatus] = useState<PostProcessModelStatus | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getNotesModelStatus()
      .then(setStatus)
      .catch((e) => console.error("Failed to get notes model status:", e));
  }, []);

  const download = async () => {
    setIsDownloading(true);
    setError(null);
    try {
      setStatus(await prepareNotesModel());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsDownloading(false);
    }
  };

  const change = async (model: string) => {
    if (!status || model === status.model_name || isDownloading) return;
    setError(null);
    try {
      const next = await setNotesModel(model);
      setStatus(next);
      if (!next.loaded) await download();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  if (!status) return null;

  return (
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
            <path d="M9 12h6m-6 4h6M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
          </svg>
        </div>
        <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Minutes &amp; Meeting Notes
        </span>
      </div>

      <div className="h-12 px-4 rounded-xl bg-surface border border-subtle flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-sm flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
            LLM Model
          </span>
          <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            Voice Memo / Meeting
          </span>
        </div>
        <div className="flex items-center gap-2 w-[140px] justify-end">
          {isDownloading ? (
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 border-2 rounded-full animate-spin"
                style={{ borderColor: "var(--border-subtle)", borderTopColor: "var(--glow-recording)" }}
              />
              <span className="font-mono text-xs" style={{ color: "var(--text-tertiary)" }}>
                Downloading...
              </span>
            </div>
          ) : (
            <>
              <select
                value={status.model_name}
                onChange={(e) => change(e.target.value)}
                className="bg-transparent font-mono text-xs appearance-none cursor-pointer focus:outline-none w-full"
                style={{ color: "var(--text-primary)" }}
              >
                {status.available_models.map((model) => (
                  <option key={model} value={model} className="bg-surface">
                    {getPostprocessModelName(model)} ({POSTPROCESS_MODEL_SIZES[model] ?? "?"})
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
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
              </svg>
            </>
          )}
        </div>
      </div>

      {!status.loaded && !isDownloading && (
        <button
          onClick={download}
          className="px-4 py-2 rounded-lg text-xs text-left transition-colors hover:bg-surface-elevated border border-subtle"
          style={{ color: "var(--text-secondary)" }}
        >
          Not downloaded yet — download now
        </button>
      )}

      {error && (
        <div
          className="px-4 py-2 rounded-lg text-xs"
          style={{ backgroundColor: "rgba(198, 125, 99, 0.15)", color: "var(--glow-recording)" }}
        >
          {error}
        </div>
      )}

      <div
        className="px-4 py-2 rounded-lg text-xs"
        style={{ backgroundColor: "rgba(198, 125, 99, 0.08)", color: "var(--text-tertiary)" }}
      >
        Writes voice-memo minutes and live meeting notes. Separate from the dictation model, so a larger, slower model only costs time here. Transcription keeps running while it works.
      </div>
    </section>
  );
}
