import { useState } from "react";

interface DictionarySectionProps {
  dictionary: Record<string, string>;
  onChange: (dictionary: Record<string, string>) => void;
}

/**
 * User dictionary editor: from → to term pairs passed to the LLM
 * post-processor as replacement hints (e.g. ASR mishearings of proper
 * nouns to their correct spelling).
 */
export function DictionarySection({ dictionary, onChange }: DictionarySectionProps) {
  const [isAdding, setIsAdding] = useState(false);
  const [newFrom, setNewFrom] = useState("");
  const [newTo, setNewTo] = useState("");

  const entries = Object.entries(dictionary).sort(([a], [b]) => a.localeCompare(b));

  const canAdd =
    newFrom.trim().length > 0 &&
    newTo.trim().length > 0 &&
    !(newFrom.trim() in dictionary);

  const addEntry = () => {
    if (!canAdd) return;
    onChange({ ...dictionary, [newFrom.trim()]: newTo.trim() });
    setNewFrom("");
    setNewTo("");
    setIsAdding(false);
  };

  const updateEntry = (from: string, to: string) => {
    onChange({ ...dictionary, [from]: to });
  };

  const removeEntry = (from: string) => {
    const next = { ...dictionary };
    delete next[from];
    onChange(next);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <span
          className="text-xs font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          Dictionary
        </span>
        <button
          onClick={() => setIsAdding(!isAdding)}
          className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
          style={{ color: "var(--text-tertiary)" }}
        >
          {isAdding ? "Cancel" : "+ Add"}
        </button>
      </div>

      <span className="text-xs px-1" style={{ color: "var(--text-tertiary)" }}>
        Correct recurring mishearings (e.g. proper nouns) during AI
        enhancement. Applied as a hint to the model — not a guaranteed literal
        replacement — and only in hotkey mode, not always-on transcription.
      </span>

      {/* Add form */}
      {isAdding && (
        <div className="flex flex-col gap-1.5 px-3 py-3 rounded-xl bg-surface border border-subtle">
          <input
            type="text"
            value={newFrom}
            onChange={(e) => setNewFrom(e.target.value)}
            placeholder="Misheard text (e.g. クロード コード)"
            className="h-8 px-2 rounded-lg bg-surface border border-subtle text-xs focus:outline-none focus:border-glow-idle"
            style={{ color: "var(--text-primary)" }}
          />
          <input
            type="text"
            value={newTo}
            onChange={(e) => setNewTo(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) addEntry();
            }}
            placeholder="Correct text (e.g. Claude Code)"
            className="h-8 px-2 rounded-lg bg-surface border border-subtle text-xs focus:outline-none focus:border-glow-idle"
            style={{ color: "var(--text-primary)" }}
          />
          {newFrom.trim() in dictionary && (
            <span className="text-xs px-1" style={{ color: "var(--glow-recording)" }}>
              This term is already in the dictionary.
            </span>
          )}
          <button
            onClick={addEntry}
            disabled={!canAdd}
            className="h-8 rounded-lg text-xs transition-colors disabled:opacity-40"
            style={{
              backgroundColor: "rgba(147, 112, 219, 0.12)",
              color: "#9370DB",
            }}
          >
            Add entry
          </button>
        </div>
      )}

      {/* Entry list */}
      {entries.length > 0 && (
        <div className="flex flex-col rounded-xl bg-surface border border-subtle divide-y divide-[var(--border-subtle)]">
          {entries.map(([from, to]) => (
            <div key={from} className="h-10 px-3 flex items-center gap-2">
              <span
                className="text-xs truncate flex-1 min-w-0"
                style={{ color: "var(--text-secondary)" }}
                title={from}
              >
                {from}
              </span>
              <svg
                className="w-3 h-3 flex-shrink-0"
                fill="none"
                stroke="var(--text-tertiary)"
                strokeWidth={2}
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
              </svg>
              <input
                type="text"
                value={to}
                onChange={(e) => updateEntry(from, e.target.value)}
                className="h-7 px-2 rounded-lg bg-transparent border border-transparent text-xs flex-1 min-w-0 focus:outline-none focus:border-glow-idle focus:bg-surface"
                style={{ color: "var(--text-primary)" }}
              />
              <button
                onClick={() => removeEntry(from)}
                className="flex-shrink-0 opacity-50 hover:opacity-100 transition-opacity"
                title="Delete"
              >
                <svg
                  className="w-3.5 h-3.5"
                  fill="none"
                  stroke="var(--text-tertiary)"
                  strokeWidth={1.5}
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
