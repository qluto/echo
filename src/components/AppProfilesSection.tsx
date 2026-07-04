import { useState } from "react";
import { AppProfile } from "../lib/tauri";
import { APP_PROFILE_PRESETS, DEFAULT_POSTPROCESS_PROMPT } from "../lib/prompts";

interface AppProfilesSectionProps {
  profiles: AppProfile[];
  onChange: (profiles: AppProfile[]) => void;
}

/** Template prompt to seed / reset a profile with, by bundle id. */
function templateFor(bundleId: string): string {
  return (
    APP_PROFILE_PRESETS.find((p) => p.bundle_id === bundleId)?.prompt ??
    DEFAULT_POSTPROCESS_PROMPT
  );
}

/**
 * Per-app prompt profiles: when the frontmost app matches a profile's
 * bundle id, its prompt replaces the system prompt (e.g. Slack drafting).
 */
export function AppProfilesSection({ profiles, onChange }: AppProfilesSectionProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [isAdding, setIsAdding] = useState(false);
  const [customBundleId, setCustomBundleId] = useState("");
  const [customName, setCustomName] = useState("");

  const updateProfile = (bundleId: string, patch: Partial<AppProfile>) => {
    onChange(
      profiles.map((p) => (p.bundle_id === bundleId ? { ...p, ...patch } : p))
    );
  };

  const removeProfile = (bundleId: string) => {
    onChange(profiles.filter((p) => p.bundle_id !== bundleId));
    if (expandedId === bundleId) setExpandedId(null);
  };

  const addProfile = (profile: AppProfile) => {
    onChange([...profiles, profile]);
    setIsAdding(false);
    setCustomBundleId("");
    setCustomName("");
    setExpandedId(profile.bundle_id);
  };

  const availablePresets = APP_PROFILE_PRESETS.filter(
    (preset) => !profiles.some((p) => p.bundle_id === preset.bundle_id)
  );
  const canAddCustom =
    customBundleId.trim().length > 0 &&
    !profiles.some((p) => p.bundle_id === customBundleId.trim());

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <span
          className="text-xs font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          App Profiles
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
        When the target app matches, its prompt replaces the system prompt —
        e.g. draft a polished Slack message from rambling speech.
      </span>

      {/* Add picker */}
      {isAdding && (
        <div className="flex flex-col gap-2 px-3 py-3 rounded-xl bg-surface border border-subtle">
          {availablePresets.map((preset) => (
            <button
              key={preset.bundle_id}
              onClick={() =>
                addProfile({
                  bundle_id: preset.bundle_id,
                  name: preset.name,
                  prompt: preset.prompt,
                  enabled: true,
                })
              }
              className="flex items-center justify-between px-3 py-2 rounded-lg border border-subtle transition-colors hover:bg-surface-elevated text-left"
            >
              <span className="text-xs" style={{ color: "var(--text-primary)" }}>
                {preset.name}
              </span>
              <span
                className="text-xs font-mono"
                style={{ color: "var(--text-tertiary)" }}
              >
                {preset.bundle_id}
              </span>
            </button>
          ))}
          <div className="flex flex-col gap-1.5">
            <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              Custom app
            </span>
            <input
              type="text"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder="Name (e.g. Mail)"
              className="h-8 px-2 rounded-lg bg-surface border border-subtle text-xs focus:outline-none focus:border-glow-idle"
              style={{ color: "var(--text-primary)" }}
            />
            <input
              type="text"
              value={customBundleId}
              onChange={(e) => setCustomBundleId(e.target.value)}
              placeholder="Bundle ID (e.g. com.apple.mail)"
              className="h-8 px-2 rounded-lg bg-surface border border-subtle text-xs font-mono focus:outline-none focus:border-glow-idle"
              style={{ color: "var(--text-primary)" }}
            />
            <button
              onClick={() =>
                addProfile({
                  bundle_id: customBundleId.trim(),
                  name: customName.trim() || customBundleId.trim(),
                  prompt: templateFor(customBundleId.trim()),
                  enabled: true,
                })
              }
              disabled={!canAddCustom}
              className="h-8 rounded-lg text-xs transition-colors disabled:opacity-40"
              style={{
                backgroundColor: "rgba(147, 112, 219, 0.12)",
                color: "#9370DB",
              }}
            >
              Add custom profile
            </button>
          </div>
        </div>
      )}

      {/* Profile list */}
      {profiles.map((profile) => {
        const isExpanded = expandedId === profile.bundle_id;
        return (
          <div
            key={profile.bundle_id}
            className="flex flex-col rounded-xl bg-surface border border-subtle"
          >
            <div className="h-11 px-3 flex items-center justify-between gap-2">
              <button
                onClick={() =>
                  setExpandedId(isExpanded ? null : profile.bundle_id)
                }
                className="flex items-center gap-2 flex-1 min-w-0 text-left"
              >
                <svg
                  className={`w-3 h-3 flex-shrink-0 transition-transform ${isExpanded ? "rotate-90" : ""}`}
                  fill="var(--text-tertiary)"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
                    clipRule="evenodd"
                  />
                </svg>
                <span
                  className="text-xs flex-shrink-0"
                  style={{ color: "var(--text-primary)" }}
                >
                  {profile.name}
                </span>
                <span
                  className="text-xs font-mono truncate"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  {profile.bundle_id}
                </span>
              </button>
              <button
                onClick={() =>
                  updateProfile(profile.bundle_id, { enabled: !profile.enabled })
                }
                className="w-10 h-5 rounded-full flex items-center flex-shrink-0 transition-all duration-200"
                style={{
                  backgroundColor: profile.enabled
                    ? "#9370DB"
                    : "var(--border-subtle)",
                  padding: "2px",
                }}
              >
                <div
                  className="w-4 h-4 rounded-full bg-white transition-transform duration-200"
                  style={{
                    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.15)",
                    transform: profile.enabled
                      ? "translateX(20px)"
                      : "translateX(0)",
                  }}
                />
              </button>
            </div>

            {isExpanded && (
              <div className="flex flex-col gap-2 px-3 pb-3">
                <textarea
                  value={profile.prompt}
                  onChange={(e) =>
                    updateProfile(profile.bundle_id, { prompt: e.target.value })
                  }
                  className="w-full h-48 px-3 py-2 rounded-lg bg-surface border border-subtle text-xs font-mono resize-none focus:outline-none focus:border-glow-idle"
                  style={{ color: "var(--text-primary)" }}
                />
                <div className="flex items-center justify-between">
                  <button
                    onClick={() =>
                      updateProfile(profile.bundle_id, {
                        prompt: templateFor(profile.bundle_id),
                      })
                    }
                    className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Reset to template
                  </button>
                  <button
                    onClick={() => removeProfile(profile.bundle_id)}
                    className="text-xs px-2 py-1 rounded transition-colors hover:bg-surface-elevated"
                    style={{ color: "var(--glow-recording)" }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
