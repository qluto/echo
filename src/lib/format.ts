/**
 * Formatting utilities shared across components.
 */

export const formatHotkey = (hotkey: string): string => {
  return hotkey
    // Remove fn when combined with function keys (fn+f12 -> f12)
    .replace(/\bfn\+?(f(?:[1-9]|1[0-9]|2[0-4]))\b/gi, "$1")
    .replace(/command/gi, "⌘")
    .replace(/ctrl/gi, "⌃")
    .replace(/control/gi, "⌃")
    .replace(/shift/gi, "⇧")
    .replace(/option/gi, "⌥")
    .replace(/alt/gi, "⌥")
    .replace(/\bfn\b/gi, "🌐") // Fn key alone
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
};

/** Seconds as `M:SS`, or `H:MM:SS` from one hour on. */
export const formatClock = (seconds: number): string => {
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = (total % 60).toString().padStart(2, "0");
  return h > 0 ? `${h}:${m.toString().padStart(2, "0")}:${s}` : `${m}:${s}`;
};
