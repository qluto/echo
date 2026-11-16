import raw from "../../CHANGELOG.md?raw";

/** One `## v0.16.0 — 2026-11-16` (or `## Unreleased`) section of CHANGELOG.md. */
export interface ChangelogSection {
  /** "0.16.0", or null for Unreleased. */
  version: string | null;
  /** The heading after "## ". */
  title: string;
  /** Lines under the heading. */
  body: string[];
}

const parse = (text: string): ChangelogSection[] =>
  text
    .split(/^## /m)
    .slice(1)
    .map((chunk) => {
      const [title, ...rest] = chunk.split("\n");
      const version = title.match(/^v?(\d+\.\d+\.\d+)/)?.[1] ?? null;
      return { version, title: title.trim(), body: rest };
    });

const SECTIONS = parse(raw);

const key = (v: string) => v.split(".").map(Number);
const newer = (a: string, b: string) => {
  const [x, y] = [key(a), key(b)];
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] > y[i];
  return false;
};

/**
 * Released sections, newest first. With `since`, only those newer than it,
 * up to `upTo` (what changed in an update from `since` to `upTo`).
 */
export function changelog(since?: string, upTo?: string): ChangelogSection[] {
  return SECTIONS.filter(
    (s) =>
      s.version !== null &&
      (!since || newer(s.version, since)) &&
      (!upTo || !newer(s.version, upTo)),
  );
}
