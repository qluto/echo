import { ReactNode } from "react";
import { changelog, ChangelogSection } from "../lib/changelog";

/** `code` and [text](url) in a changelog line, as plain styled text (no navigation). */
function inline(text: string): ReactNode[] {
  return text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .split(/(`[^`]+`)/)
    .map((part, i) =>
      part.startsWith("`") && part.endsWith("`") ? (
        <code key={i} className="font-mono text-[11px] px-1 rounded" style={{ backgroundColor: "var(--surface-muted)" }}>
          {part.slice(1, -1)}
        </code>
      ) : (
        part
      ),
    );
}

function Section({ section }: { section: ChangelogSection }) {
  const blocks: ReactNode[] = [];
  let items: ReactNode[] = [];
  const flush = () => {
    if (items.length) blocks.push(<ul key={blocks.length} className="list-disc pl-4 flex flex-col gap-1">{items}</ul>);
    items = [];
  };
  section.body.forEach((line, i) => {
    const t = line.trim();
    if (t.startsWith("- ")) {
      items.push(<li key={i}>{inline(t.slice(2))}</li>);
      return;
    }
    flush();
    if (t.startsWith("### ")) {
      blocks.push(
        <h4 key={blocks.length} className="text-[11px] font-semibold mt-1" style={{ color: "var(--text-tertiary)" }}>
          {t.slice(4)}
        </h4>,
      );
    } else if (t) {
      blocks.push(<p key={blocks.length}>{inline(t)}</p>);
    }
  });
  flush();
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
        {section.title}
      </h3>
      <div className="flex flex-col gap-1.5 text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
        {blocks}
      </div>
    </section>
  );
}

/**
 * CHANGELOG.md (bundled at build time). With `since`, only what changed after
 * that version (after an update); otherwise every release.
 */
export function ChangelogModal({ since, upTo, onClose }: { since?: string; upTo?: string; onClose: () => void }) {
  const sections = changelog(since, upTo);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ backgroundColor: "rgba(0, 0, 0, 0.25)" }} onClick={onClose}>
      <div
        className="w-full max-w-md max-h-[80vh] flex flex-col rounded-2xl border border-subtle shadow-2xl overflow-hidden"
        style={{ backgroundColor: "var(--surface)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-subtle flex-shrink-0">
          <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            {since ? `What's new in Echo ${upTo ?? ""}` : "Changelog"}
          </h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-surface-elevated transition-colors text-sm leading-none" aria-label="閉じる" style={{ color: "var(--text-tertiary)" }}>
            ×
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-5">
          {sections.length ? (
            sections.map((s) => <Section key={s.title} section={s} />)
          ) : (
            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              No changes recorded.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
