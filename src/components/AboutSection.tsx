import { useEffect, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { checkForUpdate, restartForUpdate } from "../lib/tauri";
import { ChangelogModal } from "./ChangelogModal";

/** Version, a check for updates now, and the changelog. */
export function AboutSection() {
  const [version, setVersion] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [installed, setInstalled] = useState(false);
  const [changelogOpen, setChangelogOpen] = useState(false);

  useEffect(() => {
    getVersion().then(setVersion).catch(() => {});
  }, []);

  const check = async () => {
    setChecking(true);
    setResult(null);
    try {
      const r = await checkForUpdate();
      if (r.status === "up_to_date") setResult("最新のバージョンです");
      else if (r.status === "installed") {
        setInstalled(true);
        setResult(`Echo ${r.version} をインストールしました。再起動すると反映されます`);
      } else setResult("/Applications にインストールした Echo だけがアップデートできます");
    } catch (e) {
      setResult(`確認できませんでした: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setChecking(false);
    }
  };

  const button = "text-xs px-2.5 py-1 rounded-lg border border-subtle transition-colors hover:bg-surface-elevated disabled:opacity-50";

  return (
    <section className="flex flex-col gap-3">
      <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        About
      </span>
      <div className="px-4 py-3 rounded-xl bg-surface border border-subtle flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Echo {version}
          </span>
          <div className="flex gap-1.5">
            <button onClick={() => setChangelogOpen(true)} className={button} style={{ color: "var(--text-secondary)" }}>
              変更履歴
            </button>
            {installed ? (
              <button
                onClick={() => restartForUpdate().catch((e) => setResult(String(e)))}
                className="text-xs px-2.5 py-1 rounded-lg text-white"
                style={{ backgroundColor: "var(--glow-idle)" }}
              >
                今すぐ再起動
              </button>
            ) : (
              <button onClick={check} disabled={checking} className={button} style={{ color: "var(--text-secondary)" }}>
                {checking ? "確認中…" : "アップデートを確認"}
              </button>
            )}
          </div>
        </div>
        {result && (
          <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            {result}
          </span>
        )}
      </div>
      {changelogOpen && <ChangelogModal onClose={() => setChangelogOpen(false)} />}
    </section>
  );
}
