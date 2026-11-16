import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { dismissNotice, getNotice, noticeAction, Notice, NoticeAction } from "./lib/tauri";

const ACTION_LABEL: Record<NoticeAction, string> = {
  open_meeting: "開く",
  restart_for_update: "今すぐ再起動",
  show_changelog: "変更点を見る",
};

/**
 * The popup under the menu bar icon (`notice.rs`): a title, a line of
 * detail and at most one button. It hides itself after `timeout_ms`, but not
 * while the pointer is over it.
 */
function NoticeApp() {
  const [notice, setNotice] = useState<Notice | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hovered = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismiss = useCallback((n: Notice) => {
    void dismissNotice(n.id).catch(() => {});
  }, []);

  const arm = useCallback(
    (n: Notice) => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => {
        if (!hovered.current) dismiss(n);
      }, n.timeout_ms);
    },
    [dismiss],
  );

  useEffect(() => {
    const show = (n: Notice | null) => {
      setNotice(n);
      setError(null);
      if (n) arm(n);
    };
    // A notice sent before this page finished loading
    getNotice().then(show).catch(() => {});
    const unlisten = listen<Notice>("notice", (event) => show(event.payload));
    return () => {
      unlisten.then((fn) => fn());
      if (timer.current) clearTimeout(timer.current);
    };
  }, [arm]);

  if (!notice) return null;

  const act = async () => {
    if (!notice.action) return;
    try {
      await noticeAction(notice.action);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="w-screen h-screen p-2.5 bg-transparent">
      <div
        className="w-full rounded-2xl border flex items-start gap-3 px-3.5 py-3 animate-float-in"
        style={{
          backgroundColor: "rgba(255, 255, 255, 0.97)",
          borderColor: "var(--border-subtle)",
          boxShadow: "0 6px 24px rgba(0, 0, 0, 0.12), 0 1px 3px rgba(0, 0, 0, 0.06)",
        }}
        onMouseEnter={() => {
          hovered.current = true;
        }}
        onMouseLeave={() => {
          hovered.current = false;
          arm(notice);
        }}
      >
        <div
          className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0"
          style={{ backgroundColor: "var(--glow-recording)" }}
          aria-hidden
        />
        <div className="flex-1 min-w-0 flex flex-col gap-0.5">
          <span className="text-[13px] font-semibold leading-snug" style={{ color: "var(--text-primary)" }}>
            {notice.title}
          </span>
          {(error ?? notice.body) && (
            <span
              className="text-[11px] leading-snug line-clamp-2"
              style={{ color: error ? "var(--glow-recording)" : "var(--text-secondary)" }}
            >
              {error ?? notice.body}
            </span>
          )}
          {notice.action && (
            <div className="flex justify-end mt-1.5">
              <button
                onClick={act}
                className="px-3 h-7 rounded-full text-[11px] font-medium text-white"
                style={{ backgroundColor: "var(--glow-idle)" }}
              >
                {ACTION_LABEL[notice.action]}
              </button>
            </div>
          )}
        </div>
        <button
          onClick={() => dismiss(notice)}
          aria-label="閉じる"
          className="w-5 h-5 -mr-1 -mt-0.5 rounded-full flex items-center justify-center text-sm leading-none flex-shrink-0 hover:bg-black/5"
          style={{ color: "var(--text-tertiary)" }}
        >
          ×
        </button>
      </div>
    </div>
  );
}

export default NoticeApp;
