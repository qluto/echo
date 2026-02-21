import { useEffect, useState, useRef, useCallback } from "react";
import { listen, emit } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import {
  getCurrentWindow,
  LogicalSize,
  LogicalPosition,
  currentMonitor,
} from "@tauri-apps/api/window";

type IndicatorState =
  | "idle"
  | "recording"
  | "processing"
  | "success"
  | "ambient"
  | "ambient-active";

interface FloatStatePayload {
  state: IndicatorState;
  duration: number;
}

interface RecentEntry {
  id: number;
  text: string;
  created_at: string;
}

interface HistoryPage {
  entries: RecentEntry[];
}

const NORMAL_WIDTH = 240;
const NORMAL_HEIGHT = 60;
const HOVER_WIDTH = 264;
const HOVER_HEIGHT = 360;
const BOTTOM_MARGIN = 32;

/** Resize float window and anchor its bottom edge near screen bottom. */
async function resizeAndPosition(width: number, height: number) {
  const win = getCurrentWindow();
  try {
    await win.setSize(new LogicalSize(width, height));

    const monitor = await currentMonitor();
    if (!monitor) return;
    const scaleFactor = monitor.scaleFactor;
    const screenWidth = monitor.size.width / scaleFactor;
    const screenHeight = monitor.size.height / scaleFactor;
    const x = Math.round((screenWidth - width) / 2);
    const y = Math.round(screenHeight - BOTTOM_MARGIN - height);

    await win.setPosition(new LogicalPosition(x, y));
  } catch (_) {
    /* window may not be ready */
  }
}

function formatTime(createdAt: string): string {
  const timePart = createdAt.split(" ")[1];
  if (!timePart) return "";
  return timePart.slice(0, 5);
}

function FloatApp() {
  const [state, setState] = useState<IndicatorState>("idle");
  const [duration, setDuration] = useState(0);
  const [visible, setVisible] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [recentEntries, setRecentEntries] = useState<RecentEntry[]>([]);
  const prevModeRef = useRef<"ambient" | "normal" | "hidden">("hidden");
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ---- Event listeners ----

  useEffect(() => {
    const win = getCurrentWindow();

    const unlistenState = listen<FloatStatePayload>(
      "float-state",
      async (event) => {
        const { state: newState, duration: newDuration } = event.payload;
        setState(newState);
        setDuration(newDuration);

        if (newState === "idle") {
          setTimeout(() => {
            setVisible(false);
            win.hide();
            prevModeRef.current = "hidden";
          }, 100);
        } else {
          const isAmbient =
            newState === "ambient" || newState === "ambient-active";
          const newMode = isAmbient ? "ambient" : "normal";

          if (prevModeRef.current !== newMode) {
            const [w, h] = [NORMAL_WIDTH, NORMAL_HEIGHT];
            await resizeAndPosition(w, h);
          }

          prevModeRef.current = newMode;
          setVisible(true);
          win.show();
        }
      },
    );

    win.hide();
    return () => {
      unlistenState.then((fn) => fn());
    };
  }, []);

  // Accumulate transcription entries for hover panel
  useEffect(() => {
    const unlisten = listen<RecentEntry>(
      "continuous-transcription",
      (event) => {
        setRecentEntries((prev) => [event.payload, ...prev].slice(0, 10));
      },
    );
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const isAmbientState = state === "ambient" || state === "ambient-active";

  // Backfill recent entries from DB when hover panel opens.
  useEffect(() => {
    if (!isAmbientState || !isHovered) return;
    const loadRecentEntries = async () => {
      try {
        const page = await invoke<HistoryPage>("get_transcription_history", {
          limit: 10,
          offset: 0,
        });
        if (!page?.entries?.length) return;
        setRecentEntries((prev) => {
          const merged = [...prev, ...page.entries];
          const deduped = Array.from(
            new Map(merged.map((entry) => [entry.id, entry])).values(),
          );
          return deduped
            .sort((a, b) => b.id - a.id)
            .slice(0, 10);
        });
      } catch (_) {
        // ignore history fetch errors in float window
      }
    };
    void loadRecentEntries();
  }, [isAmbientState, isHovered]);

  // Reset hover & entries on state transitions
  useEffect(() => {
    if (state !== "ambient" && state !== "ambient-active") {
      setIsHovered(false);
    }
    if (state === "idle") {
      setRecentEntries([]);
    }
  }, [state]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    };
  }, []);

  // ---- Helpers ----

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Keep ambient window compact by default, expand only while hovered.
  useEffect(() => {
    if (!visible) return;
    const [w, h] =
      isAmbientState && isHovered
        ? [HOVER_WIDTH, HOVER_HEIGHT]
        : [NORMAL_WIDTH, NORMAL_HEIGHT];
    void resizeAndPosition(w, h);
  }, [visible, isAmbientState, isHovered]);

  const handleMouseEnter = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setIsHovered(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    hoverTimeoutRef.current = setTimeout(() => {
      setIsHovered(false);
      hoverTimeoutRef.current = null;
    }, 300);
  }, []);

  const handleToggleListening = useCallback(async () => {
    await emit("request-toggle-listening", {});
  }, []);

  // ---- Render ----

  if (!visible) {
    return null;
  }

  // Hover panel (expanded view over ambient pill)
  if (isAmbientState && isHovered) {
    // Display oldest → newest (max 4), matching Pencil design order
    const displayEntries = [...recentEntries].reverse().slice(-4);

    return (
      <div
        className="h-screen w-screen flex items-end justify-center bg-transparent p-[2px]"
        style={{ paddingBottom: 15 }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div
          className="flex flex-col min-h-0 animate-float-in overflow-hidden"
          style={{
            width: "100%",
            height: "100%",
            backgroundColor: "#FFFFFF",
            borderRadius: 16,
            border: "1px solid #E8E4DF",
            boxShadow:
              "0 4px 24px rgba(0,0,0,0.07), 0 8px 48px rgba(0,0,0,0.03)",
          }}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between flex-shrink-0"
            style={{
              padding: "12px 16px",
              borderBottom: "1px solid #E8E4DF",
            }}
          >
            <div className="flex items-center gap-2">
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: 3,
                  backgroundColor: "#7C9082",
                  boxShadow: "0 0 4px rgba(124,144,130,0.5)",
                }}
              />
              <span
                style={{
                  fontFamily: "'Plus Jakarta Sans', sans-serif",
                  fontSize: 12,
                  fontWeight: 600,
                  color: "#2D2D2D",
                }}
              >
                Always-on
              </span>
            </div>
            {/* Toggle (always-on = green/active) */}
            <button
              onClick={handleToggleListening}
              style={{
                position: "relative",
                width: 40,
                height: 22,
                borderRadius: 11,
                backgroundColor: "#7C9082",
                border: "none",
                cursor: "pointer",
                padding: 0,
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: 2,
                  right: 2,
                  width: 18,
                  height: 18,
                  borderRadius: 9,
                  backgroundColor: "#FFFFFF",
                  boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
                }}
              />
            </button>
          </div>

          {/* Scroll hint */}
          <div
            className="flex items-center justify-center flex-shrink-0"
            style={{
              height: 24,
              background:
                "linear-gradient(180deg, transparent 0%, rgba(255,255,255,0.87) 30%, #FFFFFF 100%)",
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#ADADAD"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="17 11 12 6 7 11" />
              <polyline points="17 18 12 13 7 18" />
            </svg>
          </div>

          {/* History list — fills remaining space */}
          <div
            className="flex flex-col flex-1 min-h-0"
            style={{ overflowY: "auto" }}
          >
            {displayEntries.length === 0 ? (
              <div
                className="flex items-center justify-center flex-1"
                style={{
                  padding: "24px 16px",
                  color: "#ADADAD",
                  fontSize: 12,
                  fontFamily: "'Plus Jakarta Sans', sans-serif",
                }}
              >
                No transcriptions yet
              </div>
            ) : (
              displayEntries.map((entry, i) => (
                <div
                  key={entry.id}
                  className="flex flex-shrink-0"
                  style={{
                    gap: 10,
                    padding: "10px 16px",
                    borderBottom:
                      i < displayEntries.length - 1
                        ? "1px solid #F0EFEC"
                        : "none",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "'IBM Plex Mono', monospace",
                      fontSize: 10,
                      color: "#ADADAD",
                      flexShrink: 0,
                      lineHeight: 1.4,
                    }}
                  >
                    {formatTime(entry.created_at)}
                  </span>
                  <span
                    style={{
                      fontFamily: "'Plus Jakarta Sans', sans-serif",
                      fontSize: 12,
                      color: "#2D2D2D",
                      lineHeight: 1.4,
                      overflow: "hidden",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical" as const,
                    }}
                  >
                    {entry.text}
                  </span>
                </div>
              ))
            )}
          </div>

          {/* Footer bar indicator */}
          <div
            className="flex items-center justify-center flex-shrink-0"
            style={{ padding: "4px 0 6px 0" }}
          >
            <div
              className={
                state === "ambient-active" ? "animate-ambient-breathe" : ""
              }
              style={{
                width: 40,
                height: 5,
                borderRadius: 2.5,
                backgroundColor: "#7C9082",
                boxShadow:
                  "0 0 4px 1px rgba(124,144,130,0.56), 0 0 10px rgba(124,144,130,0.31)",
              }}
            />
          </div>
        </div>
      </div>
    );
  }

  // Ambient states — pill at bottom of large transparent window
  if (isAmbientState) {
    return (
      <div
        className="h-screen w-screen flex items-end justify-center bg-transparent"
        style={{ paddingBottom: 15 }}
        onMouseEnter={handleMouseEnter}
      >
        {state === "ambient" && (
          <div
            className="animate-float-in"
            style={{
              width: 44,
              height: 10,
              borderRadius: 5,
              backgroundColor: "#1A1A1C",
              boxShadow: "0 1px 4px rgba(0, 0, 0, 0.12)",
            }}
          />
        )}
        {state === "ambient-active" && (
          <div
            className="animate-float-in animate-ambient-breathe"
            style={{
              width: 44,
              height: 10,
              borderRadius: 5,
              backgroundColor: "#7C9082",
            }}
          />
        )}
      </div>
    );
  }

  // Normal states (recording / processing / success)
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-transparent">
      {state === "recording" && (
        <div
          className="flex items-center gap-3.5 h-11 px-5 pl-4 rounded-full border animate-float-in"
          style={{
            background: "#FFFFFF",
            borderColor: "var(--border-subtle)",
            boxShadow: "0 4px 20px rgba(198, 125, 99, 0.2)",
          }}
        >
          {/* Glow Orb */}
          <div
            className="w-3 h-3 rounded-md animate-glow-pulse"
            style={{
              backgroundColor: "var(--glow-recording)",
              boxShadow:
                "0 0 8px 2px var(--glow-recording), 0 0 16px var(--glow-recording-soft)",
            }}
          />

          {/* Label */}
          <span
            className="font-mono text-[11px] tracking-wide"
            style={{ color: "var(--glow-recording)" }}
          >
            transcribing
          </span>

          {/* Divider */}
          <div
            className="w-px h-4"
            style={{ backgroundColor: "var(--glow-recording-soft)" }}
          />

          {/* Duration */}
          <span
            className="text-xs font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            {formatDuration(duration)}
          </span>

          {/* Waveform */}
          <div className="flex items-center gap-[3px] h-5">
            {[6, 14, 8, 18, 10, 14].map((height, i) => (
              <div
                key={i}
                className="w-0.5 rounded-sm wave-bar"
                style={{
                  height: `${height}px`,
                  backgroundColor: "var(--glow-recording)",
                }}
              />
            ))}
          </div>
        </div>
      )}

      {state === "processing" && (
        <div
          className="flex items-center gap-3 h-11 px-5 pl-4 rounded-full border animate-float-in"
          style={{
            background: "#FFFFFF",
            borderColor: "var(--border-subtle)",
            boxShadow: "0 4px 20px rgba(212, 165, 116, 0.2)",
          }}
        >
          {/* Glow Orb */}
          <div
            className="w-3 h-3 rounded-md animate-glow-pulse"
            style={{
              backgroundColor: "var(--glow-processing)",
              boxShadow:
                "0 0 8px 2px var(--glow-processing), 0 0 16px var(--glow-processing-soft)",
            }}
          />

          {/* Label */}
          <span
            className="font-mono text-[11px] tracking-wide"
            style={{ color: "var(--glow-processing)" }}
          >
            transcribing
          </span>

          {/* Loading Dots */}
          <div className="flex items-center gap-1">
            {[1, 0.5, 0.25].map((opacity, i) => (
              <div
                key={i}
                className="w-1 h-1 rounded-full dot-pulse"
                style={{
                  backgroundColor: "var(--glow-processing)",
                  opacity,
                }}
              />
            ))}
          </div>
        </div>
      )}

      {state === "success" && (
        <div
          className="flex items-center gap-2.5 h-11 px-5 pl-4 rounded-full border animate-float-in"
          style={{
            background: "#FFFFFF",
            borderColor: "var(--border-subtle)",
            boxShadow: "0 4px 20px rgba(124, 144, 130, 0.2)",
          }}
        >
          {/* Glow Orb */}
          <div
            className="w-3 h-3 rounded-md"
            style={{
              backgroundColor: "var(--glow-success)",
              boxShadow:
                "0 0 8px 2px var(--glow-success), 0 0 16px var(--glow-success-soft)",
            }}
          />

          {/* Label */}
          <span
            className="font-mono text-[11px] tracking-wide"
            style={{ color: "var(--glow-success)" }}
          >
            inserted
          </span>

          {/* Check Icon */}
          <svg
            className="w-3.5 h-3.5"
            fill="none"
            stroke="var(--glow-success)"
            strokeWidth={2.5}
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M5 13l4 4L19 7"
            />
          </svg>
        </div>
      )}
    </div>
  );
}

export default FloatApp;
