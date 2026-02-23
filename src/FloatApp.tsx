import { useEffect, useState, useRef, useCallback } from "react";
import { listen, emit } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import {
  getCurrentWindow,
  LogicalSize,
  LogicalPosition,
  currentMonitor,
  cursorPosition,
} from "@tauri-apps/api/window";

type IndicatorState =
  | "idle"
  | "recording"
  | "processing"
  | "success"
  | "ambient"
  | "ambient-active";

type MorphPhase = "ambient" | "expanding" | "indicator" | "collapsing";

interface FloatStatePayload {
  state: IndicatorState;
  duration: number;
  isListening: boolean;
}

interface RecentEntry {
  id: number;
  text: string;
  created_at: string;
}

interface HistoryPage {
  entries: RecentEntry[];
}

const HOVER_WIDTH = 264;
const HOVER_HEIGHT = 360;
const BOTTOM_MARGIN = 12;
const AMBIENT_PILL_WIDTH = 44;
const AMBIENT_PILL_HEIGHT = 10;
const AMBIENT_PILL_RADIUS = 5;
const INDICATOR_WIDTH = 120;
const INDICATOR_HEIGHT = 44;
const INDICATOR_RADIUS = 22;

/** Ripple ring decay: first ring strongest, subsequent rings weaker */
const RIPPLE_RINGS = [
  { delay: 0,   borderWidth: 3,   opacity: 0.5  },
  { delay: 150, borderWidth: 2,   opacity: 0.3  },
  { delay: 300, borderWidth: 1.5, opacity: 0.15 },
];

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

function makeLocalTimestamp(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  return `${y}-${m}-${d} ${hh}:${mm}:${ss}`;
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/** Box shadow for indicator states */
function getIndicatorShadow(state: IndicatorState): string {
  switch (state) {
    case "recording":
      return "0 4px 20px rgba(198, 125, 99, 0.2)";
    case "processing":
      return "0 4px 20px rgba(212, 165, 116, 0.2)";
    case "success":
      return "0 4px 20px rgba(124, 144, 130, 0.2)";
    default:
      return "none";
  }
}

/** Glow color CSS variable name for a given state */
function getGlowColor(state: IndicatorState): string {
  switch (state) {
    case "recording":
      return "var(--glow-recording)";
    case "processing":
      return "var(--glow-processing)";
    case "success":
      return "var(--glow-success)";
    default:
      return "var(--glow-idle)";
  }
}

/** Base heights for wave bars (used as minimum when no audio) */
const WAVE_BAR_BASE = [4, 6, 4, 6, 4, 6];
/** Maximum heights for wave bars */
const WAVE_BAR_MAX = [10, 22, 16, 26, 18, 22];
/** Number of history samples to keep for staggered bar animation */
const LEVEL_HISTORY_SIZE = 8;
/** Which history index each bar reads from (higher = more delayed) */
const BAR_DELAY = [0, 2, 4, 1, 3, 5];

/** Wave bars that react to audio levels with per-bar time stagger */
function WaveBars({ audioLevel, glowColor }: { audioLevel: number; glowColor: string }) {
  const historyRef = useRef<number[]>(new Array(LEVEL_HISTORY_SIZE).fill(0));

  // Push new level into history ring, shifting older values
  const history = historyRef.current;
  history.pop();
  history.unshift(audioLevel);

  return (
    <div className="flex items-center gap-[3px] h-7">
      {WAVE_BAR_BASE.map((base, i) => {
        const max = WAVE_BAR_MAX[i];
        const delayed = history[BAR_DELAY[i]] ?? 0;
        const height = base + (max - base) * delayed;
        return (
          <div
            key={i}
            className="w-0.5 rounded-sm"
            style={{
              height: `${height}px`,
              backgroundColor: glowColor,
              transition: "height 60ms ease-out",
            }}
          />
        );
      })}
    </div>
  );
}

/** Render recording/processing/success indicator content */
function IndicatorContent({
  state,
  duration,
  audioLevel,
}: {
  state: IndicatorState;
  duration: number;
  audioLevel: number;
}) {
  const glowColor = getGlowColor(state);

  if (state === "recording") {
    return (
      <div className="flex items-center justify-center gap-3 h-full">
        <WaveBars audioLevel={audioLevel} glowColor={glowColor} />
        <span
          className="text-[11px] font-mono flex-shrink-0"
          style={{
            color: "var(--text-secondary)",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {formatDuration(duration)}
        </span>
      </div>
    );
  }

  if (state === "processing") {
    return (
      <div className="flex items-center justify-center gap-1.5 h-full">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-1 h-1 rounded-full dot-pulse"
            style={{ backgroundColor: glowColor }}
          />
        ))}
      </div>
    );
  }

  if (state === "success") {
    return (
      <div className="flex items-center justify-center h-full">
        <svg
          className="w-[18px] h-[18px]"
          fill="none"
          stroke={glowColor}
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
    );
  }

  return null;
}

function FloatApp() {
  const [state, setState] = useState<IndicatorState>("idle");
  const [duration, setDuration] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [visible, setVisible] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [isHoverPanelMounted, setIsHoverPanelMounted] = useState(false);
  const [recentEntries, setRecentEntries] = useState<RecentEntry[]>([]);
  const [morphPhase, setMorphPhase] = useState<MorphPhase>("ambient");
  const [audioLevel, setAudioLevel] = useState(0);
  const [showRipple, setShowRipple] = useState(false);

  const prevModeRef = useRef<"ambient" | "normal" | "hidden">("hidden");
  const prevStateRef = useRef<IndicatorState>("idle");
  const morphTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hoveredRef = useRef(false);
  const hoverPanelMountedRef = useRef(false);

  // ---- Audio level polling during recording ----
  useEffect(() => {
    if (state !== "recording") {
      setAudioLevel(0);
      return;
    }
    const intervalId = setInterval(async () => {
      try {
        const level = await invoke<number>("get_audio_level");
        setAudioLevel(level);
      } catch (_) {
        // ignore polling errors
      }
    }, 50);
    return () => clearInterval(intervalId);
  }, [state]);

  // ---- Event listeners ----

  useEffect(() => {
    const win = getCurrentWindow();

    const unlistenState = listen<FloatStatePayload>(
      "float-state",
      async (event) => {
        const {
          state: rawState,
          duration: newDuration,
          isListening: nextIsListening,
        } = event.payload;
        setDuration(newDuration);
        setIsListening(nextIsListening);

        // Normalize: when not listening, ambient states mean "idle" (no persistent indicator)
        const isAmbientLike =
          rawState === "ambient" || rawState === "ambient-active";
        const newState =
          !nextIsListening && isAmbientLike ? "idle" : rawState;
        setState(newState as IndicatorState);

        const isAmbient =
          newState === "ambient" || newState === "ambient-active" || newState === "idle";
        const newMode = isAmbient ? "ambient" : "normal";

        // Always use HOVER size (bottom-aligned pill lives in this window)
        if (prevModeRef.current === "hidden") {
          await resizeAndPosition(HOVER_WIDTH, HOVER_HEIGHT);
        }

        prevModeRef.current = newMode;
        setVisible(true);
        win.show();
      },
    );

    // Show window immediately on mount with ambient pill
    void (async () => {
      await resizeAndPosition(HOVER_WIDTH, HOVER_HEIGHT);
      setVisible(true);
      win.show();
    })();

    return () => {
      unlistenState.then((fn) => fn());
    };
  }, []);

  // Accumulate transcription entries for hover panel
  useEffect(() => {
    const unlisten = listen<RecentEntry>(
      "continuous-transcription",
      (event) => {
        const payload = event.payload;
        const normalized: RecentEntry = {
          ...payload,
          created_at: payload.created_at?.trim()
            ? payload.created_at
            : makeLocalTimestamp(),
        };
        setRecentEntries((prev) => [normalized, ...prev].slice(0, 10));
      },
    );
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const isAmbientState = state === "ambient" || state === "ambient-active" || state === "idle";

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

  // Reset hover on non-ambient state transitions
  useEffect(() => {
    if (state !== "ambient" && state !== "ambient-active" && state !== "idle") {
      setIsHovered(false);
      setIsHoverPanelMounted(false);
    }
  }, [state]);

  // ---- Ripple on success ----
  useEffect(() => {
    if (state === "success") {
      setShowRipple(true);
      const timer = setTimeout(() => setShowRipple(false), 900);
      return () => clearTimeout(timer);
    }
  }, [state]);

  // ---- Morph phase detection ----

  useEffect(() => {
    const prev = prevStateRef.current;
    prevStateRef.current = state;

    // Clear any pending morph timer
    if (morphTimerRef.current) {
      clearTimeout(morphTimerRef.current);
      morphTimerRef.current = null;
    }

    const prevIsAmbientOrIdle =
      prev === "ambient" || prev === "ambient-active" || prev === "idle";
    const curIsAmbientOrIdle =
      state === "ambient" || state === "ambient-active" || state === "idle";
    const curIsIndicator =
      state === "recording" || state === "processing" || state === "success";
    const prevIsIndicator =
      prev === "recording" || prev === "processing" || prev === "success";

    if (prevIsAmbientOrIdle && curIsIndicator) {
      // Ambient/Idle -> Indicator: start expanding (matches spring duration)
      setMorphPhase("expanding");
      morphTimerRef.current = setTimeout(() => {
        setMorphPhase("indicator");
      }, 320);
    } else if (prevIsIndicator && curIsAmbientOrIdle) {
      // Indicator -> Ambient/Idle: start collapsing
      setMorphPhase("collapsing");
      morphTimerRef.current = setTimeout(() => {
        setMorphPhase("ambient");
      }, 280);
    } else if (curIsAmbientOrIdle) {
      setMorphPhase("ambient");
    } else if (curIsIndicator) {
      setMorphPhase("indicator");
    }
  }, [state, isListening]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
      if (morphTimerRef.current) clearTimeout(morphTimerRef.current);
    };
  }, []);

  useEffect(() => {
    hoveredRef.current = isHovered;
  }, [isHovered]);

  useEffect(() => {
    hoverPanelMountedRef.current = isHoverPanelMounted;
  }, [isHoverPanelMounted]);

  // ---- Window sizing ----

  // Always use HOVER size — ambient pill lives at bottom of this window.
  useEffect(() => {
    if (!visible) return;
    void resizeAndPosition(HOVER_WIDTH, HOVER_HEIGHT);
  }, [visible]);

  // Make the window click-through when ambient and not morphing/hovered.
  useEffect(() => {
    if (!visible) return;
    const win = getCurrentWindow();
    const shouldIgnore =
      isAmbientState && !isHovered && morphPhase === "ambient";
    void win.setIgnoreCursorEvents(shouldIgnore).catch((error) => {
      console.error("Failed to set ignore cursor events:", error);
    });
  }, [visible, isAmbientState, isHovered, morphPhase]);

  // ---- Hover helpers ----

  const showHoverPanel = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    if (!hoveredRef.current) {
      hoveredRef.current = true;
      setIsHovered(true);
    }
    if (!hoverPanelMountedRef.current) {
      hoverPanelMountedRef.current = true;
      setIsHoverPanelMounted(true);
    }
  }, []);

  const hideHoverPanel = useCallback((delay = 140) => {
    if (hoveredRef.current) {
      hoveredRef.current = false;
      setIsHovered(false);
    }
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    hoverTimeoutRef.current = setTimeout(() => {
      hoverPanelMountedRef.current = false;
      setIsHoverPanelMounted(false);
      hoverTimeoutRef.current = null;
    }, delay);
  }, []);

  // Hover state machine driven by cursor position polling
  useEffect(() => {
    if (!visible || !isAmbientState) return;
    let disposed = false;

    const tick = async () => {
      if (disposed) return;
      try {
        const win = getCurrentWindow();
        const [cursor, winPos, winSize, scaleFactor] = await Promise.all([
          cursorPosition(),
          win.innerPosition(),
          win.innerSize(),
          win.scaleFactor(),
        ]);

        const wx = winPos.x;
        const wy = winPos.y;
        const ww = winSize.width;
        const wh = winSize.height;
        const cx = cursor.x;
        const cy = cursor.y;

        const insideWindow =
          cx >= wx && cx <= wx + ww && cy >= wy && cy <= wy + wh;
        const pillWidth = Math.round(AMBIENT_PILL_WIDTH * scaleFactor);
        const pillHeight = Math.round(AMBIENT_PILL_HEIGHT * scaleFactor);
        const pillBottomPadding = Math.round(15 * scaleFactor);
        const pillX = wx + Math.round((ww - pillWidth) / 2);
        const pillY = wy + (wh - pillBottomPadding - pillHeight);
        const insidePill =
          cx >= pillX &&
          cx <= pillX + pillWidth &&
          cy >= pillY &&
          cy <= pillY + pillHeight;

        if (hoverPanelMountedRef.current) {
          if (!hoveredRef.current) {
            return;
          }
          if (insideWindow) {
            showHoverPanel();
          } else {
            hideHoverPanel(80);
          }
        } else if (insidePill) {
          showHoverPanel();
        }
      } catch (_) {
        // ignore transient cursor/window query failures
      }
    };

    const intervalId = setInterval(() => {
      void tick();
    }, 60);
    void tick();

    return () => {
      disposed = true;
      clearInterval(intervalId);
    };
  }, [visible, isAmbientState, showHoverPanel, hideHoverPanel]);

  const handleToggleListening = useCallback(async () => {
    await emit("request-toggle-listening", {});
  }, []);

  // ---- Render ----

  if (!visible) {
    return null;
  }

  // ---- Unified morph render (always bottom-aligned pill) ----
  const isInIndicatorPhase =
      morphPhase === "expanding" || morphPhase === "indicator";
  const isInAmbientPhase =
    morphPhase === "ambient" || morphPhase === "collapsing";

  // Morph pill dimensions
  const pillWidth = isInIndicatorPhase ? INDICATOR_WIDTH : AMBIENT_PILL_WIDTH;
  const pillHeight = isInIndicatorPhase
    ? INDICATOR_HEIGHT
    : AMBIENT_PILL_HEIGHT;
  const pillRadius = isInIndicatorPhase ? INDICATOR_RADIUS : AMBIENT_PILL_RADIUS;
  const pillBg = isInIndicatorPhase ? "#FFFFFF" : (state === "ambient-active" ? "#7C9082" : "#1A1A1C");
  const pillBorder = isInIndicatorPhase
    ? "1px solid var(--border-subtle)"
    : "1px solid rgba(255, 255, 255, 0.25)";
  const pillShadow = isInIndicatorPhase
    ? getIndicatorShadow(state)
    : state === "ambient-active"
      ? "0 0 3px 1px rgba(124, 144, 130, 0.6), 0 0 8px rgba(124, 144, 130, 0.3)"
      : "0 1px 4px rgba(0, 0, 0, 0.12)";

  // Content is visible only when fully expanded
  const contentVisible = morphPhase === "indicator";

  const displayEntries = [...recentEntries].reverse().slice(-4);

  return (
    <div
      className="h-screen w-screen relative flex items-end justify-center bg-transparent"
      style={{ paddingBottom: 15 }}
    >
      {/* Hover panel — only in ambient phase while listening */}
      {isHoverPanelMounted && isAmbientState && morphPhase === "ambient" && (
          <div
            className="absolute inset-0 flex items-end justify-center bg-transparent p-[2px]"
            style={{ paddingBottom: 15 + AMBIENT_PILL_HEIGHT + 8 }}
          >
            <div
              className={`flex flex-col min-h-0 overflow-hidden origin-bottom ${
                isHovered
                  ? "animate-ambient-hover-in"
                  : "animate-ambient-hover-out"
              }`}
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
                <button
                  onClick={handleToggleListening}
                  aria-pressed={isListening}
                  style={{
                    position: "relative",
                    width: 40,
                    height: 22,
                    borderRadius: 11,
                    backgroundColor: isListening ? "#7C9082" : "#C8CEC9",
                    border: "none",
                    cursor: "pointer",
                    padding: 0,
                    flexShrink: 0,
                    transition: "background-color 120ms ease",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      top: 2,
                      [isListening ? "right" : "left"]: 2,
                      width: 18,
                      height: 18,
                      borderRadius: 9,
                      backgroundColor: "#FFFFFF",
                      boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
                      transition: "left 120ms ease, right 120ms ease",
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

              {/* History list */}
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
            </div>
          </div>
        )}

        {/* Morph pill with ripple wrapper */}
        <div style={{ position: "relative", flexShrink: 0 }}>
          {showRipple &&
            RIPPLE_RINGS.map((ring, i) => (
              <div
                key={`ripple-${i}`}
                className="echo-ripple-ring"
                style={{
                  width: INDICATOR_WIDTH,
                  height: INDICATOR_HEIGHT,
                  borderRadius: INDICATOR_RADIUS,
                  border: `${ring.borderWidth}px solid var(--glow-success)`,
                  animationDelay: `${ring.delay}ms`,
                  "--ripple-opacity": ring.opacity,
                } as React.CSSProperties}
              />
            ))}
          <div
            className={`morph-pill flex items-center justify-center ${
              isInAmbientPhase && state === "ambient-active" && morphPhase === "ambient"
                ? "animate-ambient-breathe"
                : ""
            }`}
            data-morph-phase={morphPhase}
            style={{
              width: pillWidth,
              height: pillHeight,
              borderRadius: pillRadius,
              backgroundColor: pillBg,
              border: pillBorder,
              boxShadow: pillShadow,
              overflow: "hidden",
              flexShrink: 0,
            }}
          >
            {/* Indicator content — fades in/out */}
            <div
              className="morph-pill-content w-full h-full"
              style={{ opacity: contentVisible ? 1 : 0 }}
            >
              <IndicatorContent state={state} duration={duration} audioLevel={audioLevel} />
            </div>
          </div>
        </div>
      </div>
  );
}

export default FloatApp;
