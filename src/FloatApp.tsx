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

const NORMAL_WIDTH = 260;
const NORMAL_HEIGHT = 60;
const HOVER_WIDTH = 264;
const HOVER_HEIGHT = 360;
const BOTTOM_MARGIN = 32;
const AMBIENT_PILL_WIDTH = 44;
const AMBIENT_PILL_HEIGHT = 10;
const AMBIENT_PILL_RADIUS = 5;
const INDICATOR_WIDTH = 240;
const INDICATOR_HEIGHT = 44;
const INDICATOR_RADIUS = 22;

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

function getGlowSoftColor(state: IndicatorState): string {
  switch (state) {
    case "recording":
      return "var(--glow-recording-soft)";
    case "processing":
      return "var(--glow-processing-soft)";
    case "success":
      return "var(--glow-success-soft)";
    default:
      return "var(--glow-idle-soft)";
  }
}

/** Render recording/processing/success indicator content */
function IndicatorContent({
  state,
  duration,
}: {
  state: IndicatorState;
  duration: number;
}) {
  const glowColor = getGlowColor(state);
  const glowSoft = getGlowSoftColor(state);

  if (state === "recording") {
    return (
      <div className="flex items-center gap-3.5 h-full px-5 pl-4">
        <div
          className="w-3 h-3 rounded-md animate-glow-pulse"
          style={{
            backgroundColor: glowColor,
            boxShadow: `0 0 8px 2px ${glowColor}, 0 0 16px ${glowSoft}`,
          }}
        />
        <span
          className="font-mono text-[11px] tracking-wide"
          style={{ color: glowColor }}
        >
          transcribing
        </span>
        <div className="w-px h-4" style={{ backgroundColor: glowSoft }} />
        <span
          className="text-xs font-medium font-mono"
          style={{
            color: "var(--text-secondary)",
            fontVariantNumeric: "tabular-nums",
            minWidth: "2.5em",
            textAlign: "right",
          }}
        >
          {formatDuration(duration)}
        </span>
        <div className="flex items-center gap-[3px] h-5">
          {[6, 14, 8, 18, 10, 14].map((height, i) => (
            <div
              key={i}
              className="w-0.5 rounded-sm wave-bar"
              style={{
                height: `${height}px`,
                backgroundColor: glowColor,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (state === "processing") {
    return (
      <div className="flex items-center gap-3 h-full px-5 pl-4">
        <div
          className="w-3 h-3 rounded-md animate-glow-pulse"
          style={{
            backgroundColor: glowColor,
            boxShadow: `0 0 8px 2px ${glowColor}, 0 0 16px ${glowSoft}`,
          }}
        />
        <span
          className="font-mono text-[11px] tracking-wide"
          style={{ color: glowColor }}
        >
          transcribing
        </span>
        <div className="flex items-center gap-1">
          {[1, 0.5, 0.25].map((opacity, i) => (
            <div
              key={i}
              className="w-1 h-1 rounded-full dot-pulse"
              style={{
                backgroundColor: glowColor,
                opacity,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (state === "success") {
    return (
      <div className="flex items-center gap-2.5 h-full px-5 pl-4">
        <div
          className="w-3 h-3 rounded-md"
          style={{
            backgroundColor: glowColor,
            boxShadow: `0 0 8px 2px ${glowColor}, 0 0 16px ${glowSoft}`,
          }}
        />
        <span
          className="font-mono text-[11px] tracking-wide"
          style={{ color: glowColor }}
        >
          inserted
        </span>
        <svg
          className="w-3.5 h-3.5"
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

  const prevModeRef = useRef<"ambient" | "normal" | "hidden">("hidden");
  const prevStateRef = useRef<IndicatorState>("idle");
  const morphTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hoveredRef = useRef(false);
  const hoverPanelMountedRef = useRef(false);

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

        if (newState === "idle") {
          // Delay hide to allow collapse animation to play out
          const hideDelay = prevModeRef.current !== "hidden" ? 350 : 100;
          setTimeout(() => {
            setVisible(false);
            win.hide();
            prevModeRef.current = "hidden";
          }, hideDelay);
        } else {
          const isAmbient =
            newState === "ambient" || newState === "ambient-active";
          const newMode = isAmbient ? "ambient" : "normal";

          // When listening, always use HOVER size (morph happens within this window)
          if (nextIsListening) {
            if (prevModeRef.current === "hidden") {
              await resizeAndPosition(HOVER_WIDTH, HOVER_HEIGHT);
            }
          } else {
            // Not listening — use normal size for indicators
            if (prevModeRef.current === "hidden" || prevModeRef.current !== newMode) {
              const [w, h] = [NORMAL_WIDTH, NORMAL_HEIGHT];
              await resizeAndPosition(w, h);
            }
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
      setIsHoverPanelMounted(false);
    }
    if (state === "idle") {
      setRecentEntries([]);
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
      // Ambient/Idle -> Indicator: start expanding
      setMorphPhase("expanding");
      morphTimerRef.current = setTimeout(() => {
        setMorphPhase("indicator");
      }, 260);
    } else if (prevIsIndicator && curIsAmbientOrIdle) {
      // Indicator -> Ambient/Idle: start collapsing
      setMorphPhase("collapsing");
      morphTimerRef.current = setTimeout(() => {
        setMorphPhase("ambient");
      }, 300);
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

  // Keep ambient window fixed as hover size to avoid flicker while expanding/collapsing.
  useEffect(() => {
    if (!visible) return;
    if (isListening) {
      // Always HOVER size when listening (morph container needs space)
      void resizeAndPosition(HOVER_WIDTH, HOVER_HEIGHT);
    } else {
      // Not listening — always use normal size
      void resizeAndPosition(NORMAL_WIDTH, NORMAL_HEIGHT);
    }
  }, [visible, isListening]);

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

  // ---- Listening mode: unified morph render ----
  if (isListening) {
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
        {/* Hover panel — only in ambient phase */}
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

        {/* Morph pill — single element that transitions between ambient pill and indicator */}
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
            <IndicatorContent state={state} duration={duration} />
          </div>
        </div>
      </div>
    );
  }

  // ---- Not listening: morph pill centered in normal-size window ----
  {
    const isInIndicatorPhase =
      morphPhase === "expanding" || morphPhase === "indicator";
    const isCollapsing = morphPhase === "collapsing";

    // Standalone morph: starts from a tiny dot (no ambient pill to morph from)
    const DOT_SIZE = 8;
    const pillWidth = isInIndicatorPhase ? INDICATOR_WIDTH : DOT_SIZE;
    const pillHeight = isInIndicatorPhase ? INDICATOR_HEIGHT : DOT_SIZE;
    const pillRadius = isInIndicatorPhase ? INDICATOR_RADIUS : DOT_SIZE / 2;
    const pillBg = isInIndicatorPhase ? "#FFFFFF" : "rgba(0, 0, 0, 0.15)";
    const pillBorder = isInIndicatorPhase
      ? "1px solid var(--border-subtle)"
      : "1px solid transparent";
    const pillShadow = isInIndicatorPhase ? getIndicatorShadow(state) : "none";

    // Opacity: visible when expanding/indicator, fades out when collapsing
    const pillOpacity = isCollapsing ? 0 : isInIndicatorPhase ? 1 : 0;
    const contentVisible = morphPhase === "indicator";

    return (
      <div className="h-screen w-screen flex items-center justify-center bg-transparent">
        <div
          className="morph-pill flex items-center justify-center"
          data-morph-phase={morphPhase}
          style={{
            width: pillWidth,
            height: pillHeight,
            borderRadius: pillRadius,
            backgroundColor: pillBg,
            border: pillBorder,
            boxShadow: pillShadow,
            opacity: pillOpacity,
            overflow: "hidden",
            flexShrink: 0,
          }}
        >
          <div
            className="morph-pill-content w-full h-full"
            style={{ opacity: contentVisible ? 1 : 0 }}
          >
            <IndicatorContent state={state} duration={duration} />
          </div>
        </div>
      </div>
    );
  }
}

export default FloatApp;
