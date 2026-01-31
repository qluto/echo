import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

type IndicatorState = "idle" | "recording" | "processing" | "success";

interface FloatStatePayload {
  state: IndicatorState;
  duration: number;
}

function FloatApp() {
  const [state, setState] = useState<IndicatorState>("idle");
  const [duration, setDuration] = useState(0);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const window = getCurrentWindow();

    // Listen for state changes from main window
    const unlistenState = listen<FloatStatePayload>("float-state", (event) => {
      const { state: newState, duration: newDuration } = event.payload;
      setState(newState);
      setDuration(newDuration);

      if (newState === "idle") {
        // Hide after a brief delay for success state
        setTimeout(() => {
          setVisible(false);
          window.hide();
        }, 100);
      } else {
        setVisible(true);
        window.show();
      }
    });

    // Initially hide the window
    window.hide();

    return () => {
      unlistenState.then((fn) => fn());
    };
  }, []);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  if (!visible) {
    return null;
  }

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-transparent">
      {state === "recording" && (
        <div
          className="flex items-center gap-3.5 h-11 px-4 pl-4 rounded-[22px] border"
          style={{
            background: "linear-gradient(180deg, #12121A 0%, #0A0A0F 100%)",
            borderColor: "rgba(255, 59, 92, 0.3)",
            boxShadow: "0 4px 40px rgba(255, 59, 92, 0.18)",
          }}
        >
          {/* Glow Orb */}
          <div
            className="w-3 h-3 rounded-md animate-glow-pulse"
            style={{
              backgroundColor: "var(--glow-recording)",
              boxShadow:
                "0 0 12px 2px var(--glow-recording), 0 0 24px var(--glow-recording-soft)",
            }}
          />

          {/* Label */}
          <span
            className="font-display text-[11px] tracking-[0.5px]"
            style={{ color: "var(--glow-recording)" }}
          >
            recording
          </span>

          {/* Divider */}
          <div
            className="w-px h-4"
            style={{ backgroundColor: "rgba(255, 59, 92, 0.3)" }}
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
          className="flex items-center gap-3 h-11 px-4 pl-4 rounded-[22px] border"
          style={{
            background: "linear-gradient(180deg, #14140A 0%, #0C0C08 100%)",
            borderColor: "rgba(255, 184, 0, 0.3)",
            boxShadow: "0 4px 40px rgba(255, 184, 0, 0.15)",
          }}
        >
          {/* Glow Orb */}
          <div
            className="w-3 h-3 rounded-md animate-glow-pulse"
            style={{
              backgroundColor: "var(--glow-processing)",
              boxShadow:
                "0 0 12px 2px var(--glow-processing), 0 0 24px var(--glow-processing-soft)",
            }}
          />

          {/* Label */}
          <span
            className="font-display text-[11px] tracking-[0.5px]"
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
          className="flex items-center gap-2.5 h-11 px-5 pl-4 rounded-[22px] border"
          style={{
            background: "linear-gradient(180deg, #0A140E 0%, #080F0A 100%)",
            borderColor: "rgba(0, 255, 148, 0.3)",
            boxShadow: "0 4px 40px rgba(0, 255, 148, 0.15)",
          }}
        >
          {/* Glow Orb */}
          <div
            className="w-3 h-3 rounded-md"
            style={{
              backgroundColor: "var(--glow-success)",
              boxShadow:
                "0 0 12px 2px var(--glow-success), 0 0 24px var(--glow-success-soft)",
            }}
          />

          {/* Label */}
          <span
            className="font-display text-[11px] tracking-[0.5px]"
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
