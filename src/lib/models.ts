/**
 * ASR model configuration and display utilities.
 * Shared between App.tsx and SettingsPanel.tsx.
 */

export const MODEL_SIZES: Record<string, string> = {
  // Qwen3-ASR models
  "mlx-community/Qwen3-ASR-1.7B-8bit": "1.7B",
  "mlx-community/Qwen3-ASR-0.6B-8bit": "0.6B",
  // Cohere Transcribe (gated; BF16 ~4GB)
  "CohereLabs/cohere-transcribe-03-2026": "2B",
  // Whisper models
  "mlx-community/whisper-large-v3-turbo": "Turbo",
  "mlx-community/whisper-large-v3": "1.5B",
  "mlx-community/whisper-medium": "769M",
  "mlx-community/whisper-small": "244M",
  "mlx-community/whisper-base": "74M",
  "mlx-community/whisper-tiny": "39M",
};

export const MODEL_ORDER = [
  "mlx-community/Qwen3-ASR-0.6B-8bit",
  "mlx-community/Qwen3-ASR-1.7B-8bit",
  "CohereLabs/cohere-transcribe-03-2026",
  "mlx-community/whisper-large-v3-turbo",
  "mlx-community/whisper-large-v3",
  "mlx-community/whisper-medium",
  "mlx-community/whisper-small",
  "mlx-community/whisper-base",
  "mlx-community/whisper-tiny",
];

/// Models that require user-provided HF authentication and license acceptance.
/// Hidden from the model picker unless the user opts in via Advanced Settings.
export const GATED_MODELS = new Set<string>([
  "CohereLabs/cohere-transcribe-03-2026",
]);

export const isGatedModel = (name: string): boolean => GATED_MODELS.has(name);

export const SUPPORTED_LANGUAGES = [
  { code: "auto", name: "Auto-detect" },
  { code: "ja", name: "Japanese" },
  { code: "en", name: "English" },
  { code: "zh", name: "Chinese" },
  { code: "ko", name: "Korean" },
  { code: "de", name: "German" },
  { code: "fr", name: "French" },
  { code: "es", name: "Spanish" },
];

export const getModelDisplayName = (name: string): string => {
  const parts = name.split("/");
  const modelPart = parts[parts.length - 1];

  if (modelPart.includes("Qwen3-ASR")) {
    return modelPart
      .replace("-8bit", "")
      .replace("Qwen3-ASR-", "Qwen3-ASR ");
  }

  if (modelPart.toLowerCase().includes("cohere-transcribe")) {
    return "Cohere Transcribe 2B";
  }

  return modelPart
    .replace("whisper-", "Whisper ")
    .split("-")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
};

export const getModelSize = (name: string): string => {
  return MODEL_SIZES[name] || "unknown";
};

export const getModelFamily = (name: string): string => {
  if (name.includes("Qwen3-ASR")) return "Qwen3";
  if (name.toLowerCase().includes("cohere-transcribe")) return "Cohere";
  if (name.includes("whisper")) return "Whisper";
  return "Unknown";
};

export const getModelShortName = (name: string): string => {
  const family = getModelFamily(name);
  const size = getModelSize(name);
  return `${family} · ${size}`;
};
