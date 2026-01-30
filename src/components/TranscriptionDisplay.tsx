import { TranscriptionResult } from "../lib/tauri";

interface TranscriptionDisplayProps {
  result: TranscriptionResult | null;
  error: string | null;
  onClear: () => void;
  onInsert: () => void;
  onCopy: () => void;
}

export function TranscriptionDisplay({
  result,
  error,
  onClear,
  onInsert,
  onCopy,
}: TranscriptionDisplayProps) {
  if (error) {
    return (
      <div className="bg-red-900/30 border border-red-700 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <svg
            className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
          <div className="flex-1">
            <p className="text-red-400 font-medium">エラーが発生しました</p>
            <p className="text-red-300 text-sm mt-1">{error}</p>
          </div>
          <button
            onClick={onClear}
            className="text-red-400 hover:text-red-300 transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-8 text-center">
        <svg
          className="w-16 h-16 text-gray-600 mx-auto mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
        </svg>
        <p className="text-gray-400">
          ホットキー <kbd className="px-2 py-1 bg-gray-700 rounded text-sm">Cmd+Shift+Space</kbd> を押して録音を開始
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">言語:</span>
          <span className="px-2 py-0.5 bg-primary-600/20 text-primary-400 text-sm rounded">
            {result.language || "自動検出"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onCopy}
            className="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded transition-colors flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            コピー
          </button>
          <button
            onClick={onInsert}
            className="px-3 py-1.5 text-sm bg-primary-600 hover:bg-primary-500 rounded transition-colors flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
              />
            </svg>
            挿入
          </button>
          <button
            onClick={onClear}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>
      <div className="p-4">
        <p className="text-lg text-white leading-relaxed whitespace-pre-wrap">
          {result.text}
        </p>
      </div>
      {result.segments && result.segments.length > 0 && (
        <div className="p-4 border-t border-gray-700 bg-gray-850">
          <p className="text-sm text-gray-400 mb-2">セグメント</p>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {result.segments.map((segment, index) => (
              <div
                key={index}
                className="flex items-start gap-2 text-sm"
              >
                <span className="text-gray-500 font-mono whitespace-nowrap">
                  {segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s
                </span>
                <span className="text-gray-300">{segment.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
