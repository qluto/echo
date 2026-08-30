# ホットキー録音中の途中経過表示（partial）試作

ブランチ: feature/streaming-partial-refine
参考: https://github.com/oboroge0/hayamimi の partial 方式（0.5s ごとに末尾 ≤8s を再デコード）

## 実装済み
- [x] `audio_capture.rs`: `start_recording_with_tap()` — 録音中の音声を 16kHz mono 512-sample フレームで
      並行配信（既存 `FrameAccumulator` を再利用。WAV 書き出しはそのまま）
- [x] `partial.rs`（新規）: `PartialDecoder` スレッド。0.5s 分の新音声ごとに末尾 ≤8s を
      `ASREngine::transcribe_samples` で再デコードし `hotkey-partial` を emit。
      エンジンは `try_lock` のみ（busy なら tick をスキップ）。Whisper 時は自動無効（`supports_partial`）
- [x] `hotkey.rs`: 押下で decoder 起動、離した直後に `stop()`（join）してから最終デコード → 古い draft が後から出ない
- [x] `transcription.rs`: `transcribe_samples(samples, lang, vad_gate)` を追加、`transcribe(path)` はそれに委譲
- [x] FloatApp: 録音中、ピルの上に draft 吹き出し（最大 3 行）を表示。録音終了で消える
- [x] cargo check / lib テスト 29 件 / npm run build OK

## 要ユーザー確認（実機）
- [ ] 体感: 話している途中に draft が追従するか、ちらつきが気にならないか
- [ ] ログ `partial: N.Ns window in M ms` が 500ms 以内か（超えるなら PARTIAL_EVERY_SEC を上げる）
- [ ] Whisper 選択時は draft が出ない（仕様）。出したい場合は別途検討
