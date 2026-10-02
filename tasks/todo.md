# Voice Memo → 議事録（録音して、止めたらまとめて書き起こす）

ブランチ: feature/voice-memo-minutes

## 仕様
- 「録音開始」で 16kHz mono WAV を app data の `recordings/` に逐次書き込み（10s ごとに flush）
- 「終了」で VAD チャンク分割（15〜28s）→ タイムスタンプ付き ASR → Qwen3 で議事録化
- 長い録音は分割ノート化 → 統合（map-reduce）
- ホットキー入力とは別ストリーム。ASR ロックはチャンク単位で取るので録音中・処理中もディクテーション可
- 録音/処理中にアプリが落ちた memo は次回起動時 `interrupted` になり再処理できる

## タスク
- [x] rust-asr: `PostProcessor::chat`（system/user/thinking/max_tokens）
- [x] `database.rs`: `memos` テーブル + CRUD
- [x] `memo.rs`: MemoRecorder / Chunker / 書き起こし / 議事録生成 / 処理スレッド
- [x] commands + AppState + lib.rs 登録
- [x] フロント: tauri.ts バインディング、useMemos、MemoSection、MemoModal
- [x] 検証: cargo test（chunker, DB, minutes の分割ロジック）、実音声 E2E（say → WAV → 議事録）、npm run build

## レビュー
- cargo test（memo/database 20 件）通過、`npm run build` 通過
- 実音声 E2E（`say` 合成の会議音声、Parakeet + Qwen3-4B、debug ビルド）:
  - 90 秒: 8 チャンク / ASR 2.7s / 議事録 1 回の LLM 呼び出しで 41s
  - 15 分: 71 チャンク / ASR 15s / 分割ノート 2 回 + 統合 1 回で 79s
- UI は Tauri API をモックしたブラウザで表示確認（待機・録音中・詳細モーダル）
- 未確認: 実マイクでの録音（この実行環境はマイク入力が取れない）。`memo_recorder_writes_wav`（ignored）で確認可能
- 既知: release プロファイルの新規ビルドで mlx-sys の Metal カーネル（fence.metal）がコンパイルエラー。今回の変更とは無関係

## 追加: 後処理モデルを Qwen3.5 へ（同サイズの現行世代）
- [x] `rust-asr/src/llm/qwen3_5.rs` + `gated_delta.rs`（Metal カーネル）を新規ポート、`PostProcessor` は model_type で分岐
- [x] 既定を `mlx-community/Qwen3.5-4B-4bit` に。設定画面に Qwen3.5 4B/2B を追加（Qwen3 も選択可のまま）
- [x] 議事録・要約は思考なしで生成（Qwen3.5 は貪欲デコードの思考モードでループする）
- 検証: クリーンアップ 3 例（自己訂正・フィラー、日英）OK。議事録 E2E は 90 秒音声で 13.7s（Qwen3: 41s）、15 分で 40.7s（Qwen3: 79s）
- 未対応: Qwen3.5-9B（重みが 2 分割）。Qwen4 / 小型の Qwen3.8 はオープンウェイトなし

## 今後の候補
- [ ] 議事録プロンプトを設定画面で編集可能にする
- [ ] システム音声（リモート会議の相手側）の取り込み、話者分離
- [ ] アプリ内での音声再生、タイトル付け・検索

---

# ホットキー録音中の途中経過表示（partial）— PR 済み

ブランチ: feature/streaming-partial-refine
参考: https://github.com/oboroge0/hayamimi の partial 方式

## 実装済み
- [x] 録音音声の 16kHz タップ（`start_recording_with_tap`）と `partial.rs` の draft デコーダ
      （0.5s ごとに末尾 ≤8s を再デコード、8s 超は最も静かな地点で区切って committed に昇格）
- [x] 離したときは committed + 末尾デコードで確定（ファイル読込・リサンプル・全体再デコード不要）
- [x] フロート UI: ピルが draft を内包して伸縮、下段に波形＋タイマー、窓自体も高さ追従（最大 40 行）
- [x] 未確定部分（末尾窓）はぼかし表示、確定で解除
- [x] 実機確認済み（ユーザー）

## 今後の候補
- [ ] 短い発話でも早めに確定表示にしたい場合は `PARTIAL_WINDOW_SEC` を下げる
- [ ] 区切り位置の認識差が気になる場合、速報挿入 → 裏で全体デコードして差し替える二段構成
