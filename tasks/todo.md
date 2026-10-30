# 会議同伴モード（Meeting Companion）

マイク（自分）+ システム音声（Zoom/Meet などの相手）をリアルタイムに書き起こし、3 分おき（60 秒オーバーラップ）に
決定事項 / ネクストアクション / 残論点を LLM で差分更新（追加・更新・削除）する。会議中の手編集はしない。

## 設計
- システム音声: Core Audio Process Tap（macOS 14.2+、`CATapDescription` global tap + private aggregate device）
  → `FrameAccumulator` で 16 kHz mono。権限は `NSAudioCaptureUsageDescription`。失敗時はマイクのみで続行（warning 表示）
- チャンク化: ソースごと。VAD なし（メモの教訓）。≥4 s で末尾 0.5 s が間なら切る / 15 s で最も静かな点で切る。
  ほぼ無音（全フレーム < -55 dBFS）は ASR しない
- ASR: 1 本のワーカー、チャンクごとに engine lock。話者ラベル 自分 / 相手
- エコー除去: 自分チャンクの時間帯に相手側が鳴っていたら、相手側の書き起こしが追いつくまで保留し、
  文字 bigram の包含率 ≥ 0.6 なら捨てる
- 議事メモ: モデルは `Facts:`（新しい発話の事実列挙）→ `Notes:`（ID 付きで全項目を書き直し、新規は `[new]`）を出力。
  コードが差分（追加・更新・種別移動・削除）を計算。入力 = 現在の項目 + 書き起こし窓（前回反映済みの 60 s は文脈扱い）。停止時に最終ラウンド
- 無音・雑音ゲート: チャンクが「直近 30 s の雑音床 +15 dB を 160 ms 以上」なら ASR、そうでなければ Silero VAD が音声と判定したときだけ ASR
- 保存: memos 行（ミックス WAV、話者付き書き起こし、項目を Markdown 化した議事録）

## コマンド / イベント契約
- `start_meeting() -> MeetingSnapshot` / `stop_meeting() -> MeetingSnapshot`（停止後は finalizing → done）
- `get_meeting_snapshot() -> MeetingSnapshot | null` / `get_meeting_status() -> MeetingStatus | null`（200 ms ポーリング）
- `open_meeting_window()`
- events: `meeting-line` (MeetingLine), `meeting-items` ({items, round}), `meeting-state` ({phase, memo_id, error, warning})

## タスク
- [x] `system_audio.rs`: Process Tap キャプチャ + ignored 実機テスト（afplay の音を取得、peak 0.157）
- [x] `meeting.rs`: Chunker / 雑音ゲート / エコー判定 / ノート解析・差分化 / 窓選択 / Markdown 化（単体テスト 10 件）
- [x] `meeting.rs`: セッション（音声スレッド・ASR ワーカー・議事メモスレッド・停止処理・memo 保存）
- [x] commands + AppState + Info.plist + capabilities + tauri.conf（meeting ウィンドウ）
- [x] フロント: `meeting.html` / `MeetingApp.tsx` / `useMeeting.ts` / tauri.ts / メイン画面のカード
- [x] 検証: cargo test（56 件）、TTS 合成会議 2 本 × Parakeet / Whisper で E2E（`meeting_end_to_end`）
- [ ] アプリ実機での通し確認（ウィンドウ操作、権限ダイアログ、イベント、memo 保存）— 未実施

## 検証結果（合成会議: `say` で 2 話者・2 トラック、30 s ごとにラウンド）
- 差分命令方式（ADD/UPDATE/DELETE）は Gemma 4 E4B では不安定: 訂正を別項目に書く、関係ない項目を DELETE、
  ADD の書式崩れで項目が消える → 「全項目を ID 付きで書き直し → コードで差分」に変更して解消
- s1（日付変更・担当交代・論点の決着・撤回）: Parakeet で最終状態が期待どおり 5/5
- s2（画面を見た上での口頭訂正「火曜ではなく水曜」・保留→決定・不採用）: 両エンジンで決定 4 件・訂正とも反映
- 雑音のみのマイク（-50 dBFS）: ゲート前は Whisper が「ご視聴ありがとうございました」を 15 行、Parakeet が「うん。」を 15 行
  → ゲート後 0 行。同じ雑音に発話を重ねても自分の発話は全件残る（両エンジン・両シナリオ）
- スピーカー漏れ（ヘッドホンなし、-10 dB）: 相手だけのチャンクは除去できるが、自分の発話と同じチャンクに混ざった分は残る

## 不具合: 議事メモ更新で "There is no Stream(cpu, 0) in current thread"
- 原因: MLX 0.32 からストリームがスレッド固有。重みは遅延ロードなので、あるスレッドでロードしたモデルを
  別スレッドで初めて評価すると失敗する。後処理 LLM は起動時にメインスレッドでロードされ、会議の議事メモスレッドで失敗
  （ボイスメモの議事録スレッド、Parakeet/Cohere を別スレッドで使う経路にも同じ潜在バグ。MLX 0.32 移行時から）
- 対策: `rust-asr/src/streams.rs`。公開エントリポイントの先頭で、呼び出しスレッドの既定ストリームを
  プロセス共通の `mlx_stream_new_thread_unsafe` ストリームに向ける（呼び出しは ASREngine の mutex で直列化済み）
- 検証: `rust-asr/tests/threads.rs`（ロードと実行を別スレッド、LLM と Parakeet）が修正前に同じエラーで失敗 → 修正後に成功。
  デコード速度は 43.2 tok/s で変化なし、既存テスト全件成功

## LLM 実行中も書き起こしを止めない + 議事録用 LLM の別設定
- 原因: ASR と LLM が `ASREngine` の mutex を共有し、LLM は生成の間ずっと保持していた
- `rust-asr/src/gpu.rs`: GPU 作業を通す FIFO チケットロック（std の Mutex は macOS で不公平で、
  デコードループが取り直し続けて ASR が 8.6 s 待たされた）。ASR はチャンク単位、LLM は prefill チャンク / デコード 1 ステップ単位で保持
- `ASREngine`: LLM を `Arc<PostProcessor>` のマップで保持し、生成はエンジンの mutex 外で実行。
  ダウンロードを含むロードも mutex 外。ディクテーション整形・履歴要約も同様
- 議事録 / 会議メモ用 LLM を別設定に（`Settings.notes_model`、設定画面「Minutes & Meeting Notes」）。
  選択時に未ダウンロードなら取得。メモ・会議の終了後、ディクテーションと別モデルなら解放
- 検証: `rust-asr/tests/threads.rs`（LLM 生成中の Parakeet 92 ms / 単独 67 ms、Whisper 遅延なし、出力は単独実行と一致）、
  `transcription_continues_during_notes_llm`（エンジン mutex 経由で生成中にチャンク 4 本、最遅 0.40 s）、既存テスト全件

## 議事メモ用 LLM: Qwen3.6-27B を採用（2026-10-29）
- 実機テストで議事メモが空: Gemma 4 E4B が Facts に事実を挙げつつ、空の Notes をそのまま写していた
  （「ネクストアクションとしては…」と明言しても空）
- 評価: 合成会議 2 本 + 実会議 1 本、mlx-lm・同一プロンプト・同一ラウンド。チェック 23 項目
  | モデル | 合計 | 1 ラウンド | サイズ |
  |---|---|---|---|
  | Gemma 4 E4B | 17（実会議 空） | 1–5 s | 5 GB |
  | Qwen3.5-9B | 15（例示の項目を混入） | 3–11 s | 6 GB |
  | Gemma 4 12B | 12（繰り返しループ） | 97 s | 7 GB |
  | Gemma 4 26B-A4B (MoE) | 20 | 2–5 s | 15 GB |
  | Qwen3.6-35B-A3B (MoE) | 20 | 3–5 s | 20 GB |
  | **Qwen3.6-27B** | **22** → プロンプト調整後 23 | 9–28 s | 16 GB |
- 実装: シャード読み込み（index.json）、Qwen3.5 ポートの非共有 lm_head、議事録専用モデル一覧、
  RAM 32 GB 以上で既定 27B、会議開始時に事前ロード。prefill チャンク 2048 → 512（27B で ASR 待ち 4.3 s → 2.1 s）
- Rust ポートは mlx-lm と s1/s2 で最終状態一致、実会議は表現差 2 か所のみ
- プロンプト調整: 合意した日程は DECISION / ラベルに「さん」を付けない / 決着した結果自体を DECISION に残す
- 候補 4 モデル（約 48 GB）は削除済み

## 既知の制約 / 残課題
- 27B は常駐 ~16 GB。32 GB 機ではディクテーション用 E4B・ASR と合わせて ~22 GB、Zoom やブラウザが重いとスワップの恐れ
- 27B でも細部の揺れはある（完了したタスクが残る、「自分さん」表記）
- 議事メモ更新の精度は Gemma 4 E4B で揺れる（決着した論点を移動せず新規の決定として足し、論点が残ることがある）
- 自分の発話と混ざったエコーは除去できない（ヘッドホン推奨。根本対策は macOS の Voice Processing によるエコーキャンセル）
- 会議中に出力デバイスを切り替えた場合の Process Tap の挙動は未確認
- 1 ラウンド目（項目ゼロ）では論点の拾いが控えめ。後続ラウンドで補われる
- 相手側は話者分離なし（全員「相手」）
