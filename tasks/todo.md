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
