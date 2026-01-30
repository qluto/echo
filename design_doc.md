# Echo - 音声入力アプリケーション 設計ドキュメント

## プロジェクト概要

**プロジェクト名**: Echo

**目的**: Apple Silicon最適化されたオフライン音声入力デスクトップアプリケーション

**主要機能**:
- Qwen3-ASRモデルを使用した高精度な音声認識
- グローバルホットキーによる音声録音
- リアルタイム文字起こし
- 任意のアプリケーションへのテキスト自動挿入
- 完全オフライン動作

**ターゲットプラットフォーム**: macOS (Apple Silicon)

**参考プロジェクト**:
- [Handy](https://github.com/cjpais/Handy) - Tauriベース音声文字化アプリケーション
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Apple Silicon最適化音声認識ライブラリ

---

## 技術スタック

### フロントエンド
- **React** 18.x - UIフレームワーク
- **TypeScript** 5.x - 型安全な開発
- **Vite** 5.x - 高速ビルドツール
- **Tailwind CSS** 3.x - ユーティリティファーストCSS

### バックエンド
- **Tauri** 2.x - デスクトップアプリケーションフレームワーク
- **Rust** 1.83+ - システムレベルプログラミング
- **cpal** 0.15 - クロスプラットフォーム音声I/O
- **hound** 3.5 - WAVファイル処理

### ML/音声認識エンジン
- **Python** 3.11+ (ARM native)
- **MLX** 0.30+ - Apple Silicon最適化MLフレームワーク
- **mlx-audio** - 音声認識ライブラリ
- **Qwen3-ASR** 1.7B (8bit量子化) - マルチリンガル音声認識モデル

### ビルド/配布
- **PyInstaller** - Pythonスクリプトのバイナリ化
- **Tauri Sidecar** - 外部バイナリのバンドル

---

## アーキテクチャ設計

### 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                        Echo.app                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐         ┌──────────────────┐          │
│  │  React UI       │◄────────┤  Tauri Core      │          │
│  │  (TypeScript)   │  Events │  (Rust)          │          │
│  └─────────────────┘         └──────────────────┘          │
│                                       │                       │
│                    ┌──────────────────┼──────────────────┐   │
│                    │                  │                  │   │
│           ┌────────▼────────┐ ┌──────▼──────┐ ┌────────▼───┐
│           │ Audio Capture   │ │  Hotkey     │ │ Sidecar    │
│           │ (cpal)          │ │  Manager    │ │ Manager    │
│           └────────┬────────┘ └─────────────┘ └────────┬───┘
│                    │                                    │    │
│                    │ WAV                          stdin/│    │
│                    │ File                        stdout │    │
│                    │                                    │    │
│                    │         ┌──────────────────────────▼──┐ │
│                    └────────►│  mlx-asr-engine             │ │
│                              │  (Python Sidecar Binary)    │ │
│                              │                             │ │
│                              │  ┌───────────────────────┐  │ │
│                              │  │ MLX Framework         │  │ │
│                              │  │ mlx-audio             │  │ │
│                              │  │ Qwen3-ASR Model       │  │ │
│                              │  └───────────────────────┘  │ │
│                              └─────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### データフロー

1. **音声録音フロー**
   ```
   ユーザー入力（ホットキー）
   → Rust Hotkey Manager
   → Audio Capture (cpal)
   → WAVファイル保存
   → Sidecar Manager に文字起こし要求
   ```

2. **文字起こしフロー**
   ```
   Sidecar Manager (Rust)
   → JSON-RPC over stdin/stdout
   → Python ASR Engine
   → MLX-Audio + Qwen3-ASR
   → 文字起こし結果をJSON返却
   → React UIに表示
   ```

3. **テキスト挿入フロー** (Handy方式)
   ```
   文字起こし結果
   → tauri-plugin-clipboard-manager でクリップボードにコピー
   → enigo クレートで Cmd+V (macOS) / Ctrl+V (Windows/Linux) をシミュレート
   → 100ms待機（システム処理時間確保）
   → アクティブアプリケーションに挿入
   ```

---

## MLX-Audio統合戦略: Sidecar方式

### 選択理由

Python依存関係の配布における以下の選択肢を検討：

| 方式 | 初期DL | オフライン | 実装難易度 |
|------|--------|-----------|-----------|
| **Sidecar (PyInstaller)** | 200MB | ✅ | ⭐⭐☆☆☆ |
| 埋め込みPython | 600MB | ✅ | ⭐⭐⭐☆☆ |
| 初回セットアップ | 80MB | ❌ | ⭐⭐☆☆☆ |
| Rust完全実装 | 100MB | ✅ | ⭐⭐⭐⭐⭐ |

**結論**: Sidecar方式を採用
- 配布サイズと実装難易度のバランスが最適
- オフライン動作を保証
- MLX-Audioの最新機能を活用可能

### Sidecar実装詳細

#### Python ASRエンジン (engine.py)

**実行モード**:
- `daemon`: 常駐モード（stdin/stdoutでJSON-RPC通信）
- `single`: 単発実行モード（コマンドライン引数）

**JSON-RPCプロトコル**:

リクエスト:
```json
{
  "command": "transcribe",
  "id": 1,
  "audio_path": "/path/to/audio.wav",
  "language": "ja"
}
```

レスポンス:
```json
{
  "id": 1,
  "result": {
    "success": true,
    "text": "こんにちは世界",
    "segments": [
      {"start": 0.0, "end": 1.5, "text": "こんにちは"},
      {"start": 1.5, "end": 2.0, "text": "世界"}
    ],
    "language": "ja"
  }
}
```

#### PyInstallerビルド設定

```bash
pyinstaller \
  --onefile \
  --name mlx-asr-engine \
  --hidden-import mlx \
  --hidden-import mlx_audio \
  --collect-all mlx \
  --collect-all mlx_audio \
  --target-arch arm64 \
  engine.py
```

**出力**: `mlx-asr-engine-aarch64-apple-darwin` (約200-300MB)

#### Tauriバンドル設定

```json
{
  "bundle": {
    "externalBin": [
      "binaries/mlx-asr-engine-aarch64-apple-darwin"
    ]
  }
}
```

---

## プロジェクト構造

```
echo/
├── src/                              # React フロントエンド
│   ├── App.tsx                       # メインアプリケーション
│   ├── components/
│   │   ├── SettingsPanel.tsx         # 設定画面
│   │   ├── RecordingOverlay.tsx      # 録音中オーバーレイ
│   │   └── TranscriptionDisplay.tsx  # 文字起こし結果表示
│   ├── lib/
│   │   └── tauri.ts                  # Tauri API ラッパー
│   ├── hooks/
│   │   └── useTranscription.ts       # 文字起こしカスタムフック
│   └── styles/
│       └── globals.css               # グローバルスタイル
│
├── src-tauri/                        # Rust バックエンド
│   ├── src/
│   │   ├── main.rs                   # エントリーポイント
│   │   ├── lib.rs                    # ライブラリルート
│   │   ├── audio_capture.rs          # 音声キャプチャ
│   │   ├── transcription.rs          # Sidecar通信
│   │   ├── hotkey.rs                 # グローバルホットキー
│   │   ├── input.rs                  # テキスト挿入（enigo使用）
│   │   └── clipboard.rs              # クリップボード操作
│   ├── binaries/                     # Sidecarバイナリ配置先
│   │   └── mlx-asr-engine-aarch64-apple-darwin
│   ├── resources/                    # バンドルリソース
│   │   └── models/                   # MLXモデル（オプション）
│   ├── icons/                        # アプリアイコン
│   ├── tauri.conf.json               # Tauri設定
│   ├── Cargo.toml                    # Rust依存関係
│   └── build.rs                      # ビルドスクリプト
│
├── python-engine/                    # Python ASRエンジン
│   ├── engine.py                     # メインスクリプト
│   ├── engine.spec                   # PyInstaller設定
│   ├── requirements.txt              # Python依存関係
│   └── build.sh                      # バイナリビルドスクリプト
│
├── scripts/
│   └── setup-dev.sh                  # 開発環境セットアップ
│
├── design_doc.md                     # 本ドキュメント
├── package.json                      # Node.js依存関係
├── vite.config.ts                    # Vite設定
├── tailwind.config.js                # Tailwind CSS設定
└── README.md                         # プロジェクト説明
```

---

## 主要機能仕様

### 1. グローバルホットキー録音

**要件**:
- カスタマイズ可能なショートカット（デフォルト: `Cmd+Shift+Space`）
- Push-to-talk方式（キー押下中のみ録音）
- システム全体で動作

**実装**:
- Rustクレート: `global-hotkey`
- macOS Accessibility権限が必要

### 2. 音声キャプチャ

**要件**:
- デフォルト入力デバイスから録音
- 16kHz モノラル WAV形式
- VAD（Voice Activity Detection）で無音除去

**実装**:
- `cpal`で音声ストリーム取得
- `hound`でWAVファイル書き出し
- オプション: `vad-rs`で無音検出

### 3. 音声認識

**要件**:
- Qwen3-ASR 1.7B (8bit量子化) 使用
- マルチリンガル対応（日本語、英語、韓国語など52言語）
- タイムスタンプ付きセグメント出力

**実装**:
- Python Sidecarで処理
- MLX-Audioライブラリ使用
- JSON-RPC通信でRustと連携

### 4. テキスト自動挿入

**要件**:
- 文字起こし結果を任意のアプリに挿入
- クリップボード経由でペースト（安定性重視）
- プラットフォーム固有のペーストショートカット対応

**実装** (Handy方式に準拠):
- **Clipboard方式**（推奨）:
  - `tauri-plugin-clipboard-manager` v2.3+ でクリップボード操作
  - `enigo` v0.6+ でキーボードイベントシミュレート
  - プラットフォーム別ペースト実装:
    - macOS: `Cmd+V`
    - Windows/Linux: `Ctrl+V` または `Shift+Insert`
  - 100ms遅延でシステム処理時間を確保
- **Direct方式**（参考）:
  - `enigo.text()` でUnicode文字を直接送信
  - キーボードレイアウト依存の問題あり（非推奨）

**参考**: [Handy Issue #439](https://github.com/cjpais/Handy/issues/439) - Direct方式のキーボードレイアウト問題

### 5. 設定管理

**要件**:
- モデル選択（Qwen3-ASR各バージョン）
- 言語設定（自動検出 or 手動指定）
- ホットキーカスタマイズ
- 音声デバイス選択

**実装**:
- Tauri Storeで永続化
- React設定画面

### 6. 録音状態UI

**要件**:
- 録音中インジケーター表示
- リアルタイムトランスクリプション（オプション）
- デバッグモード

**実装**:
- オーバーレイウィンドウ
- Tailwind CSSアニメーション

---

## 実装詳細

### Rust: ASREngine (transcription.rs)

**責務**:
- Sidecarプロセスのライフサイクル管理
- JSON-RPC通信
- エラーハンドリング

**主要API**:
```rust
impl ASREngine {
    pub fn new() -> Self;
    pub fn start(&self, sidecar_path: &str) -> Result<()>;
    pub fn stop(&self) -> Result<()>;
    pub fn transcribe(&self, audio_path: String, language: Option<String>) -> Result<TranscriptionResult>;
    pub fn ping(&self) -> Result<bool>;
}
```

### Python: ASREngine (engine.py)

**責務**:
- MLX-Audioモデルの管理
- 音声ファイルの文字起こし
- JSON-RPC通信

**主要API**:
```python
class ASREngine:
    def __init__(self, model_name: str):
    def load_model(self):
    def transcribe(self, audio_path: str, language: str = None) -> Dict[str, Any]:
```

### React: useTranscription Hook

**責務**:
- Tauri Command呼び出し
- 状態管理
- エラーハンドリング

**API例**:
```typescript
const { transcribe, result, loading, error } = useTranscription();

await transcribe('/path/to/audio.wav', 'ja');
```

### Rust: Input Module (input.rs) - Handy方式

**責務**:
- テキスト挿入処理
- キーボードイベントシミュレーション
- プラットフォーム別ペースト実装

**主要API**:
```rust
// EnigoStateをTauriの管理状態として保持
pub struct EnigoState(pub Mutex<Enigo>);

// Clipboard方式（推奨）
pub fn send_paste_ctrl_v(enigo_state: &EnigoState) -> Result<()>;
pub fn send_paste_ctrl_shift_v(enigo_state: &EnigoState) -> Result<()>; // ターミナル用
pub fn send_paste_shift_insert(enigo_state: &EnigoState) -> Result<()>; // レガシー用

// Direct方式（参考）
pub fn paste_text_direct(text: String, enigo_state: &EnigoState) -> Result<()>;
```

**実装例** (macOS Cmd+V):
```rust
pub fn send_paste_ctrl_v(enigo_state: &EnigoState) -> Result<()> {
    let mut enigo = enigo_state.0.lock().unwrap();

    #[cfg(target_os = "macos")]
    {
        enigo.key_down(Key::Meta); // Command key
        enigo.key_click(Key::Layout('v'));
        sleep(Duration::from_millis(100)); // システム処理時間確保
        enigo.key_up(Key::Meta);
    }

    #[cfg(not(target_os = "macos"))]
    {
        enigo.key_down(Key::Control);
        enigo.key_click(Key::Layout('v'));
        sleep(Duration::from_millis(100));
        enigo.key_up(Key::Control);
    }

    Ok(())
}
```

### Rust: Clipboard Module (clipboard.rs)

**責務**:
- クリップボード読み書き
- `tauri-plugin-clipboard-manager`のラッパー

**主要API**:
```rust
use tauri_plugin_clipboard_manager::ClipboardExt;

pub async fn set_clipboard_text(app: &AppHandle, text: String) -> Result<()> {
    app.clipboard().write_text(text)?;
    Ok(())
}

pub async fn get_clipboard_text(app: &AppHandle) -> Result<String> {
    let text = app.clipboard().read_text()?;
    Ok(text)
}
```

**統合フロー**:
```rust
#[tauri::command]
async fn insert_transcribed_text(
    text: String,
    app: AppHandle,
    enigo_state: State<'_, EnigoState>,
) -> Result<(), String> {
    // 1. クリップボードにコピー
    clipboard::set_clipboard_text(&app, text)
        .await
        .map_err(|e| e.to_string())?;

    // 2. 少し待機
    sleep(Duration::from_millis(50));

    // 3. Cmd+V / Ctrl+V をシミュレート
    input::send_paste_ctrl_v(&enigo_state)
        .map_err(|e| e.to_string())?;

    Ok(())
}
```

### Rust: Cargo.toml依存関係 (Handy準拠)

```toml
[dependencies]
# Tauri コア
tauri = { version = "2.9", features = ["protocol-asset"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }

# Tauri プラグイン
tauri-plugin-clipboard-manager = "2.3"
tauri-plugin-global-shortcut = "2.3"
tauri-plugin-store = "2.4"
tauri-plugin-log = "2.7"

# オーディオ処理
cpal = "0.16"
hound = "3.5"

# キーボード/マウス制御（Handy方式）
enigo = "0.6.1"
rdev = { git = "https://github.com/rustdesk-org/rdev" }

# ユーティリティ
anyhow = "1"
log = "0.4"
env_logger = "0.11"

[build-dependencies]
tauri-build = { version = "2.0", features = [] }
```

**重要な依存関係の説明**:
- `enigo` v0.6.1: キーボード/マウスイベントシミュレーション（Handyと同じバージョン）
- `tauri-plugin-clipboard-manager`: クリップボード操作
- `tauri-plugin-global-shortcut`: グローバルホットキー
- `rdev`: 入力デバイスイベントハンドリング（rustdesk forkを使用）

---

## 配布戦略

### ビルド成果物

```
Echo.app/
├── Contents/
│   ├── MacOS/
│   │   └── echo                                      # Tauriバイナリ (10-20MB)
│   ├── Resources/
│   │   └── binaries/
│   │       └── mlx-asr-engine-aarch64-apple-darwin   # Python Sidecar (200-300MB)
│   └── Info.plist
```

### 配布形式

1. **Echo.app** - macOSアプリケーションバンドル
2. **Echo.dmg** - ディスクイメージ（ドラッグ&ドロップインストール）

### 配布サイズ

- **合計**: 約 220-350MB（モデルファイル除く）
- **初回起動時**: モデル自動ダウンロード（約100-200MB追加）

### システム要件

- **OS**: macOS 14.0 (Sonoma) 以降
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **メモリ**: 8GB以上推奨
- **ストレージ**: 1GB以上の空き容量

---

## 開発フェーズ

### Phase 1: プロジェクトセットアップ (1-2日)
- [ ] Tauri v2プロジェクト作成
- [ ] React + TypeScript + Vite + Tailwind CSS設定
- [ ] Python ASRエンジンの基本実装
- [ ] PyInstallerビルドスクリプト作成

### Phase 2: 音声キャプチャ実装 (2-3日)
- [ ] cpalで音声デバイスアクセス
- [ ] WAVファイル書き出し
- [ ] 基本UI実装（録音ボタン）

### Phase 3: Sidecar統合 (3-4日)
- [ ] Python Sidecarのデーモンモード実装
- [ ] Rust側のSidecar管理実装
- [ ] JSON-RPC通信実装
- [ ] エラーハンドリング

### Phase 4: 音声認識テスト (2-3日)
- [ ] mlx-audio + Qwen3-ASR統合
- [ ] PyInstallerビルドテスト
- [ ] 文字起こし精度検証

### Phase 5: グローバルホットキー (2-3日)
- [ ] global-hotkeyクレート統合
- [ ] ショートカット登録・解除
- [ ] カスタマイズUI

### Phase 6: テキスト自動挿入 (2-3日)
- [ ] `tauri-plugin-clipboard-manager`統合
- [ ] `enigo`クレート統合
- [ ] Clipboard方式のペースト実装（Cmd+V / Ctrl+V）
- [ ] プラットフォーム別ショートカット対応
- [ ] 100ms遅延の調整とテスト

### Phase 7: UI/UX改善 (3-5日)
- [ ] 設定画面の完成
- [ ] 録音オーバーレイ
- [ ] リアルタイムフィードバック
- [ ] エラー表示

### Phase 8: 最適化とテスト (3-5日)
- [ ] パフォーマンスプロファイリング
- [ ] メモリ使用量最適化
- [ ] E2Eテスト
- [ ] バグ修正

### Phase 9: 配布準備 (2-3日)
- [ ] コード署名
- [ ] 公証 (Notarization)
- [ ] DMG作成
- [ ] ドキュメント整備

**合計開発期間**: 約3-4週間

---

## 技術的な課題と解決策

### 課題1: PyInstaller + MLXの互換性

**問題**: MLXがPyInstallerでバイナリ化できるか不明

**解決策**:
1. 早期にPoCを実施してビルド可能性を検証
2. 失敗した場合は埋め込みPython方式に切り替え
3. または、whisper.cppベースのRust実装に変更

### 課題2: リアルタイム性能

**問題**: 大きなモデルではレイテンシが高い可能性

**解決策**:
1. Qwen3-ASR 8bit量子化モデルを使用（サイズ削減）
2. VADで処理対象音声を削減
3. ストリーミング処理の実装を検討

### 課題3: macOS権限

**問題**: マイクアクセス、Accessibility権限が必要

**解決策**:
1. 初回起動時に権限要求ダイアログ表示
2. 設定ガイドの提供
3. Info.plistに権限説明を追加

### 課題4: モデルファイルの管理

**問題**: 大きなモデルファイルの配布方法

**解決策**:
1. 初回起動時に自動ダウンロード
2. または、軽量版とフル版を別配布
3. ユーザーのキャッシュディレクトリに保存

### 課題5: テキスト挿入のキーボードレイアウト問題

**問題**: `enigo`のDirect方式（`.text()`メソッド）は、キーボードレイアウトを無視してスキャンコードベースで入力する既知のバグがある（[Handy Issue #439](https://github.com/cjpais/Handy/issues/439)参照）

**具体例**: ドイツ語キーボード（QWERTZ）で"zwei"と入力すると"ywei"になる

**解決策** (Handy方式を採用):
1. **Clipboard方式を優先**: クリップボード経由でのペーストを標準とする
2. **Direct方式は使用しない**: `.text()`メソッドのバグ修正待ち
3. **プラットフォーム別ペースト実装**:
   - macOS: `Cmd+V`
   - Windows/Linux: `Ctrl+V` または `Shift+Insert`
4. **100ms遅延**: システム処理時間を確保

**参考**: Handyでは複数のペースト方式を実装し、Clipboard方式を推奨している

---

## パフォーマンス目標

### レイテンシ
- **録音開始**: <100ms
- **文字起こし開始**: <500ms
- **文字起こし完了**: 音声長の0.5-1.0倍（例: 10秒音声 → 5-10秒）

### リソース使用量
- **メモリ**: <2GB（アイドル時 <500MB）
- **CPU**: <50%（文字起こし中）
- **ディスク**: <1GB（キャッシュ含む）

---

## セキュリティとプライバシー

### 基本方針
- **完全オフライン動作**: ネットワーク通信なし（モデルダウンロード後）
- **ローカル処理**: 全ての音声データはデバイス内で処理
- **データ保持**: 音声ファイルは文字起こし後に削除（オプション設定）

### 必要な権限
- マイクアクセス権限
- Accessibility権限（キーボード/マウス制御）
- ファイルシステムアクセス（アプリデータフォルダ）

---

## テスト戦略

### 単体テスト
- Rust: `cargo test`
- Python: `pytest`
- TypeScript: `vitest`

### 統合テスト
- Tauri Command呼び出しテスト
- Sidecar通信テスト

### E2Eテスト
- ホットキー → 録音 → 文字起こし → 挿入の一連フロー

### パフォーマンステスト
- 長時間録音テスト
- メモリリークチェック
- 様々な音声品質でのテスト

---

## 参考リソース

### ドキュメント
- [Tauri v2 Documentation](https://v2.tauri.app/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)

### 参考実装
- [Handy](https://github.com/cjpais/Handy) - Tauri音声文字化アプリ（本プロジェクトの主要参考）
  - [input.rs](https://github.com/cjpais/Handy/blob/main/src-tauri/src/input.rs) - テキスト挿入実装
  - [Cargo.toml](https://github.com/cjpais/Handy/blob/main/src-tauri/Cargo.toml) - 依存関係
  - [Issue #439](https://github.com/cjpais/Handy/issues/439) - キーボードレイアウト問題
- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust Whisper実装

### ツール
- [PyInstaller](https://pyinstaller.org/)
- [python-build-standalone](https://github.com/indygreg/python-build-standalone)

---

## 今後の拡張案

### v1.1
- [ ] リアルタイムストリーミング文字起こし
- [ ] 複数言語の自動切り替え
- [ ] 話者識別（Diarization）

### v1.2
- [ ] カスタムモデルのインポート
- [ ] プラグインシステム
- [ ] クラウド同期（オプション）

### v2.0
- [ ] Windows/Linux対応
- [ ] モバイルアプリ連携
- [ ] チームコラボレーション機能

---

## ライセンス

TBD (MIT or Apache 2.0 を検討)

---

## 貢献者

- 設計: Claude (Anthropic)
- 実装: TBD

---

**最終更新**: 2026-01-30
**バージョン**: 0.1.0-draft
