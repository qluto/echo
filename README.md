# Echo - 音声入力アプリケーション

Apple Silicon最適化されたオフライン音声入力デスクトップアプリケーション

## 機能

- **高精度音声認識**: MLX-Audio + Whisper による高精度な音声認識
- **グローバルホットキー**: システム全体で動作するショートカットキー
- **リアルタイム文字起こし**: 録音終了後即座に文字起こし
- **テキスト自動挿入**: 文字起こし結果を任意のアプリケーションに挿入
- **完全オフライン**: ネットワーク不要で動作

## システム要件

- **OS**: macOS 14.0 (Sonoma) 以降
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **メモリ**: 8GB以上推奨
- **ストレージ**: 1GB以上の空き容量

## 開発環境セットアップ

### 必要なツール

- Node.js 20+
- Rust 1.83+
- Python 3.11+ (ARM native)

### インストール

```bash
# Node.js依存関係のインストール
npm install

# Rust依存関係のインストール（Tauriが自動で行います）
cd src-tauri && cargo build

# Python依存関係のインストール（ASRエンジン用）
cd python-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 開発サーバーの起動

```bash
npm run tauri:dev
```

### ビルド

```bash
# フロントエンドのビルド
npm run build

# Tauriアプリのビルド
npm run tauri:build
```

## 使い方

1. アプリを起動
2. `Cmd+Shift+Space` を押し続けて録音開始
3. 話し終わったらキーを離す
4. 自動的に文字起こしが開始
5. 結果が表示され、設定によっては自動でアクティブなアプリに挿入

## 設定

設定画面から以下の項目をカスタマイズできます：

- **ホットキー**: 録音開始/終了のショートカット
- **認識言語**: 自動検出または手動指定
- **入力デバイス**: マイク選択
- **自動挿入**: 文字起こし後の自動貼り付けの有効/無効

## 技術スタック

- **フロントエンド**: React, TypeScript, Vite, Tailwind CSS
- **バックエンド**: Tauri 2, Rust
- **音声認識**: MLX-Audio, Whisper

## ライセンス

TBD
