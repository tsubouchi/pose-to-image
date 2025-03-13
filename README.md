# AI Pose to Image Generator (AIポーズ画像ジェネレーター)

Advanced AI-powered image generation platform that converts pose images into styled character illustrations.
ポーズ画像をAIを使用してスタイライズされたキャラクターイラストに変換する高度な画像生成プラットフォーム。

## Features (主な機能)

- 🎨 Multiple style presets (Anime, Casual, Fashion Portrait)
  複数のスタイルプリセット（アニメ、カジュアル、ファッションポートレート）
- 🔄 Batch processing for multiple images
  複数画像の一括処理
- 📊 Real-time progress tracking
  リアルタイムの進捗追跡
- 🎯 Precise pose extraction and matching
  正確なポーズ抽出とマッチング
- 🌙 Modern dark mode UI
  モダンなダークモードUI

## Setup (セットアップ)

1. Clone the repository (リポジトリのクローン):
```bash
git clone https://github.com/tsubouchi/ai-pose-generator.git
cd ai-pose-generator
```

2. Install dependencies (依存パッケージのインストール):
```bash
pip install -r requirements.txt
```

3. Set up environment variables (環境変数の設定):
Create a `.env` file in the root directory with:
ルートディレクトリに`.env`ファイルを作成し、以下を設定:

```
STABILITY_KEY=your_stability_ai_key
```

## Usage (使用方法)

1. Start the application (アプリケーションの起動):
```bash
streamlit run app.py
```

2. Upload images (画像のアップロード):
   - Drop one or more images containing people
   - 人物が写っている画像を1枚以上ドロップ

3. Select generation style (生成スタイルの選択):
   - Choose from available style presets
   - 利用可能なスタイルプリセットから選択

4. View results (結果の確認):
   - Each image goes through 4 steps:
   - 各画像は4つのステップで処理:
     1. Original Image Display (元画像の表示)
     2. Pose Extraction (ポーズ抽出)
     3. Prompt Generation (プロンプト生成)
     4. Image Generation (画像生成)

## Technical Stack (技術スタック)

- **Frontend**: Streamlit
- **AI Models**: 
  - Stability AI API for image generation
  - MediaPipe for pose extraction
- **Image Processing**: PIL, OpenCV
- **UI**: Custom dark mode theme

## Requirements (必要条件)

- Python 3.8+
- Stability AI API Key
- Internet connection for API access

## Notes (注意事項)

- The system prompt is optimized for accurate pose reproduction
- システムプロンプトは正確なポーズの再現のために最適化されています
- Batch processing is available for multiple images
- 複数画像の一括処理が可能です
- All processing is done in real-time with progress tracking
- すべての処理はリアルタイムで進捗表示付きで実行されます

## License (ライセンス)

MIT License