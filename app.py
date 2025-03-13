import streamlit as st
import mediapipe as mp
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image

st.set_page_config(
    page_title="Pose to Image Generator",
    layout="wide"
)

st.title("ポーズから画像生成")

# Initialize session state for images
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'pose_image' not in st.session_state:
    st.session_state.pose_image = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

# スタイル選択のオプション
styles = {
    "アニメ調女子高生": """
    この棒人間のポーズを元に、日本のアニメスタイルの女子高生キャラクターを生成してください。
    - 制服を着た現代的な女子高生
    - アニメ調の明るい色使い
    - 自然な髪の動きと表情
    - キャラクターの個性が感じられるデザイン
    - 背景は学校や街並みなど日常的なシーン
    """,

    "カジュアルファッション（アニメ調）": """
    この棒人間のポーズを元に、カジュアルな服装の女子高生キャラクターを生成してください。
    - 私服（Tシャツ、ジーンズ、スカートなど）
    - 現代的なヘアスタイル
    - ナチュラルな表情
    - アクセサリーや小物でアクセント
    - 都会的な背景
    """,

    "ファッションポートレート（実写風）": """
    この棒人間のポーズを元に、実写風のファッションポートレートを生成してください。
    - 高解像度の写真のような仕上がり
    - 現代的なファッションスタイル
    - 自然な照明と影
    - プロフェッショナルな撮影スタジオの雰囲気
    - ソフトなボケ味のある背景
    """,

    "アウトドアポートレート（実写風）": """
    この棒人間のポーズを元に、屋外での実写風ポートレートを生成してください。
    - 自然光を活かした写真のような表現
    - カジュアルでアクティブな服装
    - 自然な表情と姿勢
    - 公園や街並みなどの屋外背景
    - 自然な環境光と影の表現
    """
}

# Top section: Drag & Drop
uploaded_file = st.file_uploader("画像をドラッグ＆ドロップしてください", type=['png', 'jpg', 'jpeg'])

# Style selection
selected_style = st.selectbox(
    "生成スタイルを選択",
    list(styles.keys())
)

# Create three columns for the images
col1, col2, col3 = st.columns(3)

if uploaded_file is not None:
    try:
        # Display original image
        with col1:
            st.header("元画像")
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            st.image(image, caption="Original Image")

        # Extract and display pose
        with col2:
            st.header("ポーズ抽出")
            pose_image = extract_pose(image)
            st.session_state.pose_image = pose_image
            st.image(pose_image, caption="Extracted Pose")

        # Generate and display new image
        with col3:
            st.header("生成画像")
            generated_image = generate_image(pose_image, styles[selected_style])
            st.session_state.generated_image = generated_image
            if generated_image is not None:
                st.image(generated_image, caption="Generated Image")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown("""
---
### 使い方:
1. 人物が写っている画像をドラッグ＆ドロップ
2. 生成したいスタイルを選択
   - アニメ調: アニメやイラスト風の画像を生成
   - 実写風: 写真のような現実的な画像を生成
3. システムが自動的にポーズを抽出し、選択したスタイルで新しい画像を生成
""")