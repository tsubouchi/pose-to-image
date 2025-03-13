import streamlit as st
import mediapipe as mp
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image_with_style
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="AI Style Transfer with Pose Matching",
    layout="wide"
)

# Custom CSS for modern dark mode design
st.markdown("""
<style>
.stApp {
    background-color: #060606;
    color: #fff;
}

.image-card {
    background-color: #0a0a0a;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(255,255,255,0.03);
    margin-bottom: 15px;
    border: 1px solid #333;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 500;
    background-color: rgba(25, 118, 210, 0.1);
    color: #64b5f6;
    margin-bottom: 5px;
}

.step-header {
    margin-bottom: 15px;
    border-bottom: 1px solid #333;
    padding-bottom: 10px;
}

.step-number {
    display: inline-block;
    width: 24px;
    height: 24px;
    background-color: #1976d2;
    border-radius: 12px;
    text-align: center;
    line-height: 24px;
    margin-right: 8px;
}

.result-container {
    padding: 15px;
    background-color: #1a1a1a;
    border-radius: 8px;
    margin-top: 20px;
}

.processing-status {
    margin-top: 10px;
    padding: 10px;
    border-radius: 4px;
    background-color: rgba(25, 118, 210, 0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("AI Style Transfer with Pose Matching")
st.markdown("""
このアプリケーションでは、2つの画像から新しい画像を生成します：
1. ポーズ画像：再現したい姿勢やポーズを含む画像
2. スタイル画像：目標とする画風や見た目の画像

AIが1枚目のポーズを2枚目の画風で再現した新しい画像を生成します。
""")

# Create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">1</span>
        <span>ポーズ参照画像をアップロード</span>
    </div>
    """, unsafe_allow_html=True)
    pose_file = st.file_uploader(
        "再現したいポーズが写っている画像をアップロード",
        type=['png', 'jpg', 'jpeg'],
        key="pose_upload"
    )
    if pose_file:
        pose_image = Image.open(pose_file)
        st.image(pose_image, caption="ポーズ参照画像", use_container_width=True)

with col2:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">2</span>
        <span>スタイル参照画像をアップロード</span>
    </div>
    """, unsafe_allow_html=True)
    style_file = st.file_uploader(
        "目標とする画風の画像をアップロード",
        type=['png', 'jpg', 'jpeg'],
        key="style_upload"
    )
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, caption="スタイル参照画像", use_container_width=True)

# Process images when both are uploaded
if pose_file and style_file:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">3</span>
        <span>生成結果</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner('画像を生成中...'):
        try:
            # Extract pose
            with st.status("ポーズを解析中...") as status:
                pose_result, pose_descriptions, landmarks = extract_pose(pose_image)
                if pose_result is None:
                    st.error("ポーズの検出に失敗しました。別の画像を試してください。")
                    st.stop()
                status.update(label="ポーズの解析が完了しました", state="complete")

            # Generate image
            with st.status("画像を生成中...") as status:
                result_image = generate_image_with_style(pose_image, style_image)
                if result_image:
                    status.update(label="画像の生成が完了しました", state="complete")
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.image(result_image, caption="生成された画像", use_container_width=True)

                    # Add download button
                    buf = io.BytesIO()
                    result_image.save(buf, format='PNG')
                    st.download_button(
                        label="生成された画像をダウンロード",
                        data=buf.getvalue(),
                        file_name="generated_pose.png",
                        mime="image/png"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"Error processing images: {str(e)}")

# Instructions
st.markdown("""
---
### 使い方:
1. ポーズ参照画像をアップロード
   - 再現したいポーズが写っている画像を選択
   - 人物が明確に写っている画像を使用

2. スタイル参照画像をアップロード
   - 目標とする画風の画像を選択
   - キャラクターデザインや画風が明確な画像を使用

3. 生成された画像を確認
   - AIが2つの画像を組み合わせて新しい画像を生成
   - 必要に応じてダウンロード可能
""")