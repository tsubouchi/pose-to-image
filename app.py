import streamlit as st
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image_with_style
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="AI Style Transfer with Pose Matching",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern dark mode design
st.markdown("""
<style>
.stApp {
    background-color: #060606;
    color: #fff;
}

.input-section {
    background-color: #0a0a0a;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #333;
    max-height: calc(25vh - 20px);
    overflow: hidden;
}

.result-container {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 15px;
    height: calc(90vh - 100px);
    overflow-y: auto;
}

.upload-header {
    font-size: 1.1em;
    font-weight: 500;
    margin-bottom: 5px;
    padding-bottom: 5px;
    border-bottom: 1px solid #333;
}

.processing-status {
    margin-top: 10px;
    padding: 10px;
    border-radius: 4px;
    background-color: rgba(25, 118, 210, 0.05);
}

.stImage {
    max-height: calc(13vh) !important;
}

div[data-testid="stImage"] {
    text-align: center;
}

div[data-testid="stImage"] > img {
    max-height: calc(13vh);
    width: auto !important;
}

/* ファイルアップローダーのサイズ調整 */
div[data-testid="stFileUploader"] {
    padding: 0.5rem;
}

div[data-testid="stFileUploader"] > div > div {
    padding: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# Create main layout with two columns
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("## Input Images")

    # Pose Image Upload Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-header">ポーズ参照画像</div>', unsafe_allow_html=True)
    pose_file = st.file_uploader(
        "再現したいポーズが写っている画像をアップロード",
        type=['png', 'jpg', 'jpeg'],
        key="pose_upload"
    )
    if pose_file:
        pose_image = Image.open(pose_file)
        st.image(pose_image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Style Image Upload Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-header">スタイル参照画像</div>', unsafe_allow_html=True)
    style_file = st.file_uploader(
        "目標とする画風や洋服の画像をアップロード",
        type=['png', 'jpg', 'jpeg'],
        key="style_upload"
    )
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("## Generated Result")

    # Process images when both are uploaded
    if pose_file and style_file:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)

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

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                logger.error(f"Error processing images: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-container" style="text-align: center; padding: 40px;">
            <p>画像を生成するには、左側で2つの画像をアップロードしてください：</p>
            <ol style="text-align: left; display: inline-block;">
                <li>ポーズ参照画像：再現したいポーズが写っている画像</li>
                <li>スタイル参照画像：目標とする画風や洋服の画像</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

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