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

st.markdown("""
<style>
.stApp {
    background-color: #060606;
    color: #fff;
}

.input-section {
    background-color: #0a0a0a;
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 8px;
    border: 1px solid #333;
}

.result-container {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 10px;
    height: 90vh;
}

.upload-header {
    font-size: 0.9em;
    margin-bottom: 2px;
    color: #ccc;
}

/* アップローダーのサイズ調整 */
div[data-testid="stFileUploader"] {
    padding: 0.25rem;
}

div[data-testid="stFileUploader"] > div > div {
    padding: 0.15rem;
}

/* 画像サイズの調整 */
div[data-testid="stImage"] img {
    max-width: 33% !important;
    max-height: auto !important;
    display: block;
    margin: 0 auto;
}

/* ヘッダーの調整 */
h2 {
    font-size: 1.1em !important;
    margin: 0 0 8px 0 !important;
    padding: 0 !important;
}

/* 説明テキストの調整 */
div.stMarkdown p {
    font-size: 0.9em;
    margin: 0;
    padding: 0;
}
</style>
""", unsafe_allow_html=True)

# Create main layout with two columns
left_col, right_col = st.columns([1, 1], gap="small")

with left_col:
    st.markdown("## Input Images")

    # Pose Image Upload Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-header">ポーズ参照画像</div>', unsafe_allow_html=True)
    pose_file = st.file_uploader(
        "再現したいポーズの画像",
        type=['png', 'jpg', 'jpeg'],
        key="pose_upload"
    )
    if pose_file:
        pose_image = Image.open(pose_file)
        st.image(pose_image, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Style Image Upload Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-header">スタイル参照画像</div>', unsafe_allow_html=True)
    style_file = st.file_uploader(
        "目標とする画風や洋服の画像",
        type=['png', 'jpg', 'jpeg'],
        key="style_upload"
    )
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, use_container_width=False)
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
        <div class="result-container">
            <p>2つの画像をアップロードしてください：</p>
            <ul style="margin: 0; padding-left: 20px;">
                <li>ポーズ参照画像：再現したいポーズ</li>
                <li>スタイル参照画像：目標とする画風や洋服</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Instructions (Minor text adjustments)
st.markdown("""
---
### 使い方:
1. ポーズ参照画像をアップロード
   - 再現したいポーズの画像を選択してください
   - 人物がはっきりと写っている画像を使用するのがおすすめです

2. スタイル参照画像をアップロード
   - 目標とする画風の画像を選択してください
   - キャラクターデザインや画風が明確な画像を使用するのがおすすめです

3. 生成された画像を確認
   - AIが2つの画像を組み合わせて新しい画像を生成します
   - 必要に応じてダウンロードできます
""")