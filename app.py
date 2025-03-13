import streamlit as st
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image_with_style
from pose_analysis import analyze_pose_for_improvements
import base64
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

/* プレビューエリアの固定サイズ設定 */
.preview-container {
    width: 100px;
    height: 100px;
    background-color: #1a1a1a;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto;
}

/* アップロードエリアのスタイル */
div[data-testid="stFileUploader"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* プレビュー画像のサイズ制御 */
.preview-image {
    max-width: 80px !important;
    max-height: 80px !important;
    object-fit: contain;
}

/* 出力画像エリアの設定 */
.output-container {
    width: 375px;
    height: 280px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.output-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

/* テキストとマージンの調整 */
div[data-testid="stMarkdown"] {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}

/* ダウンロードボタンの配置 */
div[data-testid="stDownloadButton"] button {
    margin: 4px 0 !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Create main layout with two columns
left_col, right_col = st.columns([1, 1], gap="small")

with left_col:
    st.text("Input Images")

    # ポーズ参照画像
    pose_file = st.file_uploader("再現したいポーズの画像", type=['png', 'jpg', 'jpeg'], key="pose_upload")
    if pose_file:
        pose_image = Image.open(pose_file)
        st.markdown('<div class="preview-container">', unsafe_allow_html=True)
        st.image(pose_image, width=80, output_format="PNG", clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # スタイル参照画像
    style_file = st.file_uploader("目標とする画風や洋服の画像", type=['png', 'jpg', 'jpeg'], key="style_upload")
    if style_file:
        style_image = Image.open(style_file)
        st.markdown('<div class="preview-container">', unsafe_allow_html=True)
        st.image(style_image, width=80, output_format="PNG", clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.text("Generated Result")

    if pose_file and style_file:
        try:
            result_image = None
            with st.status("🔍 ポーズを解析中...", expanded=False) as status:
                pose_result, pose_descriptions, landmarks = extract_pose(pose_image)
                if pose_result is None:
                    st.error("ポーズの検出に失敗しました。")
                    st.stop()
                status.update(label="✅ ポーズの解析が完了", state="complete")

            with st.status("🎨 画像を生成中...", expanded=False) as status:
                result_image = generate_image_with_style(pose_image, style_image)
                if result_image:
                    status.update(label="✅ 画像の生成が完了", state="complete")

            if result_image is not None:
                st.markdown('<div class="output-container">', unsafe_allow_html=True)
                st.image(result_image, output_format="PNG", clamp=True)
                st.markdown('</div>', unsafe_allow_html=True)

                buf = io.BytesIO()
                result_image.save(buf, format='PNG')
                st.download_button("💾 生成された画像をダウンロード",
                                 data=buf.getvalue(),
                                 file_name="generated_pose.png",
                                 mime="image/png")

            with st.expander("💡 AIポーズアドバイス"):
                pose_buf = io.BytesIO()
                pose_image.save(pose_buf, format='JPEG')
                pose_base64 = base64.b64encode(pose_buf.getvalue()).decode('utf-8')
                pose_analysis = analyze_pose_for_improvements(pose_base64)

                st.text("現在のポーズ")
                st.text(pose_analysis["current_pose"])

                if pose_analysis["strong_points"]:
                    st.text("✨ 良い点")
                    for point in pose_analysis["strong_points"]:
                        st.text(f"• {point}")

                if pose_analysis["suggestions"]:
                    st.text("📝 改善提案")
                    for suggestion in pose_analysis["suggestions"]:
                        st.text(f"🎯 {suggestion['point']}")
                        st.text(f"改善方法: {suggestion['suggestion']}")
                        st.text(f"理由: {suggestion['reason']}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"Error processing images: {str(e)}")
    else:
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        st.info("👈 左側で2つの画像をアップロードしてください")
        st.markdown('</div>', unsafe_allow_html=True)

with st.expander("💡 使い方"):
    st.text("1. ポーズ参照画像をアップロード\n   再現したいポーズの画像を選択")
    st.text("2. スタイル参照画像をアップロード\n   目標とする画風や洋服の画像を選択")
    st.text("3. 生成された画像を確認\n   AIが2つの画像を組み合わせて新しい画像を生成")