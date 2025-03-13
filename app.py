import streamlit as st
from PIL import Image
import io
import base64
from pose_extractor import extract_pose
from image_generator import generate_image_with_style
from pose_analysis import analyze_pose_for_improvements
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

/* 左側のプレビュー画像スタイル */
.preview-image div[data-testid="stImage"] {
    width: 80px !important;
    height: 80px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #1a1a1a;
    border-radius: 4px;
    padding: 0 !important;
    margin: 4px 0 !important;
}

.preview-image div[data-testid="stImage"] img {
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain;
}

/* 右側の出力画像スタイル */
.output-image div[data-testid="stImage"] {
    width: 375px !important;
    height: 280px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent;
    margin: 0 auto !important;
}

.output-image div[data-testid="stImage"] img {
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain;
}

/* アップロードエリアのスタイル */
div[data-testid="stFileUploader"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* テキストとマージンの調整 */
div[data-testid="stMarkdown"] {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}

/* ダウンロードボタン */
div[data-testid="stDownloadButton"] button {
    width: 100%;
    margin: 4px 0 !important;
}
</style>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="small")

with left_col:
    st.text("ポーズ参照画像")
    pose_file = st.file_uploader("再現したいポーズの画像", type=['png', 'jpg', 'jpeg'], key="pose_upload")
    if pose_file:
        pose_image = Image.open(pose_file)
        st.markdown('<div class="preview-image">', unsafe_allow_html=True)
        st.image(pose_image, width=80)
        st.markdown('</div>', unsafe_allow_html=True)

    st.text("スタイル参照画像")
    style_file = st.file_uploader("目標とする画風や洋服の画像", type=['png', 'jpg', 'jpeg'], key="style_upload")
    if style_file:
        style_image = Image.open(style_file)
        st.markdown('<div class="preview-image">', unsafe_allow_html=True)
        st.image(style_image, width=80)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
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
                st.markdown('<div class="output-image">', unsafe_allow_html=True)
                st.image(result_image)
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
        st.text("👈 左側で2つの画像をアップロードしてください")

with st.expander("💡 使い方"):
    st.text("1. ポーズ参照画像をアップロード\n   再現したいポーズの画像を選択")
    st.text("2. スタイル参照画像をアップロード\n   目標とする画風や洋服の画像を選択")
    st.text("3. 生成された画像を確認\n   AIが2つの画像を組み合わせて新しい画像を生成")