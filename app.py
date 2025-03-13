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

.preview-image {
    max-width: 80px !important;
    max-height: 80px !important;
    margin: 0 auto;
}

.generated-result {
    background-color: #0a0a0a;
    border-radius: 8px;
    padding: 1px;
    margin-top: 0px;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

/* 生成結果の画像サイズ調整 */
.generated-result div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 !important;
}

.generated-result div[data-testid="stImage"] img {
    max-width: 100% !important;
    height: 260px !important;
    object-fit: contain !important;
    margin: 0 auto !important;
}

/* コンパクトなヘッダー */
h2 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.2em !important;
    line-height: 1 !important;
}

/* アップロード部分のコンパクト化 */
div[data-testid="stFileUploader"] {
    padding: 0.15rem !important;
}

/* ダウンロードボタンの調整 */
div[data-testid="stDownloadButton"] {
    margin-top: 2px !important;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

/* ステータス表示の調整 */
div[data-testid="stStatus"] {
    padding: 0.1rem !important;
    margin: 0.1rem 0 !important;
}

/* ポーズ提案セクションのスタイル */
.pose-suggestions {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 8px;
    margin-top: 5px;
}

.suggestion-item {
    background-color: #2a2a2a;
    border-radius: 4px;
    padding: 8px;
    margin: 5px 0;
}

.strong-points {
    color: #4CAF50;
    margin: 5px 0;
}

.improvement-point {
    color: #FFC107;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Create main layout with two columns
left_col, right_col = st.columns([1, 1], gap="small")

with left_col:
    st.markdown("## Input Images")

    # Pose Image Upload Section
    st.markdown("#### ポーズ参照画像")
    pose_file = st.file_uploader(
        "再現したいポーズの画像",
        type=['png', 'jpg', 'jpeg'],
        key="pose_upload"
    )
    if pose_file:
        pose_image = Image.open(pose_file)
        st.image(pose_image, use_container_width=False, width=120)

    # Style Image Upload Section
    st.markdown("#### スタイル参照画像")
    style_file = st.file_uploader(
        "目標とする画風や洋服の画像",
        type=['png', 'jpg', 'jpeg'],
        key="style_upload"
    )
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, use_container_width=False, width=120)

with right_col:
    st.markdown("## Generated Result")

    if pose_file and style_file:
        try:
            # Initialize result_image
            result_image = None

            # Pose Analysis
            with st.status("🔍 ポーズを解析中...") as status:
                pose_result, pose_descriptions, landmarks = extract_pose(pose_image)
                if pose_result is None:
                    st.error("ポーズの検出に失敗しました。別の画像を試してください。")
                    st.stop()
                status.update(label="✅ ポーズの解析が完了", state="complete")

            with st.status("🎨 画像を生成中...") as status:
                result_image = generate_image_with_style(pose_image, style_image)
                if result_image:
                    status.update(label="✅ 画像の生成が完了", state="complete")

            # 即時に画像を表示
            if result_image is not None:
                st.markdown('<div class="generated-result">', unsafe_allow_html=True)
                st.image(result_image, width=300, use_container_width=True)

                # Download button
                buf = io.BytesIO()
                result_image.save(buf, format='PNG')
                st.download_button(
                    label="💾 生成された画像をダウンロード",
                    data=buf.getvalue(),
                    file_name="generated_pose.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # ポーズの改善提案を表示
            with st.status("🔍 ポーズを分析中...") as status:
                # Convert pose image to base64
                pose_buf = io.BytesIO()
                pose_image.save(pose_buf, format='JPEG')
                pose_base64 = base64.b64encode(pose_buf.getvalue()).decode('utf-8')

                # Get pose analysis
                pose_analysis = analyze_pose_for_improvements(pose_base64)
                status.update(label="✅ ポーズの分析が完了", state="complete")

            st.markdown('<div class="pose-suggestions">', unsafe_allow_html=True)
            st.markdown("### 💡 AIポーズアドバイス")

            # 現在のポーズの説明
            st.markdown("#### 現在のポーズ")
            st.markdown(pose_analysis["current_pose"])

            # 良い点
            if pose_analysis["strong_points"]:
                st.markdown("#### ✨ 良い点")
                for point in pose_analysis["strong_points"]:
                    st.markdown(f'<div class="strong-points">• {point}</div>', unsafe_allow_html=True)

            # 改善提案
            if pose_analysis["suggestions"]:
                st.markdown("#### 📝 改善提案")
                for suggestion in pose_analysis["suggestions"]:
                    st.markdown(
                        f"""<div class="suggestion-item">
                        <div class="improvement-point">🎯 {suggestion["point"]}</div>
                        <div>改善方法: {suggestion["suggestion"]}</div>
                        <div>理由: {suggestion["reason"]}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

            st.markdown('</div>', unsafe_allow_html=True)

            # Pose Analysis Details at the bottom
            with st.expander("🔍 ポーズ解析の詳細"):
                if pose_descriptions:
                    st.markdown("**検出されたポーズの特徴:**")
                    for key, value in pose_descriptions.items():
                        if not key.endswith("_desc"):
                            continue
                        label = key.replace("_desc", "").replace("_", " ").title()
                        st.markdown(f"- {label}: {value}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"Error processing images: {str(e)}")
    else:
        st.markdown('<div class="generated-result">', unsafe_allow_html=True)
        st.info("👈 左側で2つの画像をアップロードしてください")
        st.markdown('</div>', unsafe_allow_html=True)

# Instructions at the bottom
with st.expander("💡 使い方"):
    st.markdown("""
    1. ポーズ参照画像をアップロード
       - 再現したいポーズの画像を選択してください
       - 人物がはっきりと写っている画像を使用するのがおすすめです

    2. スタイル参照画像をアップロード
       - 目標とする画風や洋服の画像を選択してください
       - キャラクターデザインや画風が明確な画像を使用するのがおすすめです

    3. 生成された画像を確認
       - AIが2つの画像を組み合わせて新しい画像を生成します
       - 必要に応じてダウンロードできます
    """)