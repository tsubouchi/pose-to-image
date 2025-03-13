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

.upload-section {
    margin-bottom: 1rem;
}

.generated-section {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 15px;
    margin-top: 10px;
    min-height: 400px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* 生成結果の画像サイズ調整 */
.generated-section div[data-testid="stImage"] img {
    max-width: 100% !important;
    max-height: 350px !important;
    object-fit: contain;
}

/* ステータス表示の調整 */
div[data-testid="stStatus"] {
    margin-bottom: 10px !important;
}

/* 入力画像のサイズ調整 */
.upload-section div[data-testid="stImage"] img {
    max-width: 25% !important;
    max-height: 15vh !important;
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# Create main layout with two columns
left_col, right_col = st.columns([1, 1], gap="small")

with left_col:
    st.markdown("## Input Images")

    # Pose Image Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("#### ポーズ参照画像")
    pose_file = st.file_uploader(
        "再現したいポーズの画像",
        type=['png', 'jpg', 'jpeg'],
        key="pose_upload"
    )
    if pose_file:
        pose_image = Image.open(pose_file)
        st.image(pose_image)
    st.markdown('</div>', unsafe_allow_html=True)

    # Style Image Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("#### スタイル参照画像")
    style_file = st.file_uploader(
        "目標とする画風や洋服の画像",
        type=['png', 'jpg', 'jpeg'],
        key="style_upload"
    )
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("## Generated Result")

    if pose_file and style_file:
        try:
            # Process Status Area
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

            # Generated Image Display
            st.markdown('<div class="generated-section">', unsafe_allow_html=True)
            if result_image:
                st.image(result_image)

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

            # Analysis Details
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
        st.info("👈 左側で2つの画像をアップロードしてください")

# Instructions
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