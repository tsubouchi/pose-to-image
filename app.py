import streamlit as st
import mediapipe as mp
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image

st.set_page_config(
    page_title="Pose to Anime Image Generator",
    layout="wide"
)

st.title("Pose to Anime Image Generator")

# Initialize session state for images
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'pose_image' not in st.session_state:
    st.session_state.pose_image = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

# Create three columns for the images
col1, col2, col3 = st.columns(3)

# スタイル選択のオプション
styles = {
    "アニメ調": "Create an anime-style character image matching this pose. Use vibrant colors and dynamic composition.",
    "水彩画風": "Create a watercolor-style illustration matching this pose. Use soft, flowing colors and subtle textures.",
    "3Dレンダリング": "Create a 3D rendered character matching this pose. Add realistic lighting and textures.",
    "ピクセルアート": "Create a pixel art character matching this pose. Use a retro gaming style with limited color palette.",
}

with col1:
    st.header("Original Image")
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    # スタイル選択
    selected_style = st.selectbox(
        "生成スタイルを選択",
        list(styles.keys())
    )

    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            st.image(image, caption="Original Image")

            # Extract pose
            pose_image = extract_pose(image)
            st.session_state.pose_image = pose_image

            # Generate new image with selected style
            generated_image = generate_image(pose_image, styles[selected_style])
            st.session_state.generated_image = generated_image

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with col2:
    st.header("Extracted Pose")
    if st.session_state.pose_image is not None:
        st.image(st.session_state.pose_image, caption="Extracted Pose")

with col3:
    st.header("Generated Image")
    if st.session_state.generated_image is not None:
        st.image(st.session_state.generated_image, caption="Generated Image")

st.markdown("""
---
### 使い方:
1. 人物が写っている画像をアップロード
2. 生成したいスタイルを選択
3. システムが自動的にポーズを抽出し、選択したスタイルで新しい画像を生成
""")