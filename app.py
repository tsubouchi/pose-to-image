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
    page_title="AI Pose Style Transfer",
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
</style>
""", unsafe_allow_html=True)

st.title("AI Pose Style Transfer")
st.markdown("Upload a pose image and a style image to generate a new image that combines both!")

# Main processing function
def process_images(pose_image, style_image):
    try:
        # Extract pose
        pose_result, pose_descriptions, landmarks = extract_pose(pose_image)

        if pose_result is None or landmarks is None:
            st.error("Could not detect pose in the image. Please try a different image.")
            return None

        # Generate image with style
        result = generate_image_with_style(pose_image, style_image)
        return result

    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
        logger.error(f"Error processing images: {str(e)}")
        return None

# Create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">1</span>
        <span>Upload Pose Image</span>
    </div>
    """, unsafe_allow_html=True)
    pose_file = st.file_uploader("Upload an image with the pose you want to recreate", type=['png', 'jpg', 'jpeg'], key="pose_upload")
    if pose_file:
        pose_image = Image.open(pose_file)
        st.image(pose_image, caption="Pose Image", use_container_width=True)

with col2:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">2</span>
        <span>Upload Style Image</span>
    </div>
    """, unsafe_allow_html=True)
    style_file = st.file_uploader("Upload an image with the style you want to apply", type=['png', 'jpg', 'jpeg'], key="style_upload")
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, caption="Style Image", use_container_width=True)

# Process images when both are uploaded
if pose_file and style_file:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">3</span>
        <span>Generated Result</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner('Generating image...'):
        result_image = process_images(pose_image, style_image)

        if result_image:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.image(result_image, caption="Generated Image", use_container_width=True)

            # Add download button
            buf = io.BytesIO()
            result_image.save(buf, format='PNG')
            st.download_button(
                label="Download Generated Image",
                data=buf.getvalue(),
                file_name="generated_pose.png",
                mime="image/png"
            )
            st.markdown('</div>', unsafe_allow_html=True)

# Instructions
st.markdown("""
---
### How to Use:
1. Upload an image containing the pose you want to recreate
2. Upload a style image that represents the desired appearance
3. Wait for the AI to generate a new image combining both inputs
4. Download the result using the download button
""")