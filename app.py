import streamlit as st
import mediapipe as mp
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image, generate_video, generate_controlnet_openpose
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Pose to Image/Video Generator",
    layout="wide"
)

# Custom CSS for modern dark mode card design
st.markdown("""
<style>
.stApp {
    background-color: #060606;
    color: #fff;
}

.image-card {
    background-color: #0a0a0a;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 12px rgba(255,255,255,0.03);
    margin-bottom: 15px;
    border: 1px solid #333;
    position: relative;
    overflow: hidden;
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

.meta-info {
    color: #888;
    font-size: 12px;
    margin-bottom: 5px;
}

.tag {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    background-color: rgba(255,255,255,0.05);
    color: #888;
    margin-right: 4px;
    margin-bottom: 4px;
}

.step-header {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
}

.step-header h3 {
    font-size: 14px;
    margin: 0;
}

.step-icon {
    width: 24px;
    height: 24px;
    background-color: rgba(255,255,255,0.05);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 8px;
    color: #64b5f6;
    font-size: 12px;
}

.stButton>button {
    background-color: #1976d2;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 12px;
    width: 100%;
}

.stButton>button:hover {
    background-color: #1565c0;
}

/* Dark mode overrides for Streamlit elements */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stTextArea > div > div > textarea {
    background-color: #1a1a1a;
    color: white;
    border-color: #333;
}

/* Add grid background animation */
.image-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at center, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 1px) 0 0 / 40px 40px;
    pointer-events: none;
}

.image-card:hover::before {
    background-color: rgba(255,255,255,0.02);
}

/* Error and success message styling */
.stError {
    background-color: rgba(244, 67, 54, 0.1);
    color: #ff5252;
    border-color: #ff5252;
    padding: 4px 8px;
    font-size: 12px;
}

.stSuccess {
    background-color: rgba(76, 175, 80, 0.1);
    color: #69f0ae;
    border-color: #69f0ae;
    padding: 4px 8px;
    font-size: 12px;
}

/* Image container adjustments */
.stImage {
    margin: 0 !important;
    padding: 0 !important;
}

.stImage > img {
    max-height: 200px !important;
    width: auto !important;
    margin: 0 auto !important;
}

/* Prompt text area adjustments */
.stTextArea textarea {
    min-height: 100px !important;
    font-size: 12px !important;
}

/* Suggestion item styling */
.suggestion-item {
    background-color: rgba(25, 118, 210, 0.05);
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 12px;
    color: #64b5f6;
}

.suggestion-text {
    display: block;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Pose to Image/Video Generator")

# Initialize session state
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'pose_images' not in st.session_state:
    st.session_state.pose_images = []
if 'intermediate_images' not in st.session_state:
    st.session_state.intermediate_images = []
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Add mode selection
generation_mode = st.radio(
    "Select Generation Mode",
    ["Single Image", "Video Sequence"],
    help="Choose whether to generate a single image or a video sequence"
)

# Style selection options
styles = {
    "Anime Style School Uniform": {
        "base_prompt": """masterpiece, best quality, highly detailed,
full body pose with exact stick figure matching,
cute anime girl in school uniform,
sailor uniform style with pleated skirt,
clear facial features, expressive eyes,
natural indoor lighting, classroom background,
professional anime illustration""",
        "negative_prompt": """bad anatomy, bad hands, missing fingers, 
wrong pose, inaccurate pose, multiple people,
worst quality, low quality, blurry, text"""
    }
}

# File uploader
uploaded_files = st.file_uploader(
    "Drop your images here",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    key="file_uploader"
)

# Video generation parameters (if video mode selected)
if generation_mode == "Video Sequence":
    st.sidebar.subheader("Video Settings")
    fps = st.sidebar.slider("Frames per second", 15, 60, 30)
    duration = st.sidebar.slider("Duration (seconds)", 1, 10, 5)

# Style selection - fixed to school uniform for now
selected_style = "Anime Style School Uniform"

def process_image(image, idx, cols):
    """Process a single image with improved error handling."""
    try:
        # Step 2: Pose Extraction with improved error handling
        pose_image, pose_descriptions, results = extract_pose(image)

        if pose_image is None:
            st.error("""
            ポーズの検出に失敗しました。以下をお試しください：
            - 画像の明るさやコントラストを調整
            - 人物が画像の中心にいることを確認
            - 人物全体が写っていることを確認
            """)
            return False

        # Display pose image and suggestions
        st.image(pose_image, use_container_width=True)
        st.markdown('<div class="tag">Pose Detected</div>', unsafe_allow_html=True)

        if results and results.pose_landmarks:
            suggestions = get_pose_refinement_suggestions(results.pose_landmarks)

            if suggestions:
                st.markdown("""
                <div class="step-header">
                    <h4>AI Pose Suggestions</h4>
                </div>
                """, unsafe_allow_html=True)

                for key, suggestion in suggestions.items():
                    if key != "error":
                        st.markdown(f"""
                        <div class="suggestion-item">
                            <span class="suggestion-text">🔍 {suggestion}</span>
                        </div>
                        """, unsafe_allow_html=True)

        return True

    except Exception as e:
        st.error(f"画像処理中にエラーが発生しました: {str(e)}")
        logger.error(f"Error processing image: {str(e)}")
        return False


# Process uploaded images
if uploaded_files:
    try:
        # Progress container
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Process each image
        num_images = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (idx + 1) / num_images
            progress_bar.progress(progress)
            status_text.text(f"Processing image {idx + 1} of {num_images}")

            # Create image set container
            with st.container():
                st.markdown(f"""
                <div class="image-card">
                    <div class="step-header">
                        <div class="status-badge">Image Set {idx + 1}</div>
                        <div class="meta-info">Processing Time: {idx * 2 + 5}s</div>
                    </div>
                """, unsafe_allow_html=True)

                # Create columns for display
                cols = st.columns([1, 1, 1, 1, 1])

                # Step 1: Original Image
                with cols[0]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">1</div>
                        <h3>Original Image</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                # Step 2-5:  Use the new process_image function
                if not process_image(image, idx, cols):
                    continue #skip to the next image if processing failed

                # Step 3: Generation Prompt (moved here since process_image handles pose extraction)
                with cols[2]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">3</div>
                        <h3>Generation Prompt</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    generation_prompt = styles[selected_style]["base_prompt"]
                    st.text_area("Prompt", value=generation_prompt, height=300, disabled=True)
                    st.markdown('<div class="tag">Style Applied</div>', unsafe_allow_html=True)

                # Step 4: OpenPose Generation
                with cols[3]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">4</div>
                        <h3>OpenPose Generation</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        openpose_image = generate_controlnet_openpose(
                            pose_image,
                            generation_prompt
                        )
                        if openpose_image is not None:
                            st.image(openpose_image, use_container_width=True)
                            st.markdown('<div class="tag">OpenPose Generated</div>', unsafe_allow_html=True)
                        else:
                            st.error("Failed to generate OpenPose image")
                            continue
                    except Exception as e:
                        st.error(f"Error in OpenPose generation: {str(e)}")
                        logger.error(f"Error in OpenPose generation: {str(e)}")
                        continue

                # Step 5: Final Generation
                with cols[4]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">5</div>
                        <h3>Final Result</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        if generation_mode == "Single Image":
                            final_image = generate_image(
                                openpose_image,
                                generation_prompt,
                                "Convert to high quality anime while preserving pose"
                            )
                            if final_image is not None:
                                st.image(final_image, use_container_width=True)
                                st.markdown('<div class="tag">Generation Complete</div>', unsafe_allow_html=True)

                                # Add download button
                                buf = io.BytesIO()
                                final_image.save(buf, format='PNG')
                                st.download_button(
                                    label="Download Image",
                                    data=buf.getvalue(),
                                    file_name=f"generated_image_{idx+1}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        else:  # Video mode
                            video_path = generate_video(
                                [openpose_image],
                                generation_prompt,
                                fps=fps,
                                duration=duration
                            )
                            if video_path:
                                with open(video_path, 'rb') as video_file:
                                    video_bytes = video_file.read()
                                st.video(video_bytes)
                                st.markdown('<div class="tag">Video Generated</div>', unsafe_allow_html=True)
                                st.download_button(
                                    label="Download Video",
                                    data=video_bytes,
                                    file_name=f"generated_video_{idx+1}.mp4",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                    except Exception as gen_error:
                        st.error(f"Error in final generation: {str(gen_error)}")
                        logger.error(f"Error in final generation: {str(gen_error)}")

                st.markdown("</div>", unsafe_allow_html=True)

        # Clear progress indicators
        progress_bar.empty()
        status_text.success(f"Successfully processed {num_images} images!")

    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
        logger.exception(f"Error processing images: {str(e)}")

# Instructions
st.markdown("""
---
### How to Use:
1. Select generation mode (Single Image or Video)
2. Upload one or more images containing people
3. For video generation, adjust FPS and duration in the sidebar
4. Each image will be processed through five steps:
   - Step 1: Original Image Display
   - Step 2: Pose Extraction
   - Step 3: Prompt Generation
   - Step 4: OpenPose Generation
   - Step 5: Final Result (Image or Video)
5. Download your generated content using the download buttons
""")

def get_pose_refinement_suggestions(pose_landmarks):
    suggestions = {}
    for landmark in pose_landmarks:
        if landmark.x < 0.3:
            suggestions["left_side"] = "Move to the right"
        elif landmark.x > 0.7:
            suggestions["right_side"] = "Move to the left"
        if landmark.y < 0.3:
            suggestions["upper_body"] = "Stand up straight"
        elif landmark.y > 0.7:
            suggestions["lower_body"] = "Bend your knees slightly"

    return suggestions