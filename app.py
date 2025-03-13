import streamlit as st
import mediapipe as mp
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image

st.set_page_config(
    page_title="Pose to Image Generator",
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
.image-card {
    position: relative;
    overflow: hidden;
}

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
</style>
""", unsafe_allow_html=True)

st.title("Pose to Image Generator")

# Initialize session state
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'pose_images' not in st.session_state:
    st.session_state.pose_images = []
if 'intermediate_images' not in st.session_state:
    st.session_state.intermediate_images = []
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

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
    accept_multiple_files=True
)

# Style selection - fixed to school uniform
selected_style = "Anime Style School Uniform"

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
            st.markdown(f"""
            <div class="image-card">
                <div class="step-header">
                    <div class="status-badge">Image Set {idx + 1}</div>
                    <div class="meta-info">Processing Time: {idx * 2 + 5}s</div>
                </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)

            # Step 1: Original Image
            with col1:
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">1</div>
                    <h3>Original Image</h3>
                </div>
                """, unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            # Step 2: Pose Extraction
            with col2:
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">2</div>
                    <h3>Pose Extraction</h3>
                </div>
                """, unsafe_allow_html=True)
                pose_image, pose_descriptions = extract_pose(image)
                if pose_image is not None:
                    st.image(pose_image, use_container_width=True)
                    st.markdown('<div class="tag">Pose Detected</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to extract pose from the image")
                    continue

            # Step 3: Generation Prompt
            with col3:
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">3</div>
                    <h3>Generation Prompt</h3>
                </div>
                """, unsafe_allow_html=True)

                generation_prompt = f"""masterpiece, best quality, highly detailed,

Body Parts Position Description:
1. Upper Body:
- Right Arm: {pose_descriptions['right_shoulder_desc']}, elbow {pose_descriptions['right_elbow_desc']}
- Left Arm: {pose_descriptions['left_shoulder_desc']}, elbow {pose_descriptions['left_elbow_desc']}
- {pose_descriptions['spine_desc']}

2. Lower Body:
- Right Leg: hip {pose_descriptions['right_hip_desc']}, knee {pose_descriptions['right_knee_desc']}
- Left Leg: hip {pose_descriptions['left_hip_desc']}, knee {pose_descriptions['left_knee_desc']}

Style Elements:
- high quality anime art style
- school uniform with pleated skirt and sailor collar
- natural indoor lighting with soft shadows
- classroom environment background
- clear facial features and detailed eyes
- clean lineart and professional shading
"""

                st.text_area("Prompt", value=generation_prompt, height=300, disabled=True)
                st.markdown('<div class="tag">Style Applied</div>', unsafe_allow_html=True)

            # Step 4: Pose to Human
            with col4:
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">4</div>
                    <h3>Pose to Human</h3>
                </div>
                """, unsafe_allow_html=True)
                try:
                    # First pass: Convert stick figure to basic human form
                    human_pose = generate_image(
                        pose_image,
                        "Convert stick figure to human form, maintain exact pose",
                        st.session_state.system_prompt
                    )
                    if human_pose is not None:
                        st.image(human_pose, use_container_width=True)
                        st.markdown('<div class="tag">Human Form Generated</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to convert pose to human form")
                        continue
                except Exception as e:
                    st.error(f"Error in pose to human conversion: {str(e)}")
                    continue

            # Step 5: Final Style Generation
            with col5:
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">5</div>
                    <h3>Final Image</h3>
                </div>
                """, unsafe_allow_html=True)
                try:
                    # Second pass: Apply final style to human form
                    final_image = generate_image(
                        human_pose,
                        generation_prompt,
                        st.session_state.system_prompt
                    )
                    if final_image is not None:
                        st.image(final_image, use_container_width=True)
                        st.markdown('<div class="tag">Generation Complete</div>', unsafe_allow_html=True)

                        # Add download button
                        buf = io.BytesIO()
                        final_image.save(buf, format='PNG')
                        st.download_button(
                            label="Download",
                            data=buf.getvalue(),
                            file_name=f"generated_image_{idx+1}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate final image")
                except Exception as gen_error:
                    st.error(f"Error generating final image: {str(gen_error)}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Clear progress indicators
        progress_bar.empty()
        status_text.success(f"Successfully processed {num_images} images!")

    except Exception as e:
        st.error(f"Error processing images: {str(e)}")

# Instructions
st.markdown("""
---
### How to Use:
1. Upload one or more images containing people
2. The generation style is fixed to "Anime Style School Uniform"
3. Each image will be processed through five steps:
   - Step 1: Original Image Display
   - Step 2: Pose Extraction
   - Step 3: Prompt Generation
   - Step 4: Pose to Human Conversion
   - Step 5: Final Style Generation
4. Download generated images using the download buttons
""")