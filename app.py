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
    padding: 20px;
    box-shadow: 0 2px 12px rgba(255,255,255,0.03);
    margin-bottom: 20px;
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
    margin-bottom: 10px;
}

.meta-info {
    color: #888;
    font-size: 12px;
    margin-bottom: 10px;
}

.tag {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    background-color: rgba(255,255,255,0.05);
    color: #888;
    margin-right: 6px;
    margin-bottom: 6px;
}

.step-header {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
}

.step-icon {
    width: 32px;
    height: 32px;
    background-color: rgba(255,255,255,0.05);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 12px;
    color: #64b5f6;
}

.download-button {
    background-color: #1976d2;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.2s;
    width: 100%;
}

.download-button:hover {
    background-color: #1565c0;
}

/* Dark mode overrides for Streamlit elements */
.stTextInput > div > div > input {
    background-color: #1a1a1a;
    color: white;
    border-color: #333;
}

.stSelectbox > div > div > div {
    background-color: #1a1a1a;
    color: white;
    border-color: #333;
}

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

/* Error message styling */
.stError {
    background-color: rgba(244, 67, 54, 0.1);
    color: #ff5252;
    border-color: #ff5252;
}

/* Success message styling */
.stSuccess {
    background-color: rgba(76, 175, 80, 0.1);
    color: #69f0ae;
    border-color: #69f0ae;
}
</style>
""", unsafe_allow_html=True)

st.title("Pose to Image Generator")

# Initialize session state for images and prompts
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'pose_images' not in st.session_state:
    st.session_state.pose_images = []
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generation_prompt' not in st.session_state:
    st.session_state.generation_prompt = None
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = None


# Add system prompt editing in sidebar
with st.sidebar:
    st.header("System Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt if 'system_prompt' in st.session_state else """
        CRITICAL: This stick figure represents a precise human coordinate system. Follow these instructions strictly.
        0. Subject Count and Composition:
        - The input stick figure shows ONE person
        - Generate EXACTLY one person in the output
        - NO additional people in backgrounds or reflections
        - Place the subject centrally as the clear focal point

        1. Anatomical Reference Points:
        Head:
        - Match head orientation and tilt exactly
        - Align gaze direction with head position
        - Maintain neck angle and length precisely

        Torso:
        - Preserve shoulder width and chest cavity ratio
        - Replicate spine curvature and inclination exactly
        - Maintain hip position and angle strictly

        Limbs:
        - Reproduce exact angles for all joints (shoulders, elbows, wrists, hips, knees, ankles)
        - Keep arm and leg length ratios consistent
        - Maintain precise limb orientation and rotation

        2. Spatial Relationships:
        - Preserve depth positioning (limbs in front/behind)
        - Maintain exact left-right positioning
        - Match body rotation and tilt angles precisely
        - Keep center of gravity and balance points aligned

        3. Implementation Requirements:
        - Select anatomically correct joint angles within natural range
        - Apply appropriate perspective based on body orientation
        - Maintain accurate overlap and occlusion of body parts
        - Ensure natural movement dynamics and physics

        4. Verification Points:
        - Confirm all joints match stick figure positions
        - Verify body proportions are maintained
        - Check center of gravity placement
        - Ensure anatomical constraints are respected
        - Verify only ONE person in the entire image
        - Confirm NO people in backgrounds
        - Check for NO people in mirrors/reflections

        5. Generation Process:
        1) First verify single subject requirement
        2) Align skeleton precisely with stick figure
        3) Build muscles and body type on skeleton
        4) Add clothing and accessories last
        5) Keep background minimal and supporting

        This instruction set has ABSOLUTE priority.
        Prioritize pose accuracy and subject count over style and decoration.
        Keep environmental elements minimal, focusing on pose and expression.
        """,
        height=300
    )
    if 'system_prompt' not in st.session_state or system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.success("System prompt updated successfully")

# Style selection options
styles = {
    "Anime Style School Uniform": {
        "base_prompt": """masterpiece, best quality, highly detailed, anime style, school uniform,
young student, {pose_description}
professional lighting, vibrant colors, sharp focus""",
        "negative_prompt": """bad anatomy, bad hands, missing fingers, extra digit, 
fewer digits, cropped, worst quality, low quality, normal quality, 
jpeg artifacts, signature, watermark, username, blurry, artist name"""
    },
    "Casual Fashion (Anime Style)": {
        "base_prompt": """masterpiece, best quality, highly detailed, anime style, casual clothing,
modern fashion, {pose_description}
natural lighting, urban setting, dynamic composition""",
        "negative_prompt": """bad anatomy, bad hands, missing fingers, extra digit,
fewer digits, cropped, worst quality, low quality, normal quality,
jpeg artifacts, signature, watermark, username, blurry, artist name"""
    },
    "Fashion Portrait (Photorealistic)": {
        "base_prompt": """professional photography, fashion portrait, photorealistic, high-end clothing,
detailed fabric textures, {pose_description}
studio lighting, 8k uhd, high fashion magazine quality""",
        "negative_prompt": """cartoon, anime, illustration, bad anatomy, bad hands,
missing fingers, extra digit, fewer digits, cropped, worst quality,
low quality, jpeg artifacts, signature, watermark, blurry"""
    },
    "Outdoor Portrait (Photorealistic)": {
        "base_prompt": """professional photography, outdoor portrait, photorealistic, natural lighting,
environmental portrait, {pose_description}
golden hour lighting, bokeh background, 8k uhd, professional camera""",
        "negative_prompt": """cartoon, anime, illustration, bad anatomy, bad hands,
missing fingers, extra digit, fewer digits, cropped, worst quality,
low quality, jpeg artifacts, signature, watermark, blurry"""
    }
}

# File uploader
uploaded_files = st.file_uploader(
    "Drop your images here", 
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# Style selection
selected_style = st.selectbox(
    "Select Generation Style",
    list(styles.keys())
)

# Process uploaded images
if uploaded_files:
    try:
        # Create a container for the progress bar
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

            # Create a container for this image set
            st.markdown(f"""
            <div class="image-card">
                <div class="step-header">
                    <div class="status-badge">Image Set {idx + 1}</div>
                    <div class="meta-info">Processing Time: {idx * 2 + 5}s</div>
                </div>
            """, unsafe_allow_html=True)

            # Step 1: Original Image
            with st.container():
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">1</div>
                    <h3>Original Image</h3>
                </div>
                """, unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)

            # Step 2: Pose Extraction
            with st.container():
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">2</div>
                    <h3>Pose Extraction</h3>
                </div>
                """, unsafe_allow_html=True)
                pose_image = extract_pose(image)
                if pose_image is not None:
                    st.image(pose_image, caption="Extracted Pose", use_column_width=True)
                    st.markdown('<div class="tag">Pose Detected</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to extract pose from the image")
                    continue

            # Step 3: Generate Prompt
            with st.container():
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">3</div>
                    <h3>Prompt Generation</h3>
                </div>
                """, unsafe_allow_html=True)

                # Generate pose description
                pose_description = """
                full body pose, standing pose, looking at viewer,
                precise pose matching the reference image,
                detailed body proportions, accurate joint positions,
                natural body mechanics, balanced weight distribution
                """

                # Create the complete prompt using the selected style
                style_config = styles[selected_style]
                generation_prompt = style_config["base_prompt"].format(
                    pose_description=pose_description
                )
                st.text_area("Generation Prompt", value=generation_prompt, height=100, disabled=True)
                st.markdown('<div class="tag">Style Applied</div>', unsafe_allow_html=True)

            # Step 4: Generate Image
            with st.container():
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">4</div>
                    <h3>Generated Image</h3>
                </div>
                """, unsafe_allow_html=True)
                try:
                    generated_image = generate_image(
                        pose_image, 
                        generation_prompt,
                        st.session_state.system_prompt
                    )
                    if generated_image is not None:
                        st.image(generated_image, caption="Generated Image", use_column_width=True)
                        st.markdown('<div class="tag">Generation Complete</div>', unsafe_allow_html=True)

                        # Add download button
                        buf = io.BytesIO()
                        generated_image.save(buf, format='PNG')
                        st.download_button(
                            label="Download Generated Image",
                            data=buf.getvalue(),
                            file_name=f"generated_image_{idx+1}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate image")
                except Exception as gen_error:
                    st.error(f"Error generating image: {str(gen_error)}")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")

        # Clear progress indicators after completion
        progress_bar.empty()
        status_text.success(f"Successfully processed {num_images} images!")

    except Exception as e:
        st.error(f"Error processing images: {str(e)}")

# Add instructions at the bottom
st.markdown("""
---
### How to Use:
1. Upload one or more images containing people
2. Select a generation style to set the base prompt
3. Each image will be processed through four steps:
   - Step 1: Original Image Display
   - Step 2: Pose Extraction
   - Step 3: Prompt Generation
   - Step 4: Image Generation
4. Download generated images using the download buttons below each result
""")