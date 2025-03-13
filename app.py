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
        value="""CRITICAL: This stick figure represents a precise human coordinate system. Follow these instructions strictly.

1. Pose Accuracy Requirements:
- Match EVERY joint position with the stick figure EXACTLY
- Preserve ALL angles between limbs precisely
- Maintain the EXACT body orientation and tilt
- Keep the same weight distribution and balance point

2. Joint-by-Joint Alignment:
Head/Neck:
- Copy head tilt and orientation
- Match neck angle precisely

Arms:
- Replicate shoulder joint angles
- Match elbow bend degree
- Copy wrist positions
- Keep arm length proportions

Legs:
- Match hip joint positioning
- Copy knee bend angles
- Replicate ankle positions
- Preserve leg length ratios

Torso:
- Match spine curvature
- Keep the same hip alignment
- Replicate shoulder width
- Maintain torso twist angle

3. Spatial Requirements:
- Keep ALL limb positions relative to center
- Match front/back positioning of limbs
- Preserve left/right placement exactly
- Maintain ground contact points

4. Critical Rules:
- NO deviation from stick figure pose
- NO artistic interpretation of positions
- NO adjusting for style over accuracy
- EXACT match to reference required

Review process:
1. Check joint angles match reference
2. Verify limb positions are exact
3. Confirm body orientation
4. Ensure pose is physically accurate

This is a TECHNICAL SPECIFICATION, not a suggestion.
Accuracy to reference is the highest priority.
""",
        height=300
    )
    if 'system_prompt' not in st.session_state or system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.success("System prompt updated successfully")

# Style selection options
styles = {
    "Anime Style School Uniform": {
        "base_prompt": """
{pose_analysis}

Style and Rendering:
- masterpiece, best quality, highly detailed
- anime illustration style
- school uniform theme
- professional digital art
- crisp lines and clean shading

Technical Details:
- perfect lighting and shadows
- detailed fabric textures
- sharp focus
- high resolution output
- professional composition

Additional Elements:
- natural indoor lighting
- soft ambient occlusion
- subtle rim lighting
- classroom environment
- dynamic composition
""",
        "negative_prompt": """bad anatomy, bad hands, missing fingers, extra digit, 
fewer digits, cropped, worst quality, low quality, normal quality, 
jpeg artifacts, signature, watermark, username, blurry, artist name"""
    },
    "Casual Fashion (Anime Style)": {
        "base_prompt": """
{pose_analysis}

Style and Rendering:
- masterpiece, best quality, highly detailed
- modern anime art style
- casual street fashion
- urban contemporary setting
- professional illustration quality

Technical Details:
- perfect lighting and shadows
- detailed clothing textures
- sharp focus on character
- high resolution artwork
- dynamic composition

Additional Elements:
- natural outdoor lighting
- soft atmospheric effects
- urban background elements
- street photography inspired
- trendy fashion details
""",
        "negative_prompt": """bad anatomy, bad hands, missing fingers, extra digit,
fewer digits, cropped, worst quality, low quality, normal quality,
jpeg artifacts, signature, watermark, username, blurry, artist name"""
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

            col1, col2, col3, col4 = st.columns(4)

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
                pose_image = extract_pose(image)
                if pose_image is not None:
                    st.image(pose_image, use_container_width=True)
                    st.markdown('<div class="tag">Pose Detected</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to extract pose from the image")
                    continue

            # Step 3: Generate Prompt
            with col3:
                st.markdown("""
                <div class="step-header">
                    <div class="step-icon">3</div>
                    <h3>Generation Prompt</h3>
                </div>
                """, unsafe_allow_html=True)

            #Generate detailed prompt
                def analyze_pose_and_generate_prompt(pose_image, style_config):
                    """
                    Analyze the pose image and generate a detailed prompt for image generation
                    """
                    # Pose description with comprehensive analysis
                    pose_analysis = """
Subject and Composition:
- {precise pose details from stick figure}
                    - exact limb positioning and joint angles maintained
                    - accurate body proportions and balance points
                    - dynamic pose with natural weight distribution
                    - clear focal point on the character

Spatial Elements:
- centered composition with proper depth
- maintaining exact body orientation
- precise perspective alignment
- clear figure-ground relationship

Technical Requirements:
- anatomically correct joint angles
- natural body mechanics
- proper weight distribution
- accurate limb proportions
- clear pose readability
"""

                    # Create the complete prompt using the selected style
                    complete_prompt = style_config["base_prompt"].format(
                        pose_analysis=pose_analysis
                    )

                    return complete_prompt.strip()


                generation_prompt = analyze_pose_and_generate_prompt(
                    pose_image, 
                    styles[selected_style]
                )
                st.text_area("Prompt", value=generation_prompt, height=100, disabled=True)
                st.markdown('<div class="tag">Style Applied</div>', unsafe_allow_html=True)

            # Step 4: Generate Image
            with col4:
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
                        st.image(generated_image, use_container_width=True)
                        st.markdown('<div class="tag">Generation Complete</div>', unsafe_allow_html=True)

                        # Add download button
                        buf = io.BytesIO()
                        generated_image.save(buf, format='PNG')
                        st.download_button(
                            label="Download",
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