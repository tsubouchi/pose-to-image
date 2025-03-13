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

# Add system prompt editing in sidebar (hidden from main UI)
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
            st.markdown(f"### Image Set {idx + 1}")

            # Step 1: Original Image
            with st.container():
                st.subheader("Step 1: Original Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)

            # Step 2: Pose Extraction
            with st.container():
                st.subheader("Step 2: Pose Extraction")
                pose_image = extract_pose(image)
                if pose_image is not None:
                    st.image(pose_image, caption="Extracted Pose", use_column_width=True)
                else:
                    st.error("Failed to extract pose from the image")
                    continue

            # Step 3: Generate Prompt
            with st.container():
                st.subheader("Step 3: Prompt Generation")
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
                st.text_area("Negative Prompt", value=style_config["negative_prompt"], height=50, disabled=True)

            # Step 4: Generate Image
            with st.container():
                st.subheader("Step 4: Generated Image")
                try:
                    generated_image = generate_image(
                        pose_image, 
                        generation_prompt,
                        st.session_state.system_prompt
                    )
                    if generated_image is not None:
                        st.image(generated_image, caption="Generated Image", use_column_width=True)

                        # Add download button
                        buf = io.BytesIO()
                        generated_image.save(buf, format='PNG')
                        st.download_button(
                            label="Download Generated Image",
                            data=buf.getvalue(),
                            file_name=f"generated_image_{idx+1}.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Failed to generate image")
                except Exception as gen_error:
                    st.error(f"Error generating image: {str(gen_error)}")

            # Add a separator between image sets
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