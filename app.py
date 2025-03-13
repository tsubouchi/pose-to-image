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
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'pose_image' not in st.session_state:
    st.session_state.pose_image = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
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

# Top section: Drag & Drop
uploaded_file = st.file_uploader("Drop your image here", type=['png', 'jpg', 'jpeg'])

# Style selection
selected_style = st.selectbox(
    "Select Generation Style",
    list(styles.keys())
)

# Create four columns for the images
col1, col2, col3, col4 = st.columns(4)

if uploaded_file is not None:
    try:
        # Step 1: Display original image
        with col1:
            st.header("Step 1: Original Image")
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            st.image(image, caption="Original Image")

        # Step 2: Extract and display pose
        with col2:
            st.header("Step 2: Pose Extraction")
            pose_image = extract_pose(image)
            st.session_state.pose_image = pose_image
            st.image(pose_image, caption="Extracted Pose")

        # Step 3: Display and edit generation prompt
        with col3:
            st.header("Step 3: Image Prompt")
            if pose_image is not None:
                # Generate pose description
                pose_description = """
                full body pose, standing pose, looking at viewer,
                precise pose matching the reference image,
                detailed body proportions, accurate joint positions,
                natural body mechanics, balanced weight distribution
                """

                # Create the complete prompt using the selected style
                style_config = styles[selected_style]
                default_prompt = style_config["base_prompt"].format(
                    pose_description=pose_description
                )

                # Make the prompt editable
                generation_prompt = st.text_area(
                    "Edit Image Generation Prompt",
                    value=default_prompt if st.session_state.generation_prompt is None else st.session_state.generation_prompt,
                    height=400
                )
                st.session_state.generation_prompt = generation_prompt

                # Display negative prompt (read-only)
                st.text_area(
                    "Negative Prompt (Applied Automatically)",
                    value=style_config["negative_prompt"],
                    height=100,
                    disabled=True
                )

        # Step 4: Generate and display new image
        with col4:
            st.header("Step 4: Generated Image")
            if st.session_state.generation_prompt:
                generated_image = generate_image(
                    pose_image, 
                    st.session_state.generation_prompt,
                    st.session_state.system_prompt
                )
                st.session_state.generated_image = generated_image
                if generated_image is not None:
                    st.image(generated_image, caption="Generated Image")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown("""
---
### How to Use:
1. Upload an image containing a person
2. Select a generation style to set the base prompt
3. Review and customize the generation prompt:
   - The prompt describes the desired image style and characteristics
   - Edit the prompt to fine-tune the output
   - The negative prompt helps avoid common issues
4. The system will generate a new image based on your specifications
""")