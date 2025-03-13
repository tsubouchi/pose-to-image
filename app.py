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

# Initialize session state for images and system prompt
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'pose_image' not in st.session_state:
    st.session_state.pose_image = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = None
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = """
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
    """

# Style selection options
styles = {
    "Anime Style School Uniform": """
    Create an anime-style high school character based on this stick figure pose.
    - Modern school uniform design
    - Bright anime-style coloring
    - Natural hair movement and expression
    - Distinctive character design
    - School or urban background scene
    """,

    "Casual Fashion (Anime Style)": """
    Create an anime-style character in casual clothing based on this stick figure pose.
    - Casual wear (T-shirt, jeans, skirt, etc.)
    - Modern hairstyle
    - Natural expression
    - Accessories and fashion details
    - Urban background setting
    """,

    "Fashion Portrait (Photorealistic)": """
    Create a photorealistic fashion portrait based on this stick figure pose.
    - High-resolution photographic quality
    - Contemporary fashion styling
    - Natural lighting and shadows
    - Professional studio atmosphere
    - Soft bokeh background
    """,

    "Outdoor Portrait (Photorealistic)": """
    Create a photorealistic outdoor portrait based on this stick figure pose.
    - Natural lighting photography style
    - Casual and active clothing
    - Natural expression and posture
    - Park or urban outdoor setting
    - Natural environmental lighting and shadows
    """
}

# Add system prompt editing in sidebar
with st.sidebar:
    st.header("System Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=300
    )
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.success("System prompt updated successfully")

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

        # Step 3: Display prompt preview
        with col3:
            st.header("Step 3: Generation Prompt")
            if pose_image is not None:
                prompt = f"""
                System Instructions:
                {st.session_state.system_prompt}

                Style Specifications:
                {styles[selected_style]}
                """
                st.session_state.current_prompt = prompt
                st.text_area("Generation Prompt", prompt, height=400)

        # Step 4: Generate and display new image
        with col4:
            st.header("Step 4: Generated Image")
            if st.session_state.current_prompt:
                generated_image = generate_image(
                    pose_image, 
                    styles[selected_style],
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
2. Select the desired generation style
   - Anime Style: Generate anime/illustration style images
   - Photorealistic: Generate realistic photo-like images
3. Review the generation prompt and adjust if needed
4. The system will automatically extract the pose and generate a new image in the selected style
""")