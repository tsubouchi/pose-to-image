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

# File uploader - ドラッグ&ドロップ機能の修正
uploaded_files = st.file_uploader(
    "Drop your images here",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    key="file_uploader"  # Unique keyを追加
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
            with st.container():
                st.markdown(f"""
                <div class="image-card">
                    <div class="step-header">
                        <div class="status-badge">Image Set {idx + 1}</div>
                        <div class="meta-info">Processing Time: {idx * 2 + 5}s</div>
                    </div>
                """, unsafe_allow_html=True)

                # Create columns with proper spacing
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

                # Step 2: Pose Extraction
                with cols[1]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">2</div>
                        <h3>Pose Extraction</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        pose_image, pose_descriptions, results = extract_pose(image) #modified to return results
                        if pose_image is not None:
                            st.image(pose_image, use_container_width=True)
                            st.markdown('<div class="tag">Pose Detected</div>', unsafe_allow_html=True)

                            # Add pose refinement suggestions
                            if results and results.pose_landmarks:
                                suggestions = get_pose_refinement_suggestions(results.pose_landmarks)

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
                        else:
                            st.error("Error in pose extraction, using default pose")
                            pose_image = Image.new('RGB', (100,100)) #placeholder image
                            pose_descriptions = {'right_shoulder_desc':'default','right_elbow_desc':'default','left_shoulder_desc':'default','left_elbow_desc':'default','spine_desc':'default','right_hip_desc':'default','right_knee_desc':'default','left_hip_desc':'default','left_knee_desc':'default'}

                    except Exception as e:
                        st.error("Error in pose extraction, using default pose")
                        pose_image = Image.new('RGB', (100,100)) #placeholder image
                        pose_descriptions = {'right_shoulder_desc':'default','right_elbow_desc':'default','left_shoulder_desc':'default','left_elbow_desc':'default','spine_desc':'default','right_hip_desc':'default','right_knee_desc':'default','left_hip_desc':'default','left_knee_desc':'default'}
                        continue

                # Step 3: Generation Prompt
                with cols[2]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">3</div>
                        <h3>Generation Prompt</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Format the prompt with pose descriptions
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
                with cols[3]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">4</div>
                        <h3>Pose to Human</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        # First pass: Convert stick figure to realistic human form
                        human_form_prompt = f"""Important: Convert stick figure to precise human form.

CRITICAL - Exact Pose Requirements:
{', '.join([f'{k}: {v}' for k,v in pose_descriptions.items()])}

Instructions for conversion:
1. Maintain ALL joint angles and positions exactly as shown
2. Keep body proportions anatomically correct
3. Preserve the pose's distinctive characteristics
4. Focus on structural accuracy over style

Technical Requirements:
- Use neutral 3D rendering
- Simple gray background
- Clear lighting without shadows
- Focus on skeletal and muscular structure
- Ensure all limbs are correctly positioned
- Maintain center of gravity and balance points

Priority: Accuracy of pose over visual aesthetics"""

                        human_pose = generate_image(
                            pose_image,
                            human_form_prompt,
                            "Anatomically correct human figure, exact pose matching, technical reference quality"
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

                # Step 5: Generate realistic photo first, then convert to anime
                with cols[4]:
                    st.markdown("""
                    <div class="step-header">
                        <div class="step-icon">5</div>
                        <h3>Final Image</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        # First generate realistic photo
                        style_prompt = f"""System Instructions: CRITICAL - Generate high quality image from bone structure.

PHASE 1 - BONE STRUCTURE INTERPRETATION:
1. Analyze Input Structure:
- Each green line represents a specific bone/joint connection
- Structure defines the exact 3D pose positioning
- All joint angles must be perfectly matched
- Every bone length determines limb proportions

2. Precise Joint Positioning:
Upper Body:
- Right Arm: {pose_descriptions['right_shoulder_desc']}, elbow {pose_descriptions['right_elbow_desc']}
- Left Arm: {pose_descriptions['left_shoulder_desc']}, elbow {pose_descriptions['left_elbow_desc']}
- {pose_descriptions['spine_desc']}

Lower Body:
- Right Leg: hip {pose_descriptions['right_hip_desc']}, knee {pose_descriptions['right_knee_desc']}
- Left Leg: hip {pose_descriptions['left_hip_desc']}, knee {pose_descriptions['left_knee_desc']}

3. Character Creation:
Body Features:
- Young female high school student
- Natural, balanced proportions
- Graceful and dynamic pose
- Smooth, natural muscle definition
- Realistic body weight distribution

Clothing:
- High-quality sailor school uniform
- Crisp, well-fitted design
- Pleated skirt with natural flow
- Proper fabric wrinkles and folds
- Clean, detailed uniform elements

Environmental Details:
- Modern classroom setting
- Natural daylight from windows
- Soft shadows and highlights
- Clean, uncluttered background
- Professional depth of field

4. Technical Quality:
- Ultra-high resolution details
- Professional photography quality
- Perfect anatomical accuracy
- Cinematic lighting setup
- Sharp focus on subject
- Natural color grading

CRITICAL REQUIREMENTS:
- Perfect bone structure matching
- Photorealistic rendering quality
- Natural pose execution
- Comprehensive lighting setup
- Studio-quality final output

masterpiece, best quality, highly detailed, professional photograph"""

                        photo_image = generate_image(
                            pose_image,
                            style_prompt,
                            "Create ultra-high quality photo matching bone structure exactly"
                        )

                        if photo_image is not None:
                            # Convert photo to anime style
                            anime_prompt = """Transform into masterpiece anime artwork while preserving exact pose:

1. Character Style:
- Beautiful anime art style
- Clean, professional linework
- Detailed facial features
- Large, expressive eyes
- Flowing, dynamic hair
- Soft, appealing color palette

2. Uniform Details:
- Crisp sailor school uniform
- Detailed collar and cuffs
- Perfectly pleated skirt
- Subtle fabric textures
- School emblem and accessories
- Natural clothing physics

3. Environmental Elements:
- Detailed classroom background
- Soft natural lighting
- Window light effects
- Subtle ambient shadows
- Clean depth of field
- Enhanced atmosphere

4. Technical Excellence:
- High-end anime production quality
- Professional cel shading
- Perfect line consistency
- Rich color composition
- Dynamic lighting effects
- Studio-quality finish

5. Artistic Focus:
- Maintain exact pose structure
- Enhance visual appeal
- Create emotional impact
- Perfect balance of elements
- Professional composition

masterpiece, best quality, professional anime style, highly detailed"""

                            final_image = generate_image(
                                photo_image,
                                anime_prompt,
                                "Convert to highest quality anime while preserving exact pose"
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
                                st.error("Failed to convert to anime style")
                        else:
                            st.error("Failed to generate realistic photo")
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

def get_pose_refinement_suggestions(pose_landmarks):
    # Placeholder for actual pose refinement logic.  Replace with your implementation
    # This function should analyze pose_landmarks and return a dictionary of suggestions
    # Example:  {"right_shoulder": "Try to straighten your right shoulder", "left_elbow": "Bend your left elbow slightly more"}
    #  return {"error": "Pose refinement not yet implemented"} #For testing
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