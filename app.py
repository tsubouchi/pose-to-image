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

with col1:
    st.header("Original Image")
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            st.image(image, caption="Original Image")
            
            # Extract pose
            pose_image = extract_pose(image)
            st.session_state.pose_image = pose_image
            
            # Generate new image
            generated_image = generate_image(pose_image)
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
### How it works:
1. Upload an image containing a person
2. The system will extract the pose using MediaPipe
3. Gemini 2.0 Flash will generate a new anime-style image based on the pose
""")
