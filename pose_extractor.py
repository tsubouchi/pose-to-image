import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import logging

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_image_content(image):
    """
    First stage: Analyze the image content to understand the subject
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.7
    ) as pose:
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Process the image
        results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if not results.pose_landmarks:
            raise ValueError("No human subject detected in the image")

        return results, image_np.shape[:2]

def create_enhanced_stick_figure(results, image_shape, thickness_multiplier=1.5):
    """
    Second stage: Create an enhanced stick figure representation
    """
    height, width = image_shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Customize drawing specs for better visibility
    landmark_drawing_spec = mp_drawing.DrawingSpec(
        color=(50, 205, 50),  # Lime Green
        thickness=int(2 * thickness_multiplier),
        circle_radius=int(2 * thickness_multiplier)
    )
    connection_drawing_spec = mp_drawing.DrawingSpec(
        color=(30, 144, 255),  # Dodger Blue
        thickness=int(2 * thickness_multiplier)
    )

    # Draw the pose landmarks
    mp_drawing.draw_landmarks(
        canvas,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec,
        connection_drawing_spec
    )

    return canvas

def extract_pose(pil_image):
    """
    Main function: Extract pose from the input image with enhanced processing
    """
    try:
        logger.debug("Starting pose extraction process")

        # Stage 1: Analyze image content
        logger.debug("Stage 1: Analyzing image content")
        results, image_shape = analyze_image_content(pil_image)

        # Validate detection confidence
        if results.pose_landmarks:
            confidence = sum(lm.visibility for lm in results.pose_landmarks.landmark) / len(results.pose_landmarks.landmark)
            logger.debug(f"Pose detection confidence: {confidence:.2f}")
            if confidence < 0.5:
                logger.warning("Low confidence in pose detection")

        # Stage 2: Create enhanced stick figure
        logger.debug("Stage 2: Creating enhanced stick figure")
        stick_figure = create_enhanced_stick_figure(results, image_shape)

        # Convert back to PIL Image
        pose_image = Image.fromarray(stick_figure)

        logger.debug("Pose extraction completed successfully")
        return pose_image

    except Exception as e:
        logger.error(f"Error in pose extraction: {str(e)}")
        raise Exception(f"Failed to extract pose: {str(e)}")