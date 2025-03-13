import mediapipe as mp
import numpy as np
import cv2
from PIL import Image

def extract_pose(pil_image):
    """
    Extract pose from the input image and return a stick figure representation
    """
    # Convert PIL Image to numpy array
    image = np.array(pil_image)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )
    
    # Create blank canvas for stick figure
    height, width = image.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process the image
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    if results.pose_landmarks:
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            canvas,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    else:
        raise ValueError("No pose detected in the image")
    
    # Convert back to PIL Image
    pose_image = Image.fromarray(canvas)
    
    return pose_image
