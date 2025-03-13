import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import logging
import math
from typing import Dict, List, Tuple

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate the angle between three points in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2

    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def calculate_joint_angles(landmarks) -> Dict[str, float]:
    """
    Calculate all relevant joint angles from pose landmarks
    """
    # Convert landmarks to numpy arrays
    points = {}
    for idx, landmark in enumerate(landmarks.landmark):
        points[idx] = np.array([landmark.x, landmark.y, landmark.z])

    # Calculate angles for each joint
    angles = {
        # Right arm angles
        "right_shoulder": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        ),
        "right_elbow": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        ),

        # Left arm angles
        "left_shoulder": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        ),
        "left_elbow": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        ),

        # Right leg angles
        "right_hip": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        ),
        "right_knee": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
            points[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        ),

        # Left leg angles
        "left_hip": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        ),
        "left_knee": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
            points[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        ),

        # Spine angles
        "spine": calculate_angle(
            points[mp.solutions.pose.PoseLandmark.NOSE.value],
            points[mp.solutions.pose.PoseLandmark.SHOULDERS.value],
            points[mp.solutions.pose.PoseLandmark.HIPS.value]
        )
    }

    return angles

def get_pose_description(angles: Dict[str, float]) -> Dict[str, str]:
    """
    Convert numerical angles to natural language descriptions
    """
    def describe_angle(angle: float, joint_type: str) -> str:
        if joint_type == "elbow":
            if angle > 150:
                return "straight"
            elif angle > 90:
                return f"slightly bent at {angle:.1f} degrees"
            else:
                return f"bent at {angle:.1f} degrees"
        elif joint_type == "knee":
            if angle > 160:
                return "straight"
            elif angle > 110:
                return f"slightly bent at {angle:.1f} degrees"
            else:
                return f"bent at {angle:.1f} degrees"
        else:  # shoulder, hip
            return f"at {angle:.1f} degrees"

    return {
        "right_shoulder_desc": describe_angle(angles["right_shoulder"], "shoulder"),
        "right_elbow_desc": describe_angle(angles["right_elbow"], "elbow"),
        "left_shoulder_desc": describe_angle(angles["left_shoulder"], "shoulder"),
        "left_elbow_desc": describe_angle(angles["left_elbow"], "elbow"),
        "right_hip_desc": describe_angle(angles["right_hip"], "hip"),
        "right_knee_desc": describe_angle(angles["right_knee"], "knee"),
        "left_hip_desc": describe_angle(angles["left_hip"], "hip"),
        "left_knee_desc": describe_angle(angles["left_knee"], "knee"),
        "spine_desc": f"spine aligned at {angles['spine']:.1f} degrees"
    }

def extract_pose(pil_image) -> Tuple[Image.Image, Dict[str, str]]:
    """
    Extract pose from the input image and return both visualization and pose descriptions
    """
    try:
        logger.debug("Starting pose extraction process")

        # Convert PIL Image to numpy array
        image_np = np.array(pil_image)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Use the most accurate model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            # Process the image
            results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if not results.pose_landmarks:
                raise ValueError("No pose landmarks detected")

            # Calculate joint angles
            angles = calculate_joint_angles(results.pose_landmarks)
            pose_descriptions = get_pose_description(angles)

            # Create visualization
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            # Create canvas for visualization
            canvas = np.zeros(image_np.shape, dtype=np.uint8)

            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                canvas,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(50, 205, 50), thickness=4, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(30, 144, 255), thickness=2)
            )

            # Convert back to PIL Image
            pose_image = Image.fromarray(canvas)

            logger.debug("Pose extraction completed with angle calculations")
            return pose_image, pose_descriptions

    except Exception as e:
        logger.error(f"Error in pose extraction: {str(e)}")
        raise Exception(f"Failed to extract pose: {str(e)}")

def analyze_image_content(image):
    """
    First stage: Analyze the image content to understand the subject
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,  # Reduced complexity for better generalization
        min_detection_confidence=0.3,  # Lower threshold for more lenient detection
        min_tracking_confidence=0.3
    ) as pose:
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Process the image
        try:
            results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if results.pose_landmarks:
                logger.debug("Pose landmarks detected successfully")
                return results, image_np.shape[:2]
            else:
                # Try preprocessing the image
                logger.debug("Initial detection failed, trying preprocessing")
                processed_image = preprocess_image(image_np)
                results = pose.process(cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

                if results.pose_landmarks:
                    logger.debug("Pose landmarks detected after preprocessing")
                    return results, processed_image.shape[:2]
                else:
                    raise ValueError("No human pose detected in the image after preprocessing")

        except Exception as e:
            logger.error(f"Error in pose detection: {str(e)}")
            raise

def preprocess_image(image):
    """
    Preprocess the image to improve pose detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to RGB
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return rgb

def create_enhanced_stick_figure(results, image_shape, thickness_multiplier=2.0):
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
        thickness=int(3 * thickness_multiplier),
        circle_radius=int(3 * thickness_multiplier)
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