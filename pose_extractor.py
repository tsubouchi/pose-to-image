import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, Tuple

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_pose(pil_image) -> Tuple[Image.Image, Dict[str, str], any]:
    """
    Extract pose from image with improved error handling and detection
    """
    try:
        # Convert PIL Image to numpy array
        image_np = np.array(pil_image)

        # Initialize MediaPipe Pose with improved settings
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Increased from 1 to 2 for better accuracy
            min_detection_confidence=0.2,  # Lowered from 0.3 for better detection
            min_tracking_confidence=0.2,  # Added for improved tracking
            enable_segmentation=True  # Enable segmentation for better pose isolation
        ) as pose:
            # Process image with improved preprocessing
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Enhance image contrast for better detection
            enhanced_image = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=0)

            # Process enhanced image
            results = pose.process(enhanced_image)

            if not results.pose_landmarks:
                logger.warning("No pose landmarks detected, trying with different settings")
                # Second attempt with different settings
                with mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.1,
                    enable_segmentation=True
                ) as pose2:
                    results = pose2.process(image_rgb)

                    if not results.pose_landmarks:
                        logger.error("Failed to detect pose after multiple attempts")
                        return None, get_default_pose_descriptions(), None

            # Create visualization canvas
            canvas = np.zeros(image_np.shape, dtype=np.uint8)

            # Enhanced drawing settings
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                canvas,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(50, 205, 50),
                    thickness=4,
                    circle_radius=4
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(30, 144, 255),
                    thickness=2
                )
            )

            # Calculate angles for pose description
            angles = calculate_joint_angles(results.pose_landmarks)
            pose_descriptions = get_pose_description(angles)

            return Image.fromarray(canvas), pose_descriptions, results

    except Exception as e:
        logger.error(f"Error in pose extraction: {str(e)}")
        return None, get_default_pose_descriptions(), None

def get_default_pose_descriptions() -> Dict[str, str]:
    """
    Return default pose descriptions for fallback
    """
    return {
        "right_shoulder_desc": "neutral position",
        "right_elbow_desc": "slightly bent",
        "left_shoulder_desc": "neutral position",
        "left_elbow_desc": "slightly bent",
        "right_hip_desc": "neutral position",
        "right_knee_desc": "slightly bent",
        "left_hip_desc": "neutral position",
        "left_knee_desc": "slightly bent",
        "spine_desc": "spine aligned naturally"
    }

def create_basic_stick_figure(image_shape) -> np.ndarray:
    """
    Create a basic stick figure when pose detection fails
    """
    height, width = image_shape[:2] if len(image_shape) > 2 else image_shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Define basic stick figure points
    center_x = width // 2
    points = {
        'head': (center_x, height // 4),
        'shoulder': (center_x, height // 3),
        'hip': (center_x, height // 2),
        'knee': (center_x, 3 * height // 4),
        'ankle': (center_x, 7 * height // 8)
    }

    # Draw basic stick figure
    for start, end in [('head', 'shoulder'), ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]:
        cv2.line(canvas, points[start], points[end], (50, 205, 50), 2)

    return canvas

def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate the angle between three points in degrees
    """
    try:
        vector1 = point1 - point2
        vector2 = point3 - point2

        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)
    except Exception as e:
        logger.error(f"Error calculating angle: {str(e)}")
        return 90.0  # Return default angle

def calculate_joint_angles(landmarks) -> Dict[str, float]:
    """
    Calculate all relevant joint angles from pose landmarks
    """
    try:
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

            # Spine angle
            "spine": calculate_angle(
                points[mp.solutions.pose.PoseLandmark.NOSE.value],
                points[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            )
        }

        return angles
    except Exception as e:
        logger.error(f"Error calculating joint angles: {str(e)}")
        # Return default angles
        return {
            "right_shoulder": 90.0,
            "right_elbow": 90.0,
            "left_shoulder": 90.0,
            "left_elbow": 90.0,
            "right_hip": 90.0,
            "right_knee": 90.0,
            "left_hip": 90.0,
            "left_knee": 90.0,
            "spine": 90.0
        }

def get_pose_description(angles: Dict[str, float]) -> Dict[str, str]:
    """
    Convert numerical angles to natural language descriptions
    """
    try:
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
    except Exception as e:
        logger.error(f"Error creating pose descriptions: {str(e)}")
        return get_default_pose_descriptions()

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


def analyze_pose_balance(landmarks) -> Dict[str, float]:
    """
    Analyze pose balance and symmetry
    """
    try:
        # Convert landmarks to numpy arrays
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = np.array([landmark.x, landmark.y, landmark.z])

        # Calculate symmetry scores
        symmetry_scores = {
            "shoulders": calculate_symmetry(
                points[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                points[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            ),
            "elbows": calculate_symmetry(
                points[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                points[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            ),
            "hips": calculate_symmetry(
                points[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                points[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            ),
            "knees": calculate_symmetry(
                points[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
                points[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            )
        }
        return symmetry_scores
    except Exception as e:
        logger.error(f"Error analyzing pose balance: {str(e)}")
        return {}

def calculate_symmetry(right_point: np.ndarray, left_point: np.ndarray) -> float:
    """
    Calculate symmetry score between two points
    """
    try:
        # Calculate distance from center
        center = (right_point + left_point) / 2
        right_dist = np.linalg.norm(right_point - center)
        left_dist = np.linalg.norm(left_point - center)

        # Calculate symmetry score (1.0 = perfect symmetry)
        max_dist = max(right_dist, left_dist)
        if max_dist == 0:
            return 1.0
        symmetry = 1.0 - abs(right_dist - left_dist) / max_dist
        return symmetry
    except Exception as e:
        logger.error(f"Error calculating symmetry: {str(e)}")
        return 1.0

def generate_pose_suggestions(symmetry_scores: Dict[str, float], angles: Dict[str, float]) -> Dict[str, str]:
    """
    Generate pose improvement suggestions based on analysis
    """
    suggestions = {}

    try:
        # Analyze symmetry
        for part, score in symmetry_scores.items():
            if score < 0.85:  # Less than 85% symmetry
                suggestions[f"{part}_symmetry"] = f"Consider adjusting {part} alignment for better balance"

        # Analyze joint angles
        if angles.get("spine", 90) < 70:
            suggestions["spine"] = "Consider straightening your spine for better posture"

        if angles.get("right_knee", 180) < 160 and angles.get("left_knee", 180) < 160:
            suggestions["knees"] = "Deep knee bend detected - ensure stable balance"

        if angles.get("right_elbow", 180) < 90 or angles.get("left_elbow", 180) < 90:
            suggestions["elbows"] = "Sharp elbow bend - check arm positioning"

        # Add general suggestions
        if len(suggestions) == 0:
            suggestions["general"] = "Pose looks well balanced! Consider experimenting with different expressions or hand positions."

        return suggestions
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return {"error": "Unable to generate pose suggestions"}

def get_pose_refinement_suggestions(landmarks) -> Dict[str, str]:
    """
    Main function to analyze pose and provide refinement suggestions
    """
    try:
        if landmarks is None:
            return {"error": "No pose detected"}

        # Get pose measurements
        angles = calculate_joint_angles(landmarks)
        symmetry_scores = analyze_pose_balance(landmarks)

        # Generate suggestions
        suggestions = generate_pose_suggestions(symmetry_scores, angles)

        return suggestions
    except Exception as e:
        logger.error(f"Error in pose refinement analysis: {str(e)}")
        return {"error": "Failed to analyze pose"}