import math
import numpy as np
import cv2

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    try:
        return float(math.hypot(point1[0] - point2[0], point1[1] - point2[1]))
    except Exception:
        return 0.0

def calculate_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect blinks.
    """
    try:
        if not eye_landmarks or len(eye_landmarks) < 6:
            return 0.0
            
        # Vertical distances
        v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
        v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
        
        # Check for division by zero or invalid numbers
        if h <= 1e-6 or math.isnan(h) or math.isinf(h):
            return 0.0
            
        # EAR formula
        ear = (v1 + v2) / (2.0 * h)
        if math.isnan(ear) or math.isinf(ear):
            return 0.0
        return float(ear)
    except Exception:
        return 0.0

def get_head_pose(shape, image_width, image_height):
    """
    Estimate head pose given specific facial landmarks.
    
    Args:
        shape: Array of (x, y) landmarks from MediaPipe
        image_width: Width of the frame
        image_height: Height of the frame
        
    Returns:
        tuple: (pitch, yaw, roll) angles as clean floats
    """
    try:
        if not shape or len(shape) < 292:
            return 0.0, 0.0, 0.0

        image_points = np.array([
            shape[1],     # Nose tip
            shape[152],   # Chin
            shape[33],    # Left eye left corner
            shape[263],   # Right eye right corner
            shape[61],    # Left mouth corner
            shape[291]    # Right mouth corner
        ], dtype="double")
        
        # Generic 3D model points (standard approximations)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, 330.0, -65.0),         # Chin
            (-225.0, -170.0, -135.0),    # Left eye left corner
            (225.0, -170.0, -135.0),     # Right eye right corner
            (-150.0, 150.0, -125.0),     # Left mouth corner
            (150.0, 150.0, -125.0)       # Right mouth corner
        ])
        
        focal_length = float(image_width)
        center = (float(image_width) / 2.0, float(image_height) / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0.0, center[0]],
             [0.0, focal_length, center[1]],
             [0.0, 0.0, 1.0]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # Calculate Euler angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
        
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])
        
        if math.isnan(pitch) or math.isinf(pitch):
            pitch = 0.0
        if math.isnan(yaw) or math.isinf(yaw):
            yaw = 0.0
        if math.isnan(roll) or math.isinf(roll):
            roll = 0.0
            
        return pitch, yaw, roll
    except Exception:
        return 0.0, 0.0, 0.0
