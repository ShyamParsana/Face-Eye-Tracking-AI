import math
import numpy as np
import cv2

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: Tuple (x, y) or a numpy array of shape (2,)
        point2: Tuple (x, y) or a numpy array of shape (2,)
        
    Returns:
        float: Euclidean distance
    """
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def calculate_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect blinks.
    
    Args:
        eye_landmarks: List of tuples (x, y) for eye landmarks.
                       Typically 6 points: P1 to P6.
                       P1 and P4 are horizontal extremes.
                       P2, P3, P5, P6 are vertical extremes.
                       
    Returns:
        float: Computed EAR value.
    """
    # Vertical distances
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    
    # Check for division by zero
    if h == 0:
        return 0.0
        
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_head_pose(shape, image_width, image_height):
    """
    Estimate head pose given specific facial landmarks.
    
    Args:
        shape: Array of (x, y) landmarks from MediaPipe
        image_width: Width of the frame
        image_height: Height of the frame
        
    Returns:
        tuple: (pitch, yaw, roll) angles
    """
    # Select key landmarks (e.g., nose tip, chin, eye corners, mouth corners)
    # MediaPipe Face Mesh indices:
    # 1: Nose tip
    # 152: Chin
    # 33: Left eye left corner
    # 263: Right eye right corner
    # 61: Left mouth corner
    # 291: Right mouth corner
    
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
    
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0, 0, 0
        
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Calculate Euler angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
    
    # Return pitch, yaw, roll (x, y, z rotations)
    return angles[0], angles[1], angles[2]
