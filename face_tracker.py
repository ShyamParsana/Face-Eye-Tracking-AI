import cv2
import mediapipe as mp
import numpy as np
from utils import get_head_pose
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MockResults:
    def __init__(self, detection_result):
        if detection_result.face_landmarks:
            self.multi_face_landmarks = []
            for face in detection_result.face_landmarks:
                class FaceLandmarks:
                    pass
                fl = FaceLandmarks()
                fl.landmark = face
                self.multi_face_landmarks.append(fl)
        else:
            self.multi_face_landmarks = None

class FaceTracker:
    def __init__(self, logger):
        """
        Initialize the Face Tracker.
        
        Args:
            logger: EventLogger instance to record events.
        """
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.logger = logger
        
        # Counts
        self.counts = {
            "Right": 0,
            "Left": 0,
            "Up": 0,
            "Down": 0
        }
        
        # State machine variables
        self.current_state = "Neutral"
        self.neutral_threshold_pitch = 10  # degrees
        self.neutral_threshold_yaw = 10    # degrees
        
        # Confidence score (dummy calculation based on detection)
        self.confidence_score = 0.0
        self.is_counting = False
        
    def reset_counts(self):
        """Reset all face movement counts."""
        for key in self.counts:
            self.counts[key] = 0
            
    def process_frame(self, frame):
        """
        Process the frame to detect face and head pose.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            tuple: (processed_frame, results, face_direction_string)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        # Mock old results object for compatibility with other components
        results = MockResults(detection_result)
        
        direction = "Neutral"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate confidence score based on presence of landmarks
                self.confidence_score = 95.0 + (np.random.random() * 4.9) # Simulated high confidence
                
                # Get image dimensions
                h, w, _ = frame.shape
                
                # Convert landmarks to pixel coordinates
                shape = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                
                # Draw bounding box
                x_coords = [p[0] for p in shape]
                y_coords = [p[1] for p in shape]
                cv2.rectangle(frame, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), (0, 255, 0), 2)
                
                # Get head pose angles
                raw_pitch, raw_yaw, roll = get_head_pose(shape, w, h)
                
                # Apply Exponential Moving Average (EMA) smoothing to reduce jitter
                alpha = 0.3  # Smoothing factor (lower = smoother but slower, higher = faster but jittery)
                
                if not hasattr(self, 'smoothed_pitch'):
                    self.smoothed_pitch = raw_pitch
                    self.smoothed_yaw = raw_yaw
                else:
                    self.smoothed_pitch = (alpha * raw_pitch) + ((1 - alpha) * self.smoothed_pitch)
                    self.smoothed_yaw = (alpha * raw_yaw) + ((1 - alpha) * self.smoothed_yaw)
                
                # Determine face direction based on smoothed angles
                direction = self._determine_direction(self.smoothed_pitch, self.smoothed_yaw)
                
                # Draw debug text
                cv2.putText(frame, f"Pitch: {self.smoothed_pitch:.1f} Yaw: {self.smoothed_yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        else:
            self.confidence_score = 0.0
            
        return frame, results, direction
        
    def _determine_direction(self, pitch, yaw):
        """Determine head direction and update state machine."""
        
        # Initialize state variables
        if not hasattr(self, 'last_direction'):
            self.last_direction = "Neutral"
            self.direction_frames = 0
            
        direction = "Neutral"
        
        # Hysteresis thresholds
        yaw_thresh = 15
        yaw_return = 8
        pitch_thresh = 12
        pitch_return = 6
        
        # Check if we maintain the current state (hysteresis)
        if self.current_state == "Right" and yaw > yaw_return:
            direction = "Right"
        elif self.current_state == "Left" and yaw < -yaw_return:
            direction = "Left"
        elif self.current_state == "Up" and pitch > pitch_return:
            direction = "Up"
        elif self.current_state == "Down" and pitch < -pitch_return:
            direction = "Down"
        else:
            # Check entry thresholds
            if yaw > yaw_thresh:
                direction = "Right"
            elif yaw < -yaw_thresh:
                direction = "Left"
            elif pitch > pitch_thresh:
                direction = "Up"
            elif pitch < -pitch_thresh:
                direction = "Down"
                
        # Debounce logic to prevent fast counting from noise
        if direction == self.last_direction:
            self.direction_frames += 1
        else:
            self.last_direction = direction
            self.direction_frames = 0
            
        if self.direction_frames >= 5:
            stable_direction = direction
        else:
            stable_direction = self.current_state # keep current state until stable
            
        # State machine logic to avoid duplicate counts
        if stable_direction == "Neutral":
            self.current_state = "Neutral"
        elif stable_direction != "Neutral" and self.current_state != stable_direction:
            # Just entered a new state
            self.current_state = stable_direction
            if self.is_counting:
                self.counts[stable_direction] += 1
                self.logger.log_event(f"Face {stable_direction}", self.counts[stable_direction])
            
        return self.current_state
