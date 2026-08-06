import cv2
import numpy as np
from utils import calculate_ear, euclidean_distance

class EyeTracker:
    def __init__(self, logger):
        """
        Initialize the Eye Tracker.
        
        Args:
            logger: EventLogger instance to record events.
        """
        self.logger = logger
        self.is_counting = False
        
        # Counts
        self.counts = {
            "Left Blink": 0,
            "Right Blink": 0,
            "Both Blink": 0,
            "Eye Left": 0,
            "Eye Right": 0,
            "Eye Up": 0,
            "Eye Down": 0
        }
        
        # Blink state variables
        self.blink_threshold = 0.22
        self.blink_frames = 2  # Must be below threshold for 2 consecutive frames to count
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        
        self.left_blinked = False
        self.right_blinked = False
        
        # Gaze state variables
        self.current_gaze = "Center"
        
        # MediaPipe Landmark Indices
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380
        self.LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        
        # Iris Landmarks (requires refine_landmarks=True)
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]
        
        # Dynamic Calibration Baselines
        self.baseline_dx = 0.0
        self.baseline_dy = 0.0
        self.has_calibrated = False
        
    def reset_counts(self):
        """Reset all eye movement and blink counts."""
        for key in self.counts:
            self.counts[key] = 0
            
    def process_eyes(self, frame, results, face_dir="Neutral"):
        """
        Process the eye landmarks for blinks and gaze.
        
        Args:
            frame: Frame to draw on (optional)
            results: MediaPipe face mesh results
            face_dir: Current face direction string
            
        Returns:
            str: Current gaze direction.
        """
        gaze_direction = "Center"
        
        if not results.multi_face_landmarks:
            return gaze_direction
            
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            shape = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            
            # --- Blink Detection ---
            left_eye = [shape[i] for i in self.LEFT_EYE_IDXS]
            right_eye = [shape[i] for i in self.RIGHT_EYE_IDXS]
            
            raw_left_ear = calculate_ear(left_eye)
            raw_right_ear = calculate_ear(right_eye)
            
            # EMA for EAR (Eye Aspect Ratio)
            alpha_ear = 0.4
            if not hasattr(self, 'smoothed_left_ear'):
                self.smoothed_left_ear = raw_left_ear
                self.smoothed_right_ear = raw_right_ear
            else:
                self.smoothed_left_ear = (alpha_ear * raw_left_ear) + ((1 - alpha_ear) * self.smoothed_left_ear)
                self.smoothed_right_ear = (alpha_ear * raw_right_ear) + ((1 - alpha_ear) * self.smoothed_right_ear)
            
            self._handle_blinks(self.smoothed_left_ear, self.smoothed_right_ear)
            
            # --- Gaze Detection (Cartesian Logic) ---
            def get_eye_features(eye_idxs, iris_idxs):
                eye_pts = [shape[i] for i in eye_idxs if i < len(shape)]
                if not eye_pts:
                    return (0, 0), 0.0, 0.0
                    
                valid_iris = [shape[i] for i in iris_idxs if i < len(shape)]
                if valid_iris:
                    iris_pt = np.mean(valid_iris, axis=0).astype(int)
                else:
                    iris_pt = np.mean(eye_pts, axis=0).astype(int)

                xs = [pt[0] for pt in eye_pts]
                ys = [pt[1] for pt in eye_pts]
                
                left_x, right_x = min(xs), max(xs)
                top_y, bottom_y = min(ys), max(ys)
                
                eye_center_x = (left_x + right_x) / 2.0
                eye_center_y = (top_y + bottom_y) / 2.0
                
                eye_w = right_x - left_x
                eye_h = bottom_y - top_y
                
                # User's logic: 
                # Right = +x (iris moves to smaller image X) => center_x - iris_x
                # Up = +y (iris moves to smaller image Y) => center_y - iris_y
                dx = eye_center_x - iris_pt[0]
                dy = eye_center_y - iris_pt[1]
                
                # Normalize by half-width/height so max range is roughly -1 to 1
                if eye_w > 0 and eye_h > 0:
                    norm_dx = dx / (eye_w / 2.0)
                    norm_dy = dy / (eye_h / 2.0)
                else:
                    norm_dx, norm_dy = 0.0, 0.0
                    
                return iris_pt, norm_dx, norm_dy

            l_iris, l_dx, l_dy = get_eye_features(self.LEFT_EYE_IDXS, self.LEFT_IRIS)
            r_iris, r_dx, r_dy = get_eye_features(self.RIGHT_EYE_IDXS, self.RIGHT_IRIS)
            
            if frame is not None:
                # Draw Blue dots on the pupil/iris center
                if l_iris[0] > 0 and l_iris[1] > 0:
                    cv2.circle(frame, tuple(l_iris), 3, (255, 0, 0), -1)
                if r_iris[0] > 0 and r_iris[1] > 0:
                    cv2.circle(frame, tuple(r_iris), 3, (255, 0, 0), -1)
                
            # Average the dx and dy for both eyes to cancel out resting asymmetry
            raw_avg_dx = (l_dx + r_dx) / 2.0
            raw_avg_dy = (l_dy + r_dy) / 2.0
            
            # EMA for Gaze Tracking
            alpha_gaze = 0.3
            if not hasattr(self, 'smoothed_dx'):
                self.smoothed_dx = raw_avg_dx
                self.smoothed_dy = raw_avg_dy
            else:
                self.smoothed_dx = (alpha_gaze * raw_avg_dx) + ((1 - alpha_gaze) * self.smoothed_dx)
                self.smoothed_dy = (alpha_gaze * raw_avg_dy) + ((1 - alpha_gaze) * self.smoothed_dy)
            
            # Dynamic Baseline Tracking
            # When the user's face is Neutral, we assume they are generally looking forward.
            # We slowly update our "zero" center to adapt to their specific eye shape.
            if face_dir == "Neutral":
                baseline_alpha = 0.02 # Extremely slow adaptation
                if not self.has_calibrated:
                    self.baseline_dx = self.smoothed_dx
                    self.baseline_dy = self.smoothed_dy
                    self.has_calibrated = True
                else:
                    self.baseline_dx = (baseline_alpha * self.smoothed_dx) + ((1 - baseline_alpha) * self.baseline_dx)
                    self.baseline_dy = (baseline_alpha * self.smoothed_dy) + ((1 - baseline_alpha) * self.baseline_dy)
                    
            # Subtract the dynamic baseline to get perfectly centered coordinates
            adjusted_dx = self.smoothed_dx - self.baseline_dx
            adjusted_dy = self.smoothed_dy - self.baseline_dy
            
            gaze_direction = self._determine_gaze(adjusted_dx, adjusted_dy, face_dir)
            
            if frame is not None:
                # Draw yellow eye landmarks
                for pt in left_eye + right_eye:
                    cv2.circle(frame, pt, 1, (0, 255, 255), -1)
                # Debug info
                debug_text = f"X: {adjusted_dx:.2f} Y: {adjusted_dy:.2f}"
                cv2.putText(frame, debug_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
        return gaze_direction
        
    def _handle_blinks(self, left_ear, right_ear):
        """Handle blink logic with noise filtering."""
        
        # Left Eye
        if left_ear < self.blink_threshold:
            self.left_closed_frames += 1
        else:
            if self.left_closed_frames >= self.blink_frames and not self.left_blinked:
                self.left_blinked = True
            self.left_closed_frames = 0
            
        # Right Eye
        if right_ear < self.blink_threshold:
            self.right_closed_frames += 1
        else:
            if self.right_closed_frames >= self.blink_frames and not self.right_blinked:
                self.right_blinked = True
            self.right_closed_frames = 0
            
        # Logging
        if self.left_blinked and self.right_blinked:
            if self.is_counting:
                self.counts["Both Blink"] += 1
                self.logger.log_event("Both Blink", self.counts["Both Blink"])
            
            # Reset blink states
            self.left_blinked = False
            self.right_blinked = False
        else:
            if self.left_blinked:
                if self.is_counting:
                    self.counts["Left Blink"] += 1
                    self.logger.log_event("Left Blink", self.counts["Left Blink"])
                self.left_blinked = False
            elif self.right_blinked:
                if self.is_counting:
                    self.counts["Right Blink"] += 1
                    self.logger.log_event("Right Blink", self.counts["Right Blink"])
                self.right_blinked = False
            
    def left_ear_open(self, ear):
        return ear >= self.blink_threshold
        
    def right_ear_open(self, ear):
        return ear >= self.blink_threshold
        
    def _determine_gaze(self, dx, dy, face_dir):
        """
        Determine gaze direction based on user's Cartesian coordinate logic.
        +x = Right, -x = Left
        +y = Up, -y = Down
        """
        new_gaze = "Center"
        
        # If head is turned, 2D gaze is highly distorted. Ignore eye movements.
        if face_dir != "Neutral":
            pass # Keep new_gaze as "Center"
        else:
            # Thresholds for triggering the movement.
            # Adjusted to balance sensitivity
            thresh_left = 0.08
            thresh_right = 0.08
            thresh_up = 0.10
            thresh_down = 0.08
            
            # Hysteresis return thresholds (slightly lower so it doesn't flicker)
            ret_left = 0.05
            ret_right = 0.05
            ret_up = 0.07
            ret_down = 0.05
            
            # First, handle maintaining current state (Hysteresis)
            if self.current_gaze == "Left" and dx > ret_left:
                new_gaze = "Left"
            elif self.current_gaze == "Right" and dx < -ret_right:
                new_gaze = "Right"
            elif self.current_gaze == "Up" and dy > ret_up:
                new_gaze = "Up"
            elif self.current_gaze == "Down" and dy < -ret_down:
                new_gaze = "Down"
            else:
                # Calculate how far each axis has moved relative to its own threshold
                # dx > 0 is Left (swapped as requested), dx < 0 is Right
                ratio_left = dx / thresh_left if dx > 0 else 0
                ratio_right = -dx / thresh_right if dx < 0 else 0
                ratio_up = dy / thresh_up if dy > 0 else 0
                ratio_down = -dy / thresh_down if dy < 0 else 0
                
                max_ratio = max(ratio_left, ratio_right, ratio_up, ratio_down)
                
                # If any movement exceeds its threshold (ratio > 1.0)
                if max_ratio > 1.0:
                    if max_ratio == ratio_left:
                        new_gaze = "Left"
                    elif max_ratio == ratio_right:
                        new_gaze = "Right"
                    elif max_ratio == ratio_up:
                        new_gaze = "Up"
                    elif max_ratio == ratio_down:
                        new_gaze = "Down"
            
        # Debounce logic
        if not hasattr(self, 'last_gaze'):
            self.last_gaze = "Center"
            self.gaze_frames = 0
            
        if new_gaze == self.last_gaze:
            self.gaze_frames += 1
        else:
            self.last_gaze = new_gaze
            self.gaze_frames = 0
            
        # Reduced from 5 to 3 frames to make it more responsive
        if self.gaze_frames >= 3:
            stable_gaze = new_gaze
        else:
            stable_gaze = self.current_gaze
            
        # State machine to only log on change
        if stable_gaze == "Center":
            self.current_gaze = "Center"
        elif stable_gaze != "Center" and self.current_gaze != stable_gaze:
            self.current_gaze = stable_gaze
            event_name = f"Eye {stable_gaze}"
            if self.is_counting:
                self.counts[event_name] += 1
                self.logger.log_event(event_name, self.counts[event_name])
                
        return self.current_gaze
