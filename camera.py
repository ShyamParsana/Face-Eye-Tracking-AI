import cv2
import os
from datetime import datetime

class CameraManager:
    def __init__(self):
        """Initialize the Camera Manager."""
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        self.record_filename = ""
        self.assets_dir = "assets"
        
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
            
    def start_camera(self):
        """
        Detects and starts the best available camera.
        Prioritizes external cameras (index > 0) if available.
        """
        # Try finding external cameras first (usually index 1, 2, ...)
        for index in range(1, 3):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    print(f"Started external camera at index {index}")
                    return True
                cap.release()
                
        # Fallback to default camera (index 0)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                self.cap = cap
                print("Started default camera at index 0")
                return True
            cap.release()
            
        print("Error: No available cameras found.")
        return False
        
    def read_frame(self):
        """Read a frame from the camera."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame for selfie view
                frame = cv2.flip(frame, 1)
                
                # Write to video file if recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                    
                return True, frame
        return False, None
        
    def get_fps(self):
        """Get the camera's FPS (if available, else default to 30)."""
        if self.cap is not None and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30
        return 30
        
    def get_resolution(self):
        """Get camera resolution."""
        if self.cap is not None and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 640, 480
        
    def take_screenshot(self, frame):
        """Save a screenshot of the current frame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.assets_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, frame)
        return filename
        
    def toggle_recording(self):
        """Toggle video recording on/off."""
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            return False, self.record_filename
        else:
            # Start recording
            if self.cap is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.record_filename = os.path.join(self.assets_dir, f"record_{timestamp}.mp4")
                width, height = self.get_resolution()
                fps = self.get_fps()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.record_filename, fourcc, fps, (width, height))
                self.is_recording = True
                return True, self.record_filename
        return False, ""
        
    def release(self):
        """Release camera and video writer resources."""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
