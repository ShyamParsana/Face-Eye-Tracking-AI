from camera import CameraManager
from face_tracker import FaceTracker
from eye_tracker import EyeTracker
from logger import EventLogger
from gui import AppGUI

def main():
    # Initialize Core Modules
    logger = EventLogger()
    camera_mgr = CameraManager()
    
    # Start Camera
    if not camera_mgr.start_camera():
        print("Failed to start camera. Exiting...")
        return
        
    face_tracker = FaceTracker(logger)
    eye_tracker = EyeTracker(logger)
    
    # Initialize and run GUI
    app = AppGUI(camera_mgr, face_tracker, eye_tracker, logger)
    
    # Register protocol for safe closing
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    app.mainloop()
    
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
