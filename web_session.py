import asyncio
import cv2
import json
import numpy as np
import os
import struct
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from face_tracker import FaceTracker
from eye_tracker import EyeTracker
from logger import EventLogger

class SessionState:
    def __init__(self, session_id: str):
        """
        Initialize isolated tracking session for a web client.
        Preserves original FaceTracker, EyeTracker, and EventLogger instances.
        """
        self.session_id = session_id
        self.logger = EventLogger()
        self.face_tracker = FaceTracker(self.logger)
        self.eye_tracker = EyeTracker(self.logger)
        
        self.created_at = time.time()
        self.last_active = time.time()
        self.last_frame_time = time.time()
        self.fps = 30.0
        
        # Async frame queue: size=1 to guarantee zero lag by dropping stale frames
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        
        # Recording & Assets
        self.assets_dir = "assets"
        os.makedirs(self.assets_dir, exist_ok=True)
        self.is_recording = False
        self.video_writer = None
        self.record_filename = ""
        self.record_width = 640
        self.record_height = 480
        
        # Activity graph history (max 50 points, matching GUI)
        self.graph_data_face = []
        self.graph_data_eye = []
        
    def start_counting(self):
        """Enable movement counting."""
        self.face_tracker.is_counting = True
        self.eye_tracker.is_counting = True
        
    def stop_counting(self):
        """Disable movement counting."""
        self.face_tracker.is_counting = False
        self.eye_tracker.is_counting = False
        
    def reset_counts(self):
        """Reset all face and eye counts."""
        self.face_tracker.reset_counts()
        self.eye_tracker.reset_counts()
        self.graph_data_face.clear()
        self.graph_data_eye.clear()
        
    def clear_data(self):
        """Clear event logger data."""
        self.logger.clear_data()
        
    def take_screenshot(self, frame_bgr: np.ndarray) -> str:
        """Save a screenshot of the current frame to assets directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.assets_dir, f"screenshot_{timestamp}_{self.session_id[:6]}.png")
        cv2.imwrite(filename, frame_bgr)
        return filename
        
    def toggle_recording(self, width: int = 640, height: int = 480, fps: float = 30.0) -> Tuple[bool, str]:
        """Toggle video recording on/off."""
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            return False, self.record_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_filename = os.path.join(self.assets_dir, f"record_{timestamp}_{self.session_id[:6]}.mp4")
            self.record_width = width
            self.record_height = height
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.record_filename, fourcc, fps, (width, height))
            self.is_recording = True
            return True, self.record_filename
            
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a BGR frame through the preserved Python CV pipeline.
        
        Args:
            frame_bgr: Raw input BGR frame from webcam.
            
        Returns:
            tuple: (annotated_bgr_frame, telemetry_dict)
        """
        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0:
            current_instant_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_instant_fps
        self.last_frame_time = now
        self.last_active = now
        
        # Mirror frame horizontally (matches original CameraManager behavior)
        frame = cv2.flip(frame_bgr, 1)
        
        face_dir = "Neutral"
        eye_dir = "Center"
        
        try:
            # 1. Face Tracker (MediaPipe, Bounding box, Head Pose, Pitch/Yaw drawing)
            frame, face_results, face_dir = self.face_tracker.process_frame(frame)
            
            # 2. Eye Tracker (EAR, Blink detection, Iris tracking, Landmark drawing)
            eye_dir = self.eye_tracker.process_eyes(frame, face_results, face_dir)
        except Exception as cv_err:
            import traceback
            traceback.print_exc()
            cv2.putText(frame, "CV Processing Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 3. Video Writer if recording
        if self.is_recording and self.video_writer is not None:
            # Resize if needed
            h, w = frame.shape[:2]
            if (w, h) != (self.record_width, self.record_height):
                rec_frame = cv2.resize(frame, (self.record_width, self.record_height))
                self.video_writer.write(rec_frame)
            else:
                self.video_writer.write(frame)
                
        # 4. Activity Graph Tracking (matching GUI update logic)
        total_face = sum(self.face_tracker.counts.values())
        total_eye = sum(self.eye_tracker.counts.values())
        self.graph_data_face.append(total_face)
        self.graph_data_eye.append(total_eye)
        if len(self.graph_data_face) > 50:
            self.graph_data_face.pop(0)
            self.graph_data_eye.pop(0)
            
        session_duration = int(now - self.created_at)
        mins, secs = divmod(session_duration, 60)
        session_time_str = f"{mins:02d}:{secs:02d}"
        
        telemetry = {
            "fps": int(round(self.fps)),
            "session_time": session_time_str,
            "session_seconds": session_duration,
            "face_dir": face_dir,
            "eye_dir": eye_dir,
            "confidence": round(float(self.face_tracker.confidence_score), 1),
            "pitch": round(float(getattr(self.face_tracker, 'smoothed_pitch', 0.0)), 1),
            "yaw": round(float(getattr(self.face_tracker, 'smoothed_yaw', 0.0)), 1),
            "counts": {
                "Right Face Count": self.face_tracker.counts.get("Right", 0),
                "Left Face Count": self.face_tracker.counts.get("Left", 0),
                "Up Count": self.face_tracker.counts.get("Up", 0),
                "Down Count": self.face_tracker.counts.get("Down", 0),
                "Left Blink Count": self.eye_tracker.counts.get("Left Blink", 0),
                "Right Blink Count": self.eye_tracker.counts.get("Right Blink", 0),
                "Both Blink Count": self.eye_tracker.counts.get("Both Blink", 0),
                "Eye Left Count": self.eye_tracker.counts.get("Eye Left", 0),
                "Eye Right Count": self.eye_tracker.counts.get("Eye Right", 0),
                "Eye Up Count": self.eye_tracker.counts.get("Eye Up", 0),
                "Eye Down Count": self.eye_tracker.counts.get("Eye Down", 0),
            },
            "is_counting": self.face_tracker.is_counting,
            "is_recording": self.is_recording,
            "graph": {
                "face": list(self.graph_data_face),
                "eye": list(self.graph_data_eye)
            },
            "logs": self.logger.get_logs()[-100:]
        }
        
        return frame, telemetry

    def process_binary_packet(self, raw_bytes: bytes) -> bytes:
        """
        Processes an incoming binary image frame, runs the pipeline,
        and constructs an ultra-fast zero-Base64 binary response packet:
        [ 4-byte uint32 JSON length L ] [ L-byte UTF-8 JSON ] [ Raw JPEG bytes ]
        """
        # Decode image from buffer
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return b""
            
        annotated_frame, telemetry = self.process_frame(frame_bgr)
        
        # Encode annotated frame to JPEG (High quality 85 for sharp landmarks)
        ret, jpeg_buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            return b""
            
        jpeg_bytes = jpeg_buf.tobytes()
        json_bytes = json.dumps(telemetry).encode('utf-8')
        header = struct.pack("!I", len(json_bytes))
        
        # Packed binary: Header (4 bytes) + JSON telemetry + JPEG Image
        return header + json_bytes + jpeg_bytes

    def cleanup(self):
        """Release session resources."""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        
    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        return self.sessions[session_id]
        
    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            session.cleanup()
            
    def cleanup_inactive(self, max_idle_seconds: float = 900.0):
        """Clean up sessions idle for longer than max_idle_seconds (default 15 mins)."""
        now = time.time()
        to_delete = [
            sid for sid, s in self.sessions.items()
            if (now - s.last_active) > max_idle_seconds
        ]
        for sid in to_delete:
            self.remove_session(sid)

session_manager = SessionManager()
