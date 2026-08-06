import asyncio
import cv2
import json
import numpy as np
import os
import struct
import sys

from web_session import SessionManager, SessionState
from logger import EventLogger

def test_session_and_pipeline():
    print("[1/3] Testing Session State & CV Pipeline...")
    session = SessionState("test_session_123")
    
    # Create synthetic test image (640x480 black image with synthetic facial area)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_img, (320, 240), 100, (200, 200, 200), -1)
    
    ret, jpeg_buf = cv2.imencode('.jpg', test_img)
    assert ret, "JPEG encoding failed"
    raw_bytes = jpeg_buf.tobytes()
    
    # Process binary packet
    packet = session.process_binary_packet(raw_bytes)
    assert len(packet) > 4, "Packet too short"
    
    json_len = struct.unpack("!I", packet[:4])[0]
    json_bytes = packet[4:4+json_len]
    telemetry = json.loads(json_bytes.decode('utf-8'))
    
    print(f"   [OK] Telemetry decoded: FPS={telemetry['fps']}, FaceDir={telemetry['face_dir']}, EyeDir={telemetry['eye_dir']}")
    assert "counts" in telemetry, "Missing counts in telemetry"
    assert "graph" in telemetry, "Missing graph in telemetry"
    
    # Check JPEG portion
    img_bytes = packet[4+json_len:]
    assert len(img_bytes) > 0, "No JPEG bytes in packet"
    decoded_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_img is not None, "Failed to decode annotated JPEG from packet"
    print(f"   [OK] Annotated Image shape: {decoded_img.shape}")
    print("   [OK] Binary packet structure verified (4-byte length + UTF-8 JSON + JPEG bytes).")

def test_logging_and_exports():
    print("[2/3] Testing EventLogger and CSV/Excel formatting...")
    logger = EventLogger()
    logger.log_event("Face Right", 1)
    logger.log_event("Face Left", 1)
    logger.log_event("Eye Right", 1)
    logger.log_event("Both Blink", 1)
    
    df = logger.format_export_data()
    assert not df.empty, "Export DataFrame is empty"
    print(f"   [OK] Export columns: {list(df.columns)}")
    assert "Time" in df.columns, "Missing Time column"
    assert "Face Event" in df.columns, "Missing Face Event column"
    assert "Blink" in df.columns, "Missing Blink column"
    print("   [OK] Export data format conforms to multi-table specification.")

def test_fastapi_app():
    print("[3/3] Testing FastAPI routes and templates...")
    from server import app
    routes = [route.path for route in app.routes]
    print(f"   [OK] Registered routes: {routes}")
    assert "/" in routes, "Missing / route"
    assert "/demo" in routes, "Missing /demo route"
    assert "/health" in routes, "Missing /health route"
    assert "/offer" in routes, "Missing /offer route"
    assert "/ws/track/{session_id}" in routes, "Missing WebSocket route"
    assert "/api/session/{session_id}/action" in routes, "Missing action route"
    assert "/api/session/{session_id}/export/csv" in routes, "Missing CSV route"
    assert "/api/session/{session_id}/export/excel" in routes, "Missing Excel route"
    print("   [OK] All FastAPI routes successfully verified.")

if __name__ == "__main__":
    print("--- Starting System Verification ---")
    test_session_and_pipeline()
    test_logging_and_exports()
    test_fastapi_app()
    print("--- ALL TESTS PASSED SUCCESSFULLY! ---")
