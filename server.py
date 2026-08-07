import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web_session import session_manager
from webrtc_manager import webrtc_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Background startup pre-warming and cleanup task for idle sessions."""
    # Pre-warm MediaPipe model in background thread so initial user frame has 0ms model load delay
    def prewarm_model():
        try:
            from face_tracker import FaceTracker
            from eye_tracker import EyeTracker
            from logger import EventLogger
            import numpy as np
            warmup_logger = EventLogger()
            ft = FaceTracker(warmup_logger)
            et = EyeTracker(warmup_logger)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            f, res, d = ft.process_frame(dummy)
            et.process_eyes(f, res, d)
            logger.info("MediaPipe Face & Eye Tracker models pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe pre-warming note: {e}")

    asyncio.get_event_loop().run_in_executor(None, prewarm_model)

    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            try:
                session_manager.cleanup_inactive(max_idle_seconds=900)
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")

    cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info("Face & Eye Tracking AI Web Server started.")
    yield
    cleanup_task.cancel()
    await webrtc_manager.close_all()
    logger.info("Server shutdown complete.")

app = FastAPI(
    title="Face & Eye Tracking AI API",
    description="Production-ready real-time Computer Vision web application and portfolio live demo.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("assets", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ActionRequest(BaseModel):
    action: str

class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    session_id: str

@app.get("/", response_class=HTMLResponse)
@app.head("/")
@app.get("/demo", response_class=HTMLResponse)
@app.head("/demo")
async def serve_demo():
    """Serve the interactive live demo dashboard directly."""
    demo_path = os.path.join("templates", "demo.html")
    if os.path.exists(demo_path):
        with open(demo_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Demo template not found.</h1>")

@app.get("/health")
@app.head("/health")
async def health_check():
    """Health check endpoint for container orchestrators and cloud platforms."""
    return {
        "status": "healthy",
        "service": "Face & Eye Movement Tracking System",
        "webrtc_available": webrtc_manager.is_available,
        "active_sessions": len(session_manager.sessions)
    }

@app.websocket("/ws/track/{session_id}")
async def websocket_tracking_endpoint(websocket: WebSocket, session_id: str):
    """
    High-throughput, zero-Base64 binary WebSocket streaming endpoint.
    Client sends binary JPEG image -> Server processes frame through preserved Python CV pipeline ->
    Server returns packed binary [4-byte JSON length | JSON telemetry | Annotated JPEG bytes].
    """
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)
    logger.info(f"WebSocket client connected: {session_id}")

    try:
        while True:
            # Receive binary frame or text command from browser
            message = await websocket.receive()
            msg_type = message.get("type")
            
            if msg_type == "websocket.disconnect":
                logger.info(f"WebSocket client closed cleanly: {session_id}")
                break
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Process binary frame offloaded to worker thread to avoid blocking event loop
                response_packet = await asyncio.to_thread(session.process_binary_packet, raw_bytes)
                if response_packet:
                    await websocket.send_bytes(response_packet)
                    
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    cmd_type = payload.get("type")
                    if cmd_type == "action":
                        action = payload.get("action")
                        if action == "start_counting":
                            session.start_counting()
                        elif action == "stop_counting":
                            session.stop_counting()
                        elif action == "reset_counts":
                            session.reset_counts()
                        elif action == "clear_data":
                            session.clear_data()
                        elif action == "toggle_recording":
                            is_rec, fname = session.toggle_recording()
                            await websocket.send_text(json.dumps({
                                "type": "recording_state",
                                "is_recording": is_rec,
                                "filename": fname
                            }))
                    elif cmd_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "time": payload.get("time")}))
                except Exception as e:
                    logger.debug(f"WS text handle error: {e}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.warning(f"WebSocket notice for session {session_id}: {e}")
    finally:
        # Keep session alive for reconnection / HTTP fallback; background task handles cleanup
        session.last_active = time.time() if 'time' in globals() else datetime.now().timestamp()

@app.post("/api/session/{session_id}/frame")
async def process_frame_http(session_id: str, request: Request):
    """
    HTTP POST Fallback Endpoint for streaming frames when WebSockets are blocked by client firewalls/proxies.
    Accepts raw JPEG bytes and returns packed binary packet.
    """
    try:
        session = session_manager.get_or_create_session(session_id)
        raw_bytes = await request.body()
        if not raw_bytes or len(raw_bytes) < 10:
            raise HTTPException(status_code=400, detail="Invalid frame body")
            
        response_packet = await asyncio.to_thread(session.process_binary_packet, raw_bytes)
        if not response_packet:
            # If frame could not be decoded or processed, return clean empty packet
            return Response(content=b"", media_type="application/octet-stream", status_code=200)
            
        return Response(content=response_packet, media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HTTP frame processing error for {session_id}: {e}", exc_info=True)
        return Response(content=b"", media_type="application/octet-stream", status_code=200)

@app.post("/offer")
async def webrtc_offer(payload: WebRTCOfferRequest):
    """WebRTC SDP offer/answer exchange endpoint."""
    if not webrtc_manager.is_available:
        raise HTTPException(
            status_code=501,
            detail="WebRTC module (aiortc) is not available on this server host. Please use the WebSocket stream."
        )
    session = session_manager.get_or_create_session(payload.session_id)
    answer = await webrtc_manager.handle_offer(payload.sdp, payload.type, session)
    if not answer:
        raise HTTPException(status_code=500, detail="Failed to create WebRTC SDP answer")
    return answer

@app.post("/api/session/{session_id}/action")
async def handle_session_action(session_id: str, req: ActionRequest):
    """Handle control actions from frontend buttons."""
    session = session_manager.get_or_create_session(session_id)
    action = req.action.lower()
    
    if action == "start_counting":
        session.start_counting()
        return {"status": "ok", "action": "start_counting", "is_counting": True}
    elif action == "stop_counting":
        session.stop_counting()
        return {"status": "ok", "action": "stop_counting", "is_counting": False}
    elif action == "reset_counts":
        session.reset_counts()
        return {"status": "ok", "action": "reset_counts"}
    elif action == "clear_data":
        session.clear_data()
        return {"status": "ok", "action": "clear_data"}
    elif action == "toggle_recording":
        is_rec, fname = session.toggle_recording()
        return {"status": "ok", "is_recording": is_rec, "filename": fname}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

@app.get("/api/session/{session_id}/export/csv")
async def export_session_csv(session_id: str):
    """Export formatted event logs as CSV file."""
    session = session_manager.get_or_create_session(session_id)
    export_df = session.logger.format_export_data()
    
    stream = io.StringIO()
    export_df.to_csv(stream, index=False)
    csv_bytes = stream.getvalue().encode('utf-8')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"export_{timestamp}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/session/{session_id}/export/excel")
async def export_session_excel(session_id: str):
    """Export formatted event logs as Excel (.xlsx) file."""
    session = session_manager.get_or_create_session(session_id)
    export_df = session.logger.format_export_data()
    
    output = io.BytesIO()
    with export_df.to_excel(output, index=False, engine='openpyxl'):
        pass
    excel_bytes = output.getvalue()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"export_{timestamp}.xlsx"
    
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/assets/{filename}")
async def get_asset(filename: str):
    """Retrieve saved screenshots or recordings."""
    file_path = os.path.join("assets", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
