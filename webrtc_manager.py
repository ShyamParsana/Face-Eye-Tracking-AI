import asyncio
import json
import logging
from typing import Optional, Set

logger = logging.getLogger("webrtc")

try:
    import av
    from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaRelay
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    MediaStreamTrack = object
    RTCPeerConnection = None
    RTCSessionDescription = None

if WEBRTC_AVAILABLE:
    class VideoTransformTrack(MediaStreamTrack):
        """
        A video stream track that transforms incoming camera frames using
        the preserved Python FaceTracker and EyeTracker algorithms in real time.
        """
        kind = "video"

        def __init__(self, track, session_state, data_channel=None):
            super().__init__()
            self.track = track
            self.session = session_state
            self.data_channel = data_channel

        async def recv(self):
            frame = await self.track.recv()
            
            # Convert av.VideoFrame to numpy BGR image
            img_bgr = frame.to_ndarray(format="bgr24")
            
            # Process using the exact Python CV pipeline
            annotated_bgr, telemetry = self.session.process_frame(img_bgr)
            
            # Send real-time telemetry over WebRTC DataChannel if connected
            if self.data_channel and self.data_channel.readyState == "open":
                try:
                    self.data_channel.send(json.dumps(telemetry))
                except Exception as e:
                    logger.debug(f"DataChannel send error: {e}")
                    
            # Reconstruct av.VideoFrame from processed BGR ndarray
            new_frame = av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

class WebRTCManager:
    def __init__(self):
        self.pcs: Set = set()
        
    @property
    def is_available(self) -> bool:
        return WEBRTC_AVAILABLE

    async def handle_offer(self, sdp: str, sdp_type: str, session_state) -> Optional[dict]:
        """Process an incoming SDP offer and return the local SDP answer."""
        if not WEBRTC_AVAILABLE:
            return None

        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        data_channel_holder = {"channel": None}

        @pc.on("datachannel")
        def on_datachannel(channel):
            data_channel_holder["channel"] = channel

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                transform_track = VideoTransformTrack(
                    track, session_state, data_channel=data_channel_holder.get("channel")
                )
                pc.addTrack(transform_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.discard(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

    async def close_all(self):
        """Close all active WebRTC peer connections on server shutdown."""
        if not WEBRTC_AVAILABLE:
            return
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros, return_exceptions=True)
        self.pcs.clear()

webrtc_manager = WebRTCManager()
