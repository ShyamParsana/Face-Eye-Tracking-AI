/**
 * WebRTC Client Manager for Face & Eye Tracking AI
 * Negotiates real-time video track transformation with Python backend via aiortc
 */
class WebRTCClient {
    constructor(sessionId, onTelemetry, onTrack, onError) {
        this.sessionId = sessionId;
        this.onTelemetry = onTelemetry;
        this.onTrack = onTrack;
        this.onError = onError;
        this.pc = null;
        this.dataChannel = null;
    }

    async start(stream) {
        try {
            const config = {
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' }
                ]
            };

            this.pc = new RTCPeerConnection(config);

            // Data channel for real-time telemetry from Python
            this.dataChannel = this.pc.createDataChannel('telemetry');
            this.dataChannel.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (this.onTelemetry) this.onTelemetry(data);
                } catch (e) {
                    console.debug('DataChannel parse error:', e);
                }
            };

            // Add local webcam video track to peer connection
            stream.getTracks().forEach((track) => {
                this.pc.addTrack(track, stream);
            });

            // Handle incoming transformed video track from Python
            this.pc.ontrack = (event) => {
                if (event.track.kind === 'video' && this.onTrack) {
                    this.onTrack(event.streams[0]);
                }
            };

            // Create and exchange SDP offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);

            const response = await fetch('/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: this.pc.localDescription.sdp,
                    type: this.pc.localDescription.type,
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`WebRTC offer rejected: ${response.statusText}`);
            }

            const answer = await response.json();
            await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
            return true;
        } catch (err) {
            console.warn('WebRTC initialization failed, falling back to Binary WebSocket:', err);
            if (this.onError) this.onError(err);
            return false;
        }
    }

    stop() {
        if (this.dataChannel) {
            this.dataChannel.close();
            this.dataChannel = null;
        }
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
    }
}
