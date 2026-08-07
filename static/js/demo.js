/**
 * Face & Eye Tracking AI - Client Controller
 * High-Performance Zero-Base64 Binary Streaming & Responsive Dashboard Manager
 */

(function () {
    'use strict';

    // Unique Session ID
    const sessionId = 'session_' + Math.random().toString(36).substring(2, 10);

    // DOM Elements
    const permissionModal = document.getElementById('permission-modal');
    const btnGrantCamera = document.getElementById('btn-grant-camera');
    const modalError = document.getElementById('modal-error');
    
    // Mobile Drawer Elements
    const sidebarFrame = document.getElementById('sidebar-frame');
    const sidebarBackdrop = document.getElementById('sidebar-backdrop');
    const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
    const btnCloseSidebar = document.getElementById('btn-close-sidebar');

    const webcamVideo = document.getElementById('webcam-video');
    const captureCanvas = document.getElementById('capture-canvas');
    const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
    
    const annotatedImg = document.getElementById('annotated-stream');
    const videoPlaceholder = document.getElementById('video-placeholder');
    
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');
    const streamProtocol = document.getElementById('stream-protocol');
    const valLatency = document.getElementById('val-latency');
    const valTopFps = document.getElementById('val-top-fps');
    
    // Stats Elements
    const stats = {
        rightFace: document.getElementById('val-right-face'),
        leftFace: document.getElementById('val-left-face'),
        upFace: document.getElementById('val-up-face'),
        downFace: document.getElementById('val-down-face'),
        leftBlink: document.getElementById('val-left-blink'),
        rightBlink: document.getElementById('val-right-blink'),
        bothBlink: document.getElementById('val-both-blink'),
        eyeLeft: document.getElementById('val-eye-left'),
        eyeRight: document.getElementById('val-eye-right'),
        eyeUp: document.getElementById('val-eye-up'),
        eyeDown: document.getElementById('val-eye-down'),
        faceDir: document.getElementById('val-face-dir'),
        eyeDir: document.getElementById('val-eye-dir'),
        fps: document.getElementById('val-fps'),
        sessionTime: document.getElementById('val-session-time'),
    };

    // Mobile Quick Summary Elements
    const mStats = {
        faceDir: document.getElementById('m-val-face-dir'),
        eyeDir: document.getElementById('m-val-eye-dir'),
        faceTotal: document.getElementById('m-val-face-total'),
        blinkTotal: document.getElementById('m-val-blink-total'),
    };
    
    // Action Buttons
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const btnExportExcel = document.getElementById('btn-export-excel');
    const btnClearData = document.getElementById('btn-clear-data');
    const btnResetCounts = document.getElementById('btn-reset-counts');
    const btnScreenshot = document.getElementById('btn-screenshot');
    const btnRecord = document.getElementById('btn-record');
    const btnFullscreen = document.getElementById('btn-fullscreen');
    
    const logTableBody = document.getElementById('log-table-body');
    const toastContainer = document.getElementById('toast-container');

    let ws = null;
    let localStream = null;
    let isStreaming = false;
    let lastSendTime = 0;
    let isProcessingFrame = false;
    let prevBlobUrl = null;
    let isRecording = false;
    let lastLogCount = 0;

    // Mobile Drawer Logic
    if (btnToggleSidebar) {
        btnToggleSidebar.addEventListener('click', () => {
            sidebarFrame.classList.add('open');
            sidebarBackdrop.classList.add('active');
        });
    }

    if (btnCloseSidebar) {
        btnCloseSidebar.addEventListener('click', () => {
            sidebarFrame.classList.remove('open');
            sidebarBackdrop.classList.remove('active');
        });
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', () => {
            sidebarFrame.classList.remove('open');
            sidebarBackdrop.classList.remove('active');
        });
    }

    // Toast helper
    function showToast(message, duration = 3000) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // Initialize Matplotlib-matching Chart.js Graph
    const chartCanvas = document.getElementById('movement-chart');
    const movementChart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: Array(50).fill(''),
            datasets: [
                {
                    label: 'Face Movements',
                    data: [],
                    borderColor: '#00ffff',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Eye Movements',
                    data: [],
                    borderColor: '#ff9800',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'start',
                    labels: {
                        color: '#ffffff',
                        font: { family: "'Segoe UI', sans-serif", size: 10, weight: 'bold' },
                        boxWidth: 12
                    }
                },
                title: {
                    display: true,
                    text: 'Live Movement Activity',
                    color: '#ffffff',
                    font: { family: "'Segoe UI', sans-serif", size: 11, weight: 'bold' }
                }
            },
            scales: {
                x: {
                    grid: { color: '#383838' },
                    ticks: { display: false }
                },
                y: {
                    grid: { color: '#383838' },
                    ticks: { color: '#ffffff', font: { size: 9 } },
                    suggestedMin: 0,
                    suggestedMax: 10
                }
            }
        }
    });

    let firstFrameReceived = false;

    // Camera Access Initialization (Responsive for Mobile & Desktop)
    async function initCamera() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                const isSecure = window.isSecureContext !== false;
                throw new Error(
                    !isSecure
                        ? "Webcam access requires a secure connection (HTTPS). Please open the site using https://"
                        : "Your browser does not support webcam access via navigator.mediaDevices."
                );
            }

            modalError.classList.add('hidden');
            btnGrantCamera.textContent = 'Requesting camera...';
            btnGrantCamera.disabled = true;
            if (placeholderStatusText) {
                placeholderStatusText.textContent = 'Requesting camera permissions...';
            }

            const constraints = {
                video: {
                    facingMode: 'user', // Default to front camera on smartphones
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            };

            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (fallbackErr) {
                // Fallback to generic video if facingMode or constraints are unsupported
                console.warn('Fallback to basic video constraints:', fallbackErr);
                stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            }

            localStream = stream;
            webcamVideo.srcObject = stream;

            try {
                await webcamVideo.play();
            } catch (playErr) {
                console.warn('Video auto-play caught:', playErr);
            }

            captureCanvas.width = webcamVideo.videoWidth || 640;
            captureCanvas.height = webcamVideo.videoHeight || 480;

            permissionModal.classList.add('hidden');
            if (placeholderStatusText) {
                placeholderStatusText.textContent = 'Connecting to Python Computer Vision Engine...';
            }
            startWebSocketStreaming();
        } catch (err) {
            console.error('Camera access denied or prompt required:', err);
            btnGrantCamera.disabled = false;
            btnGrantCamera.textContent = 'Grant Camera Access';
            modalError.textContent = `Error accessing webcam: ${err.message || err}. Please ensure camera permissions are allowed in your browser settings.`;
            modalError.classList.remove('hidden');
            if (placeholderStatusText) {
                placeholderStatusText.textContent = `Camera Access Required: ${err.message || err}`;
            }
        }
    }

    btnGrantCamera.addEventListener('click', initCamera);

    // Auto-attempt camera start immediately when page opens
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        initCamera();
    }

    let wsRetryCount = 0;
    let streamMode = 'WS'; // 'WS' or 'HTTP'
    let isCaptureLoopRunning = false;
    let animFrameId = null;

    function ensureCaptureLoop() {
        if (!isCaptureLoopRunning) {
            isCaptureLoopRunning = true;
            captureAndSendLoop();
        }
    }

    // High-Performance Binary WebSocket Client with Auto-HTTP Fallback
    function startWebSocketStreaming() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/track/${sessionId}`;

        connectionText.textContent = 'Connecting...';
        connectionDot.className = 'status-dot';

        try {
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                wsRetryCount = 0;
                streamMode = 'WS';
                connectionDot.className = 'status-dot connected';
                connectionText.textContent = 'PYTHON 3.12 ENGINE CONNECTED';
                if (streamProtocol) streamProtocol.textContent = 'Binary WS';
                isStreaming = true;
                isProcessingFrame = false;
                ensureCaptureLoop();
            };

            ws.onmessage = (event) => {
                const now = performance.now();
                if (lastSendTime > 0) {
                    const latency = Math.round(now - lastSendTime);
                    valLatency.textContent = `${latency} ms`;
                }
                isProcessingFrame = false;

                if (event.data instanceof ArrayBuffer) {
                    unpackBinaryPacket(event.data);
                } else if (typeof event.data === 'string') {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'recording_state') {
                            isRecording = data.is_recording;
                            updateRecordingButton();
                        }
                    } catch (e) {
                        console.debug('WS text parse error:', e);
                    }
                }
            };

            ws.onerror = (err) => {
                console.warn('WebSocket status note:', err);
                isProcessingFrame = false;
            };

            ws.onclose = () => {
                wsRetryCount++;
                isProcessingFrame = false;
                
                if (wsRetryCount >= 2 && streamMode !== 'HTTP') {
                    // Activate transparent HTTP frame streaming fallback
                    console.log('Activating HTTP Streaming Fallback...');
                    streamMode = 'HTTP';
                    isStreaming = true;
                    connectionDot.className = 'status-dot connected';
                    connectionText.textContent = 'PYTHON 3.12 ENGINE (HTTP STREAM)';
                    if (streamProtocol) streamProtocol.textContent = 'HTTP Stream';
                    ensureCaptureLoop();
                    
                    // Periodic retry for WebSocket upgrade
                    setTimeout(() => {
                        if (streamMode === 'HTTP') startWebSocketStreaming();
                    }, 15000);
                } else if (streamMode === 'WS') {
                    connectionDot.className = 'status-dot';
                    connectionText.textContent = 'Connecting...';
                    setTimeout(startWebSocketStreaming, 2000);
                }
            };
        } catch (wsErr) {
            console.warn('WebSocket init exception, falling back to HTTP:', wsErr);
            streamMode = 'HTTP';
            isStreaming = true;
            connectionDot.className = 'status-dot connected';
            connectionText.textContent = 'PYTHON 3.12 ENGINE (HTTP STREAM)';
            if (streamProtocol) streamProtocol.textContent = 'HTTP Stream';
            ensureCaptureLoop();
        }
    }

    // Zero-Base64 Binary Packet Unpacker
    function unpackBinaryPacket(buffer) {
        try {
            if (!buffer || buffer.byteLength < 4) return;
            const view = new DataView(buffer);
            const jsonLength = view.getUint32(0, false); // 4-byte uint32 Big Endian
            
            if (buffer.byteLength < 4 + jsonLength) return;

            const jsonBytes = new Uint8Array(buffer, 4, jsonLength);
            const jsonStr = new TextDecoder('utf-8').decode(jsonBytes);
            const telemetry = JSON.parse(jsonStr);

            // Remaining bytes are pure JPEG image bytes
            const imageBytes = new Uint8Array(buffer, 4 + jsonLength);
            if (imageBytes.length > 0) {
                const blob = new Blob([imageBytes], { type: 'image/jpeg' });
                
                // Release memory of previous frame object URL
                if (prevBlobUrl) {
                    URL.revokeObjectURL(prevBlobUrl);
                }
                prevBlobUrl = URL.createObjectURL(blob);
                annotatedImg.src = prevBlobUrl;

                if (!firstFrameReceived) {
                    firstFrameReceived = true;
                    videoPlaceholder.style.display = 'none';
                    annotatedImg.style.opacity = '1';
                }
            }

            // Update UI with telemetry
            updateDashboard(telemetry);
        } catch (e) {
            console.error('Failed to unpack binary packet:', e);
        }
    }

    // Adaptive Frame Capture and Sending Loop
    function captureAndSendLoop() {
        if (!isStreaming) {
            isCaptureLoopRunning = false;
            return;
        }

        const now = performance.now();

        // Watchdog: If server didn't respond within 800ms, unlock the pipeline
        if (isProcessingFrame && (now - lastSendTime > 800)) {
            isProcessingFrame = false;
        }

        // Ensure webcam is actively playing
        if (webcamVideo.paused && webcamVideo.srcObject) {
            webcamVideo.play().catch(() => {});
        }

        const hasActiveStream = (webcamVideo.readyState >= 2 || webcamVideo.currentTime > 0 || (webcamVideo.srcObject && webcamVideo.srcObject.active));

        if (!isProcessingFrame && hasActiveStream) {
            isProcessingFrame = true;
            lastSendTime = now;

            const vWidth = webcamVideo.videoWidth || 640;
            const vHeight = webcamVideo.videoHeight || 480;
            if (captureCanvas.width !== vWidth || captureCanvas.height !== vHeight) {
                captureCanvas.width = vWidth;
                captureCanvas.height = vHeight;
            }

            // Draw current webcam frame onto offscreen canvas
            captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
            
            // Export canvas directly as binary JPEG blob
            captureCanvas.toBlob((blob) => {
                if (!blob) {
                    isProcessingFrame = false;
                    return;
                }

                if (streamMode === 'WS') {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        blob.arrayBuffer().then((buffer) => {
                            try {
                                ws.send(buffer);
                            } catch (sendErr) {
                                console.error('WS send error:', sendErr);
                                isProcessingFrame = false;
                            }
                        }).catch(() => {
                            isProcessingFrame = false;
                        });
                    } else {
                        // WebSocket is still connecting; do not spam HTTP requests
                        isProcessingFrame = false;
                    }
                } else if (streamMode === 'HTTP') {
                    // Transparent HTTP Frame Stream Fallback
                    fetch(`/api/session/${sessionId}/frame`, {
                        method: 'POST',
                        body: blob,
                        headers: { 'Content-Type': 'application/octet-stream' }
                    })
                    .then((res) => {
                        if (!res.ok) throw new Error(`HTTP Frame Status ${res.status}`);
                        return res.arrayBuffer();
                    })
                    .then((buffer) => {
                        const frameLatency = Math.round(performance.now() - lastSendTime);
                        valLatency.textContent = `${frameLatency} ms`;
                        isProcessingFrame = false;
                        if (buffer && buffer.byteLength > 4) {
                            unpackBinaryPacket(buffer);
                        }
                    })
                    .catch((httpErr) => {
                        console.warn('HTTP frame stream notice:', httpErr);
                        isProcessingFrame = false;
                    });
                }
            }, 'image/jpeg', 0.80);
        }

        animFrameId = requestAnimationFrame(captureAndSendLoop);
    }

    // Update UI Stats and Dashboard
    function updateDashboard(data) {
        if (!data) return;

        // Update 15 Stats matching CustomTkinter
        if (data.counts) {
            const c = data.counts;
            stats.rightFace.textContent = c['Right Face Count'] || 0;
            stats.leftFace.textContent = c['Left Face Count'] || 0;
            stats.upFace.textContent = c['Up Count'] || 0;
            stats.downFace.textContent = c['Down Count'] || 0;
            stats.leftBlink.textContent = c['Left Blink Count'] || 0;
            stats.rightBlink.textContent = c['Right Blink Count'] || 0;
            stats.bothBlink.textContent = c['Both Blink Count'] || 0;
            stats.eyeLeft.textContent = c['Eye Left Count'] || 0;
            stats.eyeRight.textContent = c['Eye Right Count'] || 0;
            stats.eyeUp.textContent = c['Eye Up Count'] || 0;
            stats.eyeDown.textContent = c['Eye Down Count'] || 0;

            // Mobile Quick Summary
            if (mStats.faceTotal) {
                mStats.faceTotal.textContent = `${c['Right Face Count'] || 0}/${c['Left Face Count'] || 0}/${c['Up Count'] || 0}/${c['Down Count'] || 0}`;
            }
            if (mStats.blinkTotal) {
                mStats.blinkTotal.textContent = `${c['Left Blink Count'] || 0}/${c['Right Blink Count'] || 0}/${c['Both Blink Count'] || 0}`;
            }
        }

        stats.faceDir.textContent = data.face_dir || '-';
        stats.eyeDir.textContent = data.eye_dir || '-';
        stats.fps.textContent = data.fps || 0;
        stats.sessionTime.textContent = data.session_time || '00:00';

        if (valTopFps) valTopFps.textContent = data.fps || 0;
        if (mStats.faceDir) mStats.faceDir.textContent = data.face_dir || '-';
        if (mStats.eyeDir) mStats.eyeDir.textContent = data.eye_dir || '-';

        // Update Live Activity Graph
        if (data.graph) {
            movementChart.data.datasets[0].data = data.graph.face || [];
            movementChart.data.datasets[1].data = data.graph.eye || [];
            movementChart.update('none'); // Update without animation for maximum performance
        }

        // Update Treeview Log Table
        if (data.logs && data.logs.length !== lastLogCount) {
            lastLogCount = data.logs.length;
            renderLogTable(data.logs);
        }
    }

    // Render Event Log Table (Replicating Tkinter Treeview, newest on top)
    function renderLogTable(logs) {
        if (!logs) return;
        logTableBody.innerHTML = '';
        const reversed = logs.slice().reverse();
        reversed.forEach((entry, idx) => {
            const tr = document.createElement('tr');
            if (idx === 0) tr.className = 'new-row';
            tr.innerHTML = `
                <td>${entry[0]}</td>
                <td>${entry[1]}</td>
                <td>${entry[2]}</td>
            `;
            logTableBody.appendChild(tr);
        });
    }

    // Button Actions & Protocol Helpers
    function sendAction(actionName) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'action', action: actionName }));
        } else {
            fetch(`/api/session/${sessionId}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: actionName })
            });
        }
    }

    btnStart.addEventListener('click', () => {
        sendAction('start_counting');
        btnStart.disabled = true;
        btnStop.disabled = false;
        showToast('Movement Counting Started');
    });

    btnStop.addEventListener('click', () => {
        sendAction('stop_counting');
        btnStart.disabled = false;
        btnStop.disabled = true;
        showToast('Movement Counting Paused');
    });

    btnClearData.addEventListener('click', () => {
        sendAction('clear_data');
        logTableBody.innerHTML = '';
        lastLogCount = 0;
        showToast('Event Log Data Cleared');
    });

    btnResetCounts.addEventListener('click', () => {
        sendAction('reset_counts');
        showToast('All Movement Counters Reset');
    });

    btnExportCsv.addEventListener('click', () => {
        window.location.href = `/api/session/${sessionId}/export/csv`;
        showToast('Exporting CSV...');
    });

    btnExportExcel.addEventListener('click', () => {
        window.location.href = `/api/session/${sessionId}/export/excel`;
        showToast('Exporting Excel (.xlsx)...');
    });

    btnScreenshot.addEventListener('click', () => {
        if (!annotatedImg.src) return;
        const a = document.createElement('a');
        a.href = annotatedImg.src;
        a.download = `screenshot_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        showToast('Screenshot Downloaded');
    });

    function updateRecordingButton() {
        if (isRecording) {
            btnRecord.textContent = 'Stop Recording';
            btnRecord.className = 'btn btn-red';
            showToast('Video Recording Started');
        } else {
            btnRecord.textContent = 'Start Recording';
            btnRecord.className = 'btn btn-blue';
            showToast('Video Recording Stopped & Saved');
        }
    }

    btnRecord.addEventListener('click', () => {
        isRecording = !isRecording;
        sendAction('toggle_recording');
        updateRecordingButton();
    });

    if (btnFullscreen) {
        btnFullscreen.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(err => console.debug(err));
                document.body.classList.add('fullscreen-active');
            } else {
                document.exitFullscreen().catch(err => console.debug(err));
                document.body.classList.remove('fullscreen-active');
            }
        });
    }

    document.addEventListener('fullscreenchange', () => {
        if (!document.fullscreenElement) {
            document.body.classList.remove('fullscreen-active');
        }
    });

})();
