# Face & Eye Tracking AI

A high-performance, real-time Computer Vision system built with Python, OpenCV, MediaPipe, CustomTkinter, and FastAPI. It tracks 3D face orientation, gaze direction, eye blinks (EAR), and head pose in real-time, offering both a **native Desktop GUI** and a **modern Web Dashboard with live video streaming**.

---

## 🌟 Key Features

- **High-Precision Tracking:** Powered by MediaPipe Face Mesh (478 3D facial landmarks) and OpenCV PnP head pose solver.
- **Direction & Blink Counters:** Real-time counting of head turns (Left/Right/Up/Down) with neutral-state latching and EAR blink detection.
- **Dual Interface:**
  - **Desktop GUI:** Built with CustomTkinter, real-time Matplotlib activity graphs, and data tables.
  - **Web Dashboard:** FastAPI backend featuring binary WebSocket streaming (<100ms latency), WebRTC support, responsive Chart.js telemetry, and mobile drawer.
- **Session Logging & Export:** Automatically logs all movements with timestamps; exportable to `.csv` and `.xlsx`.
- **Production Ready:** Fully containerized with Docker, NGINX reverse proxy, and multi-cloud CI/CD deployment scripts (Render, Railway, AWS, Azure, GCP).

---

## 📁 Project Architecture

- `main.py`: Desktop application entry point (CustomTkinter GUI).
- `server.py`: FastAPI web server entry point (WebSockets, WebRTC, REST API).
- `face_tracker.py`: 3D Head pose estimation, angle calculations, and state machine counters.
- `eye_tracker.py`: EAR blink detection, iris vectoring, and gaze direction tracking.
- `web_session.py`: Multi-client session manager isolating computer vision pipelines per browser.
- `webrtc_manager.py`: Native low-latency WebRTC video track transformer (`aiortc`).
- `camera.py`: Webcam management, auto-device selection, and video recording.
- `logger.py`: Pandas event logger with CSV/Excel export capabilities.
- `utils.py`: Geometric math, EAR calculations, and camera calibration helpers.
- `gui.py`: Desktop CustomTkinter dark-mode dashboard.
- `templates/` & `static/`: Modern landing page and real-time browser dashboard.
- `DEPLOYMENT.md`: Comprehensive cloud and container deployment guide.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.12+ installed:
```bash
git clone <your-repo-url>
cd <folder-name>
pip install -r requirements.txt
```

---

### 2. Running the Application

#### Option A: Desktop Mode (CustomTkinter)
```bash
python main.py
```

#### Option B: Web Mode (FastAPI + Browser Live Demo)
```bash
python server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
- Open `http://localhost:8000/` for the **Portfolio Showcase**.
- Open `http://localhost:8000/demo` for the **Live Webcam Tracker**.

#### Option C: Docker Compose (Production)
```bash
docker-compose up -d --build
```
- Open `http://localhost/` (served via NGINX reverse proxy).

---

## 📊 Deployment & Cloud Hosting
For deploying to **Render**, **Railway**, **AWS ECS**, **Azure App Service**, or **Google Cloud Run**, see the complete [DEPLOYMENT.md](DEPLOYMENT.md) guide.

---

## 📄 License
MIT License. Built for real-time computer vision applications.
