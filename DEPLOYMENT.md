# Face & Eye Tracking AI &bull; Production Deployment Guide

This guide provides instructions for deploying the **Face & Eye Tracking AI System** across all major cloud providers and container environments while maintaining 100% of the Python computer vision algorithms, low latency (<100ms), and 30 FPS.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Local Development & Desktop Execution](#2-local-development--desktop-execution)
3. [Docker & Docker Compose (Production)](#3-docker--docker-compose-production)
4. [Railway Deployment](#4-railway-deployment)
5. [Render Deployment](#5-render-deployment)
6. [AWS Deployment (ECS / App Runner)](#6-aws-deployment-ecs--app-runner)
7. [Azure Deployment (App Service)](#7-azure-deployment-app-service)
8. [Google Cloud Run Deployment](#8-google-cloud-run-deployment)
9. [Performance Tuning & Network Optimization](#9-performance-tuning--network-optimization)

---

## 1. Architecture Overview

- **Engine Core:** Python 3.12, MediaPipe Face Landmarker (478 3D landmarks), OpenCV PnP solver, EAR calculation, Cartesian iris vectoring.
- **Backend Framework:** FastAPI with asynchronous lifespan management.
- **Binary Transport Protocol:** Zero-Base64 packed binary WebSockets `[4-byte JSON Header Length | JSON Telemetry | Raw JPEG Bytes]` with single-slot frame queue to prevent buffer delays under network fluctuations.
- **WebRTC Support:** Native WebRTC video stream transformation using `aiortc`.
- **Frontend Dashboard:** Visually and functionally identical to the CustomTkinter desktop GUI, featuring live Matplotlib-style Chart.js graphs and Treeview data tables.

---

## 2. Local Development & Desktop Execution

### Desktop Mode (Original CustomTkinter GUI)
To run the original desktop app directly on your physical webcam:
```bash
python main.py
```

### Local Web Mode (FastAPI + Browser Live Demo)
To start the FastAPI web server locally:
```bash
pip install -r requirements.txt
python server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
1. Open `http://localhost:8000/` to view the Portfolio Landing Page.
2. Click **"Launch Live Demo"** or navigate directly to `http://localhost:8000/demo`.
3. Allow camera permissions in your browser.

---

## 3. Docker & Docker Compose (Production)

### Quick Start with Docker Compose
```bash
# Build and start services (FastAPI Backend + NGINX Proxy)
docker-compose up -d --build

# View logs
docker-compose logs -f
```
The application will be accessible at `http://localhost`.

### Standalone Docker Build
```bash
# Build container image
docker build -t face-eye-tracking-ai:latest .

# Run container
docker run -d -p 8000:8000 --name face-eye-tracking-app face-eye-tracking-ai:latest
```

---

## 4. Railway Deployment

1. Create an account on [Railway.app](https://railway.app/).
2. Connect your GitHub repository.
3. Railway automatically detects `Dockerfile` and `railway.json`.
4. Set Environment Variables in Railway Dashboard:
   - `PORT`: `8000`
5. Railway automatically provides HTTPS and WebSocket proxying.

---

## 5. Render Deployment

1. Connect your GitHub repository to [Render.com](https://render.com/).
2. Render will automatically detect `render.yaml` (Infrastructure as Code Blueprint).
3. Alternatively, create a **New Web Service**:
   - Environment: `Docker`
   - Plan: `Starter` (or higher for dedicated CPU)
   - Health Check Path: `/health`
4. Deploy service.

---

## 6. AWS Deployment (ECS / App Runner)

### AWS App Runner (Fastest container deployment)
1. Push Docker image to AWS Elastic Container Registry (ECR):
   ```bash
   aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
   docker tag face-eye-tracking-ai:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/face-eye-tracking-ai:latest
   docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/face-eye-tracking-ai:latest
   ```
2. Create an App Runner Service pointing to the ECR image.
3. Set port to `8000` and health check to `/health`.

### AWS ECS Fargate
1. Register the task definition in `aws-task-definition.json`:
   ```bash
   aws ecs register-task-definition --cli-input-json file://aws-task-definition.json
   ```
2. Create an ECS Fargate service associated with an Application Load Balancer (ALB) with WebSocket support enabled.

---

## 7. Azure Deployment (App Service)

1. Build and push image to Azure Container Registry (ACR):
   ```bash
   az acr build --registry <ACR_NAME> --image face-eye-tracking-app:latest .
   ```
2. Create Azure Web App for Containers:
   ```bash
   az webapp create --resource-group <RESOURCE_GROUP> --plan <APP_SERVICE_PLAN> \
     --name face-eye-tracking-ai --deployment-container-image-name <ACR_NAME>.azurecr.io/face-eye-tracking-app:latest
   ```
3. Enable WebSockets in Azure Portal:
   - **Configuration** &rarr; **General Settings** &rarr; **Web sockets: ON**.

---

## 8. Google Cloud Run Deployment

Deploy with a single command using Google Cloud SDK:
```bash
gcloud run deploy face-eye-tracking-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600
```
*(Cloud Run natively supports WebSockets with timeouts up to 60 minutes).*

---

## 9. Performance Tuning & Network Optimization

1. **Frame Dropping:** The backend enforces `asyncio.Queue(maxsize=1)`. If the network or client encounters latency, stale frames are discarded automatically so the processing pipeline is always in lock-step with real time.
2. **Binary Packing:** Raw JPEG buffers are encoded on the server and transferred with a 4-byte Big-Endian length header. No Base64 conversion overhead occurs on either end.
3. **JPEG Compression Quality:** Default quality is set to `85` in `web_session.py`, providing a balance between facial landmark detail and payload size (~25-35 KB per frame).
4. **Isolated Multi-User Sessions:** Every browser tab operates with its own `FaceTracker`, `EyeTracker`, and `EventLogger` instances stored in `SessionManager`.
