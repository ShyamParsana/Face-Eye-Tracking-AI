# ==============================================================================
# Face & Eye Tracking AI - Production Dockerfile
# Optimized for Python 3.12, OpenCV Headless, MediaPipe, and FastAPI
# ==============================================================================

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install OS runtime dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and model weights
COPY face_landmarker.task .
COPY utils.py .
COPY logger.py .
COPY camera.py .
COPY face_tracker.py .
COPY eye_tracker.py .
COPY web_session.py .
COPY webrtc_manager.py .
COPY server.py .
COPY static/ ./static/
COPY templates/ ./templates/

# Create assets directory
RUN mkdir -p assets

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Run FastAPI server via Uvicorn
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1 --loop uvloop --http httptools"]
