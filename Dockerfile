# Stage 1: Build virtual environment and install packages
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pynetdicom albumentations opencv-python-headless

# Stage 2: Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Expose port for FastAPI (8000) and mock DICOM Listener (11112)
EXPOSE 8000 11112

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/aura
ENV AURA_ALLOW_FALLBACK_VISION=1

CMD ["python", "aura/gateway/app.py"]
