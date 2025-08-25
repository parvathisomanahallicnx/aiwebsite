# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

WORKDIR /app

# Minimal system deps; extend if build errors request specific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy application code
COPY . .

# Service configuration
ENV PORT=8080
EXPOSE 8080

# Start the FastAPI app (honor platform-provided $PORT when present)
CMD ["sh", "-c", "uvicorn langgraph_agent_workflow:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]
