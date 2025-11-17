# Multi-stage Dockerfile for Order Flow Trading Dashboard
# Production-optimized build with minimal attack surface

FROM python:3.11-slim as base

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r trader && useradd -r -g trader trader

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY streamlit_app.py .
COPY config.py .
COPY monitoring.py .

# Create necessary directories
RUN mkdir -p logs data && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose port
EXPOSE 8501

# Start command with production optimizations
CMD ["streamlit", "run", "streamlit_app.py", \
      "--server.port=8501", \
      "--server.address=0.0.0.0", \
      "--server.headless=true", \
      "--server.enableCORS=false", \
      "--server.enableXsrfProtection=false", \
      "--server.fileWatcherType=none", \
      "--server.maxUploadSize=50", \
      "--browser.gatherUsageStats=false"]