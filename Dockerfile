# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ /app/src/
COPY .env.example .env

# Create data directories for media and event files
RUN mkdir -p /app/data/images /app/data/events

# Set the Python path to include the src directory
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Make port 8765 available to the world outside this container
EXPOSE 8765

# Run with gunicorn for production-grade concurrency.
# --workers 4 --threads 2 = 8 concurrent request handlers.
# --reload enables auto-restart on code changes (for development).
# --timeout 300 allows long-running RAG queries to complete.
# For pure local development, use: python -u src/app.py
CMD ["gunicorn", "--bind", "0.0.0.0:8765", \
     "--workers", "4", \
     "--threads", "2", \
     "--timeout", "300", \
     "--reload", \
     "--access-logfile", "-", \
     "--chdir", "/app/src", \
     "app:app"]