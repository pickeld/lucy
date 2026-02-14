# Reflex UI development image with hot reload
FROM python:3.11-slim

WORKDIR /app

# System deps for Reflex (Node.js for frontend build)
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Initialize Reflex (downloads bun/node tooling on first run)
COPY rxconfig.py .
RUN reflex init --loglevel warning 2>/dev/null || true

# Copy app source (will be overridden by volume mount in dev)
COPY . .

EXPOSE 3000 8000

# Run Reflex in dev mode (hot reload built-in)
# Use a shell entrypoint so we can bind frontend to 0.0.0.0 via next.js config
CMD ["sh", "-c", "reflex run --env dev --backend-host 0.0.0.0 --frontend-port 3000 --backend-port 8000"]
