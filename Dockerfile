FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app.py /app/
COPY static/ /app/static/
COPY requirements.txt /app/

# Create directories
RUN mkdir -p /app/data /app/cache

# Install Python dependencies properly
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV CACHE_DIR=/app/cache

# Start the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
