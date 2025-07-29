FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04  # For GPU (or python:3.12-slim for CPU only)

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run your script (uses uvicorn.run inside main.py)
CMD ["python", "main.py"]
