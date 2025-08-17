# ---- Base: CUDA runtime + Ubuntu 22.04 ----
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

WORKDIR /app

# ---- OS deps: Python 3.11 + tini for clean PID 1 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates tini git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

# ---- Python deps (serverless + ETL + Playwright Python) ----
# Ensure requirements.prod.txt contains: runpod>=0.10.0 and playwright>=1.46.0
COPY requirements.prod.txt /app/requirements.prod.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.prod.txt

# ---- Playwright browsers + system deps (Chromium) ----
RUN python -m playwright install --with-deps chromium

# (Optional) Fonts for better rendering of some sites
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ---- App code (root files + app/ package) ----
COPY . /app

# ---- Serverless entry (start handler directly; no EXPOSE, no Uvicorn) ----
ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["python","-u","/app/handler.py"]