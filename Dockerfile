# ---- Base: CUDA runtime + Ubuntu 22.04 ----
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app PLAYWRIGHT_BROWSERS_PATH=/ms-playwright PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git tini build-essential pkg-config \
    fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ---- micromamba bootstrap ----
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# ---- Create conda env with Py3.11 + FAISS GPU (CUDA 12.x range) ----
# NOTE: conda-forge exposes CUDA 12 via the virtual package `cuda-version`, not `cudatoolkit=12.1`.
ENV MAMBA_NO_BANNER=1
RUN micromamba create -y -n appenv -c conda-forge --strict-channel-priority \
    "python=3.11" \
    "faiss-gpu=1.8.*" \
    "cuda-version>=12,<13" \
    "pip" \
    && micromamba clean -a -y

# Make env visible; still prefer micromamba run for reliability in each layer
ENV MAMBA_DEFAULT_ENV=appenv
ENV PATH=/opt/conda/envs/appenv/bin:$PATH

# ---- Python deps via pip (inside appenv) ----
COPY requirements.prod.txt /app/requirements.prod.txt
RUN micromamba run -n appenv python -m pip install --no-cache-dir --upgrade pip \
    && micromamba run -n appenv python -m pip install --no-cache-dir -r /app/requirements.prod.txt

# ---- Playwright (inside appenv) ----
RUN micromamba run -n appenv python -m pip install --no-cache-dir "playwright>=1.46.0" \
    && micromamba run -n appenv python -m playwright install --with-deps chromium

# ---- App code ----
COPY . /app

# ---- Build-time verification: FAISS GPU must be available ----
RUN micromamba run -n appenv python - <<'PY'
import faiss
print("FAISS version:", faiss.__version__)
print("Has GPU:", hasattr(faiss, "StandardGpuResources"))
assert hasattr(faiss, "StandardGpuResources"), "GPU FAISS not available!"
PY

# ---- Entrypoint ----
ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["micromamba","run","-n","appenv","python","-u","/app/handler.py"]
#CMD ["python","-u","/app/handler.py"]