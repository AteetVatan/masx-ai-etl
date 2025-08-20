# ---- Base: CUDA runtime + Ubuntu 22.04 ---- 
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app PLAYWRIGHT_BROWSERS_PATH=/ms-playwright PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git tini build-essential pkg-config \
    fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# micromamba bootstrap
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# # Create env with py311 and GPU FAISS
# # ADDED: faiss-gpu from conda-forge (works with Py3.11 + CUDA 12.x)
# RUN micromamba create -y -n appenv -c conda-forge \
#     python=3.11 \
#     faiss-gpu=1.8.0 \
#     cudatoolkit=12.1 \
#     pip \
#     && micromamba clean -a -y

# Create env with py311 and GPU FAISS
# CHANGED: remove 'cudatoolkit=12.1' (not a conda-forge package for CUDA 12)
# ADDED: constrain to CUDA 12 *range* via the virtual package 'cuda-version'
ENV MAMBA_NO_BANNER=1
RUN micromamba create -y -n appenv -c conda-forge --strict-channel-priority \
        "python=3.11" \
        "faiss-gpu=1.8.*" \
        "cuda-version>=12,<13" \
        "pip" \
    && micromamba clean -a -y

# activate env for subsequent RUN/CMD
ENV MAMBA_DEFAULT_ENV=appenv
ENV PATH=/opt/conda/envs/appenv/bin:$PATH

# upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Python deps
# NOTE: requirements.prod.txt NO LONGER contains faiss-gpu (removed).
COPY requirements.prod.txt /app/requirements.prod.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.prod.txt

# Playwright
RUN python -m pip install --no-cache-dir "playwright>=1.46.0" \
    && python -m playwright install --with-deps chromium

# App code
COPY . /app

# Quick checks (GPU FAISS has StandardGpuResources)
# This should print True; if False, the build will fail and surface the issue early.
RUN python - <<'PY'
import faiss
print("FAISS version:", faiss.__version__)
print("Has GPU:", hasattr(faiss, "StandardGpuResources"))
assert hasattr(faiss, "StandardGpuResources"), "GPU FAISS not available!"
PY

ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["python","-u","/app/handler.py"]
