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
# Note: CUDA 12 via 'cuda-version' virtual package on conda-forge.
ENV MAMBA_NO_BANNER=1
RUN micromamba create -y -n appenv -c conda-forge --strict-channel-priority \
    "python=3.11" \
    "faiss-gpu=1.8.*" \
    "cuda-version>=12,<13" \
    "pip" \
    && micromamba clean -a -y

# Make env visible; still use micromamba run explicitly
ENV MAMBA_DEFAULT_ENV=appenv
ENV PATH=/opt/conda/envs/appenv/bin:$PATH
# (optional) ensure env libs are first for CUDA 12 in env
ENV LD_LIBRARY_PATH=/opt/conda/envs/appenv/lib:${LD_LIBRARY_PATH}

# ---- Pip deps (inside appenv) ----
COPY requirements.prod.txt /app/requirements.prod.txt

# 1) Add a pip constraints file that forbids faiss-cpu from PyPI
#    (prevents transitive deps from downgrading the GPU build)
RUN printf "faiss-cpu==0\nfaiss==0\n" > /app/pip-constraints.txt

RUN micromamba run -n appenv python -m pip install --no-cache-dir --upgrade pip --root-user-action=ignore \
    && micromamba run -n appenv python -m pip install --no-cache-dir -r /app/requirements.prod.txt \
    --constraint /app/pip-constraints.txt --root-user-action=ignore

# 2) Re-install conda FAISS GPU AFTER pip to overwrite any CPU wheel that slipped in
RUN micromamba install -y -n appenv -c conda-forge "faiss-gpu=1.8.*" \
    && micromamba clean -a -y

# ---- Playwright (inside appenv) ----
RUN micromamba run -n appenv python -m pip install --no-cache-dir "playwright>=1.46.0" --root-user-action=ignore \
    && micromamba run -n appenv python -m playwright install --with-deps chromium

# ---- App code ----
COPY . /app

# ---- Build-time verification: FAISS GPU must be available ----
RUN micromamba run -n appenv python - <<'PY'
import faiss, sys
print("FAISS version:", faiss.__version__)
print("faiss module:", faiss.__file__)
print("Has GPU:", hasattr(faiss, "StandardGpuResources"))
assert hasattr(faiss, "StandardGpuResources"), "GPU FAISS not available!"
PY

ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["micromamba","run","-n","appenv","python","-u","/app/handler.py"]

#CMD ["python","-u","/app/handler.py"]