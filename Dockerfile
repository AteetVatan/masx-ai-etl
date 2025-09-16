# ---- Base: CUDA runtime + Ubuntu 22.04 ----
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04



ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app PLAYWRIGHT_BROWSERS_PATH=/ms-playwright PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV TRANSFORMERS_VERBOSITY=info

WORKDIR /app

# Hugging Face cache inside container
ENV HF_HOME=/app/.hf_cache

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

# ---- Create env (Py3.11 + CUDA 12.x virtual) ----
ENV MAMBA_NO_BANNER=1
RUN micromamba create -y -n appenv -c conda-forge --strict-channel-priority \
    "python=3.11" \
    "cuda-version>=12,<13" \
    "pip" \
    && micromamba clean -a -y

# Make env visible; still call via micromamba for reliability
ENV MAMBA_DEFAULT_ENV=appenv
ENV PATH=/opt/conda/envs/appenv/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/appenv/lib:${LD_LIBRARY_PATH}

# ---- Python deps via pip (inside appenv) ----
COPY requirements.prod.txt /app/requirements.prod.txt

# Block CPU wheels that would shadow the GPU bindings
RUN printf "faiss-cpu==0\nfaiss==0\n" > /app/pip-constraints.txt


RUN micromamba run -n appenv python -m pip install --no-cache-dir --upgrade pip --root-user-action=ignore \
    && micromamba run -n appenv python -m pip install --no-cache-dir \
    -r /app/requirements.prod.txt \
    --constraint /app/pip-constraints.txt \
    --root-user-action=ignore

# ---- Install FAISS GPU from PyPI (CUDA 12 wheels) ----
# These wheels expose the 'faiss' module with GPU symbols.
RUN micromamba run -n appenv python -m pip install --no-cache-dir \
    "faiss-gpu-cu12==1.8.0.2" \
    --root-user-action=ignore

# ---- Playwright (inside appenv) ----
RUN micromamba run -n appenv python -m pip install --no-cache-dir "playwright>=1.46.0" --root-user-action=ignore \
    && micromamba run -n appenv python -m playwright install --with-deps chromium

RUN micromamba run -n appenv python - <<'PY'
import nltk
print("Downloading NLTK punkt models...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("NLTK punkt models installed")
PY


# ---- App code ----
COPY . /app

# ---- Build-time verification: FAISS GPU must be available ----
RUN micromamba run -n appenv python - <<'PY'
import faiss
print("FAISS version:", faiss.__version__)
print("faiss module:", faiss.__file__)
print("Has GPU:", hasattr(faiss, "StandardGpuResources"))
assert hasattr(faiss, "StandardGpuResources"), "GPU FAISS not available!"
PY

# Add after the FAISS verification
RUN micromamba run -n appenv python - <<'PY'
import aiohttp
print("aiohttp version:", aiohttp.__version__)
print("aiohttp available for RunPod API calls")
PY

# Pre-download models here
RUN mkdir -p $HF_HOME

RUN micromamba run -n appenv python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='$HF_HOME')"

RUN micromamba run -n appenv python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn', cache_dir='$HF_HOME'); \
    AutoTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir='$HF_HOME')"

RUN micromamba run -n appenv python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='$HF_HOME'); \
    AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='$HF_HOME')"

ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["micromamba","run","-n","appenv","python","-u","/app/handler.py"]

#CMD ["python","-u","/app/handler.py"]