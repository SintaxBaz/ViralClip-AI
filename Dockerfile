# syntax=docker/dockerfile:1

# ViralClip-AI container.
#
# Three things dominate the design:
#
#   1. ffmpeg is a system binary, not a pip package, and is needed at
#      RUNTIME -- not just to build. It goes in the final stage.
#   2. The default torch wheel on Linux is a CUDA build of roughly 2.5 GB.
#      This image installs the CPU wheel from PyTorch's cpu index instead,
#      which is around 200 MB. Build with --build-arg TORCH_INDEX= to get
#      the CUDA wheel back for a GPU host.
#   3. Model weights (~240 MB) are downloaded at BUILD time, so they sit in
#      an image layer. Otherwise the first request to every replica pays
#      for the download, and the image cannot start at all on a host
#      without egress to HuggingFace.
#
# Build:  docker build -t viralclip .
# Run:    docker run --rm -p 8000:8000 -v viralclip-data:/data viralclip

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder
# Compiles and downloads everything into a venv and a model cache, both of
# which are copied into the runtime stage. Build toolchains never ship.
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    XDG_CACHE_HOME=/opt/cache \
    HF_HOME=/opt/cache/huggingface

# Build-time only: some wheels in this tree still compile C extensions.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

WORKDIR /build

# Install torch on its own, from the CPU index. It must be a separate step
# because --index-url replaces PyPI for the whole command, and nothing else
# here is published on PyTorch's index.
ARG TORCH_VERSION=">=2.0.0,<3"
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# hadolint ignore=DL3013
RUN if [ -n "${TORCH_INDEX}" ]; then \
        pip install --no-cache-dir --index-url "${TORCH_INDEX}" "torch${TORCH_VERSION}"; \
    else \
        pip install --no-cache-dir "torch${TORCH_VERSION}"; \
    fi

# Requirements before source, so editing a .py file does not reinstall
# torch. This is the difference between a 2-second and a 10-minute rebuild.
COPY requirements.txt requirements-diarize.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Diarization is off by default: it roughly doubles the image and needs a
# HuggingFace token. Enable with --build-arg WITH_DIARIZATION=true
ARG WITH_DIARIZATION=false
RUN if [ "${WITH_DIARIZATION}" = "true" ]; then \
        pip install --no-cache-dir -r requirements-diarize.txt; \
    fi

# Bake the weights into a layer.
ARG WHISPER_MODEL=base
ENV WHISPER_MODEL=${WHISPER_MODEL}
COPY docker/preload_models.py ./docker/
RUN python docker/preload_models.py

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

# ffmpeg is required to cut clips. /healthz reports 503 without it.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG WHISPER_MODEL=base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    XDG_CACHE_HOME=/opt/cache \
    HF_HOME=/opt/cache/huggingface \
    VIRALCLIP_OUTPUT_ROOT=/data/jobs \
    VIRALCLIP_WHISPER_MODEL=${WHISPER_MODEL}

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/cache /opt/cache

# Run as a non-root user. /data is a volume mount point, so it must be
# writable by that user; /opt/cache is chowned too so a different
# VIRALCLIP_WHISPER_MODEL can still be fetched at runtime.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data/jobs \
    && chown -R app:app /data /opt/cache

WORKDIR /app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/

USER app

VOLUME ["/data"]
EXPOSE 8000

# Uses the app's own readiness check, which also verifies ffmpeg. urllib
# rather than curl, so the image needs no extra package. start-period
# covers loading the baked models at startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]

# 0.0.0.0, not 127.0.0.1: a server bound to loopback inside a container is
# unreachable from the host even with -p published.
CMD ["uvicorn", "api.app:app", \
     "--app-dir", "src", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
