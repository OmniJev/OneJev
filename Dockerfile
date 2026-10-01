# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

# --- base: qev core, no PyTorch. Used by the llama.cpp (GGUF) backend. ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY qev ./qev

RUN pip install . \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin qev \
    && mkdir -p /data/hf \
    && chown -R qev:qev /data/hf

USER qev

EXPOSE 8000

ENTRYPOINT ["qev"]
CMD ["--help"]

# --- torch: NVIDIA / CUDA server (images, video, text). ---
FROM base AS torch

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url "${TORCH_INDEX_URL}" torch torchvision \
    && pip install --extra-index-url "${TORCH_INDEX_URL}" ".[torch]"

USER qev

# --- rocm: AMD / ROCm server (images, video, text). ---
FROM base AS rocm

ARG TORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url "${TORCH_ROCM_INDEX_URL}" torch torchvision \
    && pip install --extra-index-url "${TORCH_ROCM_INDEX_URL}" ".[torch]"

USER qev
