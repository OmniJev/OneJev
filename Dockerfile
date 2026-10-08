# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server

# --- base: qev core, no PyTorch. Parent of the torch and rocm stages. ---
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

ARG TARGETARCH
# Empty picks cu128 on x86_64 and cu130 on arm64 (cu128 cannot build kernels for DGX Spark).
ARG TORCH_INDEX_URL=

USER root

RUN url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$([ "$TARGETARCH" = arm64 ] && echo cu130 || echo cu128)}" \
    && pip install --index-url "$url" torch torchvision \
    && pip install --extra-index-url "$url" ".[torch]"

USER qev

# --- rocm: AMD / ROCm server (images, video, text). ---
FROM base AS rocm

ARG TORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url "${TORCH_ROCM_INDEX_URL}" torch torchvision \
    && pip install --extra-index-url "${TORCH_ROCM_INDEX_URL}" ".[torch]"

USER qev

# --- gguf: official llama.cpp server image plus qev (GGUF models, images and text). ---
FROM ${LLAMA_IMAGE} AS gguf

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    PATH=/opt/qev/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY pyproject.toml README.md ./
COPY qev ./qev

RUN python3 -m venv /opt/qev \
    && pip install . \
    && mkdir -p /data/hf \
    && chown -R 1000:1000 /data/hf

WORKDIR /app

USER 1000

EXPOSE 8000

HEALTHCHECK NONE

ENTRYPOINT ["qev"]
CMD ["--help"]
