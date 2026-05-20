# syntax=docker/dockerfile:1.7
# Multi-stage build for x86_64 CPU-only signature cropper.
# Final image target: < 700 MB. Pin Python to 3.11 for ABI parity with
# onnxruntime-openvino wheels.

ARG PYTHON_VERSION=3.11.10

# ----- builder -----
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip wheel \
 && /opt/venv/bin/pip install .

# Bake the off-the-shelf detector weights into the image (no network at runtime).
ARG SIGCROP_SKIP_MODEL_FETCH=0
RUN if [ "$SIGCROP_SKIP_MODEL_FETCH" = "0" ]; then \
      mkdir -p /opt/sigcrop/models \
      && SIGCROP_MODEL_DIR=/opt/sigcrop/models /opt/venv/bin/python scripts/fetch_pretrained.py ; \
    else \
      echo "Skipping model fetch (SIGCROP_SKIP_MODEL_FETCH=1)"; \
      mkdir -p /opt/sigcrop/models ; \
    fi

# ----- runtime -----
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SIGCROP_MODEL_DIR=/opt/sigcrop/models \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sigcrop

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/sigcrop/models ${SIGCROP_MODEL_DIR}

USER sigcrop
WORKDIR /home/sigcrop

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/readyz || exit 1

CMD ["sigcrop-api"]
