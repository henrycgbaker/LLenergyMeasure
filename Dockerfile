# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

# Install Python 3.11 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /build

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip poetry-core build

# Install PyTorch with CUDA 12.4 (matches base image)
# This prevents pip from installing torch with mismatched CUDA version
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy project files
COPY pyproject.toml poetry.lock* README.md ./
COPY src/ ./src/

# Install the package with remaining dependencies (torch already installed)
RUN pip install --no-cache-dir .

# ============================================================================
# Stage 2: Runtime - Same base, minimal additions
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Install Python 3.11 runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Run as host user via compose `user:` directive (no baked-in user)
WORKDIR /app

# Copy virtual environment from builder (root-owned, readable by any user)
COPY --from=builder /opt/venv /opt/venv

# Set environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Clear NVIDIA's entrypoint (suppresses startup banner) and set default command
ENTRYPOINT []
CMD ["llm-energy-measure", "--help"]

# ============================================================================
# Stage 3: Dev - For VS Code devcontainer (runs as root for simplicity)
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS dev

# Install Python 3.11 and dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Create venv and install base dependencies (torch with CUDA)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && chmod -R 777 /opt/venv

# Copy source (will be overridden by workspace mount in devcontainer)
COPY . /app/

# Default to bash for interactive dev sessions
CMD ["/bin/bash"]
