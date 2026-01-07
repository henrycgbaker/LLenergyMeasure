# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

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

# Install PyTorch with CUDA 12.1 (matches base image)
# This prevents pip from installing torch with mismatched CUDA version
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY pyproject.toml poetry.lock* README.md ./
COPY src/ ./src/

# Install the package with remaining dependencies (torch already installed)
RUN pip install --no-cache-dir .

# ============================================================================
# Stage 2: Runtime - Same base, minimal additions
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Install Python 3.11 runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=app:app /opt/venv /opt/venv

# Set environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers

# Default command
ENTRYPOINT ["llm-energy-measure"]
CMD ["--help"]

# ============================================================================
# Stage 3: Dev - For VS Code devcontainer
# ============================================================================
FROM runtime AS dev

USER root

# Install dev tools (git for version control, curl for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER app

# Copy source for editable install (will be overridden by workspace mount)
COPY --chown=app:app . /app/

# Default to bash for interactive dev sessions
CMD ["/bin/bash"]
