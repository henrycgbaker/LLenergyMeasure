# Deployment Guide

Docker deployment, GPU configuration, and troubleshooting.

## Running Modes

| Mode | Best For | Setup |
|------|----------|-------|
| **Host (pip)** | Quick local dev | `pip install -e ".[pytorch]"` |
| **Docker prod** | Reproducible runs | `docker compose build base pytorch` |
| **Docker dev** | Test in container | `docker compose --profile dev build` |
| **VS Code devcontainer** | Full IDE + GPU | "Reopen in Container" |

## Host Installation

```bash
# Install with PyTorch backend (default)
pip install -e ".[pytorch]"

# Or with vLLM backend (high throughput)
pip install -e ".[vllm]"

# Or with TensorRT backend (requires Ampere+ GPU)
pip install -e ".[tensorrt]"

# Run experiment
llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100
```

## Docker

### Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA 12.4** compatible drivers (base image uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- **nvidia-container-toolkit** installed and configured
- **Privileged mode** for energy metrics (docker-compose.yml sets `privileged: true`)

### Backend Services

The project provides separate Docker services for each inference backend:

| Service | Image Tag | Use Case |
|---------|-----------|----------|
| `pytorch` | `llm-energy-measure:pytorch` | Default, most compatible |
| `vllm` | `llm-energy-measure:vllm` | High throughput with PagedAttention |
| `tensorrt` | `llm-energy-measure:tensorrt` | Maximum performance (Ampere+ GPUs) |
| `pytorch-dev` | `llm-energy-measure:pytorch-dev` | Development with PyTorch |
| `vllm-dev` | `llm-energy-measure:vllm-dev` | Development with vLLM |
| `tensorrt-dev` | `llm-energy-measure:tensorrt-dev` | Development with TensorRT |

**Note**: vLLM and TensorRT have conflicting PyTorch dependencies. Use separate images rather than installing multiple backends in one environment.

### Environment Variables

Create a `.env` file (automatically loaded via dotenv):

```bash
# Required for gated models (Llama, etc.)
HF_TOKEN=your_huggingface_token

# Run as your host user (PUID/PGID pattern like LinuxServer.io)
# This ensures files created in mounted volumes are owned by you
PUID=1000  # Your user ID (run 'id -u' to get this)
PGID=1000  # Your group ID (run 'id -g' to get this)

# Optional: GPU selection
CUDA_VISIBLE_DEVICES=0,1

# Optional: CodeCarbon logging level
CODECARBON_LOG_LEVEL=warning

# Optional: Custom paths (defaults shown)
# LLM_ENERGY_CONFIGS_DIR=./configs
# LLM_ENERGY_RESULTS_DIR=./results
# HF_HOME=~/.cache/huggingface
# TRT_ENGINE_CACHE=~/.cache/tensorrt-engines  # TensorRT only
```

**PUID/PGID Pattern**: The container starts as root (needed for GPU access and initialisation), then the entrypoint script drops privileges to your specified user before running the application. If not set, PUID/PGID are auto-detected from the mounted `/app/results` directory ownership.

**Results directory precedence:**
1. `--results-dir` CLI flag (highest)
2. `io.results_dir` in config YAML
3. `LLM_ENERGY_RESULTS_DIR` environment variable
4. Default `results/` (lowest)

### Makefile (Recommended)

The Makefile handles UID/GID mapping automatically:

```bash
# Build backends
make docker-build-pytorch     # Build PyTorch backend (default)
make docker-build-vllm        # Build vLLM backend
make docker-build-tensorrt    # Build TensorRT backend
make docker-build-all         # Build all backends
make docker-build-dev         # Build PyTorch dev image

# Run commands
make datasets                 # List available datasets
make validate CONFIG=test.yaml # Validate a config
make experiment CONFIG=test.yaml DATASET=alpaca SAMPLES=100
make lem CMD="results list"   # Run any lem command

# Interactive shells
make docker-shell             # Production shell (pytorch)
make docker-dev               # Development shell (pytorch-dev)
```

### Docker Compose

#### Building Images

Build the base image first, then the backend you need:

```bash
# Build base + PyTorch (default)
docker compose build base pytorch

# Build base + vLLM
docker compose build base vllm

# Build base + TensorRT
docker compose build base tensorrt

# Build all backends
docker compose build base pytorch vllm tensorrt

# Build dev images (include --profile dev)
docker compose --profile dev build base pytorch-dev
```

#### Running Experiments

```bash
# PyTorch backend (default)
docker compose run --rm pytorch \
  llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100

# vLLM backend
docker compose run --rm vllm \
  llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100

# TensorRT backend
docker compose run --rm tensorrt \
  llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100

# Interactive shell
docker compose run --rm pytorch /bin/bash
```

#### Development Mode

Development containers mount the source code for live editing:

```bash
# Build dev image
docker compose --profile dev build base pytorch-dev

# Interactive shell (source mounted, editable install)
docker compose --profile dev run --rm pytorch-dev

# Run command in dev container
docker compose --profile dev run --rm pytorch-dev \
  llm-energy-measure experiment /app/configs/test.yaml -d alpaca -n 10
```

### VS Code Devcontainer

1. Install [VS Code Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Set `HF_TOKEN` in your shell environment
3. Open project and use `Ctrl+Shift+P` then "Dev Containers: Reopen in Container"

The devcontainer:
- Runs as root (avoids permission issues)
- Mounts source code (edits sync instantly)
- Mounts HuggingFace cache (models persist)
- Has GPU passthrough enabled

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `~/.cache/huggingface` | `/app/.cache/huggingface` | Model cache |
| `./configs` | `/app/configs` | Experiment configs |
| `./results` | `/app/results` | Output results |
| `./scripts` | `/app/scripts` (ro) | Entrypoint scripts |
| (created in image) | `/app/.state` | Experiment state/resumption |
| `~/.cache/tensorrt-engines` | `/app/.cache/tensorrt-engines` | TensorRT engine cache (tensorrt only) |

### Backend-Specific Notes

#### vLLM

- Uses `ipc: host` for shared memory (required for multiprocessing)
- Installs its own PyTorch version (2.8+) for compatibility

#### TensorRT

- Uses `ipc: host` for shared memory (required for multiprocessing)
- Requires compute capability >= 8.0 (Ampere: A100, A10, RTX 30xx/40xx, H100, L40)
- NOT supported: V100, T4, RTX 20xx, GTX series
- Mounts engine cache for compiled TensorRT engines
- Requires MPI libraries (included in image)

## MIG GPU Support

NVIDIA MIG (Multi-Instance GPU) allows partitioning A100/H100 GPUs into isolated instances.

### What Works

- Single-process experiments on individual MIG instances
- Parallel independent experiments on different MIG instances
- MIG detection and metadata recording (`gpu_is_mig`, `gpu_mig_profile`)

### What Does NOT Work

- **Multi-process distributed inference** across MIG instances (hardware limitation)
- Using parent GPU index when MIG is enabled (must use UUID)

### Usage

```bash
# 1. List MIG instances
nvidia-smi -L

# Example output:
# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-xxx)
#   MIG 3g.20gb Device 0: (UUID: MIG-abc123)
#   MIG 3g.20gb Device 1: (UUID: MIG-def456)

# 2. Run on specific MIG instance
CUDA_VISIBLE_DEVICES=MIG-abc123 llm-energy-measure experiment config.yaml --dataset alpaca -n 100

# 3. Parallel experiments (separate terminals)
# Terminal 1:
CUDA_VISIBLE_DEVICES=MIG-abc123 llm-energy-measure experiment config1.yaml --dataset alpaca -n 100
# Terminal 2:
CUDA_VISIBLE_DEVICES=MIG-def456 llm-energy-measure experiment config2.yaml --dataset alpaca -n 100
```

### MIG with Docker

```bash
# Use specific MIG instance
CUDA_VISIBLE_DEVICES=MIG-abc123 docker compose run --rm pytorch \
  llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100

# Or set in .env
echo "CUDA_VISIBLE_DEVICES=MIG-abc123" >> .env
```

### Energy Measurement Limitation

Energy readings on MIG instances reflect the **parent GPU's total power**, not per-instance usage. This is a hardware/driver limitation. Results include `energy_measurement_warning` when running on MIG.

For accurate energy measurements:
- Use full GPUs (disable MIG)
- Or ensure no other workloads on sibling MIG instances

## Troubleshooting

### Energy Metrics are Zero

**Cause**: CodeCarbon needs NVML access, which requires privileged mode.

**Solution**: Ensure `privileged: true` in docker-compose.yml. Without compose:
```bash
docker run --privileged --gpus all ...
```

### CUDA Version Mismatch

**Symptom**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**: Ensure NVIDIA drivers support CUDA 12.4+:
```bash
nvidia-smi  # Check driver version
```

### Permission Denied on Results/State

The Docker image uses PUID/PGID to run as your host user. The entrypoint auto-detects ownership from the mounted `/app/results` directory, but you can also set explicitly:

```bash
# Option 1: Set in .env file
PUID=1000  # Replace with your user ID (run 'id -u')
PGID=1000  # Replace with your group ID (run 'id -g')

# Option 2: Pass inline
PUID=$(id -u) PGID=$(id -g) docker compose run --rm pytorch ...
```

If you previously ran as root and have root-owned files:

```bash
# Fix ownership (requires sudo or use Docker)
docker run --rm -v $(pwd)/results:/results alpine chown -R $(id -u):$(id -g) /results
```

Makefile commands (`make docker-dev`, `make experiment`) handle PUID/PGID automatically.

### MIG Device Errors

**Symptom**: `RuntimeError: CUDA error: invalid device ordinal` or `NCCL error`

**Cause**: MIG instances are hardware-isolated.

**Solutions**:

1. **Single-process on MIG** (recommended):
   ```bash
   CUDA_VISIBLE_DEVICES=MIG-abc123 llm-energy-measure experiment config.yaml --dataset alpaca -n 100
   ```

2. **Distributed inference** - use full GPUs (non-MIG)

3. **Parallel independent experiments** on separate MIG instances

### Model Downloads Every Run

**Cause**: HF cache not mounted to container.

**Solution**: The default docker-compose.yml mounts `~/.cache/huggingface`. Ensure this path exists on your host:

```bash
mkdir -p ~/.cache/huggingface
```

Or set a custom path:
```bash
export HF_HOME=/path/to/cache
docker compose run --rm pytorch ...
```

### Config Validation Errors

**Solutions**:
- Ensure YAML format (not Python)
- Validate first: `llm-energy-measure config validate config.yaml`
- Check field names match schema

### Out of Memory (OOM)

**Solutions**:
- Reduce `batching.batch_size`
- Reduce `max_input_tokens` / `max_output_tokens`
- Enable quantization: `quantization.load_in_8bit: true`
- Lower precision: `fp_precision: float16`

### vLLM/TensorRT Shared Memory Errors

**Symptom**: Errors about shared memory or IPC

**Cause**: vLLM and TensorRT require shared memory for multiprocessing.

**Solution**: The docker-compose.yml sets `ipc: host` for these services. If running manually:
```bash
docker run --ipc=host --gpus all ...
```
