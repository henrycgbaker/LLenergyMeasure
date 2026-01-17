# Deployment Guide

Docker deployment, GPU configuration, and troubleshooting.

## Running Modes

| Mode | Best For | Setup |
|------|----------|-------|
| **Host (Poetry)** | Quick local dev | `poetry install --with dev` |
| **Docker prod** | Reproducible runs | `docker compose build` |
| **Docker dev** | Test in container | `docker compose --profile dev build` |
| **VS Code devcontainer** | Full IDE + GPU | "Reopen in Container" |

## Host Installation

```bash
poetry install --with dev
poetry run llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100

# Or activate venv first:
poetry shell
llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100
```

## Docker

### Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA 12.4** compatible drivers (image uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- **nvidia-container-toolkit** installed and configured
- **Privileged mode** for energy metrics (docker-compose.yml sets `privileged: true`)

### Environment Variables

Create a `.env` file:

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
# LLM_ENERGY_RESULTS_DIR=/app/results
# LLM_ENERGY_STATE_DIR=/app/.state
```

**PUID/PGID Pattern**: The container starts as root (needed for GPU access and initialisation), then the entrypoint script drops privileges to your specified user before running the application. This ensures all files created in mounted volumes are owned by your host user, not root.

### Makefile (Recommended)

The Makefile handles UID/GID mapping automatically:

```bash
make docker-build              # Build the image
make datasets                  # List available datasets
make validate CONFIG=test.yaml # Validate a config
make experiment CONFIG=test.yaml DATASET=alpaca SAMPLES=100
make docker-shell              # Interactive shell
make docker-dev                # Development shell
```

### Docker Compose

The project uses profiles to separate production and development:

| Service | Profile | Purpose |
|---------|---------|---------|
| `llm-energy-measure-app` | (default) | Production - baked-in package |
| `llm-energy-measure-dev` | `dev` | Development - editable install |

#### Production

```bash
docker compose build

# Run experiment
docker compose run --rm llm-energy-measure-app \
  llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100

# Interactive shell
docker compose run --rm llm-energy-measure-app /bin/bash
```

#### Development

```bash
docker compose --profile dev build

# Interactive shell (source mounted)
docker compose --profile dev run --rm llm-energy-measure-dev

# Run command
docker compose --profile dev run --rm llm-energy-measure-dev \
  llm-energy-measure experiment /app/configs/test.yaml -d alpaca -n 10
```

### VS Code Devcontainer

1. Install [VS Code Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Set `HF_TOKEN` in your shell environment
3. Open project → `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

The devcontainer:
- Runs as root (avoids permission issues)
- Mounts source code (edits sync instantly)
- Mounts HuggingFace cache (models persist)
- Has GPU passthrough enabled

### Persistent Model Cache

By default, production containers lose downloaded models on exit. To persist:

```bash
# Option 1: Uncomment in docker-compose.yml
# - ${HF_HOME:-~/.cache/huggingface}:/app/.cache/huggingface

# Option 2: Use dev profile (auto-mounts cache)
make docker-dev
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `~/.cache/huggingface` | `/app/.cache/huggingface` | Model cache (optional) |
| `./configs` | `/app/configs` (ro) | Experiment configs |
| `./results` | `/app/results` | Output results |
| (created in image) | `/app/.state` | Experiment state/resumption |

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
CUDA_VISIBLE_DEVICES=MIG-abc123 docker compose run --rm llm-energy-measure-app \
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

The Docker image uses PUID/PGID to run as your host user. Set up your `.env` file:

```bash
# In your .env file
PUID=1000  # Replace with your user ID (run 'id -u')
PGID=1000  # Replace with your group ID (run 'id -g')
```

Or pass inline:

```bash
PUID=$(id -u) PGID=$(id -g) docker compose run --rm pytorch ...
```

If you previously ran as root and have root-owned files:

```bash
# Fix ownership (requires sudo or use Docker)
docker run --rm -v $(pwd)/results:/results alpine chown -R $(id -u):$(id -g) /results
```

Makefile commands (`make docker-dev`) handle PUID/PGID automatically.

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

**Cause**: HF cache is ephemeral in production containers.

**Solution**: Use dev profile or mount cache. See [Persistent Model Cache](#persistent-model-cache).

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
