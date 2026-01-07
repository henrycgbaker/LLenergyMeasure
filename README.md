# LLM Energy Measure

A Python framework for measuring LLM inference efficiency, including energy consumption, throughput, and computational metrics (FLOPs). Designed for distributed GPU benchmarking using HuggingFace models.

## Features

- **Energy Measurement**: Track GPU, CPU, and RAM energy consumption via CodeCarbon
- **Throughput Metrics**: Tokens/second, latency per token, batch processing stats
- **FLOPs Estimation**: Multiple estimation methods (calflops, architecture-based, parameter-based)
- **Multi-GPU Support**: Distributed inference via `accelerate` with per-process result tracking
- **Late Aggregation**: Raw per-GPU results are saved separately and aggregated on demand
- **Flexible Configuration**: YAML/JSON configs with inheritance support
- **Docker Ready**: GPU-enabled containerization with docker-compose

## Installation

### Local (Poetry)

```bash
# Install dependencies
poetry install

# With development dependencies
poetry install --with dev
```

### Docker

```bash
# Build the image
docker compose build

# Run with GPU access
docker compose run --rm llm-energy-measure-app llm-energy-measure --help
```

## Quick Start

### 1. Create a Configuration

Create a YAML config file (e.g., `configs/my_experiment.yaml`):

```yaml
config_name: llama2-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf

# Token limits
max_input_tokens: 512
max_output_tokens: 128

# Distributed setup
gpu_list: [0, 1]
num_processes: 2

# Precision
fp_precision: float16
```

### 2. Validate Configuration

```bash
llm-energy-measure config validate configs/my_experiment.yaml
```

### 3. Run Experiment

The `experiment` command handles `accelerate launch` automatically, reading `num_processes` from your config:

```bash
# Using built-in HuggingFace datasets (recommended)
llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100

# Using a prompts file
llm-energy-measure experiment configs/my_experiment.yaml --prompts prompts.txt
```

Or use `accelerate launch` directly for more control:

```bash
accelerate launch --num_processes 2 \
  -m llm_energy_measure.orchestration.launcher \
  --config configs/my_experiment.yaml \
  --dataset alpaca -n 100
```

### 4. Aggregate Results

After running, aggregate per-process results:

```bash
# Aggregate a specific experiment
llm-energy-measure aggregate exp_20240115_123456

# Aggregate all pending experiments
llm-energy-measure aggregate --all
```

### 5. View Results

```bash
# List all experiments
llm-energy-measure results list

# Show aggregated results
llm-energy-measure results show exp_20240115_123456

# Show per-process raw results
llm-energy-measure results show exp_20240115_123456 --raw

# Export as JSON
llm-energy-measure results show exp_20240115_123456 --json
```

## CLI Reference

```
llm-energy-measure [OPTIONS] COMMAND [ARGS]

Commands:
  experiment  Run experiment (wraps accelerate automatically)
  run         Run inference (called by accelerate launch)
  aggregate   Aggregate raw per-process results
  datasets    List available built-in datasets
  config      Configuration management
    validate  Validate a config file
    show      Display resolved configuration
  results     Results inspection
    list      List all experiments
    show      Show experiment results
```

### Built-in Datasets

Use HuggingFace datasets as prompt sources instead of text files:

```bash
# List available built-in datasets
llm-energy-measure datasets

# Run with built-in dataset
llm-energy-measure run --config config.yaml --dataset alpaca -n 100

# Use any HuggingFace dataset
llm-energy-measure run --config config.yaml \
  --dataset squad --split validation --column question -n 50
```

| Dataset | Source | Default Column |
|---------|--------|----------------|
| `alpaca` | tatsu-lab/alpaca | instruction |
| `sharegpt` | ShareGPT_Vicuna | conversations |
| `gsm8k` | gsm8k (main) | question |
| `mmlu` | cais/mmlu (all) | question |

### Global Options

| Option | Description |
|--------|-------------|
| `--version, -v` | Show version |
| `--verbose` | Enable debug logging |

## Configuration

The CLI expects **YAML** configuration files. The Python files in `configs/` are legacy research configs - see `configs/README.md` for migration instructions.

### Config Structure

```yaml
# Identity
config_name: experiment-name  # Required
model_name: org/model-name    # Required (HuggingFace path)

# Model properties
is_encoder_decoder: false
task_type: text_generation    # text_generation | translation | summarisation
inference_type: pure_generative

# Token limits
max_input_tokens: 512
max_output_tokens: 128
min_output_tokens: 0

# Input
num_input_prompts: 100
save_outputs: false

# Distributed
gpu_list: [0, 1, 2, 3]
num_processes: 4

# Batching
batching_options:
  batch_size: 8
  dynamic_batching: false

# Generation
decoder_config:
  temperature: 1.0
  top_p: 1.0
  top_k: 50

# Precision & Quantization
fp_precision: float16         # float32 | float16 | bfloat16
quantization_config:
  quantization: false
  load_in_4bit: false
  load_in_8bit: false
```

### Config Inheritance

Configs can extend base configs using `_extends`:

```yaml
# configs/base.yaml
max_input_tokens: 512
fp_precision: float16

# configs/llama2-7b.yaml
_extends: base.yaml
config_name: llama2-7b
model_name: meta-llama/Llama-2-7b-hf
```

## Docker Usage

### Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA 12.4** compatible drivers (image uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- **nvidia-container-toolkit** installed and configured
- **Privileged mode** for energy metrics (docker-compose.yml sets `privileged: true`)

### Environment Variables

Create a `.env` file with:

```bash
# Required for gated models (Llama, etc.)
HF_TOKEN=your_huggingface_token

# Optional: GPU selection (see MIG notes below)
CUDA_VISIBLE_DEVICES=0,1

# Optional: CodeCarbon logging level
CODECARBON_LOG_LEVEL=warning
```

### Quick Start with Makefile

The easiest way to run experiments in Docker:

```bash
# Build the image
make docker-build

# List available datasets
make datasets

# Validate a config
make validate CONFIG=test_tiny.yaml

# Run experiment (num_processes auto-inferred from config)
make experiment CONFIG=test_tiny.yaml DATASET=alpaca SAMPLES=100

# Interactive shell in container
make docker-shell
```

### Running with Docker Compose

The project uses Docker Compose **profiles** to separate production and development environments:

| Service | Profile | Purpose |
|---------|---------|---------|
| `llm-energy-measure-app` | (default) | Production - baked-in package |
| `llm-energy-measure-dev` | `dev` | Development - editable install with source mount |

#### Production Commands

```bash
# Build the image
docker compose build

# List available datasets
docker compose run --rm llm-energy-measure-app llm-energy-measure datasets

# Validate config
docker compose run --rm llm-energy-measure-app \
  llm-energy-measure config validate /app/configs/test_tiny.yaml

# Run experiment (recommended - auto-handles accelerate)
docker compose run --rm llm-energy-measure-app \
  llm-energy-measure experiment /app/configs/test_tiny.yaml \
  --dataset alpaca -n 100

# Or use accelerate launch directly for more control
docker compose run --rm llm-energy-measure-app \
  accelerate launch --num_processes 1 \
  -m llm_energy_measure.orchestration.launcher \
  --config /app/configs/test_tiny.yaml \
  --dataset alpaca -n 100

# View results
docker compose run --rm llm-energy-measure-app llm-energy-measure results list

# Aggregate results
docker compose run --rm llm-energy-measure-app llm-energy-measure aggregate --all

# Interactive shell
docker compose run --rm llm-energy-measure-app /bin/bash
```

#### Development Commands

```bash
# Build dev image
docker compose --profile dev build

# Interactive dev shell (source mounted, editable install)
docker compose --profile dev run --rm llm-energy-measure-dev

# Run commands in dev container
docker compose --profile dev run --rm llm-energy-measure-dev \
  llm-energy-measure experiment /app/configs/test_tiny.yaml -d alpaca -n 10
```

### Persistent Model Cache

By default in the production container, models download inside the container and are **lost when it exits**. To persist downloaded models:

```bash
# Option 1: Uncomment the HF cache mount in docker-compose.yml
# - ${HF_HOME:-~/.cache/huggingface}:/home/app/.cache/huggingface

# Option 2: Use the dev profile (auto-mounts HF cache)
docker compose --profile dev run --rm llm-energy-measure-dev \
  llm-energy-measure experiment /app/configs/test_tiny.yaml -d alpaca -n 100
```

**Note**: The dev service automatically mounts `~/.cache/huggingface` for persistent model caching.

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `~/.cache/huggingface` | `/tmp/hf_cache` | Model cache (optional mount) |
| `./configs` | `/app/configs` (ro) | Experiment configs |
| `./results` | `/app/results` | Output results |

### MIG GPU Considerations

On servers with **MIG (Multi-Instance GPU)** enabled GPUs (common on A100s), you may encounter device enumeration issues. Workarounds:

```bash
# Force use of physical GPU 0 (non-MIG device)
CUDA_VISIBLE_DEVICES=0 docker compose run --rm bench ...

# Or set in your .env file
echo "CUDA_VISIBLE_DEVICES=0" >> .env
```

## Results Structure

```
results/
├── raw/
│   └── exp_20240115_123456/
│       ├── process_0.json    # GPU 0 results
│       ├── process_1.json    # GPU 1 results
│       └── ...
└── aggregated/
    └── exp_20240115_123456.json
```

### Metrics Collected

| Category | Metrics |
|----------|---------|
| **Inference** | total_tokens, tokens_per_second, latency_per_token_ms |
| **Energy** | total_energy_j, gpu_energy_j, cpu_energy_j, emissions_kg_co2 |
| **Compute** | flops_total, flops_per_token, peak_memory_mb |

## Troubleshooting

### Energy Metrics are Zero

**Symptom**: `energy_metrics.total_energy_j` and related fields are all `0.0`.

**Cause**: CodeCarbon requires NVML access to read GPU power, which needs privileged mode in Docker.

**Solution**: Ensure `privileged: true` is set in docker-compose.yml (already configured by default). If running without docker-compose:
```bash
docker run --privileged --gpus all ...
```

### CUDA Version Mismatch

**Symptom**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Cause**: Host CUDA drivers incompatible with container's CUDA 12.4.

**Solution**: Ensure NVIDIA drivers support CUDA 12.4+. Check with:
```bash
nvidia-smi  # Shows driver version and max CUDA version
```

### Permission Denied on Results

**Symptom**: `PermissionError: [Errno 13] Permission denied: 'results/raw'`

**Cause**: The container runs as user `app` but the host `results/` directory doesn't exist or has restrictive permissions.

**Solution**: Create the results directory with appropriate permissions:
```bash
mkdir -p results
chmod 777 results  # Or use more restrictive perms with matching UID
```

Or run the container with host user ID:
```bash
docker compose run --rm --user "$(id -u):$(id -g)" llm-energy-measure-app ...
```

### MIG Device Errors

**Symptom**: `RuntimeError: CUDA error: invalid device ordinal` or only seeing MIG instances.

**Cause**: A100 or other GPUs with MIG enabled expose virtual devices differently.

**Solution**: Force use of physical GPU:
```bash
CUDA_VISIBLE_DEVICES=0 docker compose run --rm llm-energy-measure-app ...
```

### Model Download Every Run

**Symptom**: Models re-download each container run.

**Cause**: HF cache inside production container is ephemeral.

**Solution**: Use the dev profile which auto-mounts HF cache, or uncomment the cache mount in docker-compose.yml. See [Persistent Model Cache](#persistent-model-cache).

### Config Validation Errors

**Symptom**: `ValidationError` when loading config.

**Solutions**:
- Ensure you're using YAML format (not Python config files)
- Validate config before running: `llm-energy-measure config validate configs/your_config.yaml`
- Check field names match the schema in [Config Structure](#config-structure)

### Out of Memory (OOM)

**Symptom**: `CUDA out of memory` error.

**Solutions**:
- Reduce `batching_options.batch_size`
- Reduce `max_input_tokens` and `max_output_tokens`
- Use quantization: set `quantization_config.load_in_8bit: true`
- Use lower precision: `fp_precision: float16` or `bfloat16`

## Development

### Local Development

```bash
# Install dev dependencies
make dev

# Run checks (format, lint, typecheck)
make check

# Run tests
make test              # Unit tests only
make test-integration  # Integration tests
make test-all          # All tests
```

### VS Code Devcontainer

For development inside a container with full GPU access:

1. Install [VS Code Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Set `HF_TOKEN` in your shell environment
3. Open the project in VS Code and click "Reopen in Container"

The devcontainer:
- Uses the `dev` stage of the Dockerfile
- Has GPU passthrough enabled
- Mounts your host HuggingFace cache
- Installs dev dependencies and pre-commit hooks

Inside the container, run commands directly:

```bash
llm-energy-measure datasets
llm-energy-measure config validate configs/test_tiny.yaml
llm-energy-measure experiment configs/test_tiny.yaml --dataset alpaca -n 100
```

## Project Structure

```
src/llm_energy_measure/
├── cli.py                 # Typer CLI application
├── config/                # Configuration loading & models
├── core/                  # Inference engine & metrics
├── domain/                # Domain models (Pydantic)
├── orchestration/         # Experiment lifecycle management
└── results/               # Results persistence & aggregation
```

See directory-level README.md files for detailed module documentation.

## License

MIT
