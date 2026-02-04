# Getting Started

This guide walks you through your first LLM efficiency measurement experiment.

## Quick Start (5 minutes)

```bash
# Clone and install
git clone https://github.com/henrycgbaker/LLenergyMeasure
cd LLenergyMeasure
pip install -e .

# Run setup wizard (detects environment, creates config)
lem init

# Run your first experiment
lem experiment configs/examples/pytorch_example.yaml -n 10

# Or run a multi-config campaign
lem campaign configs/examples/campaign_example.yaml --dry-run
```

That's it! Base install includes PyTorch backend — no extras needed.

The `lem init` wizard:
- Detects your environment (GPU, Docker, installed backends)
- Creates `.lem-config.yaml` with your preferences
- Explains multi-backend setup (vLLM/TensorRT require Docker)
- Runs `lem doctor` to verify your setup

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **Docker** with nvidia-container-toolkit (recommended)
- **HuggingFace account** for gated models (Llama, Mistral, etc.)

## Understanding Backends

This tool supports three inference backends with different trade-offs:

| Backend | Requirements | Best For |
|---------|--------------|----------|
| **PyTorch** | Any NVIDIA GPU | Development, flexibility, widest compatibility |
| **vLLM** | NVIDIA GPU, Linux only | High throughput, continuous batching |
| **TensorRT** | Ampere+ GPU¹, CUDA 12.x, Linux | Maximum performance (compiled inference) |

¹ Ampere+ = compute capability ≥ 8.0: A100, A10, RTX 30xx/40xx, H100, L40. **NOT supported**: V100, T4, RTX 20xx, GTX.

**Important**: vLLM and TensorRT-LLM have **conflicting dependencies** and cannot coexist in the same Python environment. Docker isolates each backend.

---

## Installation Options

### Option 1: Local Install (Recommended)

The simplest way to get started for single-backend experiments.

```bash
# Clone
git clone https://github.com/henrycgbaker/LLenergyMeasure
cd LLenergyMeasure

# Install
pip install -e .                    # Includes PyTorch backend
pip install -e ".[vllm]"            # Or vLLM (Linux only)
pip install -e ".[tensorrt]"        # Or TensorRT (Ampere+ GPU)

# Check setup
lem doctor

# Run experiment
lem experiment configs/examples/pytorch_example.yaml -n 10
```

**Switching backends locally**: Create separate conda environments:
```bash
conda create -n llm-vllm python=3.10 && conda activate llm-vllm
pip install -e ".[vllm]"
```

### Option 2: Docker for Multi-Backend Campaigns

Use Docker when running campaigns that span multiple backends (vLLM and TensorRT have conflicting dependencies).

```bash
# Clone and install CLI tool locally
git clone https://github.com/henrycgbaker/LLenergyMeasure
cd LLenergyMeasure
pip install -e .

# Create .env file with your user IDs
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env

# Optional: Add HuggingFace token for gated models
echo "HF_TOKEN=hf_your_token_here" >> .env

# Build needed backend images
docker compose build pytorch vllm

# Run multi-backend campaign
lem campaign configs/examples/campaign_example.yaml
```

The `lem campaign` command automatically dispatches to Docker when multiple backends are involved.

---

## Environment Variables

For Docker mode, create a `.env` file with your user IDs for correct file permissions:

```bash
# Required for Docker
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env

# Optional: Add HuggingFace token for gated models
echo "HF_TOKEN=hf_your_token_here" >> .env
```

Get your HuggingFace token at: https://huggingface.co/settings/tokens

## Your First Experiment

### 1. Create a Configuration

Copy an example config and modify for your experiment:

```bash
cp configs/example_config.yaml configs/my_experiment.yaml
```
**Available example configs:**

| Config | Use Case |
|--------|----------|
| [pytorch_example.yaml](../configs/examples/pytorch_example.yaml) | Full PyTorch backend reference (all parameters) |
| [vllm_example.yaml](../configs/examples/vllm_example.yaml) | Full vLLM backend reference (all parameters) |
| [tensorrt_example.yaml](../configs/examples/tensorrt_example.yaml) | Full TensorRT backend reference (all parameters) |
| [campaign_example.yaml](../configs/examples/campaign_example.yaml) | Multi-config campaign comparison |

### 2. Validate Your Configuration

Ensure all the parameters are consistent (some combination of experimental parameters and inference backends etc are incompatible, this command will check)

```bash
lem config validate configs/my_experiment.yaml
```

### 3. Run the Experiment

**Poetry:**
```bash
lem experiment configs/my_experiment.yaml
```

**Docker:**
```bash
docker compose run --rm pytorch lem experiment /app/configs/my_experiment.yaml
```

**Makefile (with CLI overrides):**
```bash
make experiment CONFIG=my_experiment.yaml
```

The `experiment` command:
- Handles `accelerate launch` automatically
- Auto-aggregates results on completion
- Supports Ctrl+C for graceful interruption (resume later with `--resume`)

### 4. View Results

```bash
# List all experiments
lem results list

# Show aggregated results
lem results show exp_20240115_123456

# Export as JSON
lem results show exp_20240115_123456 --json
```

## Built-in Datasets

Use HuggingFace datasets as prompt sources instead of text files:

```bash
# List available built-in datasets
lem datasets
```

| Dataset | Source | Default Column | Notes |
|---------|--------|----------------|-------|
| `ai-energy-score` | AIEnergyScore/text_generation | text | **Default** - used when no `--dataset` specified |
| `alpaca` | tatsu-lab/alpaca | instruction | |
| `sharegpt` | ShareGPT_Vicuna | conversations | |
| `gsm8k` | gsm8k (main) | question | |
| `mmlu` | cais/mmlu (all) | question | |

**Note:** If no dataset is specified, `ai-energy-score` is used automatically for standardised energy benchmarking.

You can also use any HuggingFace dataset via CLI override:

```bash
lem experiment config.yaml --dataset squad --split validation --column question -n 50
```

## Configuration Basics

### Essential Fields

```yaml
# Identity (required)
config_name: experiment-name
model_name: org/model-name  # HuggingFace path

# Dataset (recommended - keeps everything in config)
dataset:
  name: alpaca           # Built-in alias or HuggingFace path
  sample_size: 100       # Optional: limit prompts
  split: train           # Optional: default is "train"
  column: instruction    # Optional: auto-detected if not set

# Token limits
max_input_tokens: 512
max_output_tokens: 128

# GPU setup
gpus: [0, 1]        # GPU indices

# Precision
fp_precision: float16  # float32 | float16 | bfloat16
```

### Config Inheritance

Configs can extend base configs:

```yaml
# configs/base.yaml
max_input_tokens: 512
fp_precision: float16

# configs/llama2-7b.yaml
_extends: base.yaml
config_name: llama2-7b
model_name: meta-llama/Llama-2-7b-hf
```

### Built-in Presets

Use presets for common scenarios:

```bash
# Quick validation
lem experiment --preset quick-test --model meta-llama/Llama-2-7b-hf -d alpaca -n 10

# List available presets
lem presets
```

For full configuration options, see [Configuration Guide](../src/llenergymeasure/config/README.md).

## Metrics Collected

| Category | Metrics |
|----------|---------|
| **Inference** | total_tokens, tokens_per_second, latency_per_token_ms |
| **Energy** | total_energy_j, gpu_energy_j, cpu_energy_j, emissions_kg_co2 |
| **Compute** | flops_total, flops_per_token, peak_memory_mb |
| **Latency** | ttft_mean_ms, ttft_p95_ms, itl_mean_ms, itl_p95_ms (streaming mode) |

**Note:** Latency metrics (TTFT/ITL) require `streaming: true` in config or `--streaming` CLI flag.

## Results Structure

```
results/
├── raw/
│   └── exp_20240115_123456/
│       ├── process_0.json    # GPU 0 results
│       └── .completed_0      # Completion marker
└── aggregated/
    └── exp_20240115_123456.json
```

## Common Workflows

### Resume an Interrupted Experiment

```bash
lem experiment --resume <exp_id>
```

### Resume an Interrupted Campaign

```bash
# Discover and select interrupted campaigns
lem resume

# Preview what would be resumed
lem resume --dry-run

# Clear all campaign state
lem resume --wipe
```

The `lem resume` command scans `.state/` for interrupted campaigns and presents an interactive menu if multiple are found.

### Start Fresh (Ignore Incomplete)

```bash
lem experiment configs/my_experiment.yaml --fresh
```

### Manual Aggregation

If you used `--no-aggregate` or need to re-aggregate:

```bash
lem aggregate exp_20240115_123456
lem aggregate --all  # All pending
```

## User Configuration

The `lem init` wizard creates `.lem-config.yaml` with your preferences:

```yaml
# Results directory (relative to project root)
results_dir: results

# Thermal cooling gaps between experiments
thermal_gaps:
  between_experiments: 30.0
  between_cycles: 60.0

# Docker container strategy for multi-backend campaigns
docker:
  strategy: ephemeral    # ephemeral (fresh per experiment) or persistent (faster)

# Webhook notifications
notifications:
  webhook_url: https://hooks.slack.com/services/...
  on_complete: true      # Notify when experiment completes
  on_failure: true       # Notify when experiment fails
```

**Webhook notifications** send HTTP POST requests with experiment status, useful for:
- Slack/Discord notifications for long-running campaigns
- CI/CD pipeline integration
- Custom monitoring dashboards

Run `lem init` again to update your configuration.

## Next Steps

- [CLI Reference](cli.md) - Full command documentation
- [Deployment Guide](deployment.md) - Docker, MIG GPUs, troubleshooting
- [Configuration Guide](../src/llenergymeasure/config/README.md) - All config options
