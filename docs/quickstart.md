# Getting Started

This guide walks you through your first LLM efficiency measurement experiment.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- HuggingFace account (for gated models like Llama)

## Installation

### Option 1: Local (pip)

```bash
# Clone and install
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool
cd llm-energy-measure

conda create -n llm-energy python=3.10
conda activate llm-energy
pip install -e .

# Verify
lem --help
```

### Option 2: Docker

```bash
# Clone and build
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool
cd llm-energy-measure
make docker-build-pytorch  # Or docker-build-all for all backends

# Run commands via make
make lem CMD="--help"
```

**Docker backends:**

| Backend | Build Command | Best For |
|---------|--------------|----------|
| PyTorch | `make docker-build-pytorch` | Development, flexibility (default) |
| vLLM | `make docker-build-vllm` | High throughput, continuous batching |
| TensorRT | `make docker-build-tensorrt` | Maximum performance (compiled) |
| All | `make docker-build-all` | Build all backends at once |

**Running commands in Docker:**
```bash
# Via make (recommended)
make lem CMD="experiment configs/my_experiment.yaml"

# Or directly
docker compose run --rm pytorch lem experiment /app/configs/my_experiment.yaml
```

## Environment Variables

Create a `.env` file in the project root (gitignored):

```bash
# .env
HF_TOKEN=hf_your_token_here
CUDA_VISIBLE_DEVICES=available_device_indices
```
NB: (if using poetry skip this, if using Docker:) the make command for docker compose reads `.env`, and PUID/PGID are auto-detected from the mounted directory ownership. If running docker compose directly you'll need to set PUID/PGID manually in the `.env`.

## Your First Experiment

### 1. Create a Configuration

Copy an example config and modify for your experiment:

```bash
cp configs/example_config.yaml configs/my_experiment.yaml
```
**Available example configs:**

| Config | Use Case |
|--------|----------|
| [example_config.yaml](../configs/example_config.yaml) | Minimal baseline (recommended starting point) |
| [example_pytorch.yaml](../configs/example_pytorch.yaml) | Full PyTorch backend reference (all parameters) |
| [example_vllm.yaml](../configs/example_vllm.yaml) | Full vLLM backend reference (all parameters) |
| [example_tensorrt.yaml](../configs/example_tensorrt.yaml) | Full TensorRT backend reference (all parameters) |

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

For full configuration options, see [Configuration Guide](../src/llm_energy_measure/config/README.md).

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

## Next Steps

- [CLI Reference](cli.md) - Full command documentation
- [Deployment Guide](deployment.md) - Docker, MIG GPUs, troubleshooting
- [Configuration Guide](../src/llm_energy_measure/config/README.md) - All config options
