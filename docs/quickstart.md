# Getting Started

This guide walks you through your first LLM efficiency measurement experiment.

## Quick Start (5 minutes)

```bash
# Clone and setup
git clone https://github.com/henrycgbaker/LLenergyMeasure
cd LLenergyMeasure
./setup.sh

# Run your first experiment
./lem experiment configs/examples/pytorch_example.yaml -n 10

# Or run a multi-config campaign
./lem campaign configs/examples/campaign_example.yaml --dry-run
```

That's it! The `setup.sh` script handles Docker image building, environment configuration, and creates the `lem` CLI wrapper.

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

### Option 1: Docker with setup.sh (Recommended)

The simplest way to get started. Each backend runs in an isolated container.

```bash
# Clone
git clone https://github.com/henrycgbaker/LLenergyMeasure
cd LLenergyMeasure

# One-click setup (creates .env, builds PyTorch image, creates lem wrapper)
./setup.sh

# Build other backends if needed
./setup.sh --backend vllm
./setup.sh --backend tensorrt
./setup.sh --all  # Build all backends
```

After setup, use the `lem` wrapper for all commands:
```bash
./lem experiment configs/examples/pytorch_example.yaml -n 10
./lem campaign configs/examples/campaign_example.yaml --dry-run
./lem results list
```

The `lem` wrapper automatically:
- Detects the backend from your config file
- Runs the correct Docker container
- Handles volume mounts and permissions

### Option 2: Local Development (poetry/pip)

Use this for development or if you only need **one backend**.

```bash
# Clone
git clone https://github.com/henrycgbaker/LLenergyMeasure
cd LLenergyMeasure

# Local install mode
./setup.sh --local

# Or manually:
pip install -e ".[pytorch]"     # Most compatible
pip install -e ".[vllm]"        # High throughput (Linux only)
pip install -e ".[tensorrt]"    # Highest performance (Ampere+ required)

# Verify
./lem --help
```

**Switching backends locally**: Create separate conda environments:
```bash
conda create -n llm-vllm python=3.10 && conda activate llm-vllm
pip install -e ".[vllm]"
```

---

## Environment Variables

The `setup.sh` script creates a `.env` file automatically with your user IDs for correct file permissions.

To add a HuggingFace token for gated models:

```bash
# Edit .env and add your token
HF_TOKEN=hf_your_token_here
```

Get your token at: https://huggingface.co/settings/tokens

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
- [Configuration Guide](../src/llenergymeasure/config/README.md) - All config options
