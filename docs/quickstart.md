# Getting Started

This guide walks you through your first LLM efficiency measurement experiment.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- HuggingFace account (for gated models like Llama)

## Installation

### Poetry (Recommended)

```bash
poetry install
poetry shell  # Activate the virtual environment
```

### Docker

```bash
docker compose build
```

## Your First Experiment

### 1. Create a Configuration

Create a YAML config file (e.g., `configs/my_experiment.yaml`):

```yaml
config_name: llama2-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf

# Token limits
max_input_tokens: 512
max_output_tokens: 128

# GPU setup
gpus: [0]
num_processes: 1

# Precision
fp_precision: float16
```

### 2. Validate Your Configuration

```bash
llm-energy-measure config validate configs/my_experiment.yaml
```

### 3. Run the Experiment

```bash
# Using a built-in dataset (recommended)
llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100
```

The `experiment` command:
- Handles `accelerate launch` automatically
- Auto-aggregates results on completion
- Supports Ctrl+C for graceful interruption (resume later with `--resume`)

### 4. View Results

```bash
# List all experiments
llm-energy-measure results list

# Show aggregated results
llm-energy-measure results show exp_20240115_123456

# Export as JSON
llm-energy-measure results show exp_20240115_123456 --json
```

## Built-in Datasets

Use HuggingFace datasets as prompt sources instead of text files:

```bash
# List available built-in datasets
llm-energy-measure datasets
```

| Dataset | Source | Default Column |
|---------|--------|----------------|
| `alpaca` | tatsu-lab/alpaca | instruction |
| `sharegpt` | ShareGPT_Vicuna | conversations |
| `gsm8k` | gsm8k (main) | question |
| `mmlu` | cais/mmlu (all) | question |

You can also use any HuggingFace dataset:

```bash
llm-energy-measure experiment config.yaml \
  --dataset squad --split validation --column question -n 50
```

## Configuration Basics

### Essential Fields

```yaml
# Identity (required)
config_name: experiment-name
model_name: org/model-name  # HuggingFace path

# Token limits
max_input_tokens: 512
max_output_tokens: 128

# GPU setup
gpus: [0, 1]        # GPU indices
num_processes: 2    # Should match GPU count

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
llm-energy-measure experiment --preset quick-test --model meta-llama/Llama-2-7b-hf -d alpaca -n 10

# List available presets
llm-energy-measure presets
```

For full configuration options, see [Configuration Guide](../src/llm_energy_measure/config/README.md).

## Metrics Collected

| Category | Metrics |
|----------|---------|
| **Inference** | total_tokens, tokens_per_second, latency_per_token_ms |
| **Energy** | total_energy_j, gpu_energy_j, cpu_energy_j, emissions_kg_co2 |
| **Compute** | flops_total, flops_per_token, peak_memory_mb |

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
llm-energy-measure experiment --resume <exp_id>
```

### Start Fresh (Ignore Incomplete)

```bash
llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100 --fresh
```

### Manual Aggregation

If you used `--no-aggregate` or need to re-aggregate:

```bash
llm-energy-measure aggregate exp_20240115_123456
llm-energy-measure aggregate --all  # All pending
```

## Next Steps

- [CLI Reference](cli.md) - Full command documentation
- [Deployment Guide](deployment.md) - Docker, MIG GPUs, troubleshooting
- [Configuration Guide](../src/llm_energy_measure/config/README.md) - All config options
