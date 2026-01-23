# LLenergyMeasure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A Python framework for measuring LLM inference efficiency, including energy consumption, throughput, and computational metrics (FLOPs). Designed for distributed GPU benchmarking using HuggingFace models.

## Features

- **Energy Measurement** - Track GPU, CPU, and RAM energy via CodeCarbon
- **Throughput Metrics** - Tokens/second, latency, batch processing stats
- **FLOPs Estimation** - Multiple methods (calflops, architecture-based, parameter-based - also profiler options)
- **Multi-GPU Support** - Distributed inference via `accelerate`, vLLM, or TensorRT
- **Tensor & Pipeline Parallelism** - Native PyTorch TP/PP for large model inference (TODO:  update based on vLLM / TensorRT)
- **Flexible Configuration** - YAML configs with inheritance and presets (TODO update)
- **Docker Ready** - GPU-enabled containerisation

## Implemented Testing Parameters

All parameters below are fully wired and functional.

| Category | Parameter | Values | Description |
|----------|-----------|--------|-------------|
| Core | `model_name` | any HF model | HuggingFace model path |
| Core | `fp_precision` | `float32`, `float16`, `bfloat16` | Floating point precision |
| Core | `max_input_tokens` | 1+ | Input token limit |
| Core | `max_output_tokens` | 1+ | Output token limit |
| Core | `min_output_tokens` | 0+ | Minimum output tokens |
| Core | `random_seed` | int or null | Reproducibility seed |
| Batching | `batch_size` | 1+ | Prompts per batch |
| Batching | `strategy` | `static`, `dynamic`, `sorted_static`, `sorted_dynamic` | MLPerf-aligned batching strategies |
| Batching | `max_tokens_per_batch` | int or null | Token budget (dynamic modes) |
| Decoder | `preset` | `deterministic`, `standard`, `creative`, `factual` | Sampling presets |
| Decoder | `temperature` | 0.0–2.0 | Sampling temperature |
| Decoder | `top_p` | 0.0–1.0 | Nucleus sampling |
| Decoder | `top_k` | 0+ | Top-k sampling |
| Decoder | `min_p` | 0.0–1.0 | Min probability threshold |
| Decoder | `repetition_penalty` | 0.1–10.0 | Repetition control |
| Decoder | `no_repeat_ngram_size` | 0+ | Prevent n-gram repetition |
| Quantization | `load_in_4bit` | bool | BitsAndBytes 4-bit quantization |
| Quantization | `load_in_8bit` | bool | BitsAndBytes 8-bit quantization |
| Multi-Cycle | `num_cycles` | 1–10 | Repeat experiment for statistical robustness |
| Traffic | `mode` | `constant`, `poisson` | Request arrival pattern (MLPerf-style) |
| Traffic | `target_qps` | float | Target queries per second |
| Schedule | `interval` | e.g. `6h`, `30m`, `1d` | Run frequency (daemon mode) |
| Schedule | `at` | e.g. `09:00` | Specific time of day |
| Schedule | `days` | `mon`–`sun`, `weekdays`, `weekends` | Day filter |
| Sharding | `strategy` | `none`, `tensor_parallel`, `pipeline_parallel` | Multi-GPU parallelism strategy |
| Sharding | `num_shards` | 1+ | Number of GPUs for parallelism |
| Sharding | `tp_plan` | `auto` | Tensor parallel plan (HF native, TP only) |
| Backend | `backend` | `pytorch`, `vllm` | Inference backend (see [Backends Guide](docs/backends.md)) |
| vLLM | `vllm.*` | various | vLLM-specific config (memory, KV cache, speculative) |
| PyTorch | `pytorch.*` | various | PyTorch-specific config (attention, compile, assisted gen) |

## Installation

```bash
# Poetry (recommended)
poetry install

# Docker
docker compose build
```

## Quick Start

**1. Create a config** (`configs/my_experiment.yaml`):

```yaml
# === IDENTITY ===
config_name: llama2-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf

# === MODEL PROPERTIES ===
is_encoder_decoder: false
task_type: text_generation      # text_generation | translation | summarisation
inference_type: pure_generative # pure_generative | reasoning

# === TOKEN LIMITS ===
max_input_tokens: 512
max_output_tokens: 128
min_output_tokens: 0

# === INPUT ===
num_input_prompts: 100
save_outputs: false

# === PROMPT SOURCE (alternative to --dataset CLI flag) ===
# prompts:
#   type: huggingface
#   dataset: alpaca             # Built-in: alpaca, gsm8k, mmlu, sharegpt
#   split: train
#   column: instruction

# === GPU SETUP ===
gpus: [0, 1]                    # Available GPUs on your server (e.g., [0], [0,1,2,3])
num_processes: 2                # Must be <= len(gpus)

# === PRECISION ===
fp_precision: float16           # float32 | float16 | bfloat16

# === QUANTIZATION ===
quantization:
  quantization: false
  load_in_8bit: false
  load_in_4bit: false

# === BATCHING ===
batching:
  batch_size: 4
  strategy: static              # static | dynamic | sorted_static | sorted_dynamic
  max_tokens_per_batch: null    # For dynamic strategies

# === GENERATION / DECODER ===
decoder:
  preset: null                  # deterministic | standard | creative | factual
  temperature: 1.0
  do_sample: true
  top_p: 1.0
  top_k: 50
  repetition_penalty: 1.0

# === EXPERIMENT CYCLES (for statistical robustness) ===
num_cycles: 1                   # 1-10 cycles, results aggregated with statistics
random_seed: 42                 # null = non-deterministic

# === TRAFFIC SIMULATION (MLPerf-style load testing) ===
traffic_simulation:
  enabled: false
  mode: poisson                 # constant | poisson
  target_qps: 1.0

# === SHARDING / PARALLELISM (multi-GPU large models) ===
sharding:
  strategy: none                # none | tensor_parallel | pipeline_parallel
  num_shards: 1                 # Number of GPUs for parallelism
  # tp_plan: auto               # For tensor_parallel (HF native)

# === SCHEDULED EXPERIMENTS (daemon mode) ===
schedule:
  enabled: false
  interval: "6h"                # e.g., "6h", "30m", "1d"
  # at: "09:00"                 # Specific time of day
  # days: ["mon", "wed", "fri"] # Or "weekdays", "weekends"

# === BACKEND-SPECIFIC CONFIG (optional) ===
# backend: vllm                 # Switch to vLLM backend
# vllm:
#   gpu_memory_utilization: 0.9
#   enable_prefix_caching: true
#   kv_cache_dtype: fp8

# backend: pytorch              # Default backend
# pytorch:
#   attn_implementation: flash_attention_2
#   torch_compile: reduce-overhead
```

**2. Run experiment:**

```bash
lem experiment configs/my_experiment.yaml --dataset alpaca -n 100
```

**3. View results:**

```bash
lem results list
lem results show <exp_id>
```

For a full tutorial, see the [Getting Started Guide](docs/quickstart.md).

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/quickstart.md) | Full tutorial, datasets, basic config |
| [CLI Reference](docs/cli.md) | All commands and options |
| [Backends Guide](docs/backends.md) | PyTorch vs vLLM, backend-specific config |
| [Deployment](docs/deployment.md) | Docker, MIG GPUs, troubleshooting |
| [Configuration](src/llenergymeasure/config/README.md) | Full config options, presets, validation |

## Metrics Collected

| Category | Metrics |
|----------|---------|
| **Inference** | total_tokens, tokens_per_second, latency_per_token_ms |
| **Energy** | total_energy_j, gpu_energy_j, cpu_energy_j, emissions_kg_co2 |
| **Compute** | flops_total, flops_per_token, peak_memory_mb |

## Development

```bash
make dev      # Install + pre-commit hooks
make check    # Format, lint, typecheck
make test     # Run tests
```

## Project Structure

```
src/llenergymeasure/
├── cli.py                 # Typer CLI
├── config/                # Configuration loading
├── core/                  # Inference engine & metrics
├── domain/                # Domain models (Pydantic)
├── orchestration/         # Experiment lifecycle
└── results/               # Results persistence
```

See module READMEs for detailed documentation.

## License

[MIT](LICENSE)
