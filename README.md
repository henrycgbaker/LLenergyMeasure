# LLM Energy Measure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A Python framework for measuring LLM inference efficiency, including energy consumption, throughput, and computational metrics (FLOPs). Designed for distributed GPU benchmarking using HuggingFace models.

## Features

- **Energy Measurement** - Track GPU, CPU, and RAM energy via CodeCarbon
- **Throughput Metrics** - Tokens/second, latency, batch processing stats
- **FLOPs Estimation** - Multiple methods (calflops, architecture-based, parameter-based)
- **Multi-GPU Support** - Distributed inference via `accelerate`
- **Flexible Configuration** - YAML configs with inheritance and presets
- **Docker Ready** - GPU-enabled containerisation

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
config_name: llama2-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf
max_input_tokens: 512
max_output_tokens: 128
gpus: [0]
num_processes: 1
fp_precision: float16
```

**2. Run experiment:**

```bash
llm-energy-measure experiment configs/my_experiment.yaml --dataset alpaca -n 100
```

**3. View results:**

```bash
llm-energy-measure results list
llm-energy-measure results show <exp_id>
```

For a full tutorial, see the [Getting Started Guide](docs/quickstart.md).

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/quickstart.md) | Full tutorial, datasets, basic config |
| [CLI Reference](docs/cli.md) | All commands and options |
| [Deployment](docs/deployment.md) | Docker, MIG GPUs, troubleshooting |
| [Configuration](src/llm_energy_measure/config/README.md) | Full config options, presets, validation |

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
src/llm_energy_measure/
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
