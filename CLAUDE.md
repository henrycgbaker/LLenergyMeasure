# LLM Energy Measure

LLM inference efficiency measurement framework for benchmarking energy consumption, throughput, and FLOPs across HuggingFace models.

## Architecture Overview

```
                                    CLI (typer)
                                        |
                     +------------------+------------------+
                     |                  |                  |
              config validate      run/aggregate      results list/show
                     |                  |                  |
                     v                  v                  v
              config/loader    orchestration/runner    results/repository
                     |                  |                  |
                     v                  v                  v
              config/models    core/inference         results/aggregation
                                       |
                        +--------------+--------------+
                        |              |              |
                  model_loader   energy_backends   compute_metrics
```

**Key Design Patterns:**
- **Late Aggregation**: Raw per-process results saved separately, aggregated on-demand
- **Dependency Injection**: `ExperimentOrchestrator` takes protocol-based components
- **Pydantic Models**: All configs and results are validated Pydantic models

## Key Directories

| Directory | Purpose | AI Context | Full Docs |
|-----------|---------|------------|-----------|
| `src/llm_energy_measure/` | Main package | [CLAUDE.md](src/llm_energy_measure/CLAUDE.md) | [README](src/llm_energy_measure/README.md) |
| `src/.../cli/` | CLI package (modular) | [CLAUDE.md](src/llm_energy_measure/cli/CLAUDE.md) | - |
| `src/.../config/` | Configuration system | [CLAUDE.md](src/llm_energy_measure/config/CLAUDE.md) | [README](src/llm_energy_measure/config/README.md) |
| `src/.../core/` | Inference engine, metrics | [CLAUDE.md](src/llm_energy_measure/core/CLAUDE.md) | [README](src/llm_energy_measure/core/README.md) |
| `src/.../domain/` | Domain models | [CLAUDE.md](src/llm_energy_measure/domain/CLAUDE.md) | [README](src/llm_energy_measure/domain/README.md) |
| `src/.../orchestration/` | Experiment lifecycle | [CLAUDE.md](src/llm_energy_measure/orchestration/CLAUDE.md) | [README](src/llm_energy_measure/orchestration/README.md) |
| `src/.../results/` | Results persistence | [CLAUDE.md](src/llm_energy_measure/results/CLAUDE.md) | [README](src/llm_energy_measure/results/README.md) |
| `src/.../state/` | Experiment state, transitions | [CLAUDE.md](src/llm_energy_measure/state/CLAUDE.md) | [README](src/llm_energy_measure/state/README.md) |
| `tests/` | Test suite | [CLAUDE.md](tests/CLAUDE.md) | [README](tests/README.md) |

## Quick Reference

### Core Commands
```bash
# Run experiment
llm-energy-measure experiment <config.yaml> --dataset alpaca -n 100
llm-energy-measure experiment --preset quick-test --model <model> -d alpaca

# Configuration
llm-energy-measure config validate <config.yaml>
llm-energy-measure config generate-grid base.yaml --vary batch_size=1,2,4,8

# Results
llm-energy-measure results list
llm-energy-measure results show <exp_id>
```

### Experiment Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Config file | `experiment config.yaml` | Formal experiments |
| Preset + model | `experiment --preset quick-test --model X` | Quick exploration |
| Multi-cycle | `experiment config.yaml --cycles 5` | Statistical robustness |

**Precedence**: CLI flags > Config file > Preset > Defaults

### Built-in Presets

| Preset | Purpose |
|--------|---------|
| `quick-test` | Fast validation (batch=1, max_out=32) |
| `benchmark` | Formal measurements (fp16, deterministic) |
| `throughput` | Throughput testing (batch=8, dynamic batching) |

See [config/README.md](src/llm_energy_measure/config/README.md) for full preset details.

## Running the Tool

### Backend Selection

| Backend | Requirements | Install | Use Case |
|---------|--------------|---------|----------|
| **pytorch** | Any NVIDIA GPU | `pip install -e ".[pytorch]"` | Default, most compatible |
| **vllm** | NVIDIA GPU, Linux only | `pip install -e ".[vllm]"` | High throughput |
| **tensorrt** | Ampere+ GPU, CUDA 12.x, Linux | `pip install -e ".[tensorrt]"` | Highest performance |

**⚠️ Backend Conflict**: vLLM and TensorRT-LLM have conflicting dependencies. Install only ONE at a time, or use separate Docker images.

**TensorRT GPU Requirements**: Compute capability >= 8.0 (A100, A10, RTX 30xx/40xx, H100, L40). NOT supported: V100, T4, RTX 20xx, GTX.

### Installation

```bash
# Local (pick ONE backend)
pip install -e ".[pytorch]"     # Most users
pip install -e ".[vllm]"        # High throughput
pip install -e ".[tensorrt]"    # Requires Ampere+ GPU

# Docker (recommended for isolation)
docker compose build pytorch    # Build PyTorch image
docker compose build vllm       # Build vLLM image
docker compose build tensorrt   # Build TensorRT image
```

### Running Experiments

```bash
# Local
lem experiment configs/example_pytorch.yaml --dataset alpaca -n 100

# Docker (use matching backend service)
docker compose run --rm pytorch lem experiment /app/configs/example_pytorch.yaml -n 100
docker compose run --rm vllm lem experiment /app/configs/example_vllm.yaml -n 100
```

## Development

```bash
make dev      # Install + pre-commit hooks
make check    # format + lint + typecheck
make test     # Unit tests
```

## Documentation

### User Guides (docs/)

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/quickstart.md) | Tutorial, datasets, basic config |
| [CLI Reference](docs/cli.md) | All commands and options |
| [Backends Guide](docs/backends.md) | PyTorch vs vLLM, backend-specific config |
| [Deployment](docs/deployment.md) | Docker, MIG GPUs, troubleshooting |

### Developer Docs (module READMEs)

| Topic | Location |
|-------|----------|
| Configuration system | [config/README.md](src/llm_energy_measure/config/README.md) |
| Core engine | [core/README.md](src/llm_energy_measure/core/README.md) |
| Orchestration | [orchestration/README.md](src/llm_energy_measure/orchestration/README.md) |
| Results persistence | [results/README.md](src/llm_energy_measure/results/README.md) |
