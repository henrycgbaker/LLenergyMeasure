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

| Directory | Purpose | Docs |
|-----------|---------|------|
| `src/llm_energy_measure/` | Main package | [README](src/llm_energy_measure/README.md) |
| `src/llm_energy_measure/core/` | Inference engine, model loading, metrics | [README](src/llm_energy_measure/core/README.md) |
| `src/llm_energy_measure/config/` | Configuration system | [README](src/llm_energy_measure/config/README.md) |
| `src/llm_energy_measure/domain/` | Domain models (metrics, results) | [README](src/llm_energy_measure/domain/README.md) |
| `src/llm_energy_measure/orchestration/` | Experiment lifecycle | [README](src/llm_energy_measure/orchestration/README.md) |
| `src/llm_energy_measure/results/` | Results persistence | [README](src/llm_energy_measure/results/README.md) |
| `tests/` | Unit, integration, e2e tests | [README](tests/README.md) |

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

| Mode | Setup |
|------|-------|
| Host (local) | `poetry install --with dev` |
| Docker prod | `docker compose build` |
| Docker dev | `docker compose --profile dev build` |
| VS Code devcontainer | "Reopen in Container" |

```bash
# Host
poetry run llm-energy-measure experiment configs/test.yaml --dataset alpaca -n 100

# Docker
docker compose run --rm llm-energy-measure-app llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100
```

## Development

```bash
make dev      # Install + pre-commit hooks
make check    # format + lint + typecheck
make test     # Unit tests
```

## Detailed Documentation

For comprehensive documentation on specific topics:

- **Configuration**: [config/README.md](src/llm_energy_measure/config/README.md) - Presets, decoder config, grid generation, validation
- **Core Engine**: [core/README.md](src/llm_energy_measure/core/README.md) - Inference, model loading, energy backends
- **Orchestration**: [orchestration/README.md](src/llm_energy_measure/orchestration/README.md) - Experiment lifecycle
- **Results**: [results/README.md](src/llm_energy_measure/results/README.md) - Aggregation, persistence
