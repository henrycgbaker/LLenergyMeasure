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

| Directory | Purpose |
|-----------|---------|
| `src/llm_energy_measure/` | Main package |
| `src/llm_energy_measure/core/` | Inference engine, model loading, metrics |
| `src/llm_energy_measure/config/` | Configuration loading with inheritance |
| `src/llm_energy_measure/domain/` | Domain models (metrics, results) |
| `src/llm_energy_measure/orchestration/` | Experiment lifecycle & launching |
| `src/llm_energy_measure/results/` | Results persistence & aggregation |
| `configs/` | Legacy Python configs (from research phase) |
| `tests/` | Unit, integration, e2e tests |

## Quick Reference

### CLI Commands
```bash
llm-energy-measure config validate <config.yaml>
llm-energy-measure config show <config.yaml>
llm-energy-measure aggregate <exp_id> | --all
llm-energy-measure results list [--all]
llm-energy-measure results show <exp_id> [--raw] [--json]
```

### Run Experiment
```bash
accelerate launch --num_processes N \
  -m llm_energy_measure.orchestration.launcher \
  --config <config.yaml> --prompts <prompts.txt>
```

### Development
```bash
make dev      # Install + pre-commit hooks
make check    # format + lint + typecheck
make test     # Unit tests
make test-all # All tests
```

## Detailed Documentation

- `src/llm_energy_measure/README.md` - Package overview
- `src/llm_energy_measure/core/README.md` - Core inference engine
- `src/llm_energy_measure/config/README.md` - Configuration system
- `src/llm_energy_measure/domain/README.md` - Domain models
- `src/llm_energy_measure/orchestration/README.md` - Experiment orchestration
- `src/llm_energy_measure/results/README.md` - Results handling
- `tests/README.md` - Test structure

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Typer CLI entry point |
| `constants.py` | Global constants (paths, defaults) |
| `protocols.py` | Protocol definitions for DI |
| `exceptions.py` | Custom exception hierarchy |
| `logging.py` | Loguru setup |
