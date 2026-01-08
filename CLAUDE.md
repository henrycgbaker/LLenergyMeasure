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
| `scripts/` | Docker helper scripts |
| `.devcontainer/` | VS Code devcontainer config |
| `tests/` | Unit, integration, e2e tests |

## Quick Reference

### CLI Commands
```bash
# Experiments (auto-aggregates by default)
llm-energy-measure experiment <config.yaml> --dataset alpaca -n 100
llm-energy-measure experiment --preset quick-test --model <model> -d alpaca
llm-energy-measure experiment <config.yaml> --cycles 5       # Multi-cycle for statistical robustness
llm-energy-measure experiment <config.yaml> --no-aggregate   # Skip auto-aggregation
llm-energy-measure experiment <config.yaml> --fresh          # Ignore incomplete experiments
llm-energy-measure experiment --resume <exp_id>              # Resume interrupted experiment
llm-energy-measure batch configs/*.yaml --dataset alpaca -n 100

# Configuration
llm-energy-measure config validate <config.yaml>
llm-energy-measure config show <config.yaml>
llm-energy-measure config new [--preset benchmark]
llm-energy-measure config generate-grid base.yaml --vary batch_size=1,2,4,8

# Reference
llm-energy-measure gpus                        # Show GPU topology (including MIG)
llm-energy-measure presets                     # List built-in presets
llm-energy-measure datasets                    # List built-in prompt datasets

# Results
llm-energy-measure aggregate <exp_id> | --all
llm-energy-measure aggregate <exp_id> --force  # Force partial aggregation
llm-energy-measure results list [--all]
llm-energy-measure results show <exp_id> [--raw] [--json]
```

### Experiment Workflow

**Default behaviour**: `experiment` command auto-aggregates on success:
```
experiment → subprocess (accelerate) → raw results → auto-aggregate → aggregated result
```

**Interrupt handling**: Ctrl+C saves state and shows resume instructions.

**Incomplete detection**: On startup, detects matching incomplete experiments and prompts to resume (use `--fresh` to skip).

### Run Experiment (Three Modes)

**1. Config file** (formal experiments):
```bash
llm-energy-measure experiment configs/test.yaml --dataset alpaca -n 100
```

**2. Preset + model** (quick exploration):
```bash
llm-energy-measure experiment --preset quick-test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d alpaca
```

**3. Config + CLI overrides** (parameter sweeps):
```bash
llm-energy-measure experiment config.yaml -b 8 --precision float16 --max-tokens 256
```

**Precedence**: CLI flags > Config file > Preset > Defaults

### Experiment CLI Flags
| Flag | Description |
|------|-------------|
| `--preset` | Built-in preset (quick-test, benchmark, throughput) |
| `--model / -m` | HuggingFace model name (required with --preset) |
| `--batch-size / -b` | Override batch size |
| `--precision` | Override fp_precision (float32/float16/bfloat16) |
| `--max-tokens` | Override max_output_tokens |
| `--seed` | Random seed for reproducibility |
| `--num-processes` | Override number of processes |
| `--gpu-list` | Override GPU list (comma-separated) |
| `--temperature` | Override decoder temperature |
| `--quantization/--no-quantization` | Enable/disable quantization |
| `--cycles / -c` | Multi-cycle mode (1-10 cycles for statistical robustness) |

### Multi-Cycle Experiments (Statistical Robustness)

Run experiments multiple times and compute statistics (mean, std, 95% CI):

```bash
llm-energy-measure experiment config.yaml --cycles 5 --dataset alpaca -n 100
```

**Output includes:**
- Per-cycle results: `results/raw/{exp_id}_c0/`, `results/raw/{exp_id}_c1/`, etc.
- Multi-cycle summary: `results/multi_cycle/{exp_id}.json`
- Statistics: mean ± std, 95% confidence intervals, coefficient of variation (CV)

**Academic standard**: TokenPowerBench recommends 3-10 repetitions for statistical robustness.

### Built-in Presets
| Preset | Purpose | Key Settings |
|--------|---------|--------------|
| `quick-test` | Fast validation | batch=1, max_in=64, max_out=32 |
| `benchmark` | Formal measurements | batch=1, max_in=2048, max_out=512, fp16 |
| `throughput` | Throughput testing | batch=8, dynamic batching, fp16 |

### Batch Execution
```bash
# Sequential
llm-energy-measure batch configs/*.yaml --dataset alpaca -n 100

# Parallel (4 concurrent)
llm-energy-measure batch configs/*.yaml --parallel 4 --dataset alpaca -n 100
```

### Grid Generation
```bash
llm-energy-measure config generate-grid base.yaml \
    --vary batch_size=1,2,4,8 \
    --vary fp_precision=float16,float32 \
    --output-dir ./grid/
```

### Prompt Sources

Prompts can be specified via CLI flags or in config:

```yaml
# In config file
prompt_source:
  type: huggingface
  dataset: alpaca    # Built-in: alpaca, gsm8k, mmlu, sharegpt
  sample_size: 1000
  shuffle: true
```

### Running the Tool

**Four ways to run** (choose based on your needs):

| Mode | Best For | Source | Setup |
|------|----------|--------|-------|
| Host (local) | Quick dev iteration | Local Python | `poetry install --with dev` |
| Docker prod | Reproducible runs | Baked into image | `docker compose build` |
| Docker dev | Testing in container | Mounted from host | `docker compose --profile dev build` |
| VS Code devcontainer | Full IDE + container | Mounted from host | "Reopen in Container" |

#### Host (Local Python with Poetry)
```bash
poetry install --with dev
poetry run llm-energy-measure experiment configs/test.yaml --dataset alpaca -n 100
# Or activate the venv: poetry shell
```

#### Docker Production (baked-in package)
```bash
docker compose build llm-energy-measure-app
docker compose run --rm llm-energy-measure-app llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100
docker compose run --rm llm-energy-measure-app llm-energy-measure datasets
```

#### Docker Development (mounted source)
```bash
docker compose --profile dev build llm-energy-measure-dev
docker compose --profile dev run --rm llm-energy-measure-dev  # Interactive shell
docker compose --profile dev run --rm llm-energy-measure-dev llm-energy-measure experiment /app/configs/test.yaml --dataset alpaca -n 100
```

#### VS Code Devcontainer
```bash
# Open project in VS Code → Ctrl+Shift+P → "Dev Containers: Reopen in Container"
# Then inside container:
llm-energy-measure experiment configs/test.yaml --dataset alpaca -n 100
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
| `core/dataset_loader.py` | HuggingFace dataset loading for prompts |
