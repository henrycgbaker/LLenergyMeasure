# Peer CLI Patterns Research

**Date**: 2026-02-18
**Purpose**: Inform CLI command set and UX decisions for llenergymeasure v2.0

## Summary

Extracted from existing research files (01-lm-eval-harness.md, 02-mlflow-architecture.md,
04-deployment-patterns.md, 09-broader-landscape.md).

## lm-eval-harness (EleutherAI)

**Commands**: 3 subcommands only
```bash
lm-eval run --model hf --tasks hellaswag    # primary command
lm-eval ls [tasks|groups|subtasks|tags]      # list available tasks
lm-eval validate --tasks <task1,task2>       # validate task configs
```

**Key flags for `lm-eval run`:**
- `--model/-M`: model type (default: `hf`)
- `--tasks/-t`: task names (space or comma separated)
- `--batch_size/-b`: integer, `auto`, or `auto:N`
- `--config/-C`: YAML config file path (optional; CLI args take precedence)
- `--output_path/-o`: results directory
- `--log_samples/-s`: save model inputs/outputs for analysis
- `--seed`: reproducibility

**Health/status command**: None
**Version**: No `--version` flag documented (accessed via `pip show lm_eval`)
**Zero-config**: `lm-eval run --model hf --tasks hellaswag` — minimal, works immediately
**Example configs**: YAML `--config` optional; no `.example.yaml` shipped
**Sweep/grid**: None built-in; multi-task in one run via `--tasks t1,t2,t3`; separate invocations for model sweep

## MLflow (Databricks/LF)

**Commands**: 7 subcommands
```bash
mlflow server           # launch FastAPI tracking server
mlflow experiments      # create/search/manage experiments
mlflow runs             # create/delete/describe runs
mlflow artifacts        # upload/list/download artifacts
mlflow models           # serve/predict/build Docker images
mlflow deployments      # deploy to SageMaker, Databricks, etc.
mlflow db               # schema migrations
```

**Health/status command**: None (REST API for health if server running)
**Version**: No `--version` flag; accessed via `pip show mlflow` or Python
**Zero-config**: Extremely minimal — `mlflow.log_metric("loss", 0.5)` writes to `./mlruns/` with zero setup
**Example configs**: None; config via env vars (`MLFLOW_TRACKING_URI`) or Python args
**Sweep/grid**: None built-in; MLflow is tracking, not orchestration

## Optimum-Benchmark (HuggingFace)

**Commands**: Primary Python API + Hydra YAML; CLI details sparse in research
```python
# Primary API
from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, PyTorchConfig
benchmark_config = BenchmarkConfig(name="test", ...)
report = Benchmark.launch(benchmark_config)
```

**Health/status command**: Not documented
**Version**: Not documented
**Zero-config**: Requires explicit config construction or Hydra YAML
**Sweep/grid**: Hydra sweep support (grid search via Hydra multirun)

## AIPerf (NVIDIA)

CLI tool for endpoint benchmarking. Sparse documentation in research.
Load modes: concurrency, request-rate, trace replay
Output: CSV, JSON, profile exports, PNG plots
No health/status/version pattern documented.

## Key Findings for llenergymeasure

| Pattern | lm-eval | MLflow | Optimum-BM | AIPerf | **Recommendation** |
|---------|---------|--------|------------|--------|-------------------|
| Status/health cmd | ❌ | ❌ | ❌ | ❌ | `llem config` — passive env display + onboarding (not a check/doctor command) |
| Version flag | ❌ (unusual) | ❌ | ❌ | ❌ | `--version` flag — Python best practice (Typer version_option) |
| Zero-config | ✓ (2 required flags) | ✓ (Python API, zero flags) | ✗ (requires config) | ✗ | `llem run --model X` with interactive prompts |
| Example files | ❌ (docs only) | ❌ | ❌ | ❌ | Ship `.yaml.example` files — better than most peers |
| Grid sweep | ❌ | ❌ | ✓ (Hydra) | ❌ | `sweep:` block in study.yaml — matches Hydra pattern |

**Confirmed**: No peer tool has a check/status/doctor command. This is not an industry pattern.
**Confirmed**: lm-eval's 3-subcommand model is the closest peer. Our 3-command target is industry-aligned.
**Notable**: Neither lm-eval nor MLflow use `--version` (unusual; most Python tools do). We should still include it.
