# Architecture

**Analysis Date:** 2026-01-26

## Pattern Overview

**Overall:** Layered architecture with dependency injection and pluggable backends for LLM inference efficiency measurement.

**Key Characteristics:**
- **Multi-layered**: CLI → Orchestration → Core → Domain, with strict separation of concerns
- **Pluggable backends**: PyTorch, vLLM, TensorRT via protocol-based inference engines
- **Late aggregation**: Per-process results saved independently, aggregated on-demand
- **SSOT (Single Source of Truth)**: Parameter metadata derived from Pydantic models, not static lists
- **Distributed-ready**: Accelerate/torchrun launch with per-GPU process isolation

## Layers

**Presentation Layer (CLI):**
- Purpose: User-facing command interface for experiments, config validation, results inspection
- Location: `src/llenergymeasure/cli/`
- Contains: Typer CLI apps, command handlers, Rich console formatting
- Depends on: Config system, Orchestration, Results repository
- Used by: End users via `lem` command

**Configuration Layer:**
- Purpose: Load, validate, and manage experiment configurations with preset inheritance
- Location: `src/llenergymeasure/config/`
- Contains: Pydantic models (ExperimentConfig, backend-specific configs), loader, validator, introspection (SSOT)
- Depends on: Domain models (for type hints)
- Used by: CLI, Orchestration, Runtime tests

**Orchestration Layer:**
- Purpose: Manage experiment lifecycle from config to result persistence
- Location: `src/llenergymeasure/orchestration/`
- Contains: ExperimentOrchestrator (main runner), ExperimentContext (state container), Factory (DI), Launcher (subprocess management)
- Depends on: Core (inference, metrics), Config, Domain, Results repository
- Used by: CLI experiment commands

**Core Layer:**
- Purpose: Model loading, inference execution, energy/compute metrics collection
- Location: `src/llenergymeasure/core/`
- Contains: Inference backends (PyTorch/vLLM/TensorRT), model loader, energy backends, metric collectors, FLOPs estimators
- Depends on: Domain models, Config (for model/dataset/param info)
- Used by: Orchestration, Launcher (for actual inference)

**Domain Layer:**
- Purpose: Immutable data models for configs, metrics, and results
- Location: `src/llenergymeasure/domain/`
- Contains: Pydantic models for metrics (InferenceMetrics, EnergyMetrics, ComputeMetrics), results (RawProcessResult, AggregatedResult), model info
- Depends on: None (zero external dependencies)
- Used by: All layers (ubiquitous)

**State & Results Layers:**
- Purpose: Persistent storage for experiment state and results
- Location: `src/llenergymeasure/state/`, `src/llenergymeasure/results/`
- Contains: StateManager (experiment lifecycle tracking), FileSystemRepository (result CRUD), aggregators
- Depends on: Domain models
- Used by: CLI, Orchestration

## Data Flow

**Single-Cycle Experiment:**

1. **CLI Input** (`cli/experiment.py`)
   - Load config from YAML
   - Resolve prompts (CLI > config > default)
   - Validate config
   - Parse preset overrides

2. **Orchestration Setup** (`orchestration/launcher.py`)
   - Create experiment state
   - Launch `ExperimentOrchestrator` via `accelerate launch` (single process) or `torchrun` (multi-GPU)

3. **Launcher Entry** (`orchestration/launcher.py` as `__main__`)
   - Create ExperimentContext (loads config, initializes accelerator)
   - Call `create_orchestrator()` factory
   - Resolve dataset/model/metrics components via DI
   - Call `ExperimentOrchestrator.run()`

4. **Model Loading** (`core/model_loader.py`)
   - Load HuggingFace model with quantization (AWQ, GPTQ, bitsandbytes)
   - Load tokenizer
   - Attach LoRA adapters if `config.adapter` set

5. **Inference Execution** (`core/inference_backends/`)
   - Backend-specific inference engine runs prompts in batches
   - Collects per-batch latencies, generated tokens
   - Optionally captures per-token timing for streaming latency (TTFT/ITL)

6. **Metrics Collection** (`core/compute_metrics.py`, `core/extended_metrics.py`)
   - Inference metrics (tokens, throughput)
   - Energy metrics (CodeCarbon energy tracking)
   - Compute metrics (memory, GPU utilization via pynvml)
   - Extended metrics (TPOT, token efficiency index, KV cache stats)

7. **Result Persistence** (`orchestration/runner.py`)
   - Save RawProcessResult as JSON (`results/raw/exp_ID/process_N.json`)
   - Write completion marker (`.completed_N`)

8. **Aggregation** (`results/aggregation.py`, triggered by `lem aggregate` or automatic)
   - Collect all per-process results
   - Aggregate metrics (sum tokens, average latency, max memory)
   - Compute cycle statistics for multi-cycle runs
   - Save AggregatedResult as JSON

**Multi-Cycle Experiment:**

- Each cycle runs full single-cycle flow independently
- Results collected per-cycle
- Final aggregation computes statistics across cycles (mean, std, CI)

**Multi-GPU Experiment:**

- Single config → `accelerate launch` with `num_processes=N` or `distributed_type=MULTI_GPU`
- Each GPU spawns subprocess running launcher.py
- Each subprocess saves its own raw result
- Aggregation stage combines results from all processes

## Key Abstractions

**ExperimentOrchestrator:**
- Purpose: Coordinates entire experiment lifecycle
- Examples: `src/llenergymeasure/orchestration/runner.py`
- Pattern: Accepts protocol-based components (ModelLoader, InferenceEngine, MetricsCollector, EnergyBackend, ResultsRepository) via constructor (dependency injection)
- Uses: Factory pattern for construction via `create_orchestrator()`

**InferenceBackend (Protocol):**
- Purpose: Abstract inference engine interface
- Examples: `src/llenergymeasure/core/inference_backends/pytorch.py`, `vllm.py`, `tensorrt.py`
- Pattern: Protocol-based (runtime_checkable), implements `initialize()`, `generate()`, `get_runtime_capabilities()`
- Responsibility: Handles model setup, batching strategy, CUDA context management

**EnergyBackend (Protocol):**
- Purpose: Abstract energy measurement interface
- Examples: `src/llenergymeasure/core/energy_backends/codecarbon.py`
- Pattern: Protocol-based, context manager with `__enter__`, `__exit__` for measurement periods

**FileSystemRepository:**
- Purpose: CRUD operations for experiment results
- Examples: `src/llenergymeasure/results/repository.py`
- Pattern: Single responsibility - read/write results from/to filesystem
- Uses: JSON serialization, atomic writes via temp file → rename

**Pydantic Models (SSOT):**
- Purpose: Single source of truth for parameter metadata
- Examples: `src/llenergymeasure/config/models.py`, `backend_configs.py`
- Pattern: Pydantic BaseModel with Field constraints (min, max, description)
- Uses: Introspection module auto-discovers params from model definitions

## Entry Points

**CLI Command:**
- Location: `src/llenergymeasure/cli/__init__.py` (Typer app)
- Triggers: `lem experiment <config> [--options]`
- Responsibilities: Parse args, load config, validate, call experiment command

**Experiment Command:**
- Location: `src/llenergymeasure/cli/experiment.py` (`experiment_cmd`, `run_cmd`, `aggregate_cmd`)
- Triggers: `lem experiment` or `lem run` CLI invocation
- Responsibilities: Resolve prompts, validate config, spawn subprocess via `accelerate launch`, track state, trigger aggregation

**Launcher Entry Point:**
- Location: `src/llenergymeasure/orchestration/launcher.py` (called as `python -m llenergymeasure.orchestration.launcher`)
- Triggers: `accelerate launch -m llenergymeasure.orchestration.launcher` subprocess
- Responsibilities: Initialize accelerator, create context, build components, run orchestrator, save results

**Aggregation Entry Point:**
- Location: `src/llenergymeasure/cli/results.py` (`aggregate_one` function)
- Triggers: Manual `lem aggregate <exp_id>` or automatic after all processes complete
- Responsibilities: Load per-process results, compute statistics, save aggregated result

## Error Handling

**Strategy:** Validation-first with graceful degradation for optional features.

**Patterns:**

- **Configuration errors**: `ConfigurationError` raised at load time, prevents experiment start
- **Hardware not available**: Graceful fallback (e.g., GPU sampler disabled if NVML unavailable, FLOPs estimation falls back to cheaper methods)
- **Subprocess failures**: State machine tracks process status, allows resumption via `--resume` flag
- **Missing optional metrics**: Extended metrics use nullable fields (TPOT, token_efficiency_index may be null)

**Example (from `core/extended_metrics.py`):**
```python
# Always return complete model, with None for unavailable metrics
extended = ExtendedEfficiencyMetrics(
    tpot_ms=None,  # Null if streaming disabled
    token_efficiency_index=efficiency if energy_available else None,
)
```

## Cross-Cutting Concerns

**Logging:** `src/llenergymeasure/logging.py` - Loguru-based with verbosity levels (quiet, normal, verbose). Set via CLI flags or `LLM_ENERGY_VERBOSITY` env var.

**Validation:** Three-tier approach:
1. Pydantic schema validation (automatic)
2. Custom validators in config models (cross-field constraints)
3. Runtime validation (e.g., model-specific parameter compatibility)

**Authentication:** None - local file-based operations only. HuggingFace token read from `.env` or `HF_TOKEN` env var.

**Multi-GPU Coordination:**
- Accelerate handles NCCL initialization
- Per-process unique IDs via `get_shared_unique_id()` and `get_persistent_unique_id()`
- State machine serializes experiment state across processes

**Security:**
- Config validation prevents path traversal (`src/llenergymeasure/security.py`)
- Secrets not logged (checked in logging setup)
- No shell execution in critical paths (subprocess uses list args)

---

*Architecture analysis: 2026-01-26*
