# orchestration/ - Experiment Lifecycle

Experiment orchestration, lifecycle management, and distributed launching.

## Purpose

Manages the complete experiment lifecycle from setup to teardown, including distributed launching via `accelerate` and coordinating the DI components.

## Key Files

### runner.py
Main experiment orchestrator.

**ExperimentOrchestrator** - Coordinates experiment execution:
```python
from llenergymeasure.orchestration import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator(
    model_loader=loader,
    inference_engine=engine,
    metrics_collector=collector,
    energy_backend=backend,
    repository=repository,
)

result_path = orchestrator.run(ctx, prompts)
```

Key responsibilities:
1. Load model via `ModelLoader`
2. Start energy tracking via `EnergyBackend`
3. Run inference via `InferenceEngine`
4. Collect metrics via `MetricsCollector`
5. Save raw result via `ResultsRepository`

**Note**: Does NOT aggregate - that's a separate step.

### context.py
Runtime context for experiments.

**ExperimentContext** - Holds experiment state:
```python
from llenergymeasure.orchestration import ExperimentContext, experiment_context

with experiment_context(config, accelerator) as ctx:
    # ctx.experiment_id - unique ID
    # ctx.process_index - rank in distributed setup
    # ctx.device - torch device
    # ctx.config - ExperimentConfig
    orchestrator.run(ctx, prompts)
```

### lifecycle.py
Setup and teardown utilities.

```python
from llenergymeasure.orchestration import (
    ensure_clean_start,
    warmup_model,
    cleanup_cuda,
    cleanup_distributed,
    full_cleanup,
    experiment_lifecycle,
)

# Context manager for full lifecycle
with experiment_lifecycle(config, accelerator) as ctx:
    # Auto warmup, cleanup on exit
    pass

# Individual utilities
ensure_clean_start()  # Clear GPU memory
warmup_model(model, tokenizer, device)  # Warmup passes
cleanup_cuda()  # torch.cuda.empty_cache()
cleanup_distributed(accelerator)  # Destroy process group
```

### factory.py
Dependency injection wiring.

```python
from llenergymeasure.orchestration import create_orchestrator, experiment_context

with experiment_context(config) as ctx:
    orchestrator = create_orchestrator(ctx)
    result_path = orchestrator.run(ctx, prompts)
```

**create_orchestrator(ctx)** wires up:
- `HuggingFaceModelLoader` → wraps `load_model_tokenizer()`
- `TransformersInferenceEngine` → wraps `run_inference()`
- `ThroughputMetricsCollector` → wraps `collect_compute_metrics()`
- `CodeCarbonBackend` → energy measurement
- `FileSystemRepository` → result persistence

### launcher.py
Entry point for `accelerate launch -m llenergymeasure.orchestration.launcher`.

Parses CLI args and runs experiment:
```bash
accelerate launch --num_processes 2 \
    -m llenergymeasure.orchestration.launcher \
    --config config.yaml --dataset alpaca -n 100
```

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI: lem experiment config.yaml -d alpaca -n 100 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  accelerate launch --num_processes N                            │
│    -m llenergymeasure.orchestration.launcher                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  launcher.__main__                                              │
│    1. _parse_args() → config_path, prompts                      │
│    2. load_config(config_path)                                  │
│    3. experiment_context(config) → ExperimentContext            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  create_orchestrator(ctx) - DI Wiring                           │
│    ├── HuggingFaceModelLoader                                   │
│    ├── TransformersInferenceEngine(accelerator)                 │
│    ├── ThroughputMetricsCollector(accelerator)                  │
│    ├── CodeCarbonBackend                                        │
│    └── FileSystemRepository                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ExperimentOrchestrator.run(ctx, prompts)                       │
│    1. model_loader.load(config) → (model, tokenizer)            │
│    2. energy_backend.start_tracking() → tracker                 │
│    3. inference_engine.run(model, tokenizer, prompts) → result  │
│    4. energy_backend.stop_tracking(tracker) → energy_metrics    │
│    5. metrics_collector.collect(model, result) → combined       │
│    6. repository.save_raw(experiment_id, raw_result) → path     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  results/raw/{experiment_id}/process_{N}.json                   │
└─────────────────────────────────────────────────────────────────┘
```

## Late Aggregation Pattern

### Why Late Aggregation?

When running multi-GPU experiments, `accelerate launch` spawns **N independent Python processes** - one per GPU. These processes:

1. Run in separate memory spaces (no shared state)
2. Execute the same code but on different data slices
3. Each produce their own metrics (energy, tokens, timing)
4. Cannot directly communicate complex result objects

**The problem**: There's no natural "coordinator" process that can collect and aggregate results. Each subprocess only sees its own GPU's data.

**The solution**: Each process saves its raw result independently to disk:
```
results/raw/exp_id/
├── process_0.json   # GPU 0's metrics
├── process_1.json   # GPU 1's metrics
├── .completed_0     # Marker confirming process 0 finished
├── .completed_1     # Marker confirming process 1 finished
```

After all processes complete, aggregation runs in the parent CLI process:
```python
# CLI (parent process) - runs after accelerate exits
raw_results = repo.load_all_raw(experiment_id)
aggregated = aggregate_results(experiment_id, raw_results)
repo.save_aggregated(aggregated)
```

### Benefits of This Approach

| Benefit | Description |
|---------|-------------|
| **Fault tolerance** | If one GPU fails, other results are preserved |
| **Debuggability** | Per-GPU metrics available for analysis |
| **Flexibility** | Re-aggregate with different methods without re-running |
| **Simplicity** | No inter-process communication complexity |
| **Completeness validation** | Markers ensure all N processes finished before aggregating |

### Workflow

```
┌─────────────────────────────────────────────────┐
│ CLI: lem experiment config.yaml  │
│   - Creates ExperimentState                     │
│   - Spawns subprocess via Popen                 │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ accelerate launch (subprocess)                  │
│   ├── Process 0 (GPU 0) → saves process_0.json │
│   ├── Process 1 (GPU 1) → saves process_1.json │
│   └── Each writes .completed_N marker           │
└─────────────────────────────────────────────────┘
                      │
                      ▼ (exit code 0)
┌─────────────────────────────────────────────────┐
│ CLI (parent process continues)                  │
│   - Validates all completion markers exist      │
│   - Loads raw results                           │
│   - Aggregates metrics                          │
│   - Saves aggregated result                     │
└─────────────────────────────────────────────────┘
```

### Auto-Aggregation vs Manual

By default, aggregation runs automatically after successful subprocess exit. Use `--no-aggregate` to skip this and aggregate manually later:

```bash
# Default: auto-aggregates
lem experiment config.yaml -d alpaca -n 100

# Manual aggregation
lem experiment config.yaml -d alpaca -n 100 --no-aggregate
lem aggregate exp_id
```

## Related

- See `../protocols.py` for component interfaces
- See `../results/README.md` for result persistence
- See `../core/README.md` for inference engine
