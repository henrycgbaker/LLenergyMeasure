# orchestration/ - Experiment Lifecycle

Experiment orchestration, lifecycle management, and distributed launching.

## Purpose

Manages the complete experiment lifecycle from setup to teardown, including distributed launching via `accelerate` and coordinating the DI components.

## Key Files

### runner.py
Main experiment orchestrator.

**ExperimentOrchestrator** - Coordinates experiment execution:
```python
from llm_energy_measure.orchestration import ExperimentOrchestrator

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
from llm_energy_measure.orchestration import ExperimentContext, experiment_context

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
from llm_energy_measure.orchestration import (
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

### launcher.py
Utilities for launching via `accelerate`.

```python
from llm_energy_measure.orchestration import (
    launch_experiment_accelerate,
    run_from_config,
    log_failed_experiment,
)

# Launch subprocess with accelerate
launch_experiment_accelerate(
    config_path="config.yaml",
    num_processes=4,
    prompts_file="prompts.txt",
)

# Entry point when launched via accelerate
run_from_config(config_path, prompts_file)
```

## Execution Flow

```
CLI/launcher
    |
    v
accelerate launch --num_processes N
    |
    v
launcher.run_from_config()
    |
    v
ExperimentContext (per process)
    |
    v
ExperimentOrchestrator.run()
    |
    +-> ModelLoader.load()
    +-> EnergyBackend.start_tracking()
    +-> InferenceEngine.run()
    +-> EnergyBackend.stop_tracking()
    +-> MetricsCollector.collect()
    +-> ResultsRepository.save_raw()
    |
    v
RawProcessResult saved to results/raw/exp_id/process_N.json
```

## Late Aggregation Pattern

Raw results are saved per-process. Aggregation happens later:
```bash
# After experiment completes
llm-energy-measure aggregate exp_id
```

This allows:
- Partial results if some processes fail
- Re-aggregation with different methods
- Debugging per-GPU metrics

## Related

- See `../protocols.py` for component interfaces
- See `../results/README.md` for result persistence
- See `../core/README.md` for inference engine
