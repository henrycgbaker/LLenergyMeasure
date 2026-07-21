# study/ - Multi-experiment Study Runner

Sweep expansion, cycle ordering, manifest tracking, and subprocess isolation for multi-experiment studies. Layer 4 in the six-layer architecture.

## Purpose

Implements the sweep runner that executes a `StudyConfig` (a list of `ExperimentConfig` objects) across multiple cycles. Each experiment runs in a freshly spawned subprocess for CUDA context isolation. Results travel parent ← child via `multiprocessing.Pipe`.

## Modules

| Module | Description |
|--------|-------------|
| `orchestration.py` | `orchestrate_study()` - study-layer dispatcher: preflight, study dir + manifest, single-vs-sweep branch, `StudyResult` assembly (the `api` layer is a thin adapter over it) |
| `runner.py` | `StudyRunner` - study-level run loop; drives one `ExperimentSession` per experiment and owns SIGINT, GPU locks, circuit breaker, wall-clock timeout |
| `session.py` | `ExperimentSession` protocol + offline `SubprocessSession` / `DockerSession` - context-managed engine lifetimes (setup on enter, teardown on exit, always) |
| `worker.py` | `_run_experiment_worker()` (child entry point), `_collect_result()`, process-group signalling |
| `_progress.py` | `_QueueProgressCallback` + `_consume_progress_events()` - cross-process progress bridge |
| `baseline_measure.py` | `_BaselineMixin` - baseline measurement, caching, and drift validation |
| `image_prep.py` | `_ImageMixin` - Docker image preparation and schema-fingerprint verification |
| `container_lifecycle.py` | Container naming/labels/cleanup, SIGTERM bridge, orphan reaper, failure artefacts |
| `single.py` | `run_single_experiment()` - in-process / direct-DockerRunner path for a single experiment |
| `manifest.py` | `ManifestWriter`, `StudyManifest`, `ExperimentManifestEntry` - checkpoint model |
| `gpu_memory.py` | `check_gpu_memory_residual()` - pre-dispatch GPU memory check |
| `gaps.py` | `run_gap()` - thermal gap between experiments |
| `__init__.py` | Re-exports `StudyRunner`, `ManifestWriter`, `StudyManifest`, etc. |

## StudyRunner

```python
from llenergymeasure.study import StudyRunner

runner = StudyRunner(study, manifest, study_dir, runner_specs=runner_specs)
raw_results = runner.run()  # list[ExperimentResult | dict]
```

Each experiment runs in a subprocess spawned with `multiprocessing.get_context("spawn")`:
- **spawn context** - CUDA-safe; fork causes silent CUDA corruption
- **daemon=False** - clean CUDA teardown if parent exits unexpectedly
- **Pipe-only IPC** - ExperimentResult fits in Pipe buffer for study-scale experiments
- **SIGKILL on timeout** - SIGTERM may be ignored by hung CUDA operations
- **Process group kill** - worker calls `os.setpgrp()` so all descendants (vLLM workers, MPI ranks) are killed together

## Manifest

`ManifestWriter` writes `manifest.json` after every state transition (pending → running → completed/failed). Provides foundation for `--resume` support:

```python
from llenergymeasure.study import ManifestWriter, create_study_dir

study_dir = create_study_dir(study.name, Path("results"))
manifest = ManifestWriter(study, study_dir)

manifest.mark_running(config_hash, cycle=1)
manifest.mark_completed(config_hash, cycle=1, result_file="path/to/result.json")
manifest.mark_failed(config_hash, cycle=1, error_type="EngineError", message="...")
manifest.mark_study_completed()
```

`StudyManifest` status values: `running`, `completed`, `interrupted`, `failed`.

## Cycle ordering

`ExecutionConfig.experiment_order` controls how N cycles of K experiments are ordered:

| Order | Behaviour |
|-------|-----------|
| `sequential` | All cycles of experiment 1, then all cycles of experiment 2, ... |
| `interleave` | Cycle 1 of all experiments, then cycle 2, ... (reduces thermal autocorrelation) |
| `shuffle` | Random per-cycle order, seeded from study_design_hash by default |
| `reverse` | Alternating forward/backward per cycle (counterbalanced) |
| `latin_square` | Williams balanced latin square (balances first-order carryover effects) |

## GPU memory residual check

Before each experiment dispatch, `check_gpu_memory_residual()` polls for leftover GPU memory from previous experiments. Warns if residual memory is detected (previous experiment may not have cleaned up).

## Layer constraints

- Layer 4 - may import from layers 0-3
- Can import from: `config/`, `domain/`, `device/`, `utils/`, `energy/`, `engines/`, `datasets/`, `infra/`, `harness/`, `results/`
- Cannot import from: `api/`, `cli/`

## Related

- See `orchestration.py` for `orchestrate_study()` which instantiates `StudyRunner`; `../api/_impl.py` is the thin public adapter
- See `../harness/` for the measurement lifecycle each subprocess runs
- See `../infra/docker_runner.py` for Docker dispatch path
