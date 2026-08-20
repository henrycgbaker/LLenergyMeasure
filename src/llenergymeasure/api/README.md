# api/ - Public Library API

Public Python API entry point for llenergymeasure. Layer 5 in the six-layer architecture.

## Purpose

Exposes `run_experiment()` and `run_study()` as the primary library interface. Also owns pre-flight validation and GPU index resolution. All user code and the CLI route through this layer.

## Modules

| Module | Description |
|--------|-------------|
| `_impl.py` | Thin `run_experiment()` and `run_study()` implementations; normalise input via `_to_study_config()` and delegate to `study.orchestration.orchestrate_study` |
| `_gpu.py` | `_resolve_gpu_indices()` - per-engine GPU index resolution |
| `preflight.py` | `run_preflight()` and `run_study_preflight()` - pre-experiment checks |
| `__init__.py` | Re-exports `run_experiment`, `run_study` |

## Public API

```python
from llenergymeasure import run_experiment, run_study
```

### run_experiment()

Three call forms:

```python
# YAML file path
result = run_experiment("configs/my-experiment.yaml")

# Config object
result = run_experiment(ExperimentConfig(model="gpt2", ...))

# Keyword convenience
result = run_experiment(model="gpt2", engine="pytorch", n=100)
```

Returns `ExperimentResult`. Side-effect free unless `output_dir` is set in the config.

### run_study()

```python
result = run_study("configs/sweep.yaml")
result = run_study(study_config)
```

Returns `StudyResult`. Always writes `manifest.json` to disk.

### skip_preflight

Both functions accept `skip_preflight=True` to bypass Docker and CUDA checks (useful in tests or environments where the GPU is inside the container):

```python
result = run_experiment(model="gpt2", skip_preflight=True)
```

## pre-flight checks (preflight.py)

`run_preflight(config)` runs before any GPU allocation:

1. CUDA available (`torch.cuda.is_available()`)
2. Engine package installed (`transformers`, `vllm`, or `tensorrt_llm`)
3. Model accessible (local path exists, or HuggingFace Hub reachable)
4. `build_config_probe(config).errors` - host-GPU compatibility via `EnginePlugin.check_hardware` (e.g., FP8 on non-Ada GPUs)

All failures are collected and raised together as a single `PreFlightError` so the user sees all problems at once.

`run_study_preflight(study)` adds precedence-based multi-engine Docker elevation: an engine with an explicit runner pin (env var / study YAML / user config) keeps its runner, while engines left on auto-detection are elevated to a container for isolation. Engines pinned to `process` are checked for host importability, and Docker is required only when an auto-resolved engine needs elevating. Single-engine studies pass through.

## GPU index resolution (_gpu.py)

`_resolve_gpu_indices(config)` determines which GPUs to monitor for energy measurement:

| Engine | Rule |
|---------|------|
| `vllm` | `tp_size * pp_size` GPUs |
| `tensorrt` | `tp_size` GPUs |
| `pytorch` with `device_map` | All NVML-visible GPUs |
| Otherwise | `[0]` (single-GPU default) |

## Dispatch flow

`api/` is a thin adapter; the orchestration itself lives in the study layer
(`study.orchestration.orchestrate_study`).

```
run_experiment(...)
  └─ _to_study_config()       # normalise all input forms to StudyConfig
run_study(...)
  └─ (adapter: map output_dir -> results_dir_override / resume search base)
     ├─ study.loading.resolve_study(...)               # the ONE resolution entry:
     │                                                 # dedup, design hash, cycles,
     │                                                 # equivalence groups, gaps.
     │                                                 # YAML studies pass through it
     │                                                 # inside load_study; a caller-built
     │                                                 # StudyConfig passes through it here.
     └─ study.orchestration.orchestrate_study(study)   # study-layer dispatcher
                                                       # (asserts the study is resolved)
           ├─ run_study_preflight()
           ├─ resolve_study_runners()
           ├─ create_study_dir() + ManifestWriter
           ├─ run_single_experiment()  # single experiment (study layer): harness or DockerRunner directly
           └─ _run_via_runner()        # multi-experiment: StudyRunner drives one session per experiment
```

## Layer constraints

- Layer 5 - may import from layers 0-4
- Cannot be imported by: `harness/`, `engines/`, `energy/`, `infra/`, `study/`, `device/`, `utils/`, `config/`, `domain/`, `datasets/`, `results/`
- The `cli/` layer is the only layer above `api/`

## Related

- See `../harness/` for measurement lifecycle
- See `../study/` for multi-experiment sweep execution
- See `../infra/` for Docker runner dispatch
