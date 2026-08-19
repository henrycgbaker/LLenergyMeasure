# llenergymeasure Package

Main package for the LLM efficiency measurement framework.

## Package Structure

```
llenergymeasure/
├── __init__.py         # Public API (run_experiment, run_study)
├── _version.py         # Package version (zero internal imports)
├── api/                # Public Python API (_impl.py)
├── engines/           # Engine plugins (pytorch, vllm, tensorrt) + protocol
├── cli/                # Typer CLI (run, config)
├── config/             # Configuration system (SSOT models)
├── datasets/           # Built-in prompt datasets
├── device/             # GPU info, NVML, power/thermal, gpu_indices, env metadata
├── domain/             # Domain models (ExperimentResult, etc.) - pure Pydantic
├── energy/             # Energy samplers (NVML, Zeus, CodeCarbon)
├── entrypoints/        # Docker container entry point
├── harness/            # MeasurementHarness, preflight, environment, warmup
├── infra/              # Docker runner, image registry, runner resolution
├── results/            # Results persistence, aggregation, extended metrics
├── search/             # Reserved: candidate enumeration and search policy
├── serving/            # Reserved: engine-server lifecycle vocabulary
├── study/              # Study (sweep) runner, grid expansion, preflight
└── utils/              # Shared exceptions, constants, security
```

## Layer Architecture

```
Layer 10: cli/                - Typer CLI (llem run, llem doctor)
Layer  9: api/                - Public Python API (run_experiment, run_study)
Layer  8: study/              - Study runner, grid expansion, manifest, study preflight
Layer  7: entrypoints/        - In-container process entry points
Layer  6: harness/            - MeasurementHarness, experiment preflight, env snapshot
Layer  5: results/            - Results persistence, aggregation, extended metrics
Layer  4: engines/ | energy/ | datasets/  - Engine plugins, energy samplers, prompt loading
Layer  3: infra/              - Docker runner, image registry, runner resolution
Layer  2: device/             - GPU info, NVML, power/thermal, gpu_indices, env metadata collection
Layer  1: config/ | domain/   - Config models, domain result models (pure Pydantic)
Layer  0: utils/              - Exceptions, constants, security
```

Layer rules are enforced in CI by `import-linter` (see `[tool.importlinter]` in `pyproject.toml`).

Upper layers may import lower layers but not vice versa. Packages joined by `|` on one
line are siblings at the same level and must not import each other.

`search/` and `serving/` are reserved placeholders: they carry no code yet, but the
boundary contracts already name them, so the first module landed in either one starts out
inside the rules. `search/` will take the level above `study/`; `serving/` will take the
level above `infra/`.

## Key Files

### api/
Public Python API entry point:
- `run_experiment(config, **kwargs)` - single experiment
- `run_study(config)` - multi-experiment sweep
- `save_result(result, output_dir)` - re-exported from results.persistence for CLI use

### cli/
Modular Typer CLI:
- `llem run [config.yaml]` - run single experiment or multi-experiment study
- `llem doctor [--check] [--json]` - environment health check (GPU, engines, energy, Docker, config, image schema)
- Uses `_version.py` directly for version string (not the package root)

### entrypoints/ (Layer 7)
- `container.py` - Docker container entry point
- `baseline_measure.py` - in-container idle-baseline measurement entry point
- Imports from `harness/`, `results/`, `engines/`, `infra/`, `device/`, `config/`, `domain/`,
  `utils/` - all valid downward imports. Reaching `study/` or `api/` from here is forbidden

### utils/exceptions.py
Exception hierarchy rooted at `LLEMError`:
- `ConfigError` - config loading/validation
- `EngineError` - inference engine failures
- `PreFlightError` - pre-flight check failures
- `ExperimentError` - experiment execution errors
- `StudyError` - study orchestration errors
- `DockerError` - Docker container dispatch errors

### utils/security.py
Security utilities:
- `trust_remote_code_enabled()` - read the `LLEM_TRUST_REMOTE_CODE` opt-in

## Submodules

| Module          | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `api/`          | Public Python API (run_experiment, run_study)                    |
| `engines/`     | Inference engine plugins (pytorch, vllm, tensorrt)              |
| `cli/`          | Typer CLI commands (run, config)                                 |
| `config/`       | Configuration loading, SSOT models, introspection                |
| `datasets/`     | Built-in prompt datasets                                         |
| `device/`       | GPU info, power/thermal querying, gpu_indices, env metadata      |
| `domain/`       | Pydantic models for experiments and results (pure data, no I/O)  |
| `energy/`       | Energy samplers (NVML, Zeus, CodeCarbon)                         |
| `entrypoints/`  | Docker container entry point                                     |
| `harness/`      | MeasurementHarness, experiment preflight, environment snapshot   |
| `infra/`        | Docker runner, image registry, runner resolution                 |
| `results/`      | Results persistence, aggregation, extended efficiency metrics    |
| `search/`       | Reserved: candidate enumeration policy and search driver         |
| `serving/`      | Reserved: engine-server lifecycle vocabulary                     |
| `study/`        | Study runner, grid expansion, manifest, study preflight          |
| `utils/`        | Shared exceptions, constants, security utilities                 |

## Layer-by-Layer Notes

### `cli/` (Layer 10)
- `cli/` imports only from `api/`. Must not import harness, engines, energy, infra, study, or
  results directly.
- `cli/__init__.py` imports version from `_version.py` directly (not the package root), avoiding
  the heavy `__init__.py` import chain.

### `entrypoints/` (Layer 7)
- `entrypoints/container.py` is the Docker-side entry point, started by the container runner
  with one already-resolved experiment. It may import from any lower layer, but not from
  `study/` or `api/`: re-entering orchestration from inside a measurement container would
  nest one run inside another.

### `api/` (Layer 9)
- `_impl.py` - implementation of `run_experiment` / `run_study`
- `__init__.py` - re-exports `run_experiment`, `run_study`, and `save_result`

### `study/` (Layer 8)
- `runner.py` - orchestrates multi-experiment sweeps
- `preflight.py` - study-level pre-flight validation (multi-engine Docker requirements)

### `harness/` (Layer 6) and `results/` (Layer 5)
- `harness/measurement.py` - `MeasurementHarness` measurement lifecycle (re-exported via `harness/__init__.py`)
- `harness/preflight.py` - experiment-level pre-flight checks (CUDA, engine, model)
- `harness/environment.py` - environment snapshot collection
- `harness/warmup.py` - thermal floor wait and warmup utilities
- `results/persistence.py` - `save_result()` / `load_result()` repository functions
- `results/extended_metrics.py` - efficiency metrics computation (tokens/joule, joules/token, etc.)
- Harness owns the NVML measurement window; engine compilation must never occur inside it

### `engines/` (Layer 4)
- `protocol.py` - `EnginePlugin` protocol
- `transformers.py`, `vllm.py`, `tensorrt.py` - engine implementations
- `_observed.py` - observed-runtime-data extraction (effective params, per-request metrics)
- `_cuda.py` - CUDA memory and warmup helpers
- `_errors.py` - OOM and import error helpers

### `energy/` (Layer 4)
- `base.py` - `EnergySampler` base class
- `nvml.py`, `zeus.py`, `codecarbon.py` - energy sampler implementations

### `infra/` (Layer 3)
- `docker_runner.py` - Docker dispatch (DockerRunner)
- `runner_resolution.py` - process vs container runner selection
- `image_registry.py` - Docker image registry and version tagging
- `docker_preflight.py` - Docker-level pre-flight checks

### `device/` (Layer 2)
- `gpu_info.py` - `nvml_context()`, `_resolve_gpu_indices()`, `get_gpu_architecture()`
- `power_thermal.py` - `PowerThermalSampler`, `ThrottleInfo`
- `environment.py` - hardware metadata collection via NVML and CUDA version detection
- Placed above `config/` and `domain/` because `power_thermal.py` returns `ThrottleInfo`
  from `domain/metrics.py` (valid downward import from Layer 2 to Layer 1)

### `config/` and `domain/` (Layer 1)
- Zero imports from upper layers. Pure configuration and data models.
- `domain/` models are purely Pydantic: no collection logic, no Active Record methods.
  Use `results.persistence.save_result()` / `load_result()` for persistence.

### `utils/` (Layer 0)
- `exceptions.py`, `constants.py`, `security.py` - shared utilities
- No imports from any other llenergymeasure layer

## Version Access

`_version.py` at package root contains only `__version__`. Modules needing the version string
import from `llenergymeasure._version` (not the package root) to avoid triggering the heavy
`__init__.py` import chain (which loads `api._impl` and all of its dependencies).

```python
# Correct: zero-dependency version access
from llenergymeasure._version import __version__

# Also works (public API), but triggers the full import chain
import llenergymeasure
llenergymeasure.__version__
```

## Usage

```python
from llenergymeasure import run_experiment, run_study
from llenergymeasure.config import ExperimentConfig
from llenergymeasure.domain import ExperimentResult
```

## Related

- See `cli/README.md` for CLI architecture
- See `config/README.md` for configuration system
- See `[tool.importlinter]` in `pyproject.toml` for the enforced layer and boundary rules
