# Phase 8: Testing and Integration — Research

**Researched:** 2026-02-27
**Domain:** pytest testing, protocol injection, GitHub Actions CI, two-tier test architecture
**Confidence:** HIGH

## Summary

Phase 8 is a consolidation phase that validates all subsystems built in Phases 1–7. The central challenge is not technical novelty — the testing approach is well-understood — but execution: the codebase currently has ~74 v1.x unit test files that import stale APIs and ~14 v2.0 unit test files that mostly pass. The phase must delete the v1.x tests entirely, fix the 16 currently failing v2.0 tests, and write the remaining subsystem tests that don't exist yet.

The v2.0 unit test files written during Phases 4–7 establish the correct patterns: protocol injection fakes, `make_config()`/`make_result()` factories in fixtures, `CliRunner(mix_stderr=True)` for CLI tests, and `patch()` only at public module boundaries (never deep internal modules). The GPU integration test for M1 exit criteria runs inside a Docker container with `--gpus` because CUDA is only available inside containers on this machine. The self-hosted runner approach is the right choice for GPU CI — zero cost and full control over the A100.

**Primary recommendation:** Delete all v1.x tests first (Wave 0), fix the 16 failing v2.0 tests (required fields `measurement_config_hash` and `measurement_methodology` in test factories), then write the missing subsystem tests in two waves, with the GPU integration test and CI workflows in the final wave.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Test scope and coverage**
- Start fresh — delete v1.x tests entirely, write v2.0 tests from scratch targeting new subsystems
- All subsystems get dedicated unit test files: config (schema, loader, introspection, user config), library API, PyTorch backend, energy measurement (NVML poller, FLOPs, warmup, baseline), results (schema, persistence, aggregation), CLI (run, config commands), infrastructure (protocols, state machine, resilience)
- Two tiers only: `tests/unit/` (GPU-free, fast) and `tests/integration/` (`@pytest.mark.gpu`). No e2e/ or runtime/ directories.
- No numeric coverage target — the 5 success criteria are the bar. Coverage follows naturally from testing all subsystems.

**Mock and fixture design**
- Fake protocol classes implementing InferenceBackend, EnergyBackend etc. protocols — defined in `tests/fakes.py`. Injected via constructor/function args, not `unittest.mock.patch`.
- Layered conftest: `tests/conftest.py` for shared fixtures (sample configs, tmp dirs), `tests/fakes.py` for protocol fakes, per-directory conftest.py only if needed.
- `make_config(**overrides)` factory in conftest.py — returns valid ExperimentConfig with sensible defaults. Tests override only what matters for each test case.
- Schema-driven test generation where natural (use config introspection to generate edge-case configs — boundary values, all backends, all precisions). Hardcode specific regression values where clarity matters.

**GPU integration strategy**
- Self-hosted runner on user's A100 machine (free, no GitHub GPU runner costs)
- Test model: gpt2 (124M) — matches M1 exit criterion (`llem run --model gpt2 --backend pytorch`)
- Full result validation: assert ExperimentResult has non-zero energy_total_j, valid timeseries path, tokens_per_second > 0, environment snapshot populated, schema_version '2.0'
- GPU tests run inside container (Docker with --gpus) since CUDA only available inside containers on this machine

**CI workflow design**
- Two separate workflows:
  - `ci.yml`: unit tests on every PR/push (GitHub-hosted runner, fast, free)
  - `gpu-ci.yml`: GPU integration tests on merge to main + weekly + manual (self-hosted runner)
- Unit CI checks: pytest, ruff lint + format, mypy type checking, import validation (`from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult`)
- Python version matrix: 3.10 + 3.12
- Branch protection on main: unit CI must pass before merge. GPU CI is not required (runs post-merge).

### Claude's Discretion
- Whether to extract factory functions from fakes if repetition appears (start with plain fakes)
- Exact test file naming and organisation within each tier
- Self-hosted runner setup details (runner agent configuration, labels)
- Specific pytest marks and fixture scoping decisions
- Whether path-filtered CI triggers are worth the configuration complexity

### Deferred Ideas (OUT OF SCOPE)
- Version numbering: M1 should ship as v1.x (not v2.0). v2.0 should mean all backends (vLLM, TensorRT-LLM, Docker multi-backend) are complete. Affects Phase 1's `__version__` setting and overall product versioning strategy.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STU-05 | Single experiment (`llem run experiment.yaml`) runs in-process — no subprocess | `_run()` in `_api.py` already runs in-process; integration test verifies the full pipeline executes without spawning subprocesses |
| INF-09 | Two-tier test structure: `tests/unit/` (no GPU) + `tests/integration/` (`@pytest.mark.gpu`) | Confirmed in codebase layout; pytest mark registration needs adding to `pyproject.toml`; `gpu-ci.yml` workflow is the home for GPU-marked tests |
| INF-10 | Protocol injection mocks (not `unittest.mock` patching on internal modules) | `InferenceBackend` and `EnergyBackend` protocols in `protocols.py` and `core/backends/protocol.py` are `@runtime_checkable` — fakes implement them structurally; existing Phase 7 tests show the pattern |
| INF-11 | Config introspection drives test value generation (SSOT) | `config/introspection.py` has `get_param_test_values()`, `get_experiment_config_schema()`, and `get_backend_params()` — these feed schema-driven test generation; `make_config()` can use these for boundary values |
| INF-12 | GPU CI: merge to main + weekly + manual | `gpu-ci.yml` workflow with `push` to main + `schedule` (weekly cron) + `workflow_dispatch` triggers; self-hosted runner with label `self-hosted`; Docker `--gpus all` for actual GPU access |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=8.0 (already in `[dev]`) | Test runner, fixtures, marks, parametrize | Already in pyproject.toml; the industry standard |
| pytest-cov | >=4.0 (already in `[dev]`) | Coverage reporting | Already included; useful even without a hard target |
| typer.testing.CliRunner | bundled with typer | CLI test invocation | Established pattern in existing Phase 7 tests |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock.patch | stdlib | Patch public module boundaries only | CLI tests only — patch at `llenergymeasure.cli.run.run_experiment`, never at `llenergymeasure._api._run` |
| unittest.mock.MagicMock | stdlib | Return-value stand-ins | Mock ExperimentResult return values in CLI tests; NOT for protocol fakes |
| pydantic.ValidationError | pydantic >=2.0 | Triggered by bad field values | Unit tests verify it passes through unchanged per CLI-14 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Protocol injection fakes | `unittest.mock.MagicMock` | MagicMock is implicit; fake classes are transparent — reading the fake shows exactly what it returns |
| `CliRunner(mix_stderr=True)` | `CliRunner(mix_stderr=False)` | Typer's CliRunner doesn't reliably capture `sys.stderr` in `.stderr`; `mix_stderr=True` gives `.output` as combined stream; confirmed in Phase 7 |
| Self-hosted runner for GPU | GitHub-hosted GPU runner | GitHub GPU runners cost ~$0.07/min × many minutes per run; A100 self-hosted is free |
| Docker `--gpus all` in GPU CI | Direct host CUDA | CUDA not available on host outside containers on this machine |

**Installation:**
```bash
# Already in pyproject.toml [dev] extras — no new additions needed
pip install -e ".[dev]"
```

---

## Architecture Patterns

### Recommended Project Structure

```
tests/
├── conftest.py            # Shared fixtures: make_config(), make_result(), tmp dirs
├── fakes.py               # Protocol fake classes: FakeInferenceBackend, FakeEnergyBackend
├── unit/
│   ├── __init__.py
│   ├── test_config_schema.py       # ExperimentConfig Pydantic validation
│   ├── test_config_loader.py       # load_experiment_config(), YAML parsing, ConfigError
│   ├── test_config_introspection.py # model_json_schema(), get_param_test_values()
│   ├── test_config_user_config.py  # UserConfig loading, env vars, XDG path
│   ├── test_api.py                 # run_experiment(), run_study(), __version__  [EXISTS - fix 4 tests]
│   ├── test_backend_protocol.py    # InferenceBackend protocol conformance  [EXISTS - fix 2 tests]
│   ├── test_preflight.py           # run_preflight(), PreFlightError  [EXISTS - passing]
│   ├── test_environment_snapshot.py # EnvironmentSnapshot collection  [EXISTS - passing]
│   ├── test_energy_backends_v2.py  # NVMLBackend, ZeusBackend, select_energy_backend()  [EXISTS - passing]
│   ├── test_flops_v2.py            # estimate_flops_palm()  [EXISTS - passing]
│   ├── test_warmup_v2.py           # WarmupRunner, CV convergence  [EXISTS - fix 2 tests]
│   ├── test_experiment_result_v2.py # ExperimentResult schema  [EXISTS - passing]
│   ├── test_persistence_v2.py      # to_json(), from_json(), to_parquet()  [EXISTS - passing]
│   ├── test_aggregation_v2.py      # aggregate_results()  [EXISTS - passing]
│   ├── test_cli_run.py             # llem run command  [EXISTS - passing]
│   ├── test_cli_config.py          # llem config command  [EXISTS - passing]
│   ├── test_cli_display.py         # _display.py, _vram.py  [EXISTS - passing]
│   ├── test_measurement_integration.py  # _build_result() wiring  [EXISTS - fix 2 tests]
│   ├── test_state_machine.py       # ExperimentPhase, ExperimentState, StateManager
│   ├── test_exceptions.py          # LLEMError hierarchy  [EXISTS BUT BROKEN - v1 API]
│   ├── test_resilience.py          # retry logic  [EXISTS BUT BROKEN - v1 API]
│   └── test_protocols.py           # 5 DI protocols structural conformance  [EXISTS BUT BROKEN]
└── integration/
    ├── __init__.py
    └── test_gpu_experiment.py      # @pytest.mark.gpu — real gpt2 on A100 inside Docker
```

### Pattern 1: Protocol Injection Fakes (INF-10)
**What:** Fake classes that structurally implement protocol interfaces. No `MagicMock`. Behaviour is explicit in the class body.
**When to use:** Any unit test that exercises code calling `InferenceBackend.run()`, `EnergyBackend.start_tracking()`, or `ResultsRepository.save()`
**Example:**
```python
# tests/fakes.py
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult

class FakeInferenceBackend:
    """Minimal InferenceBackend fake — returns a pre-built ExperimentResult."""

    name = "fake"

    def __init__(self, result: ExperimentResult | None = None):
        self._result = result  # set in test via make_result()

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        if self._result is None:
            raise ValueError("FakeInferenceBackend: set .result before calling run()")
        return self._result


class FakeEnergyBackend:
    """Minimal EnergyBackend fake — returns fixed EnergyMeasurement."""

    name = "fake-energy"

    def start_tracking(self):
        return object()  # opaque tracker handle

    def stop_tracking(self, tracker) -> "EnergyMeasurement":
        from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement
        return EnergyMeasurement(total_j=10.0, duration_sec=5.0)
```

### Pattern 2: make_config() and make_result() Factories (conftest.py)
**What:** Factory functions in `tests/conftest.py` that return valid model objects with sensible defaults. Tests override only what they care about.
**When to use:** Every test that needs an `ExperimentConfig` or `ExperimentResult`

```python
# tests/conftest.py
import pytest
from datetime import datetime
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult


def make_config(**overrides) -> ExperimentConfig:
    """Return a valid ExperimentConfig with sensible defaults."""
    defaults = {
        "model": "gpt2",
        "backend": "pytorch",
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def make_result(**overrides) -> ExperimentResult:
    """Return a valid ExperimentResult with sensible defaults."""
    from llenergymeasure.domain.experiment import AggregationMetadata
    defaults = {
        "experiment_id": "test-001",
        "measurement_config_hash": "abc123def456",  # 16-char hex
        "measurement_methodology": "total",
        "aggregation": AggregationMetadata(num_processes=1),
        "total_tokens": 1000,
        "total_energy_j": 10.0,
        "total_inference_time_sec": 5.0,
        "avg_tokens_per_second": 200.0,
        "avg_energy_per_token_j": 0.01,
        "total_flops": 1e9,
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return make_config()


@pytest.fixture
def tmp_results_dir(tmp_path):
    return tmp_path / "results"
```

**Critical note:** The 16 currently failing v2.0 tests all fail because their `ExperimentResult` construction omits the two required fields: `measurement_config_hash` (a 16-char hex str) and `measurement_methodology` (Literal `"total"`, `"steady_state"`, `"windowed"`). The fix is to add these fields to the `make_result()` factory in each test file — or better, standardise them in the shared `conftest.py` factory.

### Pattern 3: @pytest.mark.gpu for Integration Tests
**What:** All GPU-requiring tests marked `@pytest.mark.gpu`. Run only in `gpu-ci.yml`.
**When to use:** Any test that loads a real model, runs real inference, or calls `torch.cuda.*`

```python
# tests/integration/test_gpu_experiment.py
import pytest

@pytest.mark.gpu
class TestGPUExperiment:
    """GPU integration tests — require @pytest.mark.gpu, run inside Docker container."""

    def test_run_gpt2_pytorch_returns_valid_result(self, tmp_path):
        """llem run --model gpt2 --backend pytorch produces a valid ExperimentResult."""
        from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult

        config = ExperimentConfig(
            model="gpt2",
            backend="pytorch",
            n=10,
            output_dir=str(tmp_path),
        )
        result = run_experiment(config)

        assert isinstance(result, ExperimentResult)
        assert result.schema_version == "2.0"
        assert result.total_energy_j > 0
        assert result.avg_tokens_per_second > 0
        assert result.environment_snapshot is not None
        assert result.measurement_config_hash  # non-empty
```

### Pattern 4: pytest marks registration in pyproject.toml
**What:** Register the `gpu` mark to silence pytest warnings.
**Required change to pyproject.toml:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "gpu: marks tests as requiring GPU hardware (deselect with '-m \"not gpu\"')",
]
```

### Pattern 5: CI Workflow Structure

**`ci.yml`** (GitHub-hosted, unit tests only):
```yaml
name: CI

on:
  push:
    branches: [main, "gsd/*", "feature/*", "fix/*"]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -m "not gpu" -v --tb=short
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      - run: mypy src/
      - run: python -c "from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult"
```

**`gpu-ci.yml`** (self-hosted runner, GPU integration tests):
```yaml
name: GPU CI

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 2 * * 1"   # weekly Monday 02:00 UTC
  workflow_dispatch:        # manual trigger

jobs:
  gpu-integration:
    runs-on: [self-hosted, gpu]   # label must match runner registration
    steps:
      - uses: actions/checkout@v4
      - name: Build GPU test image
        run: docker build -f docker/Dockerfile.pytorch -t llem-test:pytorch .
      - name: Run GPU integration tests
        run: |
          docker run --rm --gpus all \
            -v ${{ github.workspace }}:/workspace \
            -w /workspace \
            llem-test:pytorch \
            pytest tests/integration/ -m gpu -v --tb=short
```

### Anti-Patterns to Avoid

- **`unittest.mock.patch` on internal modules:** Never patch `llenergymeasure._api._run` or `llenergymeasure.core.backends.pytorch.PyTorchBackend.run`. Patch at public call sites only (`llenergymeasure.cli.run.run_experiment`). Internal patching is brittle and defeats the purpose of the architecture.
- **MagicMock for protocol fakes:** `MagicMock()` with no spec returns truthy for every attribute, hiding real bugs. Use explicit fake classes in `tests/fakes.py` — they fail loudly when the contract is violated.
- **GPU calls in `tests/unit/`:** Any `import torch` that calls `torch.cuda.*` will fail on GitHub-hosted runners. Use `importlib.util.find_spec("torch")` guards in unit tests, or better, mock the torch import entirely via fakes.
- **Ordering the "delete v1.x tests" step last:** V1.x tests import stale APIs that no longer exist (e.g. `BUILTIN_DATASETS`, `DistributedError`, `DockerConfig`). If they run first, collection errors pollute the test suite. Delete them in Wave 0 — before writing anything new.
- **Monkeypatching `conftest.py` fixtures across tiers:** Keep `tests/conftest.py` minimal (factories + basic fixtures). GPU-specific fixtures belong in `tests/integration/conftest.py` only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Protocol conformance checking | Custom `isinstance` validators | `@runtime_checkable` Protocol + `isinstance(fake, InferenceBackend)` | Already works; protocols are `@runtime_checkable` in `protocols.py` and `core/backends/protocol.py` |
| Test value generation for ExperimentConfig fields | Hardcoded lists of values | `config/introspection.py` — `get_param_test_values()`, `get_experiment_config_schema()` | Already exists; INF-11 requires it; avoids drift when schema changes |
| CLI test harness | Custom subprocess runner | `typer.testing.CliRunner(mix_stderr=True)` | Already used in Phase 7 tests; handles stdin/stdout/stderr correctly |
| GPU availability detection in integration tests | `import torch; torch.cuda.is_available()` | `@pytest.mark.gpu` + CI filter (`-m "not gpu"`) | Unit CI never sees GPU marks; integration CI runs in Docker where CUDA is available |

**Key insight:** The test infrastructure concerns in this phase are mostly solved. The real work is writing the tests themselves and wiring the CI. Don't build abstractions that don't exist yet (factories, fixtures) before seeing where repetition actually emerges.

---

## Common Pitfalls

### Pitfall 1: ExperimentResult Required Fields Missing in Test Factories
**What goes wrong:** 16 currently failing tests build `ExperimentResult(...)` without `measurement_config_hash` and `measurement_methodology` — both are required fields with `...` defaults (no default).
**Why it happens:** These fields were added to `ExperimentResult` in Phase 6 after the test files were written. The test factories weren't updated.
**How to avoid:** Fix `make_result()` factories in the failing test files first. Better yet, add a canonical `make_result()` in `tests/conftest.py` and replace all ad-hoc `ExperimentResult(...)` constructions in tests.
**Warning signs:** `pydantic_core.ValidationError: Field required` in tests that construct `ExperimentResult` directly.

### Pitfall 2: V1.x Test Files Blocking Collection
**What goes wrong:** `tests/unit/test_exceptions.py` imports `DistributedError` which doesn't exist in the v2.0 exception hierarchy. `tests/unit/test_config_models.py` imports `BUILTIN_DATASETS` which was removed. pytest aborts collection with `ImportError`.
**Why it happens:** 23 unit test files (and 3 integration files) import v1.x APIs that were deleted/renamed in v2.0.
**How to avoid:** Delete ALL v1.x tests in Wave 0 before running `pytest tests/unit/ -v`. Only keep the 14 v2.0-compatible files already identified.
**Warning signs:** `ImportError: cannot import name 'X'` during collection.

### Pitfall 3: GPU Test Running on GitHub-Hosted Runner
**What goes wrong:** `@pytest.mark.gpu` tests appear in the regular pytest run and import `torch.cuda`, crashing on the GitHub-hosted runner.
**Why it happens:** No `pytest.mark.gpu` registration + no `-m "not gpu"` deselection in unit CI.
**How to avoid:** Register the mark in `pyproject.toml`. Add `-m "not gpu"` to the unit CI `pytest` invocation. GPU tests only run in `gpu-ci.yml` on the self-hosted runner inside Docker.
**Warning signs:** `torch.cuda.is_available()` returning `False` causing test failures on GitHub runners.

### Pitfall 4: Self-Hosted Runner Label Mismatch
**What goes wrong:** `gpu-ci.yml` uses `runs-on: [self-hosted, gpu]` but the runner was registered with different labels.
**Why it happens:** GitHub Actions matches job to runners by exact label set. If the runner was registered with label `self-hosted` only (no `gpu` label), the job queues forever.
**How to avoid:** During runner setup, explicitly add the `gpu` label: `./config.sh --labels self-hosted,gpu`. Or use just `runs-on: self-hosted` and add the `gpu` label during registration.
**Warning signs:** Job stays in "Queued" state indefinitely in GitHub Actions UI.

### Pitfall 5: Docker Image Not Built Before GPU Test Run
**What goes wrong:** `gpu-ci.yml` tries to `docker run llem-test:pytorch` but the image doesn't exist on the self-hosted runner.
**Why it happens:** Self-hosted runners start fresh or the image hasn't been built since last code change.
**How to avoid:** Always include a `docker build` step before `docker run` in `gpu-ci.yml`. Use `--no-cache` occasionally to catch dependency drift. The build step is cheap (layers cached after first run).
**Warning signs:** `docker: No such image: llem-test:pytorch`.

### Pitfall 6: test_backend_protocol.py KeyError on load_in_4bit
**What goes wrong:** Two tests in `test_backend_protocol.py` fail with `KeyError: 'load_in_4bit'`.
**Why it happens:** The test accesses `pytorch_config` fields that exist but may be nested differently after the Phase 4.1 parameter audit renamed fields or moved them into `BitsAndBytesConfig`.
**How to avoid:** Check the actual `PyTorchConfig` field names in `config/backend_configs.py`. The `BitsAndBytesConfig` nesting means quantization params are under `pytorch.bnb_config` not directly on `PyTorchConfig`.
**Warning signs:** `KeyError` on quantization-related fields in protocol tests.

---

## Code Examples

Verified patterns from the existing v2.0 codebase:

### Existing make_result pattern (needs updating)
```python
# Current pattern in tests/unit/test_api.py (working)
def _make_experiment_result(**overrides) -> ExperimentResult:
    from llenergymeasure.domain.experiment import AggregationMetadata
    defaults = {
        "experiment_id": "test-001",
        "aggregation": AggregationMetadata(num_processes=1),
        "measurement_config_hash": "abc123def45678",  # REQUIRED — 16-char hex
        "measurement_methodology": "total",            # REQUIRED
        "total_tokens": 1000,
        "total_energy_j": 10.0,
        "total_inference_time_sec": 5.0,
        "avg_tokens_per_second": 200.0,
        "avg_energy_per_token_j": 0.01,
        "total_flops": 1e9,
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)
```

### Protocol conformance check (runtime_checkable)
```python
# Source: protocols.py + InferenceBackend protocol
from llenergymeasure.core.backends.protocol import InferenceBackend
from tests.fakes import FakeInferenceBackend

def test_fake_satisfies_protocol():
    fake = FakeInferenceBackend(result=make_result())
    assert isinstance(fake, InferenceBackend)  # works because @runtime_checkable
```

### Schema-driven test generation (INF-11)
```python
# Source: config/introspection.py get_param_test_values()
from llenergymeasure.config.introspection import get_param_test_values, get_backend_params

def test_all_precisions_valid():
    """Schema-driven: test all valid precision values from introspection."""
    from llenergymeasure.config.ssot import PRECISION_SUPPORT
    for precision in PRECISION_SUPPORT["pytorch"]:
        config = make_config(precision=precision)
        assert config.precision == precision
```

### Patching public boundary only (CLI tests)
```python
# Source: tests/unit/test_cli_run.py (Phase 7)
# Patch at the CLI module's import, not at the implementation
with patch("llenergymeasure.cli.run.run_experiment") as mock_run:
    mock_run.return_value = make_result()
    result = runner.invoke(app, ["run", "--model", "gpt2"])
assert result.exit_code == 0
```

### GPU integration test (new — inside Docker)
```python
# tests/integration/test_gpu_experiment.py
import pytest
from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult

@pytest.mark.gpu
def test_gpt2_pytorch_returns_valid_result(tmp_path):
    config = ExperimentConfig(model="gpt2", backend="pytorch", n=10)
    result = run_experiment(config)
    assert isinstance(result, ExperimentResult)
    assert result.schema_version == "2.0"
    assert result.total_energy_j > 0
    assert result.avg_tokens_per_second > 0
    assert result.environment_snapshot is not None
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1.x test files testing v1.x APIs | v2.0 test files targeting new subsystem APIs | Phases 4–7 wrote 14 new test files | 23 v1.x test files now break at collection; must delete |
| `unittest.mock.patch` deep in stack | Protocol injection fakes in `tests/fakes.py` | Architecture decision from CONTEXT.md | Fakes are transparent; no hidden behaviour |
| `CliRunner()` (default) | `CliRunner(mix_stderr=True)` | Phase 7 P02 | Reliable output capture for typer tests |
| `runs-on: ubuntu-latest` for all | Split: GitHub-hosted for unit, self-hosted for GPU | CONTEXT.md CI decision | Zero cost GPU CI; fast free unit CI |
| Ad-hoc ExperimentResult construction | `make_result()` factory with required fields | Phase 8 (this phase) | Fixes 16 failing tests; prevents future required-field drift |

**Deprecated/outdated:**
- All 23 v1.x test files in `tests/unit/` (plus 3 in `tests/integration/`) that import stale APIs — delete entirely. Full list: `test_campaign_group_by.py`, `test_resume.py` (cli/), `test_campaign_config.py`, `test_webhook.py`, `test_grid.py`, `test_manifest.py`, `test_bootstrap.py`, `test_campaign.py`, `test_config_loader.py`, `test_config_models.py`, `test_constants.py`, `test_container_strategy.py`, `test_core_traffic.py`, `test_dataset_loader.py`, `test_exceptions.py`, `test_inference_generation.py`, `test_orchestration_lifecycle.py`, `test_resilience.py`, `test_results_aggregation.py`, `test_schema_version.py`, `test_security.py`, `test_state.py`, `test_user_config.py` and integration: `test_config_aggregation_pipeline.py`, `test_config_params_wired.py`, `test_error_handling.py`.
- `poetry run pytest` Makefile targets — Makefile uses `poetry run`, but the project uses `hatchling` + `pip install -e`. These targets will fail. Replace with plain `pytest`.
- `tests/e2e/`, `tests/runtime/`, `tests/fixtures/`, `tests/configs/`, `tests/conftest_backends.py` — these are v1.x artefacts. Per CONTEXT.md decision: two tiers only. Delete or repurpose.

---

## Open Questions

1. **Self-hosted runner label**
   - What we know: Runner must be configured with labels matching `gpu-ci.yml` `runs-on` specification.
   - What's unclear: Whether the runner has already been set up and what labels it uses.
   - Recommendation: Use `runs-on: self-hosted` initially (single label). If multiple self-hosted runners exist, add a `gpu` label during registration: `./config.sh --labels self-hosted,gpu,linux`.

2. **v1.x test files to keep in `tests/integration/` and other subdirs**
   - What we know: `test_cli_workflows.py`, `test_docker_naming.py`, `test_entrypoint_puid.py`, `test_repository_operations.py` collect without errors but use v1.x domain models.
   - What's unclear: Whether any of these test something still relevant to v2.0 that warrants rewriting vs. pure deletion.
   - Recommendation: Delete all. The GPU integration test and CLI success-criteria tests in Phase 8 replace them.

3. **`tests/unit/test_config_introspection.py` compatibility**
   - What we know: The file exists and imports `get_params_requiring_gpu_capability`, `get_special_test_models`, `get_mutual_exclusions` — functions that may not exist in v2.0 `introspection.py`.
   - What's unclear: Whether the introspection module still has these v1.x functions.
   - Recommendation: Check which functions are v2.0 before deciding to rewrite vs. delete. Only `get_backend_params()`, `get_param_test_values()`, `get_experiment_config_schema()` are confirmed v2.0 per the module docstring.

---

## Validation Architecture

Phase 8 IS the validation phase. The `nyquist_validation` workflow flag is not set in `config.json`, so this section follows the standard format for a testing phase.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/unit/ -m "not gpu" -x -q` |
| Full suite command | `pytest tests/unit/ -m "not gpu" -v` |
| GPU suite command | `pytest tests/integration/ -m gpu -v` (inside Docker on self-hosted) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STU-05 | Single experiment runs in-process (no subprocess) | unit | `pytest tests/unit/test_api.py::test_run_uses_in_process_runner -x` | ❌ Wave 1 |
| INF-09 | Two-tier test structure (`tests/unit/` + `tests/integration/`) | structural | `pytest tests/unit/ -m "not gpu" && pytest tests/integration/ -m gpu` | ✅ dirs exist, marks need registering |
| INF-10 | Protocol injection fakes (no unittest.mock.patch on internals) | unit | `pytest tests/unit/ -k "fake or protocol" -x` | ❌ Wave 1 (fakes.py) |
| INF-11 | Config introspection drives test value generation | unit | `pytest tests/unit/test_config_introspection.py -x` | ❌ Wave 1 (rewrite needed) |
| INF-12 | GPU CI: merge to main + weekly + manual | CI config | `gh workflow run gpu-ci.yml` | ❌ Wave 2 (`.github/workflows/gpu-ci.yml`) |

### Wave 0 Gaps (setup tasks before writing tests)
- [ ] Delete 23 v1.x `tests/unit/` files that break at collection
- [ ] Delete 3 v1.x `tests/integration/` files that break at collection
- [ ] Remove v1.x test directories (`tests/e2e/`, `tests/runtime/`, `tests/fixtures/`, `tests/configs/`)
- [ ] Add `markers` section to `pyproject.toml` `[tool.pytest.ini_options]`
- [ ] Add `make_config()` and `make_result()` canonical factories to `tests/conftest.py`
- [ ] Create `tests/fakes.py` with `FakeInferenceBackend` and `FakeEnergyBackend`

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/llenergymeasure/` — all module APIs verified by reading source files
- Direct test inspection: `tests/unit/test_cli_run.py`, `test_cli_config.py`, `test_api.py` — Phase 7 patterns confirmed
- `pytest` run output: `242 passed, 16 failed` confirmed with `pytest` v8.4.2
- `.planning/phases/08-testing-and-integration/08-CONTEXT.md` — user decisions locked

### Secondary (MEDIUM confidence)
- GitHub Actions self-hosted runner documentation: standard `runs-on: [self-hosted, label]` syntax is stable and widely documented
- Docker `--gpus all` flag: confirmed working pattern for NVIDIA GPU containers

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — confirmed by direct codebase inspection; pytest already installed
- Architecture: HIGH — patterns established in Phases 4–7 unit tests; no new patterns needed
- Pitfalls: HIGH — failures confirmed by running `pytest` and inspecting error messages

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable pytest + GitHub Actions syntax)
