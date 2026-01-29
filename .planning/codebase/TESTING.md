# Testing Patterns

**Analysis Date:** 2026-01-26

## Test Framework

**Runner:**
- Framework: Pytest (configured via `pyproject.toml`)
- Config: `tool.pytest.ini_options` in `pyproject.toml`
- Test discovery: `tests/test_*.py` and `test_*()` functions

**Assertion Library:**
- PyTest built-in assertions (no extra library needed)
- Use `pytest.approx()` for floating-point comparisons
- Use `pytest.raises()` for exception testing

**Run Commands:**
```bash
make test                           # Unit tests only (tests/unit/)
make test-integration               # Integration tests (tests/integration/)
make test-all                       # Unit + integration (excludes tests/runtime/)
make test-runtime                   # Runtime tests with GPU (tests/runtime/)
make test-runtime-quick             # Runtime tests, quick subset

# Direct pytest
poetry run pytest tests/unit/ -v
poetry run pytest tests/ -v --ignore=tests/runtime/
poetry run pytest tests/runtime/ -v --backend pytorch
```

**Addopts:**
- Verbosity: `-v` (shows individual test names)
- Traceback: `--tb=short` (compact error output)

## Test File Organization

**Location:**
- Unit tests: `tests/unit/` (no GPU, isolated component tests)
- Integration tests: `tests/integration/` (component interaction, simulated results)
- E2E tests: `tests/e2e/` (full workflow tests, no GPU)
- Runtime tests: `tests/runtime/` (GPU-required, parameter validation)
- Fixtures: `tests/conftest.py` and `tests/fixtures/`
- Fixtures per test module: `tests/unit/conftest.py`, `tests/runtime/conftest.py`

**Naming:**
- Test files: `test_<module>.py` (mirrors source structure)
  - `test_core_inference.py` → tests for `src/llenergymeasure/core/inference.py`
  - `test_config_models.py` → tests for `src/llenergymeasure/config/models.py`
  - `test_orchestration_runner.py` → tests for `src/llenergymeasure/orchestration/runner.py`
- Test classes: `Test<Feature>` (e.g., `TestCalculateInferenceMetrics`, `TestConfigModels`)
- Test functions: `test_<descriptor>()` (e.g., `test_basic_metrics()`, `test_empty_latencies()`)

**Structure:**
```
tests/
├── conftest.py                          # Global pytest config, ANSI stripping
├── conftest_backends.py                 # MockBackend for testing
├── unit/
│   ├── test_config_*.py
│   ├── test_core_*.py
│   ├── test_domain_*.py
│   ├── test_orchestration_*.py
│   └── test_results_*.py
├── integration/
│   ├── test_config_aggregation_pipeline.py
│   ├── test_cli_workflows.py
│   ├── test_error_handling.py
│   └── test_config_params_wired.py
├── e2e/
│   └── test_cli_e2e.py
├── runtime/
│   ├── conftest.py                      # GPU fixtures, backend detection
│   ├── test_all_params.py               # SSOT param discovery + testing
│   ├── test_runtime_params.py           # Pytest wrapper for parametrised tests
│   └── discover_params.py               # Param discovery utility
└── fixtures/
    └── __init__.py
```

## Test Structure

**Suite Organization:**
- Pytest automatically discovers `Test*` classes and `test_*()` functions
- Group related tests in classes for organization (not required for execution)
- Use fixtures for shared setup (see Fixtures section)

**Patterns:**

Example from `tests/unit/test_config_models.py`:
```python
import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import TrafficSimulation


class TestTrafficSimulation:
    """Tests for TrafficSimulation (MLPerf-style traffic patterns)."""

    def test_defaults(self):
        """Verify default values."""
        config = TrafficSimulation()
        assert config.enabled is False
        assert config.mode == "poisson"
        assert config.target_qps == 1.0

    def test_target_qps_must_be_positive(self):
        """Validate constraint: target_qps > 0."""
        with pytest.raises(ValidationError):
            TrafficSimulation(target_qps=0)
        with pytest.raises(ValidationError):
            TrafficSimulation(target_qps=-1.0)

    def test_seed_for_reproducibility(self):
        """Verify seed enables reproducible Poisson arrivals."""
        config = TrafficSimulation(enabled=True, mode="poisson", seed=42)
        assert config.seed == 42
```

**Setup/Teardown:**
- Use `@pytest.fixture` for setup (preferred over `setUp()`/`tearDown()`)
- Fixtures are function-scoped by default (created per test)
- Use `scope="module"` or `scope="session"` for expensive setups
- Use `tmp_path` fixture for temporary directories (auto-cleaned)

## Mocking

**Framework:** Python `unittest.mock` (built-in)

**Patterns:**
```python
from unittest.mock import MagicMock, PropertyMock

# Simple mock
loader = MagicMock()
loader.load.return_value = (MagicMock(), MagicMock())
loader.load.assert_called_once()

# Property mock (for read-only properties)
mock_device = MagicMock()
type(mock_device).index = PropertyMock(return_value=0)

# Verify call order
components["model_loader"].load.assert_called_once()
components["energy_backend"].start_tracking.assert_called_once()
components["inference_engine"].run.assert_called_once()
```

Example from `tests/unit/test_orchestration_runner.py`:
```python
@pytest.fixture
def mock_components():
    """Create mock components for the orchestrator."""
    loader = MagicMock()
    loader.load.return_value = (MagicMock(), MagicMock())

    inference_engine = MagicMock()
    inference_engine.run.return_value = MagicMock()

    energy_backend = MagicMock()
    energy_backend.start_tracking.return_value = MagicMock()
    energy_backend.stop_tracking.return_value = EnergyMetrics(
        total_energy_j=10.0,
        gpu_power_w=100.0,
        duration_sec=1.0,
    )

    repository = MagicMock()
    repository.save_raw.return_value = Path("/tmp/result.json")

    return {
        "model_loader": loader,
        "inference_engine": inference_engine,
        "energy_backend": energy_backend,
        "repository": repository,
    }
```

**What to Mock:**
- External dependencies (model loader, energy backend, repository)
- GPU operations (to run tests without GPU)
- Network calls or slow I/O operations
- Components being tested in isolation

**What NOT to Mock:**
- The class under test (test the real implementation)
- Pydantic models (test validation is working)
- Pure utility functions
- Business logic of dependencies

## Fixtures and Factories

**Test Data:**
Fixtures for common test objects:

```python
@pytest.fixture
def sample_config():
    """Create a sample experiment config."""
    return ExperimentConfig(
        config_name="test_config",
        model_name="test/model",
        gpus=[0],
        num_processes=1,
    )

@pytest.fixture
def sample_raw_result():
    """Create a sample raw process result."""
    return RawProcessResult(
        experiment_id="test_exp_001",
        process_index=0,
        gpu_id=0,
        config_name="test_config",
        model_name="test-model",
        timestamps=Timestamps(
            start=datetime(2024, 1, 1, 10, 0, 0),
            end=datetime(2024, 1, 1, 10, 1, 0),
            duration_sec=60.0,
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=1000,
            input_tokens=500,
            output_tokens=500,
            inference_time_sec=60.0,
            tokens_per_second=16.67,
            latency_per_token_ms=60.0,
        ),
        energy_metrics=EnergyMetrics(
            total_energy_j=100.0,
            gpu_energy_j=80.0,
            cpu_energy_j=20.0,
            duration_sec=60.0,
        ),
        compute_metrics=ComputeMetrics(
            flops_total=1e12,
            flops_per_second=1.67e10,
            flops_method="calflops",
            flops_confidence="high",
        ),
    )
```

**Location:**
- Global fixtures: `tests/conftest.py`
- Test-type-specific fixtures: `tests/unit/conftest.py`, `tests/runtime/conftest.py`
- Test-file-local fixtures: Inside test file as `@pytest.fixture`

## Coverage

**Requirements:** Not strictly enforced, but coverage tracking enabled

**View Coverage:**
```bash
poetry run pytest tests/unit/ --cov=src/llenergymeasure --cov-report=html
# Then open htmlcov/index.html
```

**Configuration:**
- `pytest-cov` included in `[dev]` extras
- Coverage reports: HTML (htmlcov/), term-missing

## Test Types

**Unit Tests (`tests/unit/`):**
- Fast, isolated component tests
- No GPU required
- Use mocks for external dependencies
- Run with `make test`
- Examples:
  - `test_config_models.py`: Pydantic validation
  - `test_core_inference.py`: Metrics calculation
  - `test_exceptions.py`: Exception hierarchy
  - `test_orchestration_runner.py`: Component orchestration (mocked)

**Integration Tests (`tests/integration/`):**
- Component interaction tests
- May touch file system (using `tmp_path`)
- No GPU required (use simulated results)
- Run with `make test-integration`
- Examples:
  - `test_config_aggregation_pipeline.py`: Config → aggregation → export pipeline
  - `test_cli_workflows.py`: CLI command interaction with real repository
  - `test_error_handling.py`: Error propagation across layers
  - `test_config_params_wired.py`: Backend param wiring

**E2E Tests (`tests/e2e/`):**
- Full workflow tests (config → execution → results)
- Simulate inference (no real GPU calls)
- Run with `make test-all`
- Example:
  - `test_cli_e2e.py`: End-to-end CLI workflow

**Runtime Tests (`tests/runtime/`):**
- GPU-required parameter validation tests
- CANONICAL parameter testing (SSOT-derived)
- Use `--backend pytorch`, `--backend vllm`, etc. to test specific backends
- Run with `make test-runtime`

### Runtime Parameter Testing

The `tests/runtime/` directory contains tests that run real inference:

**Key files:**
- `test_all_params.py`: Standalone + pytest test for all backend params
- `test_runtime_params.py`: Pytest wrapper with parametrisation
- `discover_params.py`: Utility for auto-discovering params from SSOT

**Param discovery (SSOT):**
Parameters are auto-discovered from Pydantic models in `config/backend_configs.py`, not hardcoded:

```bash
# List auto-discovered params for a backend
python -m tests.runtime.test_all_params --list-params --backend pytorch

# Run tests for specific params
pytest tests/runtime/ -v --backend pytorch
pytest tests/runtime/ -v --backend vllm

# Quick mode (fewer parameter variations)
pytest tests/runtime/ -v --quick

# Run standalone for comprehensive sweeps
python -m tests.runtime.test_all_params --backend pytorch
```

**Skip markers:**
- `@pytest.mark.requires_gpu` - Requires CUDA GPU
- `@pytest.mark.requires_vllm` - Requires vLLM installed
- `@pytest.mark.requires_tensorrt` - Requires TensorRT-LLM installed
- `@pytest.mark.slow` - Takes >1 minute

Example:
```python
@pytest.mark.requires_vllm
def test_vllm_tensor_parallel():
    """Test vLLM tensor parallelism (vLLM only)."""
    ...
```

## Common Patterns

**Floating-point Comparisons:**
```python
# Use pytest.approx for float comparisons (handles tiny precision errors)
assert result.tokens_per_second == pytest.approx(1000.0)
assert result.latency_per_token_ms == pytest.approx(10.0, rel=1e-2)  # 1% tolerance
```

**Async Testing:**
Not currently used (all code is synchronous).

**Error Testing:**
```python
# Test that error is raised with correct message
with pytest.raises(ConfigurationError, match="model_name is required"):
    ExperimentConfig(config_name="test")

# Test specific exception type
with pytest.raises(ValidationError) as exc_info:
    TrafficSimulation(target_qps=0)
assert "greater than 0" in str(exc_info.value)
```

**Parametrised Tests:**
```python
# Test multiple scenarios with @pytest.mark.parametrize
@pytest.mark.parametrize(
    "exc_class,message",
    [
        (ConfigurationError, "Invalid config"),
        (ModelLoadError, "Model not found"),
    ],
)
def test_exception_with_message(self, exc_class, message):
    err = exc_class(message)
    assert str(err) == message
```

**Temporary Directories:**
```python
# Use tmp_path fixture (auto-cleaned)
def test_save_results(tmp_path):
    result_file = tmp_path / "result.json"
    repository.save_raw(result, result_file)
    assert result_file.exists()
```

## Test Execution in CI

**CI Pipeline (`make ci`):**
1. Code formatting and linting: `make check`
2. Unit + integration tests: `make test-all`
3. Documentation freshness: `make check-docs`

**Local pre-commit:**
- Ruff format + lint run before each commit
- Type checking: `mypy src/`

## Special Test Configuration

**Global Pytest Setup (`tests/conftest.py`):**
```python
# Disable Rich colors for consistent ANSI-free output
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

# Utility: strip ANSI codes from CLI output
def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)
```

**CLI Testing (`tests/unit/test_cli.py`):**
```python
from typer.testing import CliRunner

# Use CliRunner with NO_COLOR env var for consistent output
runner = CliRunner(env={"NO_COLOR": "1"})

# Invoke CLI commands
result = runner.invoke(app, ["config", "validate", str(config_file)])
assert result.exit_code == 0
assert "Valid configuration" in result.stdout
```

---

*Testing analysis: 2026-01-26*
