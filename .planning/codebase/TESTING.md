# Testing Patterns

**Analysis Date:** 2026-02-05

## Test Framework

**Runner:**
- Framework: Pytest 8.0+
- Config location: `tool.pytest.ini_options` in `pyproject.toml`
- Test discovery: `testpaths = ["tests"]`, `python_files = ["test_*.py"]`, `python_functions = ["test_*"]`

**Assertion Library:**
- Pytest built-in assertions (no additional library)
- Use `pytest.approx()` for floating-point comparisons
- Use `pytest.raises()` for exception testing with optional `match` parameter

**Run Commands:**
```bash
make test                           # Unit tests only (tests/unit/)
make test-integration               # Integration tests (tests/integration/)
make test-all                       # Unit + integration (excludes tests/runtime/)

# Runtime tests (GPU required, Docker dispatch)
make test-runtime                   # PyTorch backend via Docker
make test-runtime-vllm              # vLLM backend via Docker
make test-runtime-tensorrt          # TensorRT backend via Docker
make test-runtime-all               # All backends via Docker
make test-runtime-quick             # Quick mode (fewer params)
make test-runtime-list              # List SSOT-discovered params
make test-runtime-check             # Check Docker setup

# Direct pytest (local environment)
poetry run pytest tests/unit/ -v
poetry run pytest tests/integration/ -v
poetry run pytest tests/ -v --ignore=tests/runtime/
pytest tests/runtime/ -v --backend pytorch --quick
```

**Pytest Configuration:**
- Verbosity: `-v` (shows individual test names)
- Traceback: `--tb=short` (compact error output)
- Addopts in `pyproject.toml`: `-v --tb=short`

## Test File Organization

**Location:**
```
tests/
├── conftest.py                          # Global fixtures: ANSI stripping, NO_COLOR env
├── conftest_backends.py                 # MockBackend for testing without GPU
├── fixtures/                            # Shared test fixtures (currently minimal)
├── unit/                                # Fast, isolated component tests (64 files)
│   ├── cli/                             # CLI command tests
│   ├── config/                          # Configuration tests
│   ├── notifications/                   # Webhook tests
│   ├── orchestration/                   # Orchestration tests
│   ├── results/                         # Results tests
│   └── test_*.py                        # Core module tests
├── integration/                         # Component interaction tests (6 files)
│   ├── test_config_aggregation_pipeline.py
│   ├── test_cli_workflows.py
│   ├── test_error_handling.py
│   ├── test_config_params_wired.py
│   ├── test_docker_naming.py
│   └── test_entrypoint_puid.py
├── e2e/                                 # Full workflow tests (1 file)
│   └── test_cli_e2e.py
└── runtime/                             # GPU-required param tests (4 files)
    ├── conftest.py                      # GPU fixtures, backend detection
    ├── test_all_params.py               # SSOT param discovery + testing (CANONICAL)
    ├── test_runtime_params.py           # Pytest wrapper for parametrized tests
    ├── discover_params.py               # Param discovery utility
    └── issues.yaml                      # Known issues for smoke test exceptions
```

**Naming:**
- Test files: `test_<module>.py` (mirrors source structure)
  - `test_core_inference.py` → tests `src/llenergymeasure/core/inference.py`
  - `test_config_models.py` → tests `src/llenergymeasure/config/models.py`
  - `test_orchestration_runner.py` → tests `src/llenergymeasure/orchestration/runner.py`
- Test classes: `Test<Feature>` (e.g., `TestCalculateInferenceMetrics`, `TestTrafficSimulation`)
- Test functions: `test_<descriptor>()` (e.g., `test_basic_metrics()`, `test_target_qps_must_be_positive()`)
- Subdirectory mirrors: `tests/unit/cli/test_resume.py` → `src/llenergymeasure/cli/resume.py`

**Structure Pattern:**
- Tests mirror source tree: `tests/unit/test_<module>_<submodule>.py`
- Examples:
  - `tests/unit/test_core_inference.py` → `src/llenergymeasure/core/inference.py`
  - `tests/unit/test_config_backend_configs.py` → `src/llenergymeasure/config/backend_configs.py`
  - `tests/unit/cli/test_docker_gpu_propagation.py` → `src/llenergymeasure/cli/` (GPU env propagation)

## Test Structure

**Suite Organization:**
```python
import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import TrafficSimulation


class TestTrafficSimulation:
    """Tests for TrafficSimulation (MLPerf-style traffic patterns)."""

    def test_defaults(self):
        """Verify default values match specification."""
        config = TrafficSimulation()
        assert config.enabled is False
        assert config.mode == "poisson"
        assert config.target_qps == 1.0
        assert config.seed is None

    def test_target_qps_must_be_positive(self):
        """target_qps must be > 0."""
        with pytest.raises(ValidationError):
            TrafficSimulation(target_qps=0)
        with pytest.raises(ValidationError):
            TrafficSimulation(target_qps=-1.0)

    def test_seed_for_reproducibility(self):
        """seed allows reproducible Poisson arrivals."""
        config = TrafficSimulation(enabled=True, mode="poisson", seed=42)
        assert config.seed == 42
```

**Patterns:**
- Group related tests in classes (e.g., `TestTrafficSimulation`, `TestDecoderConfig`)
- Use descriptive docstrings for each test method
- One assertion concept per test (multiple asserts for related properties OK)
- Setup/teardown via `@pytest.fixture` (preferred over `setUp()`/`tearDown()`)

**Fixtures:**
- Function-scoped by default (created per test)
- Use `scope="module"` or `scope="session"` for expensive setups
- Use `tmp_path` fixture for temporary directories (auto-cleaned)

## Mocking

**Framework:** Python `unittest.mock` (built-in)

**Patterns:**
```python
from unittest.mock import MagicMock, patch

# Simple mock
mock_accelerator = MagicMock()
mock_accelerator.device = torch.device("cpu")
mock_accelerator.is_main_process = True
mock_accelerator.process_index = 0
mock_accelerator.num_processes = 1

# Mock return values
loader = MagicMock()
loader.load.return_value = (MagicMock(), MagicMock())

# Verify calls
loader.load.assert_called_once()
energy_backend.start_tracking.assert_called_once()
```

**Example from `tests/unit/test_orchestration_context.py`:**
```python
@pytest.fixture
def mock_accelerator():
    """Create a mock Accelerator."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.is_main_process = True
    accelerator.process_index = 0
    accelerator.num_processes = 1
    return accelerator
```

**What to Mock:**
- GPU operations (torch.cuda.*, pynvml calls) - to run tests without GPU
- External dependencies (model loader, energy backend, repository)
- Network calls or slow I/O operations
- Components being tested in isolation
- Backend implementations (use `MockBackend` from `conftest_backends.py`)

**What NOT to Mock:**
- The class under test (test real implementation)
- Pydantic models (test validation works)
- Pure utility functions
- Domain logic

**MockBackend for Testing:**
```python
# tests/conftest_backends.py
class MockBackend:
    """Test backend that simulates inference without GPU."""

    def __init__(
        self,
        latency_per_prompt_ms: float = 10.0,
        output_tokens_per_prompt: int = 50,
        fail_on_inference: bool = False,
    ) -> None:
        ...
```

## Fixtures and Factories

**Test Data Fixtures:**
```python
@pytest.fixture
def sample_config():
    """Create a sample experiment config for testing."""
    return ExperimentConfig(
        config_name="test_config",
        model_name="test/model",
        gpus=[0],
        pytorch=PyTorchConfig(num_processes=1),
    )

@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary results directory with repository."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return FileSystemRepository(base_dir=results_dir)

@pytest.fixture
def mock_cuda():
    """Mock CUDA availability for non-GPU testing."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1):
        yield
```

**Location:**
- Global fixtures: `tests/conftest.py`
- Backend mocks: `tests/conftest_backends.py`
- Test-type-specific: `tests/unit/conftest.py`, `tests/runtime/conftest.py` (if needed)
- Test-file-local: Inside test file as `@pytest.fixture`

**Common Fixtures:**
- `sample_config()` - Valid ExperimentConfig
- `temp_results_dir(tmp_path)` - FileSystemRepository in temp dir
- `mock_accelerator()` - Mock Accelerator object
- `mock_cuda()` - Mock GPU availability

## Coverage

**Requirements:** Not strictly enforced, but coverage tracking enabled

**View Coverage:**
```bash
poetry run pytest tests/unit/ --cov=src/llenergymeasure --cov-report=html
# Open htmlcov/index.html
```

**Configuration:**
- `pytest-cov` included in `[dev]` extras
- Coverage reports: HTML (htmlcov/), term-missing

## Test Types

### Unit Tests (`tests/unit/`)

**Characteristics:**
- Fast, isolated component tests (64 test files)
- No GPU required
- Use mocks for external dependencies
- Run with `make test`

**Examples:**
- `test_config_models.py`: Pydantic validation (6 test classes)
- `test_core_inference.py`: Metrics calculation (`TestCalculateInferenceMetrics`)
- `test_exceptions.py`: Exception hierarchy
- `test_orchestration_runner.py`: Component orchestration (mocked)
- `test_config_introspection.py`: SSOT introspection (13 test classes)
- `test_resilience.py`: Retry logic (`TestRetryOnError`, `TestCleanupGpuMemory`)

**Test count:** ~265 test classes across 64 files

### Integration Tests (`tests/integration/`)

**Characteristics:**
- Component interaction tests (6 files)
- May touch file system (using `tmp_path`)
- No GPU required (use simulated results)
- Run with `make test-integration`

**Examples:**
- `test_config_aggregation_pipeline.py`: Config → aggregation → export pipeline
- `test_cli_workflows.py`: CLI command interaction with real repository
- `test_error_handling.py`: Error propagation across layers
- `test_config_params_wired.py`: Backend param wiring end-to-end
- `test_docker_naming.py`: Docker container naming
- `test_entrypoint_puid.py`: PUID/PGID handling

### E2E Tests (`tests/e2e/`)

**Characteristics:**
- Full workflow tests (1 file)
- Simulate inference (no real GPU calls)
- Run with `make test-all`

**Example:**
- `test_cli_e2e.py`: End-to-end CLI workflow with MockBackend

### Runtime Tests (`tests/runtime/`)

**Characteristics:**
- GPU-required parameter validation tests
- CANONICAL parameter testing (SSOT-derived)
- Use Docker dispatch to correct backend containers
- Run with `make test-runtime*` commands

**Key files:**
- `test_all_params.py`: Standalone + pytest, SSOT param discovery (150+ lines)
- `test_runtime_params.py`: Pytest wrapper with parametrization
- `discover_params.py`: Utility for auto-discovering params
- `conftest.py`: GPU fixtures, backend detection
- `issues.yaml`: Known issues for smoke test exceptions

**SSOT Parameter Discovery:**
Parameters are auto-discovered from Pydantic models in `config/backend_configs.py`, not hardcoded:

```python
from llenergymeasure.config.introspection import get_backend_params

# Auto-discover all params for a backend
params = get_backend_params("pytorch")
# Returns: {'pytorch.batch_size': {...}, 'pytorch.attn_implementation': {...}, ...}
```

**Runtime Test Orchestration:**
```bash
# Uses SSOT introspection + Docker dispatch (same pattern as `lem campaign`)
python scripts/runtime-test-orchestrator.py --backend pytorch
python scripts/runtime-test-orchestrator.py --backend all    # All 3 backends
python scripts/runtime-test-orchestrator.py --quick          # Quick mode
python scripts/runtime-test-orchestrator.py --list-params    # List all params
python scripts/runtime-test-orchestrator.py --check-docker   # Check Docker setup
```

**Skip Markers:**
```python
@pytest.mark.requires_gpu         # Requires CUDA GPU
@pytest.mark.requires_vllm        # Requires vLLM installed
@pytest.mark.requires_tensorrt    # Requires TensorRT-LLM installed
@pytest.mark.slow                 # Takes >1 minute
```

**Example:**
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

**Error Testing:**
```python
# Test that error is raised with correct message
with pytest.raises(ConfigurationError, match="model_name is required"):
    ExperimentConfig(config_name="test")

# Test specific exception type
with pytest.raises(ValidationError) as exc_info:
    TrafficSimulation(target_qps=0)
assert "greater than 0" in str(exc_info.value)

# Test custom exception attributes
with pytest.raises(BackendNotAvailableError) as exc_info:
    backend.initialize(config, runtime)
assert exc_info.value.backend == "vllm"
assert "pip install" in exc_info.value.install_hint
```

**Parametrized Tests:**
```python
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

    # Load and verify
    loaded = repository.load_raw(result_file)
    assert loaded.experiment_id == result.experiment_id
```

**Async Testing:**
Not currently used (all code is synchronous).

## Special Test Configuration

**Global Pytest Setup (`tests/conftest.py`):**
```python
"""Pytest configuration and fixtures for LLM Bench tests."""

import os
import re

# Disable Rich colors in tests to ensure consistent output for assertions
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"  # Additional terminal hint


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)
```

**CLI Testing Pattern:**
```python
from typer.testing import CliRunner

# Use CliRunner with NO_COLOR env var for consistent output
runner = CliRunner(env={"NO_COLOR": "1"})

# Invoke CLI commands
result = runner.invoke(app, ["config", "validate", str(config_file)])
assert result.exit_code == 0
assert "Valid configuration" in result.stdout
```

**MockBackend Registration:**
```python
# tests/conftest_backends.py
def register_mock_backend() -> None:
    """Register MockBackend in the backend registry for testing."""
    from llenergymeasure.core.inference_backends import register_backend

    register_backend("mock", MockBackend)
```

## Test Execution in CI

**CI Pipeline (`make ci`):**
1. Code formatting and linting: `make check`
   - `ruff format`
   - `ruff check --fix`
   - `mypy src/`
2. Unit + integration tests: `make test-all`
3. Documentation freshness: `make check-docs`

**Local pre-commit:**
- Ruff format + lint run before each commit
- MyPy type checking on `src/`
- SSOT doc regeneration (when introspection files change)

## Runtime Parameter Testing (SSOT Architecture)

**Workflow:**

```
┌────────────────────────────────────────────────────────────┐
│              Runtime Test SSOT Workflow                     │
└────────────────────────────────────────────────────────────┘

1. Pydantic Models (backend_configs.py)
   ↓
2. introspection.py: get_backend_params("pytorch")
   ↓
3. test_all_params.py: Auto-discover params + test values
   ↓
4. runtime-test-orchestrator.py: Docker dispatch to correct container
   ↓
5. lem experiment <config> runs in Docker container
   ↓
6. Verify: output tokens, throughput, energy metrics
   ↓
7. Report: results/test_results_<backend>.json
```

**Adding New Parameters:**
1. Add to Pydantic model: `backend_configs.py` or `models.py`
2. Tests auto-discover via introspection (no manual update)
3. Run `make generate-docs` to update documentation
4. Run `make test-runtime-<backend>` to validate

**Test Validation:**
```python
# From tests/runtime/test_all_params.py
def verify_inference_results(result: dict[str, Any]) -> ValidationResult:
    """Verify that inference actually ran (not just config validation)."""

    # Check output tokens generated
    if result.get("output_tokens", 0) <= 0:
        return ValidationResult(
            param_applied=False,
            validation_status="NO_EFFECT",
            ...
        )

    # Check metrics are reasonable
    if result.get("tokens_per_second", 0) <= 0:
        return ValidationResult(validation_status="UNVERIFIED", ...)

    return ValidationResult(validation_status="VERIFIED", ...)
```

**Test Models:**
- Default: `Qwen/Qwen2.5-0.5B` (small, fast, no auth required)
- Quantized (AWQ): `Qwen/Qwen2.5-0.5B-Instruct-AWQ`
- Quantized (GPTQ): `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4`
- Sample size: 5 prompts, max_output: 32 tokens
- Timeout: 300 seconds per test

---

*Testing analysis: 2026-02-05*
