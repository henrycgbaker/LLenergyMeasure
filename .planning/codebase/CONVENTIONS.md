# Coding Conventions

**Analysis Date:** 2026-01-26

## Naming Patterns

**Files:**
- Module files: `snake_case` (e.g., `config_loader.py`, `inference_backends.py`)
- Test files: `test_<module>.py` (e.g., `test_core_inference.py`, `test_config_models.py`)
- Special files: `__init__.py`, `conftest.py` for pytest fixtures

**Functions:**
- Function names: `snake_case` (e.g., `calculate_inference_metrics()`, `setup_logging()`)
- Private/internal functions: `_leading_underscore()` (e.g., `_get_verbosity_from_env()`)
- Test functions: `test_<descriptor>()` (e.g., `test_basic_metrics()`, `test_empty_latencies()`)

**Classes:**
- Class names: `PascalCase` (e.g., `ExperimentConfig`, `InferenceMetrics`, `ModelLoader`)
- Abstract/Protocol classes: `PascalCase` with naming convention (e.g., `BackendProtocol`, `EnergyBackendProtocol`)
- Pydantic models: `PascalCase` (e.g., `TrafficSimulation`, `PyTorchConfig`, `RawProcessResult`)

**Variables:**
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MAX_NEW_TOKENS`, `SCHEMA_VERSION`, `PRESETS`)
- Instance variables: `snake_case` (e.g., `experiment_id`, `total_energy_j`, `model_name`)
- Private instance variables: `_leading_underscore` (e.g., `_latency_ms`, `_output_tokens`, `_config`)

**Types:**
- Type hints: Python 3.10+ syntax (e.g., `list[str]`, `dict[str, float]`, `X | None` not `Optional[X]`)
- Union types: `X | Y` syntax (e.g., `str | None`, `int | float`)
- Generic containers: `list[T]`, `dict[K, V]` not `List[T]`, `Dict[K, V]`

## Code Style

**Formatting:**
- Formatter: Ruff (`poetry run ruff format`)
- Line length: 100 characters
- Quote style: Double quotes (`"..."`)
- Indent style: 4 spaces

**Linting:**
- Linter: Ruff (`poetry run ruff check`)
- Rules enabled: `["E", "F", "I", "UP", "B", "SIM", "RUF"]`
- Line length rule ignored: `E501` (handled by formatter)

**Enforcement:**
- Pre-commit hook: Ruff formatting and linting
- CI: `make check` runs format, lint, typecheck
- Type checking: MyPy with strict mode enabled

## Import Organization

**Order:**
1. Standard library (`os`, `sys`, `pathlib`, `typing`, etc.)
2. Third-party packages (`pydantic`, `torch`, `loguru`, `typer`, etc.)
3. Local application imports (relative imports from `llenergymeasure.*`)

**Path Aliases:**
- No path aliases currently configured
- Use absolute imports from package root (e.g., `from llenergymeasure.config.models import ExperimentConfig`)

**Patterns:**
- Prefer explicit imports: `from module import ClassName` (not `from module import *`)
- Use `TYPE_CHECKING` guard for type-only imports: `if TYPE_CHECKING: from ... import ...`
- Type stubs: `from typing import TYPE_CHECKING`

Example:
```python
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from loguru import logger

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.metrics import InferenceMetrics

if TYPE_CHECKING:
    from llenergymeasure.core.inference import InferenceEngine
```

## Error Handling

**Exception Hierarchy:**
- Base exception: `LLMBenchError` in `llenergymeasure/exceptions.py`
- Domain errors: `ConfigurationError`, `ModelLoadError`, `InferenceError`, `EnergyTrackingError`, `AggregationError`, `DistributedError`
- Retryable errors: `RetryableError` (transient errors like OOM, GPU communication)
- Backend errors: `BackendError`, `BackendNotAvailableError`, `BackendInitializationError`, `BackendInferenceError`, `BackendTimeoutError`, `BackendConfigError`
- State machine errors: `InvalidStateTransitionError`

**Pattern:**
```python
from llenergymeasure.exceptions import ConfigurationError, RetryableError

# Domain error
if not config.model_name:
    raise ConfigurationError("model_name is required")

# Retryable error (for transient failures)
try:
    result = inference_engine.run(prompts)
except torch.cuda.OutOfMemoryError as e:
    raise RetryableError(f"GPU OOM: {e}", max_retries=3) from e
```

**Guidelines:**
- Catch specific exceptions, not generic `Exception`
- Use custom exceptions for domain-specific errors
- Include context in error messages: parameter name, expected values, actual values
- Use `from e` for exception chaining to preserve stack traces
- Backend errors include `backend` and optional `install_hint` for diagnostics

## Logging

**Framework:** Loguru (`loguru.logger`)

**Setup:**
- Configure via `setup_logging()` in `llenergymeasure/logging.py`
- Verbosity levels: `quiet` (WARNING+), `normal` (INFO+), `verbose` (DEBUG+)
- Environment variable: `LLM_ENERGY_VERBOSITY` (default: "normal")

**Patterns:**
```python
from llenergymeasure.logging import get_logger

logger = get_logger(__name__)  # Bind module name

# Log at appropriate levels
logger.debug("Loading model from cache")
logger.info(f"Starting inference on {len(prompts)} prompts")
logger.warning(f"GPU memory {used_gb:.1f}GB exceeds warning threshold")
logger.error(f"Inference failed: {error}")
```

**Guidelines:**
- Use `get_logger(__name__)` for module-level loggers (binds module name automatically)
- Avoid logging secrets, tokens, or PII
- Use f-strings for message formatting
- Structured logging: include keys in logs for machine parsing
- Log at start/end of major operations (model loading, inference, aggregation)

## Comments

**When to Comment:**
- Complex algorithms or non-obvious logic
- Workarounds for known limitations or bugs
- Links to related documentation or issues
- Explanations of "why", not "what" (code shows what)

**JSDoc/TSDoc:**
- Use Google-style docstrings for public functions and classes
- Include Args, Returns, Raises sections
- Include brief description for simple/obvious functions

**Pattern:**
```python
def calculate_inference_metrics(
    num_prompts: int,
    latencies_ms: list[float],
    total_input_tokens: int,
    total_generated_tokens: int,
) -> InferenceMetrics:
    """Calculate inference performance metrics.

    Args:
        num_prompts: Number of prompts processed.
        latencies_ms: List of per-batch latencies in milliseconds.
        total_input_tokens: Total input tokens processed.
        total_generated_tokens: Total tokens generated.

    Returns:
        InferenceMetrics with calculated values.
    """
    ...
```

## Function Design

**Size:**
- Keep functions focused and single-responsibility
- Aim for <50 lines when practical
- Long functions should be broken into smaller helper functions

**Parameters:**
- Use type hints for all parameters (required)
- Keep parameter count <= 5 (use objects/dataclasses for many params)
- Use keyword arguments for optional parameters
- Use `*` to force keyword-only args when useful

**Return Values:**
- Always include return type hint
- Return early to reduce nesting
- Use structured return types (Pydantic models, dataclasses) for multiple values
- Return `None` explicitly when function has side effects only

**Pattern:**
```python
def run_inference(
    prompts: list[str],
    config: ExperimentConfig,
    *,  # Force keyword args
    timeout_sec: float = 300.0,
) -> InferenceMetrics:
    """Run inference and collect metrics."""

    # Early return for validation
    if not prompts:
        return InferenceMetrics.empty()

    # Main logic
    ...
    return InferenceMetrics(...)
```

## Module Design

**Exports:**
- Use `__init__.py` to expose public API
- Keep internal modules private (prefixed with `_` or in subdirectories)
- Document public API in module docstring

**Barrel Files:**
- Use in `llenergymeasure/` and submodules for re-exporting commonly-used classes
- Pattern: `from module import ClassName` in `__init__.py`

**Example (`llenergymeasure/__init__.py`):**
```python
"""LLenergyMeasure - LLM inference efficiency measurement framework."""

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.metrics import InferenceMetrics, EnergyMetrics

__all__ = ["ExperimentConfig", "InferenceMetrics", "EnergyMetrics"]
```

## Pydantic Models

**Pattern:**
- All configuration and result objects are Pydantic `BaseModel`
- Use `Field()` for validation, defaults, and documentation
- Include `description` for all fields (used in CLI help, docs)
- Validate at instantiation time (Pydantic v2)

**Example from `config/models.py`:**
```python
class TrafficSimulation(BaseModel):
    """Simulated traffic pattern for inference load."""

    enabled: bool = Field(
        default=False,
        description="Enable traffic simulation",
    )
    mode: Literal["poisson", "constant"] = Field(
        default="poisson",
        description="Traffic arrival pattern",
    )
    target_qps: float = Field(
        default=1.0,
        gt=0,  # Greater than
        description="Target queries per second",
    )
```

## Dependency Injection

**Pattern:**
- Components implement protocol interfaces (e.g., `EnergyBackendProtocol`)
- Components injected via class constructors
- Factories create instances based on configuration

**Example from `orchestration/runner.py`:**
```python
class ExperimentOrchestrator:
    def __init__(
        self,
        model_loader: ModelLoaderProtocol,
        inference_engine: InferenceEngineProtocol,
        metrics_collector: MetricsCollectorProtocol,
        energy_backend: EnergyBackendProtocol,
        repository: RepositoryProtocol,
    ):
        self._loader = model_loader
        self._inference = inference_engine
        self._metrics = metrics_collector
        self._energy = energy_backend
        self._repository = repository
```

---

*Convention analysis: 2026-01-26*
