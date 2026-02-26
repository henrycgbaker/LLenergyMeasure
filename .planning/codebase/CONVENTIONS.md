# Coding Conventions

**Analysis Date:** 2026-02-05

## Naming Patterns

**Files:**
- Snake_case for Python modules: `model_loader.py`, `experiment_state.py`, `backend_configs.py`
- Prefix `test_` for all test files: `test_core_inference.py`, `test_config_models.py`
- Backend implementations: `pytorch.py`, `vllm.py`, `tensorrt.py` (in `core/inference_backends/`)
- Protocol/interface files: `protocols.py` (contains Protocol classes)
- Private modules: No leading underscore pattern (use subdirectories for organization)

**Functions:**
- Snake_case: `calculate_inference_metrics()`, `load_prompts_from_source()`, `setup_logging()`
- Private functions with leading underscore: `_get_verbosity_from_env()`, `_is_json_output_mode()`
- Test methods: `test_*` pattern (pytest requirement): `test_basic_metrics()`, `test_defaults()`
- Validators: `@field_validator`, `@model_validator` for Pydantic models

**Variables:**
- Snake_case: `experiment_id`, `total_tokens`, `backend_metadata`, `inference_time_sec`
- Constants: UPPERCASE_WITH_UNDERSCORES: `TEST_MODEL = "Qwen/Qwen2.5-0.5B"`, `DEFAULT_DATASET = "ai-energy-score"`, `CURRENT_SCHEMA_VERSION = "3.0.0"`
- Private variables: leading underscore: `_latency_ms`, `_initialized`, `_config`
- Environment variables: UPPERCASE: `LLM_ENERGY_VERBOSITY`, `VLLM_LOGGING_LEVEL`

**Classes:**
- PascalCase: `ExperimentConfig`, `ModelInfo`, `InferenceMetrics`, `ExperimentOrchestrator`
- Protocol classes: Suffix or name contains `Protocol`: `EnergyBackendProtocol`, `InferenceBackend` (interface)
- Test classes: Prefix with `Test`: `class TestCalculateInferenceMetrics:`, `class TestTrafficSimulation:`
- Exception classes: Suffix with `Error`: `ConfigurationError`, `RetryableError`, `BackendConfigError`
- Pydantic models: Descriptive names: `TrafficSimulation`, `PyTorchConfig`, `RawProcessResult`

**Types:**
- Type aliases: PascalCase or descriptive: `VerbosityType = Literal["quiet", "normal", "verbose"]`
- Generic type vars: Single uppercase letter: `T = TypeVar("T")`

## Code Style

**Formatting:**
- Tool: Ruff (replaces Black + isort)
- Line length: 100 characters (configured in `pyproject.toml`)
- Quote style: Double quotes (`"..."`)
- Indent style: 4 spaces

**Linting:**
- Tool: Ruff (`ruff check`)
- Selected rules: `["E", "F", "I", "UP", "B", "SIM", "RUF"]`
  - E: pycodestyle errors
  - F: pyflakes
  - I: isort (import sorting)
  - UP: pyupgrade (modern Python syntax)
  - B: flake8-bugbear (likely bugs)
  - SIM: flake8-simplify
  - RUF: ruff-specific rules
- Ignored: `E501` (line length, handled by formatter)

**Type Checking:**
- Tool: mypy
- Mode: Strict (`strict = true` in `pyproject.toml`)
- Python version: 3.10+
- Relaxed checks:
  - `disallow_untyped_calls = false` (for torch, transformers)
  - `warn_unused_ignores = false` (cross-environment compatibility)
- Excludes: `experiment_core_utils`, `experiment_orchestration_utils`, `configs/`

**Pre-commit Enforcement:**
- All style/lint/type checks run automatically via `.pre-commit-config.yaml`
- Hooks:
  1. Branch protection (main branch)
  2. trailing-whitespace, end-of-file-fixer
  3. check-yaml, check-added-large-files, check-merge-conflict
  4. ruff (lint + fix)
  5. ruff-format
  6. mypy (src/ only)
  7. SSOT doc regeneration (when introspection files change)
- Manual run: `pre-commit run --all-files`
- Install: `pre-commit install`

## Import Organization

**Order (enforced by ruff):**
1. Future annotations: `from __future__ import annotations`
2. Standard library imports (sorted alphabetically)
3. Third-party imports (sorted alphabetically)
4. Local application imports (sorted alphabetically)

**Example from `src/llenergymeasure/core/model_loader.py`:**
```python
from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from llenergymeasure.config.backend_configs import PyTorchConfig
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.exceptions import ConfigurationError
```

**Import patterns:**
- Use `TYPE_CHECKING` guard for circular imports and type-only imports
- Prefer explicit imports: `from module import Class` over `import module`
- No wildcard imports: Never `from module import *`
- Group local imports by logical package
- Import order within each section: alphabetical

**Path Aliases:**
- None configured
- All imports use full package paths: `from llenergymeasure.config.models import ExperimentConfig`

## Type Annotations

**Required for:**
- All public function signatures
- All class attributes (via dataclass fields or Pydantic Field())
- Public API methods and callbacks

**Style:**
- Python 3.10+ syntax: `list[str]` not `List[str]`, `dict[str, float]` not `Dict[str, float]`
- Union syntax: `str | None` not `Optional[str]`, `int | float` not `Union[int, float]`
- Use `Any` sparingly, prefer specific types
- Future annotations enabled globally: `from __future__ import annotations`

**Example from `src/llenergymeasure/core/inference.py`:**
```python
def calculate_inference_metrics(
    num_prompts: int,
    latencies_ms: list[float],
    total_input_tokens: int,
    total_generated_tokens: int,
) -> InferenceMetrics:
    """Calculate inference metrics from raw timing data."""
```

**Protocols for abstraction:**
```python
# src/llenergymeasure/protocols.py
from typing import Protocol

class EnergyBackendProtocol(Protocol):
    def start(self) -> None: ...
    def stop(self) -> dict[str, float]: ...
```

**TYPE_CHECKING imports:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from llenergymeasure.core.inference import InferenceEngine
```

## Error Handling

**Exception hierarchy:**
```python
# src/llenergymeasure/exceptions.py
LLMBenchError                    # Base exception
├── ConfigurationError           # Invalid or missing configuration
├── ModelLoadError              # Failed to load model/tokenizer
├── InferenceError              # Error during model inference
├── EnergyTrackingError         # Error in energy measurement
├── AggregationError            # Error aggregating results
├── DistributedError            # Multi-GPU setup errors
├── RetryableError              # Transient errors (OOM, GPU issues)
│   └── max_retries attribute
├── BackendError                # Backend-specific base
│   ├── BackendNotAvailableError     # Dependencies missing
│   ├── BackendInitializationError   # Model loading failed
│   ├── BackendInferenceError        # Inference execution error
│   ├── BackendTimeoutError          # Exceeded timeout
│   └── BackendConfigError           # Invalid backend param
└── InvalidStateTransitionError  # State machine violations
```

**Patterns:**
- Raise custom exceptions, not generic `ValueError`/`RuntimeError` in public API
- Use `RetryableError` for transient failures (OOM, GPU communication errors)
- Backend errors include context: `BackendConfigError(backend="vllm", param="batch_size", message=...)`
- State transitions raise `InvalidStateTransitionError(from_status="running", to_status="pending", entity="experiment")`
- Use exception chaining: `raise ModelLoadError(...) from e`

**Example from `src/llenergymeasure/exceptions.py`:**
```python
class BackendNotAvailableError(BackendError):
    """Backend is not installed or not usable."""

    def __init__(self, backend: str, install_hint: str | None = None):
        msg = f"Backend '{backend}' is not available"
        if install_hint:
            msg += f". Install with: {install_hint}"
        super().__init__(msg)
        self.backend = backend
        self.install_hint = install_hint
```

**No bare except:**
- Always catch specific exceptions
- Use `finally` for cleanup, not `except` without re-raise
- Warnings via `warnings.warn()` for non-critical issues

## Logging

**Framework:** Loguru (`from loguru import logger`)

**Configuration:**
- Setup: `setup_logging()` in `src/llenergymeasure/logging.py`
- Verbosity modes (set via `LLM_ENERGY_VERBOSITY` env var or `--quiet`/`--verbose` flags):
  - `quiet`: WARNING+ only, no progress bars, simplified format
  - `normal`: INFO+, progress bars, simplified format (default)
  - `verbose`: DEBUG+, full format with timestamps and module names

**Log levels:**
- `DEBUG`: Internal state, function entry/exit (verbose mode only)
- `INFO`: User-facing status updates, experiment progress
- `WARNING`: Recoverable issues, deprecated features, config warnings
- `ERROR`: Errors that prevent operation but don't crash
- `CRITICAL`: Fatal errors (rarely used)

**Format patterns:**
```python
# Structured logging with context
logger.info("Experiment started: {experiment_id}", experiment_id=exp_id)

# F-string for complex messages
logger.debug(f"bitsandbytes {bnb_version}: 4bit={supports_4bit}, 8bit={supports_8bit}")

# Context in warnings/errors
logger.warning(f"GPU cleanup failed: {e}")
logger.error(f"All {max_retries + 1} attempts failed: {e}")
```

**Backend filtering:**
- Noisy backends (vLLM, TensorRT, transformers, ray) suppressed in normal/quiet mode
- Configured in `BACKEND_NOISY_LOGGERS` list: `["vllm", "tensorrt", "transformers", "ray", "accelerate"]`
- Verbose mode shows all backend logs at DEBUG level
- Environment variables set for backends: `VLLM_LOGGING_LEVEL`, `TLLM_LOG_LEVEL`

**Per-experiment log files:**
- Full DEBUG logs written to `results/<exp_id>/logs/<exp_id>.log` (independent of console verbosity)
- Rotation: 50 MB, retention: 7 days, thread-safe (`enqueue=True`)
- Format: Always uses `VERBOSE_FORMAT` (timestamps + module names)

**Example from `src/llenergymeasure/logging.py`:**
```python
VERBOSE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

SIMPLE_FORMAT = "<level>{level: <8}</level> | <level>{message}</level>"
```

## Docstrings

**Format:** Google-style

**Complex function:**
```python
def initialize(self, config: ExperimentConfig, runtime: BackendRuntime) -> None:
    """Initialize the inference backend with model and configuration.

    Args:
        config: Experiment configuration containing model name and parameters.
        runtime: Runtime context with device info and distributed settings.

    Raises:
        BackendInitializationError: If model loading fails.
    """
```

**Simple function:**
```python
def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
```

**Required sections for complex functions:**
- Brief description (first line)
- Args: Parameter descriptions with types (if not obvious from signature)
- Returns: Return value description (if non-trivial)
- Raises: Exceptions that may be raised (for public API only)

**Module docstrings:**
- All modules have 1-3 sentence docstring at top
- Key modules include architecture notes

**Example from `src/llenergymeasure/config/models.py`:**
```python
"""Configuration models for LLM Bench experiments.

This module defines the Tier 1 (Universal) configuration that applies identically
across all backends. Backend-specific parameters live in backend_configs.py.
"""
```

**Class docstrings:**
```python
class TrafficSimulation(BaseModel):
    """MLPerf-style traffic simulation for realistic load testing.

    Modes:
    - constant: Fixed inter-arrival time (1/target_qps seconds)
    - poisson: Exponential inter-arrival times (MLPerf server scenario)
    """
```

## Pydantic Patterns

**Model definition:**
```python
class TrafficSimulation(BaseModel):
    """MLPerf-style traffic simulation for realistic load testing."""

    enabled: bool = Field(default=False, description="Enable traffic simulation")
    mode: Literal["constant", "poisson"] = Field(
        default="poisson",
        description="Traffic arrival pattern (MLPerf terminology)"
    )
    target_qps: float = Field(
        default=1.0,
        gt=0,
        description="Target queries per second"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility"
    )
```

**Validation:**
- Use `Field(..., gt=0)`, `Field(..., ge=0)`, `Field(..., le=100)` for numeric constraints
- Use `@field_validator` for single-field validation
- Use `@model_validator` for cross-field validation
- Literal types for enums: `Literal["constant", "poisson"]`
- Custom validation in validators (not in model body)

**Example validator:**
```python
@field_validator("target_qps")
@classmethod
def validate_qps(cls, v: float) -> float:
    if v <= 0:
        raise ValueError("target_qps must be positive")
    return v
```

**Configuration precedence (enforced in config loader):**
- CLI flags > Config file > Preset > Pydantic defaults

## Function Design

**Size:**
- Keep functions under 100 lines
- Extract helper functions for complex logic
- One primary responsibility per function

**Parameters:**
- Use keyword-only arguments for clarity when >2 params: `def func(required: str, *, optional: int = 0)`
- Avoid mutable defaults (use `None` + inline assignment: `items = items or []`)
- Use dataclasses/Pydantic models for >4 related parameters
- Type hints required for all parameters

**Return values:**
- Prefer domain objects (Pydantic models, dataclasses) over dicts/tuples
- Use `None` for "no result", not empty dict/list (unless collection is semantic)
- Always include return type hint (even for `-> None`)

**Example from `src/llenergymeasure/core/inference.py`:**
```python
def calculate_inference_metrics(
    num_prompts: int,
    latencies_ms: list[float],
    total_input_tokens: int,
    total_generated_tokens: int,
) -> InferenceMetrics:
    """Calculate inference metrics from raw timing data.

    Returns domain object, not dict.
    """
    return InferenceMetrics(
        input_tokens=total_input_tokens,
        output_tokens=total_generated_tokens,
        total_tokens=total_input_tokens + total_generated_tokens,
        inference_time_sec=sum(latencies_ms) / 1000.0,
        tokens_per_second=tps,
        latency_per_token_ms=latency_per_token,
    )
```

## Module Design

**Exports:**
- Use `__all__` to control public API
- Re-export key items in `__init__.py` for convenience

**Example from `src/llenergymeasure/orchestration/__init__.py`:**
```python
from llenergymeasure.orchestration.context import (
    ExperimentContext,
    experiment_context,
)
from llenergymeasure.orchestration.factory import (
    build_energy_backend,
    build_inference_engine,
)
from llenergymeasure.orchestration.runner import ExperimentOrchestrator

__all__ = [
    "ExperimentContext",
    "experiment_context",
    "build_energy_backend",
    "build_inference_engine",
    "ExperimentOrchestrator",
]
```

**Barrel files:**
- Used in `__init__.py` to create package-level API
- Selective exports only (not re-exporting everything)
- Flatten deeply nested structures for external API

## SSOT Architecture

**Single Source of Truth:** Pydantic models are the canonical source for all parameter metadata

**Introspection module:** `src/llenergymeasure/config/introspection.py`
- Auto-discovers parameters from Pydantic models (`backend_configs.py`, `models.py`)
- Provides: test values, constraints, mutual exclusions, skip conditions
- Used by: runtime tests (`tests/runtime/`), doc generators (`scripts/generate_*.py`), CLI tools

**Key functions:**
- `get_backend_params(backend: str)` - All params for a backend (from Pydantic model)
- `get_param_test_values(param: str)` - Test values derived from Field constraints
- `get_streaming_constraints()` - Params affected by streaming=True
- `get_mutual_exclusions()` - Incompatible param combinations
- `get_param_skip_conditions(param: str)` - Hardware/GPU requirements for testing

**When adding parameters:**
1. Add to Pydantic model in `backend_configs.py` or `models.py`
2. Add Field metadata: `Field(default=..., description="...", gt=0, ...)`
3. Run `make generate-docs` to regenerate docs
4. Tests auto-discover via introspection (no manual update needed)

**No parallel lists:** Test values, constraints, documentation all derived from Pydantic Field metadata

**Pre-commit integration:** Docs regenerate automatically when introspection sources change

## Comments

**When to Comment:**
- Complex algorithms: Explain the "why", not the "what"
- Non-obvious performance optimizations
- Workarounds for library bugs/limitations
- Business logic that isn't self-documenting
- TODOs for future work: `TODO: Add AWQ support when transformers 4.36+ available`

**What NOT to comment:**
- Self-explanatory code: Don't write `# Increment counter` for `count += 1`
- Type information (use type hints instead)
- Function purpose (use docstring)
- Obvious operations

**Example from `tests/conftest_backends.py`:**
```python
# Sleep 10% of simulated time to avoid blocking tests
time.sleep(total_latency_sec * 0.1)
```

**Example from `src/llenergymeasure/core/model_loader.py`:**
```python
# bitsandbytes 0.39.0+ supports 4-bit quantization (QLoRA)
supports_4bit = parsed_version >= version.parse("0.39.0")
```

---

*Convention analysis: 2026-02-05*
