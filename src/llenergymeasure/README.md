# llenergymeasure Package

Main package for the LLM efficiency measurement framework.

## Package Structure

```
llenergymeasure/
├── cli.py              # Typer CLI application
├── constants.py        # Global constants
├── exceptions.py       # Custom exceptions
├── logging.py          # Loguru configuration
├── protocols.py        # Protocol definitions (interfaces)
├── resilience.py       # Retry/circuit breaker utilities
├── security.py         # Path sanitization, validation
├── config/             # Configuration system
├── core/               # Inference & metrics engine
├── domain/             # Domain models (Pydantic)
├── orchestration/      # Experiment lifecycle
├── results/            # Results persistence
└── state/              # Experiment state management
```

## Key Files

### cli.py
Typer-based CLI with subcommands:
- `run` - Run experiment (placeholder, actual launch via accelerate)
- `aggregate` - Aggregate raw results from multi-GPU runs
- `config validate/show` - Configuration management
- `results list/show` - Results inspection

### constants.py
Framework constants:
- `DEFAULT_RESULTS_DIR` - Results output path
- `SCHEMA_VERSION` - Result file schema version
- Inference defaults (tokens, temperature)

### protocols.py
Protocol definitions for dependency injection:
- `ModelLoader` - Load model/tokenizer
- `InferenceEngine` - Run inference
- `EnergyBackend` - Energy tracking
- `MetricsCollector` - Collect metrics
- `ResultsRepository` - Persist results

### exceptions.py
Custom exceptions:
- `LLMBenchError` - Base exception
- `ConfigurationError` - Config loading/validation
- `InferenceError` - Inference failures
- `EnergyTrackingError` - Energy measurement issues

### security.py
Security utilities:
- `sanitize_experiment_id()` - Sanitize IDs for filesystem
- `is_safe_path()` - Prevent path traversal

## Submodules

| Module | Description |
|--------|-------------|
| `config/` | Configuration loading with inheritance support |
| `core/` | Inference engine, model loading, FLOPs, energy backends |
| `domain/` | Pydantic models for metrics and results |
| `orchestration/` | Experiment runner, launcher, lifecycle |
| `results/` | FileSystemRepository, aggregation logic |
| `state/` | Experiment state tracking |

## Usage

```python
from llenergymeasure.config import load_config, ExperimentConfig
from llenergymeasure.core import run_inference, load_model_tokenizer
from llenergymeasure.domain import RawProcessResult, AggregatedResult
from llenergymeasure.orchestration import ExperimentOrchestrator
from llenergymeasure.results import FileSystemRepository
```

## Related

- See `core/README.md` for inference engine details
- See `config/README.md` for configuration system
- See `orchestration/README.md` for experiment lifecycle
