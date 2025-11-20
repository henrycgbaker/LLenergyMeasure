# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-dev] - 2025-11-20

### 🎉 Major Refactor

Complete rewrite and modernization of the LLM Efficiency Measurement Tool.

### Added

- **New Package Structure**: Modern `src/` layout with proper Python packaging
- **Pydantic Configuration**: Type-safe configuration models replacing nested dictionaries
- **Fixed FLOPs Calculator**: Accurate FLOPs calculation for quantized models
  - Implemented `FLOPsCalculator` class with caching
  - Architectural estimation fallback
  - Per-model accuracy (no more hardcoded values!)
- **Comprehensive Testing**: pytest infrastructure with fixtures and markers
  - Unit tests for configuration and FLOPs calculator
  - Test fixtures for models and configurations
  - CI/CD with GitHub Actions
- **Type Hints**: Full type annotations throughout codebase
- **Modern Tooling**:
  - Ruff for linting and formatting
  - MyPy for type checking
  - Pre-commit hooks
  - Rich for beautiful CLI output
- **Better Logging**: Structured logging with Rich integration
- **Documentation**: Comprehensive docs for all modules

### Changed

- **Breaking**: Configuration now uses Pydantic models instead of dictionaries
  - Migration function provided: `ExperimentConfig.from_legacy_dict()`
- **Breaking**: Package renamed to `llm_efficiency` (from root imports)
- **Breaking**: New import paths: `from llm_efficiency.config import ExperimentConfig`
- **Improved**: FLOPs calculation accuracy (fixes critical v1.0 bug)
- **Improved**: Dependency management with pyproject.toml
- **Improved**: Code organization and modularity

### Fixed

- **Critical**: Quantized models now get accurate FLOPs values (was hardcoded in v1.0)
- **Critical**: FLOPs calculation properly handles different model architectures
- Type safety issues from dictionary-based configs
- Missing validation for configuration parameters

### Removed

- Hardcoded `cached_flops_for_quantised_models` from config
- Alphabetic prefixes from module names (a_, b_, c_)
- Debug print statements (replaced with logging)
- Duplicate code across modules

## [1.0.0] - 2024-XX-XX

### Initial Release

Original thesis code with core functionality:

- Energy and performance metrics collection
- Support for multiple models and configurations
- Distributed inference with Hugging Face Accelerate
- Persistent progress tracking
- Multiple experiment modes

**Known Issues**:
- Quantized model FLOPs uses hardcoded value for all models
- Limited test coverage
- Dictionary-based configuration without validation
- Legacy code organization

---

## Migration Guide: v1.0 → v2.0

### Configuration Changes

**v1.0:**
```python
config = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "fp_precision": "float16",
    "batching_options": {
        "batch_size___fixed_batching": 16
    }
}
```

**v2.0:**
```python
from llm_efficiency.config import ExperimentConfig, BatchingConfig

config = ExperimentConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    precision="float16",
    batching=BatchingConfig(batch_size=16)
)
```

**Or migrate automatically:**
```python
from llm_efficiency.config import ExperimentConfig

old_config = {...}  # v1.0 dictionary
new_config = ExperimentConfig.from_legacy_dict(old_config)
```

### Import Changes

**v1.0:**
```python
from experiment_core_utils.h_metrics_compute import get_flops
from configs.a_default_config import base_config
```

**v2.0:**
```python
from llm_efficiency.metrics import FLOPsCalculator
from llm_efficiency.config import ExperimentConfig
```

### FLOPs Calculation Changes

**v1.0:**
```python
# Used hardcoded value for ALL quantized models
flops = config.quantization_config.get("cached_flops_for_quantised_models")
```

**v2.0:**
```python
# Accurate FLOPs per model
calculator = FLOPsCalculator()
flops = calculator.get_flops(
    model=model,
    model_name=config.model_name,
    sequence_length=128,
    device=device,
    is_quantized=config.quantization.enabled
)
```

---

**Author**: Henry Baker
