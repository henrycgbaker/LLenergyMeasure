# Changelog

All notable changes to this project will be documented in this file.

**Versioning**: X.Y.Z
- **X** = Major stable release
- **Y** = New functionality
- **Z** = Patches/bug fixes

---

## [1.2.0] - 2025-11-20

### 🎉 Complete Core Infrastructure

Major architectural improvements with all core modules implemented.

### Added

**Core Modules:**
- Distributed computing with Accelerate (`core/distributed.py`)
- Inference engine with batching (`core/inference.py`)
- Model loading with quantization support (`core/model_loader.py`)
- Data/prompt processing (`data/prompts.py`)
- Energy tracking with CodeCarbon (`metrics/energy.py`)

**Results Management (Major Improvement):**
- Single-file results storage (vs 8+ files in v1.0.0)
- Type-safe dataclasses (ExperimentResults, InferenceMetrics, etc.)
- ResultsManager for easy save/load/aggregate
- Automatic efficiency metrics calculation
- Simple CSV export
- Summary statistics

**Testing:**
- Integration tests for full workflow
- 40+ unit tests total
- CI/CD with GitHub Actions

**Examples:**
- Complete workflow demonstration
- Quantization comparison showing FLOPs fix

### Changed

- **Results storage**: Single JSON per experiment (was 8+ files)
- **Results aggregation**: Simple, robust dataclass-based approach
- All modules now use proper logging (not print statements)
- Full type hints throughout

### Fixed

- **Critical**: Quantized model FLOPs calculation (accurate per-model)
- Results aggregation complexity and fragility

---

## [1.1.0] - 2025-11-20

### 🎯 Foundation & Bug Fixes

Initial refactoring with critical bug fixes.

### Added

**Infrastructure:**
- Modern package structure (`src/llm_efficiency/`)
- Type-safe Pydantic configuration models
- FLOPs calculator with caching (`metrics/compute.py`)
- Comprehensive testing infrastructure (pytest)
- Modern tooling (Ruff, MyPy, pre-commit hooks)
- Full documentation (README, module docs, refactoring strategy)

**Configuration System:**
- `ExperimentConfig` with validation
- Sub-configs: `BatchingConfig`, `QuantizationConfig`, `DecoderConfig`, etc.
- Migration helper: `ExperimentConfig.from_legacy_dict()`

**Testing:**
- 25 configuration tests
- 15 FLOPs calculator tests
- Test fixtures and markers

**Documentation:**
- README.md
- QUICKSTART.md
- REFACTORING_STRATEGY.md
- Module-specific READMEs

### Changed

- **Breaking**: Configuration uses Pydantic models (was nested dicts)
- **Breaking**: New import paths (`from llm_efficiency.config`)
- Package renamed to `llm-efficiency`
- Dependencies reduced from 305 to ~12 core packages

### Fixed

- **Critical**: Quantized models now get accurate FLOPs (was hardcoded: 52,638,582,308,864 for ALL models!)
- FLOPs calculation supports multiple fallback strategies
- Configuration validation (was missing)

---

## [1.0.0] - 2024

### Initial Release

Original thesis code with core functionality.

**Features:**
- Energy and performance metrics collection
- Support for multiple models and configurations
- Distributed inference with Hugging Face Accelerate
- Persistent progress tracking
- Multiple experiment modes

**Known Issues:**
- ❌ Quantized model FLOPs uses same hardcoded value for all models
- ❌ Results scattered across 8+ JSON files per experiment
- ❌ No test coverage
- ❌ Dictionary-based configuration without validation
- ❌ No type hints

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| **1.2.0** | 2025-11-20 | Core modules + clean results management |
| **1.1.0** | 2025-11-20 | Foundation + FLOPs fix + testing |
| **1.0.0** | 2024 | Original thesis code |

---

## Migration Guide: v1.0.0 → v1.2.0

### Major Improvements

1. **FLOPs Calculation**: Now accurate for all models (quantized and non-quantized)
2. **Results Storage**: Single file vs 8+ files
3. **Type Safety**: Pydantic models with validation
4. **Clean APIs**: Simple, intuitive interfaces

### Configuration Changes

**v1.0.0:**
```python
config = {
    "model_name": "TinyLlama-1.1B",
    "fp_precision": "float16",
    "batching_options": {"batch_size___fixed_batching": 16},
    "quantization_config": {
        "cached_flops_for_quantised_models": 52638582308864  # WRONG!
    }
}
```

**v1.2.0:**
```python
from llm_efficiency.config import ExperimentConfig, BatchingConfig

config = ExperimentConfig(
    model_name="TinyLlama-1.1B",
    precision="float16",  # cleaner!
    batching=BatchingConfig(batch_size=16),  # typed!
    # No hardcoded FLOPs - automatic!
)
```

### Results Changes

**v1.0.0:**
```python
# 8+ separate JSON files to manage
results/raw_results/0001/
├── 0001_1_experiment_setup.json
├── 0001_2_experiment_variables.json
├── 0001_3_model_architecture.json
├── 0001_4_inference_metrics.json
├── 0001_5_compute_metrics.json
├── 0001_6_local_energy_results_process_0.json
├── 0001_7_global_energy_results.json
└── 0001_8_text_output.json
```

**v1.2.0:**
```python
# Single file, clean API
from llm_efficiency.storage import ResultsManager

manager = ResultsManager()
manager.save_experiment(results)  # One file: 0001.json
results = manager.load_experiment("0001")  # Easy!
```

### Import Changes

| v1.0.0 | v1.2.0 |
|--------|---------|
| `from experiment_core_utils.b_model_loader import ...` | `from llm_efficiency.core import load_model_and_tokenizer` |
| `from experiment_core_utils.h_metrics_compute import get_flops` | `from llm_efficiency.metrics import FLOPsCalculator` |
| `from configs.a_default_config import base_config` | `from llm_efficiency.config import ExperimentConfig` |

---

**Author:** Henry Baker

