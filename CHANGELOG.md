# Changelog

All notable changes to this project will be documented in this file.

**Versioning**: X.Y.Z
- **X** = Major stable release
- **Y** = New functionality
- **Z** = Patches/bug fixes

---

## [2.0.0] - 2025-11-20

### 🎉 Stable Release

First stable production release with comprehensive features, documentation, and testing.

### Summary

v2.0.0 represents the culmination of systematic refactoring from the original thesis codebase. This version provides a production-ready framework for LLM efficiency measurement with comprehensive documentation, robust logging, advanced analysis tools, and extensive testing.

### API Stability

Starting with v2.0.0, the public API is considered stable with semantic versioning guarantees.

### Key Features

**Core:**
- Multi-precision support (FP32, FP16, BF16, FP8)
- Quantization (4-bit NF4/FP4, 8-bit INT8)
- Energy tracking with CO2 emissions
- FLOPs calculation (accurate per-model)
- Performance profiling and caching

**CLI:**
- Rich terminal interface
- Interactive configuration wizard
- Experiment management and export

**Analysis:**
- Experiment comparison with rankings
- Statistical analysis and outlier detection
- Efficiency scoring and reporting

**Developer Experience:**
- 10+ comprehensive examples
- Detailed usage guide and API docs
- Type-safe Pydantic configuration
- Structured logging with JSON output
- Custom exception hierarchy
- 204+ unit tests

### Version History

- v1.6.0: Documentation & Examples
- v1.7.0: Enhanced Logging
- v1.8.0: Advanced Analysis
- v1.9.0: Stability & Polish
- v2.0.0: Stable Release

### Statistics

- Lines of Code: ~15,000+ (src + tests + examples)
- Documentation: ~6,000+ lines
- Test Coverage: 204+ tests
- Examples: 10+ scripts

### Backwards Compatibility

✅ Fully backwards compatible with v1.x
- All configurations work without changes
- Legacy functions preserved

---

## [1.9.0] - 2025-11-20

### 🛡️ Stability & Polish

Production readiness improvements with comprehensive input validation and API stability.

### Added

**Validation Module (`utils/validation.py`):**
- `validate_positive_int()` - Validate positive integers
- `validate_positive_float()` - Validate positive floats
- `validate_in_range()` - Range validation
- `validate_choice()` - Enum/choice validation
- `validate_path_exists()` - Path validation
- `validate_model_name()` - Model name format validation
- `validate_precision()` - Precision type validation
- `validate_batch_config()` - Batch configuration validation
- `validate_quantization_config()` - Quantization settings validation

### Features

**Input Validation:**
```python
from llm_efficiency.utils import validate_batch_config, validate_precision

# Validate configuration
validate_batch_config(batch_size=8, num_batches=100)
validate_precision("float16")  # Validates against allowed types
```

### Changed

- Enhanced error messages with actionable suggestions
- Updated utils/__init__.py with validation exports
- Improved configuration validation across all modules

### Fixed

- Edge case handling in configuration parsing
- Improved error messages for invalid inputs
- Better validation of quantization settings

---

## [1.8.0] - 2025-11-20

### 📊 Advanced Analysis & Comparison

Comprehensive analysis utilities for experiment comparison and statistical analysis.

### Added

**Analysis Module (`analysis/` package):**
- `comparison.py` - Experiment comparison utilities
  - `compare_experiments()` - Compare multiple experiments with rankings
  - `compare_models()` - Group and compare by model
  - `compare_configurations()` - Configuration-based comparison
  - `generate_comparison_report()` - Automated report generation
  - `ComparisonResult` dataclass with best performers

- `statistics.py` - Statistical analysis
  - `calculate_statistics()` - Mean, median, stdev for metrics
  - `detect_outliers()` - Statistical outlier detection
  - `calculate_efficiency_score()` - Weighted efficiency scoring
  - `rank_experiments()` - Multi-metric ranking

### Features

**Experiment Comparison:**
```python
from llm_efficiency.analysis import compare_experiments, generate_comparison_report

comparison = compare_experiments(
    experiment_ids=["exp1", "exp2", "exp3"],
    results_dir=Path("./results"),
)

print(f"Fastest: {comparison.fastest.config.model_name}")
print(f"Most efficient: {comparison.most_efficient.config.model_name}")

report = generate_comparison_report(comparison, output_file=Path("report.txt"))
```

**Statistical Analysis:**
```python
from llm_efficiency.analysis import calculate_statistics, detect_outliers, rank_experiments

# Calculate statistics
stats = calculate_statistics(experiments)
print(f"Mean throughput: {stats['throughput']['mean']:.2f}")

# Detect outliers
outliers, normal = detect_outliers(experiments, metric="throughput", threshold=2.0)

# Rank by efficiency
ranked = rank_experiments(experiments, metric="efficiency")
for exp, score in ranked[:5]:
    print(f"{exp.config.model_name}: {score:.1f}")
```

### Changed

- Added `analysis` package to project structure
- Updated exports for new analysis utilities

---

## [1.7.0] - 2025-11-20

### 📝 Enhanced Logging System

Comprehensive logging improvements with structured logging, JSON output, and progress tracking.

### Added

**Structured Logging (`utils/logging.py` enhanced):**
- `JSONFormatter` - JSON formatter for machine-readable logs
  - ISO, Unix, or readable timestamp formats
  - Automatic exception formatting with tracebacks
  - Extra fields support for context
- `StructuredLogger` - Context-aware logger
  - Automatic context injection in all log messages
  - `add_context()` / `remove_context()` methods
  - `context_scope()` context manager for temporary fields
  - Nested context scope support
- `LoggingConfig` - Centralized configuration manager
  - Multiple output formats: rich, json, standard
  - Per-module log level configuration
  - File and console output
  - Third-party logger suppression (codecarbon, transformers, torch)

**New Logging Functions:**
- `configure_logging()` - Modern configuration API
- `get_structured_logger()` - Get context-aware logger
- `set_module_level()` - Runtime module level adjustment
- `log_execution_time()` - Context manager for timing operations
- `log_progress()` - Progress tracking with ETA calculation
- `log_metrics()` - Dictionary metrics logging

**Test Coverage:**
- `test_logging.py` - 18 comprehensive tests
  - JSONFormatter testing
  - StructuredLogger testing
  - Configuration testing
  - Context manager testing
  - **Total test count: 204 (up from 186, +10% increase)**

### Features

**Structured Logging Example:**
```python
from llm_efficiency.utils import get_structured_logger, configure_logging

# Configure with JSON output
configure_logging(
    level="INFO",
    format="json",
    output_file=Path("./logs/app.log"),
    module_levels={
        "llm_efficiency.core": "DEBUG",
        "llm_efficiency.metrics": "WARNING",
    }
)

# Get structured logger with context
logger = get_structured_logger(__name__, context={"experiment_id": "exp123"})

# All logs automatically include context
logger.info("Starting inference", extra={"batch_size": 8})
# Output: {"timestamp": "...", "experiment_id": "exp123", "batch_size": 8, ...}

# Temporary context
with logger.context_scope(phase="warmup"):
    logger.info("Warming up model")
```

**Progress Tracking:**
```python
from llm_efficiency.utils import log_progress

with log_progress(logger, "processing_batches", total=100) as update:
    for i in range(100):
        # Do work
        update(i + 1)
        # Logs: "processing_batches: 50/100 (50.0%) - 10.5 items/s - ETA: 4.8s"
```

**Execution Timing:**
```python
from llm_efficiency.utils import log_execution_time

with log_execution_time(logger, "model_loading"):
    model = load_model()
# Logs: "Completed model_loading in 12.345s"
```

### Changed

- Enhanced existing `setup_logging()` and `get_logger()` for backwards compatibility
- Updated `utils/__init__.py` with new logging exports
- Improved third-party logger suppression

### Performance Improvements

- JSON logging for structured log analysis
- Configurable log intervals to reduce overhead
- Per-module log levels for targeted debugging

---

## [1.6.0] - 2025-11-20

### 📚 Documentation & Examples Release

Comprehensive documentation release with 10+ example scripts, detailed usage guide, and modernized README.

### Added

**Example Scripts (`examples/` directory):**
- `01_basic_experiment.py` - Simple experiment workflow for beginners
- `02_cli_usage.sh` - Complete CLI command reference and examples
- `03_profiling_example.py` - Performance profiling demonstrations (5 examples)
- `04_caching_example.py` - Caching strategies and patterns (5 examples)
- `05_multi_model_comparison.py` - Automated multi-model benchmarking
- `06_quantization_comparison.py` - Quantization methods comparison (FP16, 8-bit, 4-bit NF4/FP4)
- `07_custom_config.py` - Advanced configuration patterns (5 examples)
- `08_results_analysis.py` - Results loading, analysis, and export (5 examples)
- `09_error_handling.py` - Error handling and recovery patterns (5 examples)
- `10_advanced_workflow.py` - Production-ready benchmark suite with caching and profiling
- `examples/README.md` - Comprehensive examples documentation with learning paths

**Documentation:**
- `USAGE_GUIDE.md` - Complete 600+ line usage guide covering:
  - Installation and setup
  - Configuration (Python API and YAML)
  - Running experiments (single, multi-model, quantization comparison)
  - Results analysis and export
  - CLI reference (all commands with examples)
  - Python API reference (all classes and functions)
  - Advanced topics (multi-GPU, custom datasets, reproducibility)
  - Best practices and troubleshooting

**Updated README.md:**
- Modernized for v1.5.0+ architecture
- Added badges (Python version, license)
- Quick start examples (both Python API and CLI)
- Example scripts table with difficulty levels
- Updated architecture diagram reflecting src/ layout
- Advanced features section (profiling, caching, error handling)
- Complete CLI reference
- Testing instructions
- Development setup guide
- Version 1.5.0 release notes

### Features

**Example Highlights:**

*Basic Experiment:*
```python
from llm_efficiency.config import ExperimentConfig
from llm_efficiency.core.experiment import run_experiment

config = ExperimentConfig(
    model_name="gpt2",
    precision="float16",
    batch_size=4,
    num_batches=20,
    output_dir=Path("./results"),
)

result = run_experiment(config)
print(f"Throughput: {result.metrics.tokens_per_second:.2f} tok/s")
```

*Multi-Model Comparison:*
```bash
python examples/05_multi_model_comparison.py
# Compares multiple models, generates comprehensive report
# Output: comparison_report.txt + comparison_report.json
```

*Advanced Workflow:*
```python
# From examples/10_advanced_workflow.py
suite = BenchmarkSuite(
    output_dir=Path("./results/benchmark"),
    cache_dir=Path("./cache"),
)
results = suite.run_benchmark_suite(
    models=["gpt2", "gpt2-medium"],
    precisions=["float16"],
    quantization_configs=[...],
)
suite.generate_comprehensive_report("report.txt")
```

**Learning Paths:**
- **Beginner** (30 min): Basic experiment → CLI usage → Config → Analysis
- **Intermediate** (1 hour): +Model comparison → Profiling → Caching → Error handling
- **Advanced** (2 hours): +Quantization → Advanced workflows → Custom benchmarks

### Changed

- README.md fully rewritten for modern package structure
  - Removed outdated thesis-style references (MAIN_*.py files)
  - Added modern src/ package layout
  - Updated installation instructions
  - Added CLI-first approach
  - Linked to all 10 examples
- Examples directory structure with comprehensive README
- Improved onboarding experience for new users

### Documentation

**Coverage:**
- 10+ fully documented example scripts (~4000 lines of example code)
- Comprehensive USAGE_GUIDE.md (600+ lines)
- Updated README.md (480 lines)
- Examples README (450 lines)
- **Total documentation: ~5500+ lines**

**Topics Covered:**
- Installation and setup
- Basic usage (Python API + CLI)
- Configuration (programmatic + YAML)
- Model comparison and benchmarking
- Quantization evaluation
- Performance profiling integration
- Caching strategies
- Results analysis and export
- Error handling and recovery
- Production workflows
- Best practices
- Troubleshooting

### Removed

- Outdated README sections referencing old architecture
- Legacy file structure references

---

## [1.5.0] - 2025-11-20

### ⚡ Performance & Caching Optimizations

Performance-focused release with comprehensive profiling and caching utilities.

### Added

**Performance Profiling (`utils/profiling.py`):**
- `PerformanceProfiler` class for tracking execution time and resource usage
- `@profile_function` decorator for automatic function profiling
- `timer()` context manager for simple timing
- `get_memory_usage()` - current memory statistics
- `get_cpu_usage()` - current CPU utilization
- Profile results export to JSON
- Detailed performance summaries with percentages

**Advanced Caching (`utils/cache.py`):**
- `LRUCacheWithTTL` - LRU cache with time-to-live support
  - Automatic expiration of old entries
  - Size-based eviction (LRU policy)
  - Hit rate tracking
  - Persistent save/load to disk
- `DiskCache` - File-based persistent caching
  - Ideal for large objects
  - TTL support
  - Disk usage monitoring
- `@cached_with_ttl` decorator for function result caching

**Test Coverage:**
- `test_profiling.py` - 12 tests for profiling utilities
- `test_cache.py` - 25 tests for caching utilities
- **Total test count: 186 (up from 149, +25% increase)**

### Features

**Profiling Example:**
```python
from llm_efficiency.utils import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("model_loading"):
    model = load_model(...)

with profiler.profile("inference"):
    outputs = model.generate(...)

profiler.print_summary()  # Shows timing breakdown
profiler.save_results("profile.json")
```

**Caching Example:**
```python
from llm_efficiency.utils import LRUCacheWithTTL, cached_with_ttl

# Manual caching
cache = LRUCacheWithTTL(maxsize=100, ttl=3600)
cache.set("key", expensive_result)
result = cache.get("key")

# Decorator caching
@cached_with_ttl(ttl=3600, maxsize=100)
def expensive_computation(x):
    return complex_calc(x)
```

### Changed

- Enhanced `utils/__init__.py` with profiling and caching exports
- Dependencies already include `psutil>=5.9.0` (no changes needed)

### Performance Improvements

- Function-level profiling for optimization insights
- Memory usage tracking to identify leaks
- LRU cache reduces redundant computations
- Disk cache for large, reusable data structures
- Hit rate monitoring for cache effectiveness

---

## [1.4.0] - 2025-11-20

### 🛡️ Production Polish & Error Handling

Comprehensive improvements for production readiness with robust error handling, retry logic, and pickle export support.

### Added

**Export Features:**
- Pickle (`.pkl`) export format for fast binary serialization
- `export_to_pickle()` - Bulk export to pickle
- `load_from_pickle()` - Load experiments from pickle
- `save_experiment_pickle()` / `load_experiment_pickle()` - Per-experiment pickle support
- Unified `export()` method supporting CSV, JSON, and pickle formats
- CLI support: `llm-efficiency export results.pkl --format pickle`

**Error Handling & Retry Logic:**
- Custom exception hierarchy (`LLMEfficiencyError` base class)
  - `ModelLoadingError`, `InferenceError`, `ConfigurationError`
  - `DataError`, `MetricsError`, `StorageError`
  - `NetworkError`, `QuantizationError`
- Retry utilities (`utils/retry.py`)
  - `@retry_with_exponential_backoff` decorator
  - `retry_on_exception()` function
  - `RetryContext` context manager
- Automatic retry for network operations (4 retries with exponential backoff)
  - Model downloading
  - Tokenizer loading
  - Dataset fetching

**Enhanced Model Loading:**
- Comprehensive error messages with actionable suggestions
- OOM detection with helpful recovery hints
- Quantization validation (CUDA availability, bitsandbytes version)
- Automatic retry on network failures (2s → 4s → 8s → 16s delays)
- Precision validation with warnings for unsupported types

**New Test Coverage:**
- `test_retry.py` - 15 tests for retry utilities
- `test_exceptions.py` - 12 tests for exception hierarchy
- 8 additional tests for pickle functionality in `test_results.py`
- **Total test count: 149 (up from 114)**

### Changed

- **Model loader** (`core/model_loader.py`): Now throws specific exceptions instead of generic errors
- **CLI export command**: Supports `--format` flag (csv/pickle/json)
- **Error messages**: More informative with suggested fixes
- **Logging**: Enhanced with success/failure states

### Fixed

- Network timeout handling during model downloads
- Better error messages for quantization failures
- Graceful fallback for unsupported precision types

---

## [1.3.0] - 2025-11-20

### 🚀 Modern CLI with Typer + Rich

Added comprehensive command-line interface with beautiful terminal output.

### Added

**CLI Features:**
- `llm-efficiency run` - Run experiments with rich progress bars
- `llm-efficiency list` - List all experiments in formatted table
- `llm-efficiency show <id>` - Display detailed experiment results
- `llm-efficiency export <file>` - Export experiments to CSV
- `llm-efficiency init` - Interactive configuration wizard
- `llm-efficiency summary` - Generate summary statistics
- Beautiful terminal output with Rich (colors, tables, panels)
- Progress indicators during long operations
- Version flag (`--version`)
- Verbose logging option (`--verbose`)

**Dependencies:**
- Added `typer[all]>=0.9.0` for CLI framework
- Added `rich>=13.7.0` for terminal formatting

### Changed

- Console entry point: `llm-efficiency` command now available after install
- All CLI commands use modern Typer framework with type hints
- Interactive prompts with validation and defaults

---

## [1.2.1] - 2025-11-20

### 🧪 Comprehensive Test Coverage

Expanded test suite with extensive unit tests for all core modules.

### Added

**New Test Files (73 additional tests):**
- `test_distributed.py` - 15 tests for distributed computing
- `test_model_loader.py` - 7 tests for model loading
- `test_inference.py` - 9 tests for inference engine
- `test_prompts.py` - 14 tests for data processing
- `test_energy.py` - 8 tests for energy tracking
- `test_results.py` - 20 tests for results management

### Changed

- **Test coverage**: Increased from 41 to 114 total tests
- All core modules now have dedicated unit tests
- Better separation of unit vs integration tests
- Improved test fixtures and mocking

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
| **1.5.0** | 2025-11-20 | Performance profiling + advanced caching (186 tests) |
| **1.4.0** | 2025-11-20 | Production polish + error handling + pickle export |
| **1.3.0** | 2025-11-20 | Modern CLI with Typer + Rich |
| **1.2.1** | 2025-11-20 | Comprehensive test coverage (114 tests) |
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

