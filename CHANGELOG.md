# Changelog

All notable changes to this project are documented here.

## [v1.16.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.16.0) (2025-01-07)

Production-ready containerization with GPU support.

### Added
- Multi-stage Dockerfile with nvidia/cuda:12.4.1 base image
- Docker Compose with dev/prod profiles
- VS Code devcontainer configuration
- Privileged mode for NVML energy metrics access

### Improved
- CI workflow reliability with concurrency groups
- Tests now run on both `src/` and `tests/` directories
- Simplified dev container setup (runs as root for simplicity)

### Fixed
- Docker CUDA version mismatch with host drivers
- Volume permission issues in containers
- Deprecation warnings (torch_dtype, TRANSFORMERS_CACHE, CodeCarbon pandas)
- nvidia-smi GPU utilization parsing edge cases

---

## [v1.15.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.15.0) (2025-12-21)

Comprehensive test coverage and quality documentation.

### Added
- 8 end-to-end CLI tests covering complete benchmark workflows
- 47 integration tests for config → aggregation → export pipeline
- Measurement methodology documentation (`docs/methodology.md`)
- Total test count: 416 passing tests

### Removed
- `requirements.txt` in favor of Poetry-managed dependencies

---

## [v1.13.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.13.0) (2025-12-21)

CLI interface and experiment orchestration.

### Added
- Typer-based CLI with subcommands:
  - `experiment`: Run experiments (wraps accelerate automatically)
  - `aggregate`: Combine raw per-process results
  - `config validate/show`: Configuration inspection
  - `results list/show`: Results inspection with `--raw` and `--json` options
  - `datasets`: List built-in HuggingFace datasets
- `ExperimentOrchestrator` with clean dependency injection (~100 lines)
- `ExperimentContext` dataclass for runtime state management
- Accelerate launcher with retry logic

### Removed
- Legacy `MAIN_*.py` entry points (6 files)

---

## [v1.10.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.10.0) (2025-12-20)

Package rename and v2 architecture.

### Breaking Changes
- **Package renamed**: `llm-bench` → `llm-energy-measure`
- All imports now use `llm_energy_measure` instead of `llm_bench`

### Added
- Energy backend plugin registry with CodeCarbon auto-registration
- `FlopsEstimator` with 3-strategy fallback chain (calflops → architecture → parameter estimate)
- Results aggregation with temporal overlap and GPU attribution verification
- CSV/JSON export functionality
- Core modules: distributed setup, model loading, prompt processing, inference engine
- Pydantic domain models for all configs and results
- Protocol-based interfaces for extensibility
- Comprehensive type hints and docstrings
- 296 unit tests

### Changed
- Migrated from legacy `experiment_core_utils` to modular architecture
- Replaced print statements with Loguru structured logging

---

## [v1.0.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.0.0) (2025-12-16)

Research phase complete - stable multi-model benchmarking.

### Features
- Multi-model experiment support with config scenarios
- Experiment suite CSV export with naming conventions
- Failed experiment detection and cycle tracking
- Minimum output length enforcement
- Large model stability improvements
- Code annotations and documentation

### Research Capabilities
- Data wrangling and cleaning pipelines
- Plotting functionality for analysis
- FLOPs caching for repeated model runs

---

## [v0.5.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v0.5.0) (2025-03-22)

Core measurement functionality complete.

### Added
- Distributed results aggregation across GPUs
- FLOPs calculation with quantization support
- Robust process cleanup
- Optimum benchmark integration

### Improved
- Distributed execution stability
- Code organization with major restructuring

---

## [v0.1.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v0.1.0) (2025-03-02)

Initial project setup.

### Added
- Basic text generation script
- Initial project structure
- Development roadmap (TODO)
