# Changelog

All notable changes to this project are documented here.

## [v1.16.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.16.0) (2025-01-07)

Production-ready containerisation with full GPU support and streamlined developer experience.

### Added
- **Multi-stage Dockerfile** with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base image
  - Builder stage for dependency compilation
  - Runtime stage for production deployment (~3GB image)
  - Dev stage for local development with editable installs
- **Docker Compose profiles** separating production and development workflows
  - `llm-energy-measure-app`: Production service with baked-in package
  - `llm-energy-measure-dev`: Development service with source mounting
- **VS Code devcontainer** configuration for seamless IDE integration
  - GPU passthrough with `--gpus all`
  - Privileged mode for NVML energy metrics
  - Pre-configured Python extensions (Pylance, Ruff)
- **Makefile targets** for common Docker operations (`make docker-build`, `make experiment`, `make datasets`)

### Improved
- CI workflow reliability with concurrency groups preventing parallel releases
- Test runner now validates both `src/` and `tests/` directories
- Dev container runs as root, eliminating permission complexity with virtual environments
- Documentation expanded with "Running the Tool" section covering all four execution modes

### Fixed
- Docker CUDA 12.4 base image now matches host driver requirements
- Volume permission errors resolved by running dev containers as root
- Deprecated `torch_dtype` parameter replaced with `dtype` in model loading
- Removed obsolete `TRANSFORMERS_CACHE` environment variable (superseded by `HF_HOME`)
- CodeCarbon pandas `FutureWarning` suppressed via targeted filter
- `nvidia-smi` GPU utilisation parsing handles `[N/A]` values gracefully

---

## [v1.15.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.15.0) (2025-12-21)

Comprehensive test coverage ensuring reliability across all components.

### Added
- **End-to-end CLI tests** (8 tests) validating complete benchmark workflows
  - Config validation through to results aggregation
  - Dataset listing and prompt source configuration
  - Error handling for invalid inputs
- **Integration tests** (47 tests) covering non-GPU workflows
  - Configuration loading with inheritance chains
  - Results repository file operations lifecycle
  - CLI command parsing and execution
  - Aggregation pipeline from raw results to exports
- **Methodology documentation** (`docs/methodology.md`) explaining measurement approach
  - Energy tracking via CodeCarbon with NVML backend
  - FLOPs estimation strategies and confidence levels
  - Distributed GPU result aggregation logic

### Changed
- Total test count: **416 passing tests** (unit + integration + e2e)
- All tests run without GPU access using mocked/simulated data

### Removed
- `requirements.txt` (306 frozen packages) — all dependencies now managed via Poetry lockfile

---

## [v1.13.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.13.0) (2025-12-21)

User-friendly command-line interface replacing legacy entry points.

### Added
- **Typer-based CLI** (`llm-energy-measure`) with intuitive subcommands:
  - `experiment <config> --dataset <name> -n <samples>`: Run experiments with automatic `accelerate launch` wrapping
  - `aggregate <exp_id> | --all [--force]`: Combine raw per-process JSON results into aggregated metrics
  - `config validate <file>`: Check configuration syntax and required fields
  - `config show <file>`: Display resolved configuration with inheritance applied
  - `results list [--all]`: Show available experiment runs
  - `results show <exp_id> [--raw] [--json]`: Inspect experiment results
  - `datasets`: List built-in HuggingFace datasets (alpaca, gsm8k, mmlu, sharegpt)
- **ExperimentOrchestrator** (~100 lines) with clean dependency injection
  - Accepts protocol-based components for energy backend, model loader, inference engine
  - Manages experiment lifecycle from config loading through result persistence
- **ExperimentContext** dataclass encapsulating runtime state
  - Accelerator instance, model, tokenizer, prompts
  - Automatic cleanup of distributed resources on context exit
- **Accelerate launcher** with configurable retry logic for transient failures
- **25 CLI tests** and **27 orchestration unit tests**

### Removed
- Legacy `MAIN_*.py` entry points (6 files) — all functionality now accessible via CLI

### Usage Examples
```bash
# Run experiment with built-in dataset
llm-energy-measure experiment configs/llama2-7b.yaml --dataset alpaca -n 1000

# Aggregate all pending results
llm-energy-measure aggregate --all

# Export results as JSON
llm-energy-measure results show exp_20240115_123456 --json
```

---

## [v1.10.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.10.0) (2025-12-20)

Major architectural refactor establishing clean module boundaries.

### Breaking Changes
- **Package renamed**: `llm-bench` → `llm-energy-measure`
- All imports now use `llm_energy_measure` instead of `llm_bench`

### Added
- **Energy backend plugin registry** with automatic CodeCarbon registration
  - `register_backend()`, `get_backend()`, `list_backends()` API
  - Protocol-based interface for custom energy tracking backends
- **FlopsEstimator** with three-strategy fallback chain:
  1. `calflops` (high confidence) — direct computation graph measurement
  2. `architecture` (medium confidence) — calculation from `model.config` parameters
  3. `parameter_estimate` (low confidence) — approximation via `2 × params × seq_len`
  - Returns `FlopsResult` with value, method, confidence level, and precision
- **Results aggregation** with verification checks:
  - Temporal overlap detection (ensures concurrent GPU execution)
  - GPU attribution verification (prevents double-counting across processes)
  - Derived efficiency metrics (tokens/joule, FLOPs/watt)
- **Export functionality** for CSV and JSON formats
  - Flattened Pydantic models with logical column ordering
  - `ResultsExporter` class for unified export interface
- **Core modules** migrated from legacy `experiment_core_utils`:
  - `distributed.py`: Accelerator setup, unique ID generation, barrier synchronisation
  - `model_loader.py`: HuggingFace model/tokeniser loading with BitsAndBytes quantisation
  - `prompts.py`: Prompt filtering, sorting, tokenisation, batching strategies
  - `inference.py`: Batch inference engine with memory-efficient generation
  - `compute_metrics.py`: FLOPs calculation, peak memory stats, GPU utilisation tracking
  - `energy_backends/codecarbon.py`: CodeCarbon wrapper implementing `EnergyBackend` protocol
- **Pydantic domain models** for all configurations and results
  - `ExperimentConfig`, `BatchingOptions`, `DecoderConfig`, `QuantisationConfig`
  - `EnergyMetrics`, `InferenceMetrics`, `ComputeMetrics`, `RawProcessResult`, `AggregatedResult`
- **296 unit tests** covering all new modules

### Changed
- Replaced `print()` statements with Loguru structured logging throughout
- Comprehensive type hints and docstrings on all public interfaces
- BitsAndBytes quantisation correctly reports fp16 precision in FLOPs calculations

---

## [v1.0.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v1.0.0) (2025-12-16)

Research phase complete — stable multi-model benchmarking validated on production hardware.

### Features
- **Multi-model experiment support** with scenario-based configuration
  - Run sequential experiments across model families (Llama, Mistral, Phi, etc.)
  - Scenario configs defining model × precision × batch size combinations
- **Experiment suite CSV export** with consistent naming conventions
  - Timestamped output files with model name and config hash
  - Append mode for incremental experiment runs
- **Failed experiment detection** with cycle tracking
  - Automatic retry on transient failures
  - Quarantine of consistently failing configurations
- **Minimum output token enforcement** ensuring comparable generation lengths
- **Large model stability improvements**
  - Gradient checkpointing for memory-constrained runs
  - Proper CUDA cache clearing between experiments

### Research Capabilities
- **Data wrangling pipelines** for experiment result analysis
  - Pandas-based cleaning and normalisation
  - Outlier detection and filtering
- **Plotting functionality** for efficiency metrics visualisation
  - Tokens/second vs energy consumption scatter plots
  - Model size vs efficiency Pareto frontiers
- **FLOPs caching** preventing redundant calculations for repeated model runs

### Validation
- Tested with 1B and 3B parameter models on A100 GPUs
- Verified energy measurements against manual nvidia-smi readings
- Cross-validated FLOPs estimates with published model specifications

---

## [v0.5.0](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/releases/tag/v0.5.0) (2025-03-22)

Core measurement functionality establishing the foundation for all subsequent development.

### Added
- **Distributed results aggregation** across multiple GPUs
  - Per-process JSON result files with process rank metadata
  - Aggregation logic summing energy, averaging throughput
  - Support for 1-8 GPU configurations
- **FLOPs calculation** with quantisation awareness
  - Correct handling of INT8/INT4 operations
  - Integration with `calflops` library
- **Robust process cleanup** preventing zombie processes
  - Signal handlers for graceful shutdown
  - Distributed barrier synchronisation before exit
- **Optimum benchmark integration** for standardised measurements

### Improved
- **Distributed execution stability**
  - Proper NCCL initialisation and teardown
  - Timeout handling for stalled processes
- **Code organisation** with major directory restructuring
  - Separation of config, core, and result handling
  - Modular utility functions
