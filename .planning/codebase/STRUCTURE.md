# Codebase Structure

**Analysis Date:** 2026-02-05

## Directory Layout

```
llm-efficiency-measurement-tool/
├── src/llenergymeasure/         # Main package
│   ├── cli/                     # CLI commands and display
│   │   ├── display/             # Rich console formatting
│   │   ├── __init__.py          # App setup, command registration
│   │   ├── experiment.py        # Experiment commands
│   │   ├── campaign.py          # Campaign orchestration
│   │   ├── resume.py            # Resume functionality
│   │   ├── init_cmd.py          # Interactive campaign setup
│   │   ├── config.py            # Config commands
│   │   ├── results.py           # Results commands
│   │   ├── listing.py           # Discovery commands
│   │   ├── doctor.py            # System diagnostics
│   │   └── batch.py             # Batch execution
│   ├── config/                  # Configuration system
│   │   ├── models.py            # Universal config (tier 1)
│   │   ├── backend_configs.py   # Backend-specific config (tier 2)
│   │   ├── loader.py            # YAML loading + inheritance
│   │   ├── introspection.py     # SSOT parameter discovery
│   │   ├── validation.py        # Config validation
│   │   ├── docker_detection.py  # Docker environment detection
│   │   ├── backend_detection.py # Backend availability detection
│   │   ├── campaign_config.py   # Campaign configuration model
│   │   └── user_config.py       # User preferences
│   ├── core/                    # Inference, metrics, model loading
│   │   ├── inference_backends/  # Backend implementations
│   │   │   ├── protocols.py     # InferenceBackend protocol
│   │   │   ├── shared.py        # Common utilities
│   │   │   ├── pytorch.py       # PyTorch backend
│   │   │   ├── vllm.py          # vLLM backend
│   │   │   └── tensorrt.py      # TensorRT backend
│   │   ├── energy_backends/     # Energy measurement backends
│   │   │   ├── base.py          # Protocol definitions
│   │   │   └── codecarbon.py    # CodeCarbon implementation
│   │   ├── model_loader.py      # Model/tokenizer loading
│   │   ├── inference.py         # Inference metrics calculation
│   │   ├── compute_metrics.py   # Memory/GPU metrics
│   │   ├── extended_metrics.py  # Efficiency metrics computation
│   │   ├── gpu_utilisation.py   # GPU sampler (pynvml)
│   │   ├── gpu_info.py          # GPU detection, MIG handling
│   │   ├── flops.py             # FLOPs estimation
│   │   ├── parallelism.py       # Parallelism validation
│   │   ├── dataset_loader.py    # Dataset loading
│   │   └── distributed.py       # Multi-GPU utilities
│   ├── domain/                  # Domain models (Pydantic)
│   │   ├── experiment.py        # Results (RawProcessResult, AggregatedResult)
│   │   ├── metrics.py           # Metric models
│   │   ├── model_info.py        # Model/GPU metadata
│   │   └── environment.py       # Environment metadata
│   ├── orchestration/           # Experiment lifecycle
│   │   ├── runner.py            # ExperimentOrchestrator
│   │   ├── campaign.py          # CampaignRunner
│   │   ├── container.py         # ContainerManager (persistent Docker)
│   │   ├── manifest.py          # Campaign progress tracking
│   │   ├── context.py           # ExperimentContext
│   │   ├── factory.py           # Dependency injection
│   │   ├── launcher.py          # Subprocess entry point
│   │   ├── lifecycle.py         # CUDA cleanup, lifecycle utilities
│   │   └── grid.py              # Grid expansion utilities
│   ├── results/                 # Results persistence, aggregation
│   │   ├── repository.py        # FileSystemRepository (CRUD)
│   │   ├── aggregation.py       # Result combining
│   │   ├── exporters.py         # CSV/JSON export
│   │   └── timeseries.py        # Time-series export
│   ├── state/                   # Experiment state tracking
│   │   └── experiment_state.py  # StateManager, state machine
│   ├── notifications/           # Notification backends
│   │   └── webhook.py           # Webhook notifications
│   ├── constants.py             # Global constants, presets
│   ├── exceptions.py            # Custom exception types
│   ├── logging.py               # Logging setup
│   ├── protocols.py             # Protocol definitions for DI
│   ├── resilience.py            # Retry logic
│   ├── security.py              # Path validation
│   └── progress.py              # Progress bar utilities
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests (fast, mocked)
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   ├── runtime/                 # GPU-required parameter tests
│   │   ├── test_all_params.py   # Comprehensive param validation
│   │   └── discover_params.py   # Introspection utility
│   ├── fixtures/                # Shared test fixtures
│   └── conftest.py              # Pytest configuration
├── configs/                     # Example experiment configs
│   └── examples/                # Reference configurations
│       ├── pytorch_example.yaml
│       ├── vllm_example.yaml
│       └── tensorrt_example.yaml
├── docs/                        # User documentation
│   ├── quickstart.md            # Getting started guide
│   ├── cli.md                   # CLI reference
│   ├── backends.md              # Backend selection guide
│   └── deployment.md            # Docker and deployment
├── docker/                      # Docker build context
│   ├── Dockerfile.pytorch       # PyTorch backend
│   ├── Dockerfile.vllm          # vLLM backend
│   └── Dockerfile.tensorrt      # TensorRT backend
├── scripts/                     # Utility scripts
│   ├── generate_*.py            # Doc/config generators
│   └── *.sh                     # Setup, analysis scripts
├── .state/                      # Experiment state (runtime)
├── results/                     # Results storage (runtime)
│   ├── raw/                     # Per-process raw results
│   └── aggregated/              # Aggregated results
├── .planning/                   # GSD planning documents
│   ├── codebase/                # Codebase analysis
│   └── phases/                  # Phase planning
├── pyproject.toml               # Poetry project config
├── docker-compose.yml           # Multi-backend compose
├── Makefile                     # Development targets
├── CLAUDE.md                    # Architecture overview
└── README.md                    # Project README
```

## Directory Purposes

**`src/llenergymeasure/cli/`:**
- Purpose: Command-line interface and user-facing display
- Contains: Typer command handlers, Rich console formatting, campaign coordination, Docker dispatch
- Key files:
  - `__init__.py` - App setup, command registration via `_register_commands()`
  - `experiment.py` - Core experiment commands (run, aggregate)
  - `campaign.py` - Multi-config campaign orchestration
  - `resume.py` - Resume interrupted experiments
  - `init_cmd.py` - Interactive campaign setup wizard
  - `display/` - Rich console utilities (tables, summaries, results)
- Pattern: Each command group in separate file, imported and registered in `__init__.py`

**`src/llenergymeasure/config/`:**
- Purpose: Configuration loading, validation, preset management, backend/Docker detection
- Contains: Pydantic models (universal + backend-specific), loader with inheritance, SSOT introspection
- Key files:
  - `models.py` - Universal ExperimentConfig (tier 1: decoder, IO, traffic)
  - `backend_configs.py` - Backend-specific configs (tier 2: PyTorchConfig, VLLMConfig, TensorRTConfig)
  - `loader.py` - YAML loading with `_extends` inheritance resolution
  - `introspection.py` - SSOT module: `get_backend_params()`, `get_streaming_constraints()`
  - `docker_detection.py` - `is_inside_docker()`, `should_use_docker_for_campaign()`
  - `backend_detection.py` - `is_backend_available()`, `get_available_backends()`
  - `campaign_config.py` - Campaign configuration model
- Pattern: Two-tier config architecture (universal + backend-native), YAML → loader → Pydantic validation

**`src/llenergymeasure/core/`:**
- Purpose: Inference execution, model loading, metrics collection, backend implementations
- Contains: Backend implementations, model loader, energy measurement, FLOPs estimation, metric computation
- Key subdirectories:
  - `inference_backends/` - Protocol-based backend implementations
  - `energy_backends/` - Energy measurement backends
- Key files:
  - `inference_backends/pytorch.py` - HuggingFace Transformers + Accelerate
  - `inference_backends/vllm.py` - vLLM with PagedAttention
  - `inference_backends/tensorrt.py` - TensorRT-LLM
  - `inference_backends/protocols.py` - InferenceBackend protocol, RuntimeCapabilities
  - `inference_backends/shared.py` - Common utilities across backends
  - `model_loader.py` - HF model/tokenizer loading with quantization + LoRA
  - `extended_metrics.py` - Efficiency metrics (TPOT, memory, GPU utilisation)
  - `gpu_utilisation.py` - Background GPU sampler via pynvml
  - `parallelism.py` - Parallelism validation (GPU count vs config)
- Pattern: Protocol-based backends, RuntimeCapabilities declare launch mode and CUDA management

**`src/llenergymeasure/domain/`:**
- Purpose: Immutable data models for entire system
- Contains: Pydantic BaseModels for configs, metrics, results, model metadata
- Key files:
  - `metrics.py` - InferenceMetrics, EnergyMetrics, ComputeMetrics, ExtendedEfficiencyMetrics
  - `experiment.py` - RawProcessResult, AggregatedResult, Timestamps
  - `model_info.py` - ModelInfo, GPUInfo
  - `environment.py` - EnvironmentMetadata
- Pattern: All models use Pydantic validation, zero external dependencies, used by all layers

**`src/llenergymeasure/orchestration/`:**
- Purpose: Experiment lifecycle management, subprocess coordination, campaign execution, container management
- Contains: Orchestrator, campaign runner, container manager, context, factory, launcher
- Key files:
  - `runner.py` - ExperimentOrchestrator (main experiment runner)
  - `campaign.py` - CampaignRunner (multi-config orchestration with warmup/gaps)
  - `container.py` - ContainerManager (persistent Docker containers for campaigns)
  - `manifest.py` - CampaignManifest (progress tracking)
  - `context.py` - ExperimentContext (runtime state container)
  - `factory.py` - Dependency injection wiring via `create_components()`
  - `launcher.py` - Subprocess entry point (called by accelerate launch)
  - `lifecycle.py` - CUDA cleanup, distributed cleanup
  - `grid.py` - Grid expansion utilities
- Pattern: Dependency injection via Factory, one orchestrator per process, launcher spawned by accelerate

**`src/llenergymeasure/results/`:**
- Purpose: Results persistence, late aggregation, CSV/JSON export
- Contains: Repository (CRUD), aggregation logic, exporters
- Key files:
  - `repository.py` - FileSystemRepository (load/save results)
  - `aggregation.py` - Result combining, extended metrics late aggregation
  - `exporters.py` - CSV/JSON output
  - `timeseries.py` - Time-series export
- Pattern: Raw results stored per-process, aggregated on-demand via separate step, atomic writes

**`src/llenergymeasure/state/`:**
- Purpose: Experiment state tracking with resumption support
- Contains: StateManager (persistent state), ExperimentState (status tracking), state machine
- Key files:
  - `experiment_state.py` - StateManager, ExperimentState, state transition validation
- Pattern: JSON state files in `.state/` directory, validated transitions via `EXPERIMENT_VALID_TRANSITIONS`

**`src/llenergymeasure/notifications/`:**
- Purpose: Notification backends for experiment completion
- Contains: Webhook implementation
- Key files:
  - `webhook.py` - Webhook notification backend
- Pattern: Protocol-based notification backends

**`tests/unit/`:**
- Purpose: Fast, isolated component tests
- Contains: Test files mirroring source structure
- Pattern: Mock heavy dependencies (GPU, file system), fast execution

**`tests/integration/`:**
- Purpose: Component interaction tests
- Contains: Tests for config loading pipeline, state transitions, result aggregation
- Pattern: May touch real filesystem (in tmp dirs), real Pydantic validation

**`tests/runtime/`:**
- Purpose: GPU-required parameter validation tests
- Contains: SSOT-based parameter testing, backend-specific inference validation
- Key files:
  - `test_all_params.py` - Comprehensive standalone test (can run without pytest)
  - `discover_params.py` - Introspection utility for parameter discovery
- Pattern: Run actual inference with various parameter combinations, strict validation

**`tests/e2e/`:**
- Purpose: Full workflow tests without GPU
- Contains: Simulated results, CLI workflow tests
- Pattern: Use mocked inference results, test state machine transitions

**`configs/examples/`:**
- Purpose: Reference experiment configurations
- Contains: YAML configs for PyTorch, vLLM, TensorRT examples
- Pattern: One config per backend, demonstrate common parameter combinations

**`docs/`:**
- Purpose: User-facing documentation
- Contains: Quickstart guide, CLI reference, backend selection guide, deployment instructions
- Pattern: Markdown files, some generated docs auto-updated from SSOT

**`docker/`:**
- Purpose: Docker build context
- Contains: Dockerfiles for each backend
- Pattern: Multi-stage builds, pinned base images, non-root users

**`scripts/`:**
- Purpose: Development and documentation generation utilities
- Contains: Doc generators, setup scripts
- Pattern: Python scripts for doc generation, shell scripts for setup

## Key File Locations

**Entry Points:**
- CLI app: `src/llenergymeasure/cli/__init__.py` - Creates Typer app, registers commands
- Launcher subprocess: `src/llenergymeasure/orchestration/launcher.py` - Entry for accelerate launch
- Main experiment: `src/llenergymeasure/cli/experiment.py:experiment_cmd`
- Campaign: `src/llenergymeasure/cli/campaign.py:campaign_cmd`
- Resume: `src/llenergymeasure/cli/resume.py:resume_cmd`

**Configuration:**
- Universal config: `src/llenergymeasure/config/models.py` - ExperimentConfig (tier 1)
- Backend configs: `src/llenergymeasure/config/backend_configs.py` - PyTorchConfig, VLLMConfig, TensorRTConfig (tier 2)
- Config loader: `src/llenergymeasure/config/loader.py` - YAML loading with inheritance
- SSOT introspection: `src/llenergymeasure/config/introspection.py` - Auto-discover param metadata
- Docker detection: `src/llenergymeasure/config/docker_detection.py` - `is_inside_docker()`, dispatch logic
- Backend detection: `src/llenergymeasure/config/backend_detection.py` - `is_backend_available()`

**Core Logic:**
- Model loading: `src/llenergymeasure/core/model_loader.py` - HuggingFace loading with quantization
- Inference backends: `src/llenergymeasure/core/inference_backends/pytorch.py`, `vllm.py`, `tensorrt.py`
- Backend protocol: `src/llenergymeasure/core/inference_backends/protocols.py` - InferenceBackend, RuntimeCapabilities
- FLOPs estimation: `src/llenergymeasure/core/flops.py` - Multiple estimation strategies
- Extended metrics: `src/llenergymeasure/core/extended_metrics.py` - Efficiency analysis computation
- GPU utilities: `src/llenergymeasure/core/gpu_utilisation.py` - Background sampler
- Parallelism validation: `src/llenergymeasure/core/parallelism.py` - GPU count checks

**Orchestration:**
- Orchestrator: `src/llenergymeasure/orchestration/runner.py` - ExperimentOrchestrator class
- Campaign runner: `src/llenergymeasure/orchestration/campaign.py` - CampaignRunner
- Container manager: `src/llenergymeasure/orchestration/container.py` - ContainerManager
- Factory: `src/llenergymeasure/orchestration/factory.py` - Dependency injection wiring
- Context: `src/llenergymeasure/orchestration/context.py` - ExperimentContext
- Lifecycle: `src/llenergymeasure/orchestration/lifecycle.py` - Cleanup utilities

**Results:**
- Repository: `src/llenergymeasure/results/repository.py` - FileSystemRepository (CRUD)
- Aggregation: `src/llenergymeasure/results/aggregation.py` - Result combining, late aggregation
- Models: `src/llenergymeasure/domain/experiment.py` - RawProcessResult, AggregatedResult

**State:**
- State manager: `src/llenergymeasure/state/experiment_state.py` - StateManager, ExperimentState
- State storage: `.state/` directory - JSON experiment state files

**Testing:**
- Fixtures: `tests/conftest.py`
- Parameter discovery: `tests/runtime/discover_params.py`
- Canonical test: `tests/runtime/test_all_params.py` (standalone or pytest)

## Naming Conventions

**Files:**
- Python modules: `*.py` (lowercase with underscores)
- Test files: `test_*.py` (pytest discovers automatically)
- Pytest config: `conftest.py`
- Config files: `*_example.yaml`, `*_preset.yaml` in `configs/`
- Documentation: `CLAUDE.md` in each package for local docs, `README.md` for detailed info

**Directories:**
- Lowercase with underscores (snake_case): `inference_backends/`, `energy_backends/`
- Hierarchical: domain knowledge separated into focused modules
- Plural for collections: `tests/`, `configs/`, `scripts/`, `docs/`

**Classes:**
- PascalCase: `ExperimentOrchestrator`, `InferenceEngine`, `FileSystemRepository`
- Protocol suffixes: `*Protocol` or bare names (InferenceBackend, RuntimeCapabilities)
- Exception suffixes: `*Error`, `*Exception` (ConfigurationError, InvalidStateTransitionError)

**Functions:**
- snake_case: `create_orchestrator()`, `load_config()`, `get_flops_estimator()`
- Verb-first for actions: `run_experiment()`, `collect_metrics()`, `save_results()`
- Boolean checks: `is_backend_available()`, `should_use_docker_for_campaign()`

**Constants:**
- SCREAMING_SNAKE_CASE: `SCHEMA_VERSION`, `COMPLETION_MARKER_PREFIX`, `DEFAULT_DATASET`
- Located in `constants.py` or at module top-level

**Type/Protocol Names:**
- PascalCase: `ModelLoader`, `InferenceEngine`, `EnergyBackend`
- Singular: `MetricsCollector` (handles one concept)

## Where to Add New Code

**New Backend (Inference):**
- Implementation: `src/llenergymeasure/core/inference_backends/my_backend.py`
- Config model: Add `MyBackendConfig` to `src/llenergymeasure/config/backend_configs.py`
- Protocol: Implement `InferenceBackend` from `protocols.py`
- RuntimeCapabilities: Define launch mode and CUDA management
- Registration: Update factory in `src/llenergymeasure/orchestration/factory.py`
- Detection: Add to `KNOWN_BACKENDS` in `src/llenergymeasure/config/backend_detection.py`
- Tests: `tests/runtime/test_all_params.py` auto-discovers new params via introspection
- Example config: `configs/examples/my_backend_example.yaml`
- Docker: `docker/Dockerfile.my_backend`, add service to `docker-compose.yml`

**New Energy Backend:**
- Implementation: `src/llenergymeasure/core/energy_backends/my_energy.py`
- Protocol: Implement `EnergyBackend` from `src/llenergymeasure/protocols.py`
- Registration: Update factory to choose between backends

**New Metric:**
- Domain model: Add field to `ExtendedEfficiencyMetrics` in `src/llenergymeasure/domain/metrics.py`
- Computation: Add logic to `src/llenergymeasure/core/extended_metrics.py`
- Raw data collection: Add fields to `RawProcessResult` if needs late aggregation
- Aggregation: Update `src/llenergymeasure/results/aggregation.py` if needs multi-process aggregation

**New CLI Command:**
- Implementation: Create `src/llenergymeasure/cli/my_command.py`
- Function with Typer decorators: `@app.command()`
- Registration: Add to `_register_commands()` in `src/llenergymeasure/cli/__init__.py`
- Display helpers: Use `src/llenergymeasure/cli/display/` module for Rich output

**New Configuration Parameter:**
- Universal param: Add to `src/llenergymeasure/config/models.py` (tier 1: decoder, IO, traffic)
- Backend-specific param: Add to `backend_configs.py` (tier 2: PyTorchConfig/VLLMConfig/TensorRTConfig)
- Validation: Add custom validators in same file if cross-field constraints
- Documentation: Run `make generate-docs` to auto-generate docs from introspection
- Introspection: No manual updates needed, `introspection.py` auto-discovers via Pydantic

**New Domain Model:**
- File: `src/llenergymeasure/domain/my_model.py` (or extend existing if related)
- Pattern: Pydantic BaseModel with Field descriptions
- Export: Add to `__init__.py` in domain module

**New CLI Display Component:**
- File: Add to `src/llenergymeasure/cli/display/` (tables.py, summaries.py, results.py)
- Pattern: Rich console formatting, use shared console instance from `display/console.py`

**New State in State Machine:**
- Update `ExperimentStatus` enum in `state/experiment_state.py`
- Update `EXPERIMENT_VALID_TRANSITIONS` dict with allowed transitions
- Add transition logic to `StateManager` if needed

**New Campaign Strategy:**
- Implementation: Add to `orchestration/campaign.py:CampaignRunner.generate_execution_order()`
- CLI option: Add to `cli/campaign.py:campaign_cmd` `--structure` choices

**Utilities:**
- Shared helpers: `src/llenergymeasure/` top-level (logging.py, exceptions.py, security.py, progress.py)
- Backend-agnostic: `src/llenergymeasure/core/` subdirectories for specific concerns (prompts.py, distributed.py)

## Special Directories

**`.state/`:**
- Purpose: Experiment state storage for resumption
- Generated: Yes (created at runtime)
- Committed: No (.gitignore)
- Contents: JSON state files, one per experiment (`<exp_id>.json`)
- Cleanup: Delete to reset experiment state, safe to remove
- Location: Configurable via `LLM_ENERGY_STATE_DIR` env var

**`results/`:**
- Purpose: Experiment results storage
- Generated: Yes (created at first experiment)
- Committed: No (.gitignore)
- Contents:
  - `raw/<exp_id>/process_N.json` - Per-process raw results
  - `raw/<exp_id>/.completed_N` - Completion markers
  - `aggregated/<exp_id>.json` - Aggregated multi-process result
- Cleanup: Safe to delete, experiments can be re-run
- Location: Configurable via `LLM_ENERGY_RESULTS_DIR` env var

**`.planning/`:**
- Purpose: GSD (Get Stuff Done) planning documents
- Generated: By GSD orchestrator
- Committed: Yes (documentation)
- Contents:
  - `codebase/` - Codebase analysis (ARCHITECTURE.md, STRUCTURE.md, etc.)
  - `phases/` - Phase planning documents
- Pattern: Refreshed when codebase changes significantly

**`docker/`:**
- Purpose: Docker build context
- Contains: Dockerfiles for each backend (pytorch, vllm, tensorrt)
- Pattern: Multi-stage builds, pinned base images, non-root users
- Used by: `docker-compose.yml` for building backend images

**`configs/examples/`:**
- Purpose: Reference configurations
- Contains: Example YAML configs demonstrating backend usage
- Pattern: One config per backend, show common parameter combinations
- Used by: Documentation, new users learning the tool

**`scripts/`:**
- Purpose: Development utilities
- Contains:
  - `generate_*.py` - Doc generators (auto-update from SSOT)
  - `*.sh` - Setup and analysis scripts
- Pattern: Standalone Python scripts, often use introspection module

---

*Structure analysis: 2026-02-05*
