# Codebase Structure

**Analysis Date:** 2026-01-26

## Directory Layout

```
llm-efficiency-measurement-tool/
├── src/llenergymeasure/         # Main package
│   ├── cli/                     # CLI commands and display
│   ├── config/                  # Configuration system
│   ├── core/                    # Inference, metrics, model loading
│   ├── domain/                  # Domain models (Pydantic)
│   ├── orchestration/           # Experiment lifecycle
│   ├── results/                 # Results persistence, aggregation
│   ├── state/                   # Experiment state tracking
│   ├── constants.py             # Global constants
│   ├── exceptions.py            # Custom exception types
│   ├── logging.py               # Logging setup
│   ├── protocols.py             # Protocol definitions for DI
│   ├── resilience.py            # Retry logic
│   ├── security.py              # Path validation
│   └── progress.py              # Progress bar utilities
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   ├── runtime/                 # GPU-required parameter tests
│   ├── fixtures/                # Shared test fixtures
│   └── conftest.py              # Pytest configuration
├── configs/                     # Example experiment configs
│   ├── examples/                # Reference configurations
│   └── presets/                 # Built-in preset definitions
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
├── .state/                      # Experiment state storage (created at runtime)
├── results/                     # Results storage (created at runtime)
│   ├── raw/                     # Per-process raw results
│   └── aggregated/              # Aggregated results
├── .planning/                   # GSD planning documents
├── pyproject.toml               # Poetry project config
├── docker-compose.yml           # Multi-backend compose
├── Makefile                     # Development targets
├── CLAUDE.md                    # Architecture overview
└── README.md                    # Project README
```

## Directory Purposes

**`src/llenergymeasure/cli/`:**
- Purpose: Command-line interface and user-facing display
- Contains: Typer command handlers, Rich console formatting, command registration
- Key files: `__init__.py` (app setup), `experiment.py` (core commands), `display/` (formatting)
- Pattern: Each command group in separate file (config.py, results.py, etc.), imported in __init__.py

**`src/llenergymeasure/config/`:**
- Purpose: Configuration loading, validation, and preset management
- Contains: Pydantic models (universal + backend-specific), loader with inheritance, SSOT introspection
- Key files: `models.py` (universal ExperimentConfig), `backend_configs.py` (PyTorchConfig, VLLMConfig, TensorRTConfig), `introspection.py` (SSOT module)
- Pattern: YAML config → loader → Pydantic validation → runtime access

**`src/llenergymeasure/core/`:**
- Purpose: Inference execution and metrics collection
- Contains: Backend implementations, model loader, energy measurement, FLOPs estimation, metric computation
- Key subdirectories:
  - `inference_backends/` - Protocol-based backend implementations (pytorch.py, vllm.py, tensorrt.py)
  - `energy_backends/` - Energy measurement backends (CodeCarbon)
- Key files: `model_loader.py` (model/tokenizer loading), `inference.py` (metrics calculation), `extended_metrics.py` (efficiency metrics)

**`src/llenergymeasure/domain/`:**
- Purpose: Immutable data models for entire system
- Contains: Pydantic BaseModels for configs, metrics, results, model metadata
- Key files: `metrics.py` (inference/energy/compute metrics), `experiment.py` (raw/aggregated results), `model_info.py` (GPU/model metadata)
- Pattern: All models use Pydantic validation, used by all layers

**`src/llenergymeasure/orchestration/`:**
- Purpose: Experiment lifecycle management and subprocess coordination
- Contains: ExperimentOrchestrator (main runner), ExperimentContext, Factory (DI), Launcher, Lifecycle utilities
- Key files: `runner.py` (orchestrator), `context.py` (context container), `factory.py` (DI wiring), `launcher.py` (subprocess entry point)
- Pattern: Dependency injection via Factory, one orchestrator per process, launcher spawned by accelerate

**`src/llenergymeasure/results/`:**
- Purpose: Results persistence and aggregation
- Contains: Repository (CRUD), aggregation logic, CSV/JSON exporters
- Key files: `repository.py` (FileSystemRepository), `aggregation.py` (result combining), `exporters.py` (CSV/JSON output)
- Pattern: Raw results stored per-process, aggregated on-demand via separate step

**`src/llenergymeasure/state/`:**
- Purpose: Experiment state tracking with resumption support
- Contains: StateManager (persistent state), ExperimentState (status tracking), state machine
- Key files: `experiment_state.py` (StateManager + ExperimentState)
- Pattern: JSON state files in `.state/` directory, validated state transitions

**`tests/unit/`:**
- Purpose: Fast, isolated component tests
- Contains: Test files mirroring source structure (test_config_*.py, test_core_*.py, etc.)
- Pattern: Mock heavy dependencies (GPU, file system), fast execution

**`tests/integration/`:**
- Purpose: Component interaction tests
- Contains: Tests for config loading pipeline, state transitions, result aggregation
- Pattern: May touch real filesystem (in tmp dirs), real Pydantic validation

**`tests/runtime/`:**
- Purpose: GPU-required parameter validation tests
- Contains: SSOT-based parameter testing, backend-specific inference validation
- Key files: `test_all_params.py` (comprehensive standalone test), `discover_params.py` (introspection utility)
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
- Pattern: Markdown files, generated docs auto-updated from SSOT

## Key File Locations

**Entry Points:**

- CLI app: `src/llenergymeasure/cli/__init__.py` - Creates Typer app, registers commands
- Launcher subprocess: `src/llenergymeasure/orchestration/launcher.py` - Entry for accelerate launch
- Main experiment: `src/llenergymeasure/cli/experiment.py` - experiment_cmd function

**Configuration:**

- Universal config: `src/llenergymeasure/config/models.py` - ExperimentConfig (shared across all backends)
- Backend configs: `src/llenergymeasure/config/backend_configs.py` - PyTorchConfig, VLLMConfig, TensorRTConfig
- Config loader: `src/llenergymeasure/config/loader.py` - YAML loading with inheritance
- SSOT introspection: `src/llenergymeasure/config/introspection.py` - Auto-discover param metadata from models

**Core Logic:**

- Model loading: `src/llenergymeasure/core/model_loader.py` - HuggingFace loading with quantization
- Inference backends: `src/llenergymeasure/core/inference_backends/*.py` (pytorch.py, vllm.py, tensorrt.py)
- FLOPs estimation: `src/llenergymeasure/core/flops.py` - Multiple estimation strategies
- Extended metrics: `src/llenergymeasure/core/extended_metrics.py` - Efficiency analysis

**Orchestration:**

- Orchestrator: `src/llenergymeasure/orchestration/runner.py` - ExperimentOrchestrator class
- Factory: `src/llenergymeasure/orchestration/factory.py` - Dependency injection wiring
- Context: `src/llenergymeasure/orchestration/context.py` - ExperimentContext
- Lifecycle: `src/llenergymeasure/orchestration/lifecycle.py` - State machine, cleanup

**Results:**

- Repository: `src/llenergymeasure/results/repository.py` - FileSystemRepository (CRUD)
- Aggregation: `src/llenergymeasure/results/aggregation.py` - Result combining logic
- Models: `src/llenergymeasure/domain/experiment.py` - RawProcessResult, AggregatedResult

**State:**

- State manager: `src/llenergymeasure/state/experiment_state.py` - StateManager, ExperimentState
- State storage: `.state/` directory - JSON experiment state files

**Testing:**

- Fixtures: `tests/conftest.py`, `tests/conftest_backends.py`
- Parameter discovery: `tests/runtime/discover_params.py`
- Canonical test: `tests/runtime/test_all_params.py` (can run standalone or via pytest)

## Naming Conventions

**Files:**

- `*.py` for Python modules
- `test_*.py` for test files (pytest discovers automatically)
- `conftest.py` for pytest configuration/fixtures
- Config files: `*_example.yaml`, `*_preset.yaml` in `configs/`
- CLAUDE.md in each package for local documentation

**Directories:**

- Lowercase with underscores (snake_case): `inference_backends/`, `energy_backends/`
- Hierarchical: domain knowledge separated into focused modules
- Plural for collections: `tests/`, `configs/`, `scripts/`, `docs/`

**Classes:**

- PascalCase: `ExperimentOrchestrator`, `InferenceEngine`, `FileSystemRepository`
- Protocol suffixes: `*Protocol` or just bare names (RuntimeCapabilities, EnergyBackend)
- Exception suffixes: `*Error`, `*Exception` (ConfigurationError, InvalidStateTransitionError)

**Functions:**

- snake_case: `create_orchestrator()`, `load_config()`, `get_flops_estimator()`
- Verb-first for actions: `run_experiment()`, `collect_metrics()`, `save_results()`

**Constants:**

- SCREAMING_SNAKE_CASE: `SCHEMA_VERSION`, `COMPLETION_MARKER_PREFIX`
- Located in `constants.py` or at module top-level

**Type/Protocol Names:**

- PascalCase: `ModelLoader`, `InferenceEngine` (protocol-based)
- Singular: `MetricsCollector` (handles one concept)

## Where to Add New Code

**New Backend (Inference):**
- Implementation: `src/llenergymeasure/core/inference_backends/my_backend.py`
- Config model: Add `MyBackendConfig` to `src/llenergymeasure/config/backend_configs.py`
- Registration: Update factory in `src/llenergymeasure/orchestration/factory.py`
- Tests: `tests/runtime/test_all_params.py` auto-discovers new params via introspection
- Example config: `configs/examples/my_backend_example.yaml`

**New Energy Backend:**
- Implementation: `src/llenergymeasure/core/energy_backends/my_energy.py`
- Pattern: Implement EnergyBackendProtocol from `src/llenergymeasure/protocols.py`
- Registration: Update factory to choose between backends

**New Metric:**
- Add field to `ExtendedEfficiencyMetrics` in `src/llenergymeasure/domain/metrics.py`
- Computation: Add logic to `src/llenergymeasure/core/extended_metrics.py`
- Aggregation: Update `src/llenergymeasure/results/aggregation.py` if needs multi-process aggregation

**New CLI Command:**
- Implementation: Create `src/llenergymeasure/cli/my_command.py`
- Create function with Typer decorators
- Registration: Add to `_register_commands()` in `src/llenergymeasure/cli/__init__.py`
- Display helpers: Use `src/llenergymeasure/cli/display/` module for Rich output

**New Configuration Parameter:**
- Add to Pydantic model: `src/llenergymeasure/config/models.py` (universal) or `backend_configs.py` (backend-specific)
- Validation: Add custom validators in same file if cross-field constraints
- Documentation: Run `make generate-docs` to auto-generate docs from introspection

**New Domain Model:**
- File: `src/llenergymeasure/domain/my_model.py` (or extend existing file if related)
- Pattern: Pydantic BaseModel with Field descriptions
- Export: Add to `__init__.py` in domain module

**Utilities:**

- Shared helpers: `src/llenergymeasure/` top-level (logging.py, exceptions.py, security.py, progress.py)
- Backend-agnostic: `src/llenergymeasure/core/` subdirectories for specific concerns (prompts.py, distributed.py)

## Special Directories

**`.state/`:**
- Purpose: Experiment state storage for resumption
- Generated: Yes (created at runtime)
- Committed: No (.gitignore)
- Contents: JSON state files, one per experiment
- Cleanup: Delete to reset experiment state

**`results/`:**
- Purpose: Experiment results storage
- Generated: Yes (created at first experiment)
- Committed: No (.gitignore)
- Contents:
  - `raw/exp_ID/process_N.json` - Per-process raw results
  - `aggregated/exp_ID.json` - Aggregated multi-process result
- Cleanup: Safe to delete, experiments can be re-run

**`.planning/`:**
- Purpose: GSD (Ghost Surgeon Doctor) planning documents
- Generated: By GSD orchestrator
- Committed: Yes (documentation)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md

**`docker/`:**
- Purpose: Docker build context
- Contains: Dockerfiles for each backend (pytorch, vllm, tensorrt)
- Pattern: Multi-stage builds, pinned base images, non-root users

---

*Structure analysis: 2026-01-26*
