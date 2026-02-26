# Architecture

**Analysis Date:** 2026-02-05

## Pattern Overview

**Overall:** Layered architecture with dependency injection, protocol-based abstraction, and campaign execution model

**Key Characteristics:**
- Clean layer separation: CLI → Orchestration → Core → Domain
- Protocol-based interfaces enable pluggable backends (PyTorch, vLLM, TensorRT)
- Late aggregation pattern: raw per-process results saved separately, aggregated on-demand
- Campaign execution with dual container strategy (ephemeral vs persistent Docker dispatch)
- State machine for experiment lifecycle with validated transitions and resume capability
- SSOT (Single Source of Truth) configuration via Pydantic introspection
- Backend-native configuration architecture (tier 1 universal + tier 2 backend-specific)

## Layers

**CLI Layer:**
- Purpose: User interface, command dispatch, output formatting, campaign coordination
- Location: `src/llenergymeasure/cli/`
- Contains: Typer commands, display utilities (Rich console), argument parsing, Docker dispatch logic
- Depends on: Orchestration, Config, Results, State
- Used by: End users via `lem` command
- Key modules: `experiment.py` (single experiments), `campaign.py` (multi-config campaigns), `display/` (formatting)

**Orchestration Layer:**
- Purpose: Experiment lifecycle management, distributed execution, campaign coordination, Docker container management
- Location: `src/llenergymeasure/orchestration/`
- Contains: `ExperimentOrchestrator`, `CampaignRunner`, `ContainerManager`, `launcher.py`, factory (DI)
- Depends on: Core, Domain, Config, Results, State
- Used by: CLI layer, launcher subprocess
- Key modules: `runner.py` (single experiment), `campaign.py` (multi-config orchestration), `container.py` (persistent Docker), `launcher.py` (subprocess entry)

**Core Layer:**
- Purpose: Inference execution, model loading, metrics collection, backend implementations
- Location: `src/llenergymeasure/core/`
- Contains: Inference backends (PyTorch/vLLM/TensorRT), model loader, energy backends, metrics collectors, GPU utilities
- Depends on: Domain, Config
- Used by: Orchestration layer
- Key modules: `inference_backends/` (backend implementations), `model_loader.py`, `extended_metrics.py`, `gpu_utilisation.py`

**Domain Layer:**
- Purpose: Data models, type definitions, validation rules (pure models, zero dependencies)
- Location: `src/llenergymeasure/domain/`
- Contains: Pydantic models for configs, results, metrics
- Depends on: Nothing
- Used by: All other layers
- Key modules: `experiment.py` (results), `metrics.py` (metric models), `model_info.py`

**Configuration Layer:**
- Purpose: Config loading, validation, inheritance, backend detection, SSOT introspection
- Location: `src/llenergymeasure/config/`
- Contains: `ConfigLoader`, backend configs, introspection module, Docker/backend detection
- Depends on: Domain
- Used by: All layers for configuration access
- Key modules: `loader.py`, `models.py` (universal), `backend_configs.py` (backend-specific), `introspection.py` (SSOT), `docker_detection.py`, `backend_detection.py`

**State Management Layer:**
- Purpose: Experiment state persistence with resume capability, validated state transitions
- Location: `src/llenergymeasure/state/`
- Contains: `StateManager`, `ExperimentState`, state machine transitions
- Depends on: Domain
- Used by: Orchestration, CLI
- Key modules: `experiment_state.py` (StateManager + state machine)

**Results Layer:**
- Purpose: Results persistence, late aggregation, export (CSV/JSON)
- Location: `src/llenergymeasure/results/`
- Contains: `FileSystemRepository`, `ResultAggregator`, exporters
- Depends on: Domain
- Used by: Orchestration, CLI
- Key modules: `repository.py`, `aggregation.py`, `exporters.py`

## Data Flow

**Single Experiment Execution (Local):**

1. User runs `lem experiment config.yaml -d alpaca -n 100`
2. CLI (`cli/experiment.py`) loads config, resolves prompts, validates
3. Backend detection: `is_backend_available()` checks if backend installed locally
4. State check: `StateManager.find_matching_experiment()` looks for incomplete run
5. If resume available: prompt user, load previous state
6. CLI spawns `accelerate launch orchestration/launcher.py --config-path ...`
7. `launcher.py` entry point:
   - Creates `ExperimentContext` with Accelerator
   - Gets backend via `get_backend(config.backend)`
   - Checks `backend.get_runtime_capabilities()` for CUDA management
   - Calls `factory.create_components()` for DI wiring
   - Creates `ExperimentOrchestrator` with injected components
8. `ExperimentOrchestrator.run()`:
   - Backend `initialize()` (loads model/tokenizer)
   - Backend `run_inference()` (executes prompts)
   - Metrics collection (inference, energy, compute, extended)
   - Saves `RawProcessResult` to `results/raw/<exp_id>/process_N.json`
   - Writes completion marker `.completed_N`
9. Main process detects all markers → triggers aggregation
10. `ResultAggregator` loads all raw results, computes combined metrics
11. `AggregatedResult` saved to `results/aggregated/<exp_id>.json`
12. State transition: RUNNING → COMPLETED → AGGREGATED

**Single Experiment Execution (Docker):**

1. Same CLI entry, but backend not available locally
2. CLI detects Docker needed via `should_use_docker_for_campaign([backend])`
3. Ephemeral container strategy: `docker compose run --rm <backend> lem experiment ...`
4. GPU routing: `CUDA_VISIBLE_DEVICES` propagated to container via env
5. Inside container: same local flow (launcher → orchestrator → results)
6. Results written to mounted volume, accessible from host

**Campaign Execution (Multi-Config):**

1. User runs `lem campaign config1.yaml config2.yaml config3.yaml --cycles 3 --structure interleaved`
2. CLI (`cli/campaign.py`) loads all configs, creates `CampaignConfig`
3. Backend detection: collect unique backends from all configs
4. Dispatch decision:
   - Single backend + installed locally → run locally
   - Multi-backend OR backend not installed → dispatch to Docker
5. Local execution:
   - `CampaignRunner` generates execution order (interleaved/shuffled/grouped)
   - For each experiment: warmup → run → wait gap → next
6. Docker execution (persistent container strategy):
   - `ContainerManager.start_all()` → `docker compose up -d <services>`
   - Container warmup delay (let services initialize)
   - For each experiment:
     - Warmup prompts via `docker compose exec <service> lem warmup ...`
     - Wait config gap (thermal cooldown)
     - Run experiment via `docker compose exec <service> lem experiment ...`
     - Collect result
     - Wait cycle gap if cycle complete
   - `ContainerManager.stop_all()` → `docker compose down`
7. `CampaignManifest` tracks progress (enables resume)
8. Results collected for cross-config comparison

**GPU Routing (Multi-GPU Systems):**

1. User specifies parallelism in config:
   - PyTorch: `pytorch.num_processes=4` → Accelerate data parallelism
   - vLLM: `vllm.tensor_parallel_size=4` → internal tensor parallelism
   - TensorRT: `tensorrt.tp_size=4` → internal tensor parallelism
2. Launcher determines GPU allocation:
   - PyTorch: `accelerate launch --num_processes=4` → each process gets one GPU
   - vLLM/TensorRT: single process, backend manages GPU distribution
3. Docker dispatch:
   - Host: `CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose run ...`
   - Container: env var propagated → GPUs visible to backend
4. Validation: `core/parallelism.py` checks sufficient GPUs available

**State Management and Resume:**

1. Experiment start: StateManager creates state with INITIALISED status
2. Before inference: transition to RUNNING
3. Each process completion: update `ProcessProgress` entry
4. All processes complete: transition to COMPLETED
5. Aggregation complete: transition to AGGREGATED
6. On failure/interrupt: transition to FAILED/INTERRUPTED
7. Resume flow:
   - `lem resume` or `lem experiment --resume`
   - StateManager finds INTERRUPTED/FAILED experiments
   - Config hash matching ensures same configuration
   - State reloads, experiment continues from last checkpoint

## Key Abstractions

**InferenceBackend Protocol:**
- Purpose: Unified interface for different inference engines
- Examples: `core/inference_backends/pytorch.py`, `vllm.py`, `tensorrt.py`
- Pattern: Each backend implements `initialize()`, `run_inference()`, `cleanup()`, `get_runtime_capabilities()`, `validate_config()`
- Runtime capabilities: Backends declare launch mode (ACCELERATE/TORCHRUN/DIRECT) and CUDA management

**RuntimeCapabilities:**
- Purpose: Declare backend launch requirements and CUDA management
- Examples:
  - PyTorch: `LaunchMode.ACCELERATE`, `CudaManagement.ORCHESTRATOR`
  - vLLM: `LaunchMode.DIRECT`, `CudaManagement.BACKEND`
  - TensorRT: `LaunchMode.DIRECT`, `CudaManagement.BACKEND`
- Pattern: Queried before initialization to avoid CUDA fork issues

**ExperimentContext:**
- Purpose: Encapsulate experiment runtime state
- Examples: `orchestration/context.py`
- Pattern: Created via context manager or factory, holds accelerator, device, config, provenance
- Contains: experiment_id, config, accelerator, device, process info, provenance tracking

**ExperimentOrchestrator:**
- Purpose: Coordinate experiment lifecycle from model loading to result persistence
- Examples: `orchestration/runner.py`
- Pattern: Dependency injection via factory, protocol-based components
- Responsibilities: Backend initialization, inference execution, metrics collection, result saving

**CampaignRunner:**
- Purpose: Multi-config experiment orchestration with scheduling
- Examples: `orchestration/campaign.py`
- Pattern: Generates execution order, manages warmup/gaps, tracks progress
- Strategies: interleaved (fair comparison), shuffled (random order), grouped (all cycles per config)

**ContainerManager:**
- Purpose: Persistent Docker container lifecycle for campaign execution
- Examples: `orchestration/container.py`
- Pattern: `docker compose up -d` → `exec` commands → `docker compose down`
- Benefits: No container startup overhead per experiment, warm containers

**StateManager:**
- Purpose: Atomic state persistence with validated transitions
- Examples: `state/experiment_state.py`
- Pattern: Load/save with file locking, transition validation, config hashing for resume
- State machine: INITIALISED → RUNNING → COMPLETED → AGGREGATED (with FAILED/INTERRUPTED recovery)

**ConfigLoader:**
- Purpose: Config inheritance and resolution with preset application
- Examples: `config/loader.py`
- Pattern: Resolves `_extends` chains, applies presets, validates, tracks provenance
- Precedence: CLI flags > Config file > Preset > Defaults

**SSOT Introspection:**
- Purpose: Derive parameter metadata from Pydantic models (auto-discovery)
- Examples: `config/introspection.py`
- Pattern: `get_backend_params()`, `get_streaming_constraints()` inspect Pydantic Field definitions
- Used by: Runtime tests, doc generators, CLI validation

**FileSystemRepository:**
- Purpose: CRUD operations for experiment results
- Examples: `results/repository.py`
- Pattern: Single responsibility, JSON serialization, atomic writes (temp → rename)
- Storage: `results/raw/<exp_id>/process_N.json`, `results/aggregated/<exp_id>.json`

## Entry Points

**CLI Entry:**
- Location: `src/llenergymeasure/cli/__init__.py`
- Triggers: User runs `lem <command>`
- Responsibilities: Load .env, setup logging, register commands, parse arguments, determine verbosity

**Experiment Command:**
- Location: `src/llenergymeasure/cli/experiment.py:experiment_cmd`
- Triggers: `lem experiment <config> [options]`
- Responsibilities: Load config, resolve prompts, validate, detect backend, spawn launcher or Docker, track state

**Campaign Command:**
- Location: `src/llenergymeasure/cli/campaign.py:campaign_cmd`
- Triggers: `lem campaign <configs> --cycles N`
- Responsibilities: Load all configs, determine dispatch strategy, coordinate execution, track progress

**Launcher Entry:**
- Location: `src/llenergymeasure/orchestration/launcher.py`
- Triggers: `accelerate launch -m llenergymeasure.orchestration.launcher`
- Responsibilities: Create context, wire dependencies via factory, run orchestrator, save results

**Docker Entry:**
- Location: Container service definitions in `docker-compose.yml`
- Triggers: `docker compose run <backend>` or `docker compose exec <backend>`
- Responsibilities: Isolated backend execution with GPU routing, volume mounts for results

**Resume Entry:**
- Location: `src/llenergymeasure/cli/resume.py:resume_cmd`
- Triggers: `lem resume` or `lem experiment --resume`
- Responsibilities: Find interrupted experiments, prompt user, reload state, continue execution

**Init Entry:**
- Location: `src/llenergymeasure/cli/init_cmd.py:init_cmd`
- Triggers: `lem init`
- Responsibilities: Interactive campaign setup, generate campaign YAML

## Error Handling

**Strategy:** Layered validation with graceful degradation for optional features

**Patterns:**
- **Config validation**: Three-tier (Pydantic schema → custom validators → runtime validation)
- **Backend availability**: `is_backend_available()` checks imports, Docker fallback available
- **State transitions**: `InvalidStateTransitionError` raised for illegal transitions
- **Energy tracking failures**: Non-fatal, results include `energy_tracking_failed=True`
- **Extended metrics**: Nullable fields (TPOT, GPU utilisation) for unavailable measurements
- **Resume capability**: State machine enables recovery from FAILED/INTERRUPTED states
- **Docker dispatch**: Automatic fallback when backend not installed locally
- **GPU allocation**: Validation in `core/parallelism.py` prevents oversubscription

**Custom Exceptions:**
- `ConfigurationError`: Invalid config or missing requirements
- `BackendNotAvailableError`: Backend not installed or not supported
- `BackendInitializationError`: Model loading or backend setup failed
- `BackendInferenceError`: Inference execution failed
- `InvalidStateTransitionError`: Illegal state machine transition
- `AggregationError`: Result aggregation failed

**Resilience:**
- `resilience.py` provides retry decorators with exponential backoff
- Campaign execution continues on single experiment failure (tracked in manifest)
- Process completion markers enable partial result recovery

## Cross-Cutting Concerns

**Logging:**
- Loguru-based structured logging in `logging.py`
- Verbosity levels: quiet (warnings only), normal (info + warnings), verbose (debug + timestamps)
- Environment variable: `LLM_ENERGY_VERBOSITY` propagated to subprocesses
- JSON output mode: `LLM_ENERGY_JSON_OUTPUT=true` for machine-readable results
- Secrets filtering: HF tokens and API keys redacted in logs

**Validation:**
- Pydantic models: Automatic schema validation with Field constraints
- Custom validators: Cross-field constraints in config models
- Backend validation: `InferenceBackend.validate_config()` returns list of warnings
- Path security: `security.py` prevents path traversal attacks
- State transitions: `EXPERIMENT_VALID_TRANSITIONS` dict enforces valid state changes

**CUDA Management:**
- Backend capabilities declare who initializes CUDA context
- PyTorch (orchestrator-managed): Safe to call `torch.cuda.*` before `initialize()`
- vLLM/TensorRT (backend-managed): Orchestrator must NOT call CUDA functions (fork safety)
- Detection: `RuntimeCapabilities.orchestrator_may_call_cuda` property
- Context creation: `_get_orchestrator_may_call_cuda()` checks capabilities before Accelerator init

**GPU Routing:**
- Environment variable: `CUDA_VISIBLE_DEVICES` controls GPU visibility
- Docker propagation: Host env vars passed to container via compose env section
- Validation: `core/parallelism.py` validates GPU count vs parallelism config
- MIG detection: `core/gpu_info.py` detects MIG instances, warns about power measurement limitations

**Parallelism:**
- PyTorch: Accelerate data parallelism (`pytorch.num_processes`, each process gets one GPU)
- vLLM: Internal tensor/pipeline parallelism (`vllm.tensor_parallel_size`, `pipeline_parallel_size`)
- TensorRT: Internal tensor/pipeline parallelism (`tensorrt.tp_size`, `pp_size`)
- Launch mode: Declared via `RuntimeCapabilities` (ACCELERATE vs DIRECT)

**Energy Tracking:**
- Protocol-based backends: `EnergyBackend` in `protocols.py`
- CodeCarbon implementation: `core/energy_backends/codecarbon.py`
- Baseline subtraction: Idle power measured and subtracted from raw measurements
- MIG limitations: Per-instance power not available, parent GPU power used with warning
- Failure handling: `energy_tracking_failed` flag, placeholder metrics (0.0)

**Configuration Provenance:**
- Full parameter tracking: `{value, source, source_detail}` for each parameter
- Sources: CLI, config file, preset, default
- Preset chain: List of presets applied in order (for inheritance tracking)
- Effective config: Full resolved config serialized in results for reproducibility
- CLI overrides: Separate dict tracking which params overridden via CLI flags

**Container Strategy:**
- Ephemeral (default): `docker compose run --rm` per experiment (full isolation)
- Persistent (campaign): `docker compose up -d` → `exec` commands → `down` (speed)
- Detection: `is_inside_docker()` checks `/.dockerenv` and `/proc/1/cgroup`
- Dispatch decision: `should_use_docker_for_campaign()` based on backend availability

**Backend Detection:**
- Import-based: `is_backend_available()` attempts import, catches errors
- Known backends: `["pytorch", "vllm", "tensorrt"]`
- Docker recommendation: vLLM/TensorRT suggest Docker in install hints
- Runtime check: Backends can also check GPU compute capability

---

*Architecture analysis: 2026-02-05*
