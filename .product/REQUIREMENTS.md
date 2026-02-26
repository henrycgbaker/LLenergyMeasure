# v2.0 Requirements

**Last updated**: 2026-02-26
**Status**: Authoritative — derived from all `decisions/`, `designs/`, and `preservation_audit/` documents
**Downstream**: `.planning/` phases and GSD execution implement these requirements

---

## How to Read This Document

- **Req IDs** use subsystem prefixes: `LA` (library API), `CFG` (config), `CM` (core measurement), `RES` (results), `CLI` (CLI), `STU` (study), `INF` (infrastructure)
- **Milestone tags**: `M1` (core single-experiment), `M2` (study/sweep), `M3` (Docker multi-backend), `M4` (advanced features). See [ROADMAP.md](ROADMAP.md) for milestone definitions.
- **Source**: every requirement cites its decision/design doc. If it's not in a decision doc, it's not a requirement.
- **Scoping principle**: carry forward all working v1.x code; only defer things needing significant new implementation.

---

## 1. Library API

> Source: [designs/library-api.md](designs/library-api.md), [decisions/experiment-study-architecture.md](decisions/experiment-study-architecture.md)

| ID | Requirement | Milestone |
|----|-------------|-----------|
| LA-01 | `run_experiment(config: str \| Path \| ExperimentConfig \| None, **kwargs) -> ExperimentResult` | M1 |
| LA-02 | `run_study(config: str \| Path \| StudyConfig) -> StudyResult` | M2 |
| LA-03 | Internal: `_run(StudyConfig) -> StudyResult` always (Option C). Both public functions wrap/unwrap. | M1 |
| LA-04 | `run_experiment()` is side-effect-free when `output_dir` not specified (no disk writes) | M1 |
| LA-05 | `run_study()` always writes manifest to disk (documented exception to side-effect-free) | M2 |
| LA-06 | No union return types. Each function returns exactly one type. | M1 |
| LA-07 | `__init__.py` exports: `run_experiment`, `run_study`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult`, `__version__` | M1 |
| LA-08 | Everything NOT in `__init__.py` is internal — no SemVer guarantee | M1 |
| LA-09 | One minor version deprecation window before removing any `__all__` export | M1 |
| LA-10 | `__version__: str = "2.0.0"` | M1 |

---

## 2. Config

> Sources: [designs/experiment-config.md](designs/experiment-config.md), [decisions/config-architecture.md](decisions/config-architecture.md), [designs/study-yaml.md](designs/study-yaml.md), [designs/user-config.md](designs/user-config.md)

### 2.1 ExperimentConfig

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CFG-01 | Single `ExperimentConfig` Pydantic model with composition (not inheritance). `extra="forbid"`. | M1 |
| CFG-02 | Field renames: `model_name`→`model`, `fp_precision`→`precision`, `num_input_prompts`→`n`. Clean break, no shims. | M1 |
| CFG-03 | `extra:`→`passthrough_kwargs: dict[str, Any] \| None = None` (declared field, not wildcard) | M1 |
| CFG-04 | Shared fields: `model: str`, `backend: Literal[...]`, `precision` (default `bf16`), `dataset` (default `aienergyscore`), `n: int = 100`, `random_seed: int = 42` | M1 |
| CFG-05 | Sub-configs: `decoder: DecoderConfig`, `warmup: WarmupConfig`, `baseline: BaselineConfig` | M1 |
| CFG-06 | Backend sections: `pytorch: PyTorchConfig \| None`, `vllm: VLLMConfig \| None`, `tensorrt: TensorRTConfig \| None` — all optional, `None` = omitted | M1 |
| CFG-07 | `lora: LoRAConfig \| None = None` — optional adapter support (carry-forward from v1.x) | M1 |
| CFG-08 | `PRECISION_SUPPORT` and `DECODING_SUPPORT` dicts as SSOT in `config/ssot.py` | M1 |
| CFG-09 | Cross-validators: precision vs backend, decoding vs backend, backend section vs `backend` field consistency | M1 |
| CFG-10 | None-as-sentinel on backend section fields: `None` = "use backend's own default" | M1 |

### 2.2 StudyConfig and Sweep

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CFG-11 | `StudyConfig` = thin resolved container: `list[ExperimentConfig]` + `ExecutionConfig` | M2 |
| CFG-12 | Sweep resolution at YAML parse time, before Pydantic validation | M2 |
| CFG-13 | Dotted notation sweep keys: `pytorch.batch_size: [1, 8]` — backend-scoped grid | M2 |
| CFG-14 | Three modes: grid sweep (Cartesian), explicit `experiments:` list, combined | M2 |
| CFG-15 | `ExecutionConfig`: `n_cycles` (Pydantic default=1, CLI effective default=3), `cycle_order`, `config_gap_seconds`, `cycle_gap_seconds` | M2 |
| CFG-16 | `study_design_hash` = SHA-256[:16] of sweep+experiments only (execution block excluded) | M2 |
| CFG-17 | Single run = degenerate `StudyConfig(experiments=[config])` | M1 |

### 2.3 YAML Loading

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CFG-18 | `load_experiment_config()` and `load_study_config()` in `config/loader.py` | M1 |
| CFG-19 | `yaml.safe_load` only — no arbitrary Python execution | M1 |
| CFG-20 | YAML parse errors → `ConfigError` with file path context. Pydantic `ValidationError` passes through unchanged. | M1 |

### 2.4 User Config

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CFG-21 | XDG path: `~/.config/llenergymeasure/config.yaml` via `platformdirs` | M1 |
| CFG-22 | Missing file → all defaults apply, no error | M1 |
| CFG-23 | Sections: `output:`, `runners:`, `measurement:`, `execution:` | M1 |
| CFG-24 | Runner precedence: env var `LLEM_RUNNER_<BACKEND>` → user config → `local` default | M1 |
| CFG-25 | Env vars: `LLEM_CARBON_INTENSITY`, `LLEM_DATACENTER_PUE`, `LLEM_NO_PROMPT` | M1 |

### 2.5 Introspection

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CFG-26 | `config/introspection.py`: `model_json_schema()` + domain layers (test values, constraint metadata, backend routing). ~250 LOC. | M1 |

---

## 3. Core Measurement

> Sources: [designs/architecture.md](designs/architecture.md), [designs/energy-backends.md](designs/energy-backends.md), [decisions/warmup-strategy.md](decisions/warmup-strategy.md), [decisions/flops-estimation.md](decisions/flops-estimation.md)

### 3.1 Inference Backends

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-01 | PyTorch inference backend (local) | M1 |
| CM-02 | vLLM inference backend (Docker) | M3 |
| CM-03 | TensorRT-LLM inference backend (Docker) | M3 |
| CM-04 | `InferenceBackend` Protocol in `core/backends/protocol.py` | M1 |
| CM-05 | Backend default: `pytorch` when multiple installed | M1 |
| CM-06 | P0 fix: PyTorch `model_kwargs` bug (L375) | M1 |
| CM-07 | P0 fix: vLLM streaming broken | M3 |
| CM-08 | P0 fix: Docker execution broken | M3 |
| CM-09 | P0 fix: vLLM `--shm-size` missing | M3 |
| CM-10 | Multi-backend study without Docker → hard error at pre-flight | M2 |

### 3.2 Energy Measurement

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-11 | NVML poller (`nvml.py`) always available — base install | M1 |
| CM-12 | Zeus backend optional (`llenergymeasure[zeus]`) | M1 |
| CM-13 | CodeCarbon backend optional (`llenergymeasure[codecarbon]`) | M1 |
| CM-14 | Energy backend priority: Zeus → NVML → CodeCarbon. Mutual exclusion enforced. | M1 |
| CM-15 | `torch.cuda.synchronize()` before every measurement stop | M1 |
| CM-16 | Timeseries: 1 Hz sampling, sidecar `timeseries.parquet` | M1 |

### 3.3 Baseline Power

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-17 | Idle GPU baseline measurement before warmup (`BaselineConfig.enabled: bool = True`) | M1 |
| CM-18 | `baseline_power_w` stored in ExperimentResult | M1 |
| CM-19 | `energy_adjusted_j = energy_total_j - (baseline_power_w × duration_sec)` | M1 |
| CM-20 | Baseline cache with session-level TTL (carry-forward from v1.x) | M1 |

### 3.4 Warmup

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-21 | Fixed-count default: `n_warmup: 5`, full-length prompts, reduced output | M1 |
| CM-22 | Thermal floor: 60s wait after warmup (configurable down to 30s) | M1 |
| CM-23 | CV-based convergence as opt-in: `convergence_detection: true` in WarmupConfig | M1 |
| CM-24 | `WarmupResult` with 6 fields: `converged`, `final_cv`, `iterations_completed`, `target_cv`, `max_prompts`, `latencies_ms` | M1 |

### 3.5 FLOPs and Metrics

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-25 | Primary metrics: `energy_per_output_token` (J/token) and `tokens_per_second` | M1 |
| CM-26 | FLOPs = reference metadata, not primary metric. PaLM formula (2 × N_params × tokens). | M1 |
| CM-27 | `FlopsResult` with `method: str` and `confidence: Literal["high", "medium", "low"]` | M1 |
| CM-28 | Warmup tokens excluded from FLOPs calculation | M1 |

### 3.6 Pre-flight

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-29 | Pre-flight checks: GPU available, backend installed, dataset accessible, VRAM estimate | M1 |
| CM-30 | Pre-flight failure → `PreFlightError`. All failures reported at once, not one at a time. | M1 |
| CM-31 | GPU persistence mode: pre-flight warning (not blocking error) | M1 |

### 3.7 Environment

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CM-32 | `EnvironmentSnapshot` auto-captured at experiment start: Python version, CUDA version, driver version, GPU names/VRAM, pip freeze, tool version | M1 |
| CM-33 | CUDA version: multi-source detection (torch → version.txt → nvcc → None) | M1 |
| CM-34 | Thermal throttle detection (carry-forward from v1.x) | M1 |

---

## 4. Results

> Sources: [designs/result-schema.md](designs/result-schema.md), [decisions/output-storage.md](decisions/output-storage.md)

### 4.1 ExperimentResult Schema

| ID | Requirement | Milestone |
|----|-------------|-----------|
| RES-01 | `ExperimentResult` (renamed from `AggregatedResult`). All v2.0 fields ship together. | M1 |
| RES-02 | `measurement_config_hash: str` — SHA-256[:16], environment snapshot excluded | M1 |
| RES-03 | `measurement_methodology: Literal["total", "steady_state", "windowed"]` | M1 |
| RES-04 | `steady_state_window: tuple[float, float] \| None` | M1 |
| RES-05 | `schema_version: str = "2.0"` | M1 |
| RES-06 | `baseline_power_w`, `energy_adjusted_j`, `energy_per_device_j` | M1 |
| RES-07 | `EnergyBreakdown` nested model (raw_j, adjusted_j, baseline provenance) — carry-forward from v1.x | M1 |
| RES-08 | `reproducibility_notes: str` — fixed disclaimer about NVML accuracy | M1 |
| RES-09 | `environment_snapshot: EnvironmentSnapshot` | M1 |
| RES-10 | `measurement_warnings: list[str]` — short duration, low baseline confidence, etc. | M1 |
| RES-11 | `warmup_excluded_samples: int \| None` | M1 |
| RES-12 | Process completeness validation (marker files + 4-check) — scoped to PyTorch multi-GPU internal | M1 |

### 4.2 StudyResult Schema

| ID | Requirement | Milestone |
|----|-------------|-----------|
| RES-13 | `StudyResult`: `study_design_hash`, `measurement_protocol`, `result_files: list[str]`, `summary: StudySummary` | M2 |
| RES-14 | `StudyManifest`: in-progress checkpoint, distinct from `StudyResult` | M2 |
| RES-15 | `result_files` contains paths, not embedded results | M2 |

### 4.3 Persistence and Export

| ID | Requirement | Milestone |
|----|-------------|-----------|
| RES-16 | Output always in subdirectory: `{name}_{timestamp}/result.json` + `timeseries.parquet` | M1 |
| RES-17 | Collision policy: append `_1`, `_2` counter — never overwrite | M1 |
| RES-18 | JSON = always primary. Parquet = timeseries sidecar. CSV = opt-in `--export-csv`. | M1 |
| RES-19 | `to_json()`, `to_parquet()`, `from_json()` in `results/persistence.py` | M1 |
| RES-20 | Late aggregation (per-process → ExperimentResult) in `results/aggregation.py` | M1 |
| RES-21 | Unified output layout: all backends → one `ExperimentResult`. PyTorch multi-GPU raw files internal (`.state/`). | M1 |

---

## 5. CLI

> Sources: [designs/cli-commands.md](designs/cli-commands.md), [decisions/cli-ux.md](decisions/cli-ux.md), [designs/observability.md](designs/observability.md), [decisions/error-handling.md](decisions/error-handling.md)

### 5.1 Commands

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CLI-01 | 2 commands + 1 flag: `llem run`, `llem config`, `llem --version` | M1 |
| CLI-02 | Entry point renamed `lem` → `llem`. No backward-compat shim. | M1 |
| CLI-03 | `llem run [CONFIG] [OPTIONS]` — auto-detects single vs study YAML | M1 |
| CLI-04 | `llem run` flags: `--model`, `--backend`, `--dataset`, `-n`, `--batch-size`, `--precision`, `--output`, `--dry-run`, `--quiet`, `--verbose` | M1 |
| CLI-05 | Study-mode flags: `--cycles`, `--no-gaps`, `--order`, `--resume` | M2 |
| CLI-06 | `llem config [--verbose]` — environment display + setup guidance | M1 |
| CLI-07 | `--dry-run`: L1 (Pydantic validation, always) + L2 (VRAM estimate, grid preview) | M1 |

### 5.2 Display

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CLI-08 | Plain text output (~200 LOC). tqdm for progress. No Rich dependency. | M1 |
| CLI-09 | Output routing: progress → `stderr`, final summary → `stdout` | M1 |
| CLI-10 | `--quiet`: suppress progress, keep final summary. `--verbose`: add subprocess events. | M1 |
| CLI-11 | Thermal gap countdown display during inter-experiment pauses | M2 |

### 5.3 Error Handling

| ID | Requirement | Milestone |
|----|-------------|-----------|
| CLI-12 | Exit codes: 0 (success), 1 (error), 2 (usage/config), 130 (SIGINT) | M1 |
| CLI-13 | `LLEMError` → `ConfigError`, `BackendError`, `PreFlightError`, `ExperimentError`, `StudyError` | M1 |
| CLI-14 | Pydantic `ValidationError` passes through unchanged | M1 |

---

## 6. Study

> Sources: [designs/study-yaml.md](designs/study-yaml.md), [decisions/study-execution-model.md](decisions/study-execution-model.md), [decisions/experiment-isolation.md](decisions/experiment-isolation.md), [designs/study-resume.md](designs/study-resume.md)

### 6.1 Execution

| ID | Requirement | Milestone |
|----|-------------|-----------|
| STU-01 | `StudyRunner` iterates experiments, spawns subprocesses (`multiprocessing.get_context("spawn")`) | M2 |
| STU-02 | IPC via `multiprocessing.Pipe`; file-based fallback for results >1MB | M2 |
| STU-03 | `daemon=False` on subprocess (clean CUDA teardown) | M2 |
| STU-04 | Timeout via `p.join(timeout=...)` + SIGKILL on timeout | M2 |
| STU-05 | Single experiment (`llem run experiment.yaml`) runs in-process — no subprocess | M1 |
| STU-06 | Config gap between experiments from user config (machine-local) | M2 |
| STU-07 | `cycle_order: sequential \| interleaved` — interleaved round-robins across configs | M2 |

### 6.2 Manifest and Resume

| ID | Requirement | Milestone |
|----|-------------|-----------|
| STU-08 | `StudyManifest` written after each experiment completes (checkpoint pattern) | M2 |
| STU-09 | `ManifestWriter`: `mark_running()`, `mark_completed()`, `mark_failed()` — atomic writes | M2 |
| STU-10 | `--resume` flag implementation | M4 |
| STU-11 | Resume identity: `measurement_config_hash` of resolved ExperimentConfig | M4 |

---

## 7. Infrastructure

> Sources: [designs/packaging.md](designs/packaging.md), [designs/testing.md](designs/testing.md), [decisions/installation.md](decisions/installation.md), [decisions/docker-execution.md](decisions/docker-execution.md)

### 7.1 Packaging

| ID | Requirement | Milestone |
|----|-------------|-----------|
| INF-01 | `pyproject.toml`, `src/` layout, hatchling build | M1 |
| INF-02 | Base deps: `pydantic>=2.0`, `typer>=0.9`, `pyyaml>=6.0`, `platformdirs>=3.0`, `nvidia-ml-py`, `pyarrow>=14.0`, `tqdm` | M1 |
| INF-03 | Extras: `[pytorch]`, `[vllm]`, `[tensorrt]`, `[zeus]`, `[codecarbon]`, `[webhooks]` | M1 |
| INF-04 | No `[all]` extra (vLLM + TRT are process-incompatible) | M1 |
| INF-05 | Entry point: `llem` only. `lem` removed. | M1 |

### 7.2 Protocols and State

| ID | Requirement | Milestone |
|----|-------------|-----------|
| INF-06 | `protocols.py`: `ModelLoader`, `InferenceEngine`, `MetricsCollector`, `EnergyBackend`, `ResultsRepository` — carry-forward from v1.x | M1 |
| INF-07 | State machine: 3 states (`INITIALISING`, `MEASURING`, `DONE`) + `failed: bool` | M1 |
| INF-08 | `StateManager`, `find_by_config_hash()`, `cleanup_stale()`, atomic state writes | M1 |

### 7.3 Testing

| ID | Requirement | Milestone |
|----|-------------|-----------|
| INF-09 | Two-tier: `tests/unit/` (no GPU) + `tests/integration/` (`@pytest.mark.gpu`) | M1 |
| INF-10 | Protocol injection mocks (not `unittest.mock` patching) | M1 |
| INF-11 | Config introspection drives test value generation (SSOT) | M1 |
| INF-12 | GPU CI: merge to main + weekly + manual + path-filtered PRs | M1 |

### 7.4 Docker

| ID | Requirement | Milestone |
|----|-------------|-----------|
| INF-13 | Ephemeral `docker run` per experiment (container lifecycle = experiment lifecycle) | M3 |
| INF-14 | Config via env var injection; results via shared volume | M3 |
| INF-15 | TRT engine cache: `~/.llenergymeasure/trt-engines/{hash}/` | M3 |
| INF-16 | Official images: `llenergymeasure/{backend}:{version}-cuda{major}` | M3 |
| INF-17 | Docker treated as system dependency, checked at pre-flight | M3 |

### 7.5 Resilience

| ID | Requirement | Milestone |
|----|-------------|-----------|
| INF-18 | Retry logic (carry-forward from v1.x `resilience.py`) | M1 |
| INF-19 | Subprocess lifecycle management (carry-forward from v1.x) | M1 |
| INF-20 | Path sanitisation and security (carry-forward from v1.x) | M1 |

---

## 8. Explicitly Deferred

| ID | What | Deferred To | Reason | Source |
|----|------|-------------|--------|--------|
| DEF-01 | `confidence_intervals` field (bootstrap CIs) | v2.1 | Raw measurement is primary; CIs downstream analysis | P-17 |
| DEF-02 | `_extends:` YAML inheritance | Post-v2.0 | Too complex; 0/0 peer tools at v1 stage | `config-architecture.md` |
| DEF-03 | Singularity/Apptainer runner | v2.1+ | `NotImplementedError` in v2.0 | `user-config.md` |
| DEF-04 | `llem results push` (shareability/upload) | Post-v2.0 | Trust model unresolved | `future-versions.md` |
| DEF-05 | lm-eval integration (`quality_per_joule`) | v3.0 | Separate version | `future-versions.md` |
| DEF-06 | Web platform | v4.0 | Separate product | `web-platform.md` |
| DEF-07 | Study-level cross-config aggregation | Post-v2.0 | StudyResult = manifest only; stats downstream | P-02 |
| DEF-08 | Persistent Docker containers | v2.1+ | Ephemeral-only in v2.0 | `docker-execution.md` |
| DEF-09 | Attention FLOPs in estimation | v2.1 | Documented limitation for >2048 tokens | `flops-estimation.md` |

---

## 9. Explicitly Cut

| ID | What | Reason | Source |
|----|------|--------|--------|
| CUT-01 | `NormalisedMetrics` + `precision_factor` | Normalising hides the efficiency differences the tool exists to measure | P-05 |
| CUT-02 | Unified quantisation abstraction (P-04) | Backend-specific params in backend sections; no translation layer | P-04 |
| CUT-03 | Unified speculative decoding config (P-10) | Same policy as P-04 | P-10 |
| CUT-04 | Naming/alias registry (P-16) | Clean break at v2.0; no deprecation shims | P-16 |
| CUT-05 | `PRESETS` dict registry (N-X07) | 0/8 peers use named preset registries; opinionated Pydantic defaults instead | N-X07 |
| CUT-06 | Batch processing command (N-X06) | `llem run study.yaml` covers the use case; 0/8 peers have `batch` subcommand | N-X06 |
| CUT-07 | `--profile` flag | 0/5 peers use named rigour profiles; direct flags only | `cli-ux.md` |
| CUT-08 | `llem init` command | No peer tool has it; not needed | `installation.md` |
| CUT-09 | 1,524 lines confirmed dead code | Phase 4 audit | `AUDIT-REPORT.md` |

---

## Open Items (Resolve Before M1)

| Item | Status | Reference |
|------|--------|-----------|
| Create `aienergyscore.jsonl` built-in dataset file | TODO | `NEEDS_ADDRESSING.md` |
| Confirm `peak_memory_mb` measurement semantics | TODO | `.planning/todos/` |
| Fix PyTorch P0 bug (model_kwargs L375) | TODO | `versioning-roadmap.md` |
