# v2.0 Roadmap

**Last updated**: 2026-02-26
**Status**: Authoritative — milestones derived from [REQUIREMENTS.md](REQUIREMENTS.md)
**Upstream**: `decisions/`, `designs/`, `preservation_audit/`
**Downstream**: `.planning/` phases implement these milestones

---

## Principle

v2.0 is delivered across **4 incremental milestones**. Each milestone ships a usable product.
No separate v2.1/v2.2 versions exist — all milestones contribute to the single v2.0 release.

> Carry forward all working v1.x code. Only defer things needing significant new implementation.
> If it's trivial and important, include it.

---

## Milestone Overview

```
M1  Core Single-Experiment     llem run --model X works (PyTorch, local)
M2  Study / Sweep              llem run study.yaml works (multi-experiment, subprocess isolation)
M3  Docker Multi-Backend       vLLM + TensorRT-LLM via Docker, cross-backend studies
M4  Advanced Features          Traffic simulation, streaming latency, study resume, webhooks
```

Each milestone has **entry criteria** (what must exist before starting) and **exit criteria**
(what must work before the milestone is considered complete).

---

## M1 — Core Single-Experiment

> **Ships**: `llem run --model meta-llama/Llama-3.1-8B` produces a complete `ExperimentResult`
> with energy measurement, baseline correction, warmup, FLOPs, environment snapshot,
> JSON + Parquet output.

### Scope

**Library restructure**
- `src/llenergymeasure/` package with `__init__.py` public API
- `run_experiment()`, `run_study()` (study returns single-experiment degenerate case)
- `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult` exports
- `_api.py` implementation, `protocols.py` DI interfaces

**Config**
- `ExperimentConfig` composition model (all field renames, backend sections, validators)
- `config/loader.py` (YAML → Pydantic), `config/ssot.py` (PRECISION_SUPPORT, DECODING_SUPPORT)
- `config/introspection.py` (SSOT: model_json_schema → test values, docs, CLI)
- User config at `~/.config/llenergymeasure/config.yaml`
- Error wrapping: YAML errors → `ConfigError`, Pydantic `ValidationError` passes through

**Core measurement**
- PyTorch inference backend (P0 bug fix: model_kwargs)
- Energy backends: NVML (base), Zeus (optional), CodeCarbon (optional)
- Energy backend mutual exclusion (Zeus → NVML → CodeCarbon priority)
- Baseline power measurement + `energy_adjusted_j`
- Warmup: fixed-count default (n=5), CV-based opt-in, thermal floor (60s)
- FLOPs estimation (PaLM formula, reference metadata, confidence field)
- `EnvironmentSnapshot` auto-capture (GPU, CUDA, pip freeze, tool version)
- Thermal throttle detection, GPU persistence mode warning
- Timeseries: 1 Hz sampling → `timeseries.parquet`
- Pre-flight validation (GPU, backend, dataset, VRAM estimate)

**Results**
- `ExperimentResult` schema: all v2.0 fields (config hash, methodology, baseline, warnings, environment)
- `EnergyBreakdown` nested model
- `WarmupResult` (6 fields)
- Output: `{name}_{timestamp}/result.json` + `timeseries.parquet`
- Collision handling (counter suffix, never overwrite)
- Late aggregation (per-process → ExperimentResult)
- `to_json()`, `to_parquet()`, `from_json()` persistence API

**CLI**
- `llem run [CONFIG] [OPTIONS]` — single experiment
- `llem config [--verbose]` — environment display
- `llem --version`
- `--dry-run`: L1 (Pydantic) + L2 (VRAM estimate)
- Plain text display (~200 LOC, tqdm progress)
- `--quiet`, `--verbose` verbosity levels
- `LLEMError` hierarchy, exit codes 0/1/2/130
- Entry point: `llem` (no `lem` shim)

**Infrastructure**
- `pyproject.toml` with extras (`[pytorch]`, `[zeus]`, `[codecarbon]`)
- State machine: 3 states + `failed: bool`
- `protocols.py` (5 DI protocols)
- Testing: `tests/unit/` + `tests/integration/`, protocol mocks
- Dead code removal (1,524 lines)
- Carry-forward: resilience/retry, subprocess lifecycle, path sanitisation

### Entry criteria
- All `.product/` decisions finalised (done)
- v1.x codebase available as reference

### Exit criteria
- `llem run --model gpt2 --backend pytorch` produces valid `ExperimentResult` JSON
- `llem run experiment.yaml` loads YAML, validates, runs, writes output
- `llem config` shows environment state
- `llem --version` prints `2.0.0`
- Unit tests pass without GPU
- Integration tests pass with GPU (PyTorch backend)

### Requirements
LA-01 through LA-10, CFG-01 through CFG-10, CFG-17 through CFG-26,
CM-01, CM-04 through CM-06, CM-11 through CM-34, RES-01 through RES-12,
RES-16 through RES-21, CLI-01 through CLI-04, CLI-06 through CLI-14,
STU-05, INF-01 through INF-12, INF-18 through INF-20

---

## M2 — Study / Sweep

> **Ships**: `llem run study.yaml` runs multi-experiment studies with sweep grammar,
> subprocess isolation, cycle ordering, thermal gaps, manifest checkpointing,
> and `--dry-run` grid preview.

### Scope

**Study runner**
- `StudyRunner` in `study/runner.py` — subprocess orchestration
- `multiprocessing.get_context("spawn")` for CUDA compatibility
- IPC via `Pipe` + file-based fallback
- `daemon=False`, timeout + SIGKILL on hung subprocess
- Config gap and cycle gap between experiments

**Sweep grammar**
- `study/grid.py`: `sweep:` → `list[ExperimentConfig]` expansion
- Dotted notation (`pytorch.batch_size: [1, 8]`) — backend-scoped grid
- Three modes: grid sweep, explicit `experiments:` list, combined
- Scoped key for non-present backend → `ValidationError`

**StudyConfig + ExecutionConfig**
- `StudyConfig` = `list[ExperimentConfig]` + `ExecutionConfig`
- `ExecutionConfig`: `n_cycles`, `cycle_order` (sequential/interleaved), gap seconds
- `study_design_hash` (execution block excluded)

**Results and manifest**
- `StudyResult` schema: summary, result_files, measurement_protocol
- `StudyManifest` checkpoint (written after each experiment)
- `ManifestWriter`: `mark_running()`, `mark_completed()`, `mark_failed()`

**CLI**
- `llem run study.yaml` — auto-detected by loader
- Study-mode flags: `--cycles`, `--no-gaps`, `--order`
- `--dry-run` grid preview: show all experiments, estimated VRAM, total runs
- Multi-experiment display with thermal gap countdown
- CSV export: `--export-csv`

### Entry criteria
- M1 complete (single experiment works end-to-end)

### Exit criteria
- `llem run study.yaml` with `sweep:` block produces correct experiment grid
- Subprocess isolation: each experiment runs in its own process
- `study_manifest.json` written incrementally
- `StudyResult` written at completion with correct `result_files` paths
- `--dry-run` shows grid preview with VRAM estimates
- Interleaved cycle ordering works correctly

### Requirements
LA-02, LA-05, CFG-11 through CFG-16, RES-13 through RES-15,
CLI-05, CLI-11, STU-01 through STU-09, CM-10

---

## M3 — Docker Multi-Backend

> **Ships**: vLLM and TensorRT-LLM backends work via Docker. Cross-backend studies
> (`backend: [pytorch, vllm]`) auto-detect Docker requirement and run.

### Scope

**Docker runner**
- Ephemeral `docker run` per experiment
- Config via env var injection, results via shared volume
- Container lifecycle = experiment lifecycle
- Auto-runner detection: multi-backend → Docker required
- TRT engine cache at `~/.llenergymeasure/trt-engines/{hash}/`

**Backends**
- vLLM inference backend (activated)
- TensorRT-LLM inference backend (activated)
- P0 fixes: vLLM streaming, Docker broken, vLLM shm-size

**Docker images**
- Official images: `llenergymeasure/{backend}:{version}-cuda{major}`
- Base images from upstream (pytorch, vllm, nvidia)
- GitHub Actions build + publish on release tag

**Pre-flight**
- Docker availability check
- Backend image availability check
- Multi-backend without Docker → hard error with guidance

### Entry criteria
- M2 complete (study runner works with subprocess isolation)
- Docker available on dev machine

### Exit criteria
- `llem run study.yaml` with `backend: [pytorch, vllm]` runs Docker containers
- vLLM and TRT backends produce valid `ExperimentResult`
- TRT engine cache prevents re-compilation
- Multi-backend without Docker → clear error message

### Requirements
CM-02, CM-03, CM-07 through CM-10, INF-13 through INF-17

---

## M4 — Advanced Features

> **Ships**: Traffic simulation, streaming latency instrumentation, study resume,
> webhook notifications, LoRA instrumentation improvements.

### Scope

**Traffic simulation**
- Poisson and constant-rate arrival patterns
- MLPerf-standard traffic generation
- `TrafficConfig` in `ExperimentConfig`

**Streaming latency**
- TTFT/ITL collection for streaming backends (vLLM native, PyTorch instrumented)
- `LatencyMeasurementMode` enum (true streaming vs simulated)

**Study resume**
- `--resume PATH` flag implementation
- Resume identity via `measurement_config_hash`
- Skip completed experiments, re-run failed

**Webhook notifications**
- `llenergymeasure[webhooks]` optional extra (`httpx` dependency)
- Useful error if `httpx` not installed
- Fire-and-forget on experiment completion

### Entry criteria
- M3 complete (all backends working)

### Exit criteria
- Traffic simulation produces repeatable load patterns
- TTFT/ITL metrics populated in ExperimentResult for streaming backends
- `--resume` skips completed experiments and reruns failed
- Webhook fires on experiment completion

### Requirements
STU-10, STU-11

---

## Beyond v2.0

| Version | Scope | Key Deliverable |
|---------|-------|-----------------|
| **v2.1** | Bootstrap CIs, attention FLOPs, Singularity runner | `confidence_intervals` field populated |
| **v3.0** | lm-eval integration | `quality_per_joule` metric |
| **v4.0** | Web platform | Static leaderboard → dynamic API → live features |

See [decisions/versioning-roadmap.md](decisions/versioning-roadmap.md) and
[decisions/future-versions.md](decisions/future-versions.md) for full detail.

---

## Dependency Graph

```
M1 ──→ M2 ──→ M3 ──→ M4
                       │
                       ▼
                   v2.0 release
```

M1 through M4 are strictly sequential — each builds on the previous.
The v2.0 release ships after M4 is complete.
