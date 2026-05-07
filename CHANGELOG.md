# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x pre-release series).
Minor version bumps (`0.x.0`) mark milestone completions. Breaking changes can occur between any 0.x release.

## [Unreleased]

## [v0.10.0] - TBD

Post-v0.9.0 work: engine-coupling restructure, engine-invariants pipeline, Docusaurus docs site, Z-engines layout, and CI hardening.

### Breaking Changes

- **`engine: pytorch` renamed to `engine: transformers`** throughout YAML, CLI, and Python API.
  The `pytorch` identifier has been renamed to `transformers` - the engine runs HuggingFace
  Transformers `.generate()`. PyTorch is the tensor substrate, not the engine, and renaming aligns
  with `pip install transformers` and the library that owns the inference API.

  Migrate with:
  ```bash
  sed -i 's/engine: pytorch/engine: transformers/g; s/^pytorch:/transformers:/g' your-study.yaml
  ```

  Affected: YAML engine value, YAML section key, `PyTorchConfig` class, `ENGINE_PYTORCH` constant,
  `[pytorch]` extra, `LLEM_RUNNER_PYTORCH`/`LLEM_IMAGE_PYTORCH` env vars, Docker image tags.
  Preserved (PyTorch the library - unchanged): `import torch`, `torch_dtype`, `pytorch/pytorch:*`
  base image, `PYTORCH_VERSION` build args, `torch_compile_backend` field. ([#261])

- **`backend:` field and `--backend` flag renamed to `engine:`** in YAML configs, CLI, and result
  JSON. Aligns terminology with how vLLM, TRT-LLM, and HuggingFace use "engine" natively.

  Migrate with:
  ```bash
  sed -i 's/^\(\s*\)backend:/\1engine:/g' your-study.yaml
  ```

  Affected: YAML field, CLI flag (`-b` becomes `-e`), result JSON fields `"backend"` and
  `"backend_version"`, Python symbols `BackendPlugin`, `BackendError`, `BACKEND_*` constants,
  `get_backend()`, `detect_default_backend()`. ([#260])

- **`tensorrt.tp_size` renamed to `tensorrt.tensor_parallel_size`** to match `TrtLlmArgs` native
  naming. `transformers.tp_size` is unchanged (follows the `accelerate` convention). ([#269])

- **Typed-field curation for engine configs.** Applies the maximalist rubric "type anything with a
  plausible energy/throughput/latency path" to each engine's Pydantic surface. Dropped fields
  remain settable via YAML (`extra="allow"` passthrough unless noted). ([#270])

  Transformers: drops `revision` (reproducibility metadata) and `trust_remote_code` (security
  toggle); adds `allow_tf32`, `autocast_enabled`, `autocast_dtype`, `low_cpu_mem_usage`.

  vLLM: drops `sampling.max_tokens` and `beam_search.max_tokens` (duplicates of
  `ExperimentConfig.max_output_tokens`); adds `num_scheduler_steps`, `max_seq_len_to_capture`,
  `distributed_executor_backend`; replaces flat speculative fields with nested `VLLMSpeculativeConfig`.

  TensorRT-LLM: drops `engine_path`, `TensorRTCalibConfig`, `TensorRTBuildCacheConfig`,
  `sampling.return_perf_metrics`, and `backend: Literal["trt"]`; adds `pipeline_parallel_size`
  and `max_num_tokens`.

- **Engines (vLLM, TensorRT-LLM) now run exclusively inside Docker.** Host extras `[vllm]` and
  `[tensorrt]` removed. Only `[transformers]` remains host-installable. ([#498])

- **`dtype:` and `decoder:` fields migrated into per-engine sub-configs.** Top-level
  `ExperimentConfig.dtype` and `ExperimentConfig.decoder` have moved to each engine's own
  configuration section. ([#290], [#291])

- **`--dtype` and `--batch-size` CLI flags removed.** Both fields are now set via YAML config only.
  ([#292])

- **`precision:` field renamed to `dtype:`** with standard value strings (e.g. `float16`, `bfloat16`
  instead of the prior enum). ([#196])

### Added

- `llem doctor` CLI command reports per-engine image status (OK / MISMATCH / UNVERIFIED /
  UNREACHABLE) and exits non-zero on mismatch for CI gating. ([#256])
- Host/container schema fingerprint verification: Docker images stamped at build time with a
  `llem.expconf.schema.fingerprint` OCI label. Mismatches abort with a rebuild hint. Bypassable
  via `LLEM_SKIP_IMAGE_CHECK=1`. ([#256])
- `SchemaLoader` class (`llenergymeasure.config.SchemaLoader`) reads vendored engine schemas via
  `importlib.resources` with per-instance caching and major-version envelope validation. ([#268])
- Engine parameter discovery script (`scripts/discover_engine_schemas.py`) introspects installed
  engine packages inside their Docker images. Supports `vllm`, `tensorrt`, `transformers`, and
  `--all`. ([#266])
- Vendored engine parameter schemas at `src/llenergymeasure/engines/{vllm,tensorrt,transformers}/`.
  Regenerate with `make discover-schema ENGINE=<engine>`. ([#266])
- Per-engine sub-package layout (`src/llenergymeasure/engines/<engine>/`) co-locating runtime data,
  schema JSON, and engine invariants YAML. ([#570])
- Per-engine SSOT for library version pins (`engine_versions/`) used by Renovate, Dockerfiles,
  and the invariant-mining pipeline. ([#477])
- Engine invariants mining pipeline: static and dynamic miners for all three engines extract
  validation rules as a reproducible corpus. ([#375], [#434], [#444])
- Vendor-replay CI gate validates corpus against live engine packages; TensorRT gate runs on
  self-hosted GPU runner. ([#414], [#440], [#447])
- `probe` primitive for binary miner reusability check. ([#482])
- `ConfigProbe` protocol and per-engine `probe_config()` implementations. ([#293])
- Configurable per-experiment timeout via `study_execution.experiment_timeout_seconds` (default
  600 s), replacing the previous `max(n_prompts * 2, 600)` heuristic. Both local and Docker paths
  honour the same field. ([#250])
- Disk-persisted baseline power cache with configurable strategy and TTL enforcement. ([#242], [#243])
- Per-study JSONL log capturing runtime warnings and container stderr. ([#395])
- `llem report-gaps` command proposes corpus rules from runtime observations. ([#397])
- Study robustness features: circuit breaker, resume-on-failure, GPU locks, container lifecycle
  management. ([#214])
- Live per-experiment progress display with Rich panels and sub-bullet heartbeats. ([#152], [#165])
- `.env`-based runtime config and configurable `device_map` default. ([#275])
- `trust_remote_code` opt-in via `LLEM_TRUST_REMOTE_CODE` env var. ([#274])
- TRT-LLM build cache configurable via `LLEM_TRT_BUILD_CACHE_{ENABLED,DIR}` env vars. ([#277])
- Tensor parallelism fields (`tp_plan`, `tp_size`) for the Transformers engine. ([#161])
- Cross-field operators in vendored-rules loader. ([#410])
- Docusaurus documentation site at `website/` serving user, methodology, API, and architecture
  docs. ([#566])
- Per-engine discovered-schema Markdown digest rendered to `docs/`. ([#560])
- Architecture documentation suite in `docs/architecture/`. ([#433])
- Per-engine engine-invariants and engine-schemas CI workflows with cross-pipeline coordination
  (consolidated from predecessor mine + vendor + parameter-discovery workflows). ([#484], [#486])
- Engine-pipeline orchestrator (`engine-pipeline.yml`) as single reusable workflow entry point.
  ([#514], [#573])
- Cloudflare Pages PR preview deploy workflow. ([#575])
- SSOT audit trail and GHCR image retention policies. ([#546])

### Changed

- Re-typed `tensorrt.backend` as `Literal["trt", "pytorch", "_autodeploy"] | None` (reverses a
  prior incorrect curation-pass drop; `None` lets TRT-LLM auto-pick the runtime path). ([#276])
- Engine-invariants pipeline consolidated from separate mine + vendor + parameter-discovery
  workflows into a single orchestrated flow with sequential downstream pipelines. ([#484], [#573])
- `study_execution` field names updated (execution fields renamed, `reverse`/`latin_square`
  ordering modes added). ([#190])
- Dataset restructured into nested `DatasetConfig` sub-model. ([#195])
- `OutputConfig` extracted from `ExperimentConfig` as a separate sub-model. ([#203])
- `EnergyConfig` flattened to `energy_sampler` + `gpu_telemetry` fields. ([#201])
- `study_name` field replaces generic `name` field in study configs. ([#182])
- `n_prompts` default reduced to 50; `max_output_tokens` default bumped to 256. ([#175], [#213])
- Renovate customManager retargeted from Dockerfile ARGs to `engine_versions/` SSOT. ([#481])
- First-party `Dockerfile.vllm` and `Dockerfile.tensorrt` replaced with upstream-direct images
  plus volume mounts. ([#509])

### Fixed

- `ImportError: cuKernelGetName` when importing `tensorrt_llm`: LD_LIBRARY_PATH ordering placed
  the bundled compat CUDA 12.2 library ahead of the host-driver mount. Fixed by prepending
  `/usr/local/cuda/compat/lib` so the host-driver mount takes precedence. ([#264])
- Miner `added_at` timestamp lost on re-mine; f-string `message_template` fields now rendered
  correctly. ([#523])
- `Dockerfile.transformers` stale references to the old `[pytorch]` extra and header comments
  corrected. ([#265])
- Config hash mismatch in Docker study runs resolved. ([#176])
- Non-matching engine sections stripped correctly during multi-engine grid expansion. ([#171])
- Docker auto-elevation enforced for multi-engine studies. ([#172])
- Baseline cache path resolved before Docker bind-mount. ([#248])

### Removed

- Internal helper `llenergymeasure.study.runner._calculate_timeout` (replaced by direct config
  reads). ([#529])
- First-party `Dockerfile.vllm` and `Dockerfile.tensorrt` engine images. ([#509])
- Predecessor CI workflows: `auto-mine.yml`, `vendor-tensorrt.yml`, `vendor-vllm.yml`,
  `parameter-discovery.yml`, and predecessors. ([#483], [#485])


## [v0.9.0] - 2026-03-20

Docker infrastructure, vLLM engine, TensorRT-LLM engine, package restructure, test hardening, and CI.

### Added

- NVML GPU memory residual check before experiment dispatch (threshold 1 GB), preventing
  stale-process contamination. ([#24], [#26])
- Docker runner infrastructure: container lifecycle management, volume mounts, GPU index
  resolution. ([#27], [#124])
- Docker pre-flight environment checks. ([#28])
- TensorRT-LLM Docker image rewrite with CUDA 12.6.2 upgrade. ([#114])
- `TensorRTConfig` expanded to full TRT-LLM parameter schema. ([#115])
- `mpirun` injection for TensorRT-LLM tensor parallelism. ([#116])
- `BackendPlugin.validate_config` protocol method. ([#121])
- `TensorRTBackend` implementation registered in `get_backend()`. ([#122])
- `TensorRTConfig.engine_path` for pre-compiled engine loading. ([#143])
- 9-layer import-linter architecture enforcement in CI. ([#135], [#144])

### Changed

- Package restructured with file moves, import rewrites, and layer boundary fixes. ([#133], [#134])
- Prompt loading moved outside the NVML measurement window. ([#145])
- Shared backend helpers extracted; dead warmup code removed. ([#140])
- Test suite restructured; `importorskip` guards added for optional dependencies. ([#137], [#138])

### Fixed

- `accelerate` restored as a `[pytorch]` optional dependency (accidentally dropped). ([#132])
- Runner mode auto-detection (local vs Docker) on startup. ([#146])
- Silent `NVMLError`, payload detection, and empty `gpu_indices` guard. ([#141])

### Removed

- Dead code, stale type annotations, and unused dependencies. ([#130])


## [v0.8.0] - 2026-02-27

Multi-experiment study sweeps.

### Added

- `run_study()` public API for multi-experiment studies. ([#23])
- `StudyConfig` with sweep grammar (grid and cycle ordering). ([#23])
- YAML-driven parameter sweeps across models, engines, and precisions. ([#23])
- `StudyRunner` with sequential experiment dispatch. ([#23])
- Study-level aggregation and result collection. ([#23])
- Manifest-based progress tracking with resume support. ([#23])


## [v0.7.0] - 2026-02-27

First end-to-end single-experiment release.

### Added

- `run_experiment()` public API. ([#22])
- `ExperimentConfig` to `ExperimentResult` pipeline. ([#22])
- Energy measurement via CodeCarbon and Zeus backends. ([#22])
- Extended metrics: TPOT, TEI, memory efficiency. ([#22])
- Streaming latency measurement (TTFT / ITL). ([#22])
- Results persistence in Parquet format. ([#22])


---

## Historical (pre-0.x)

> The entries below predate the current 0.x versioning scheme introduced in early 2026.
> They describe the research prototype and early CLI rewrites that were restructured and
> re-versioned starting from v0.1.0. Version numbers v1.x and v2.0.0 referenced here are
> legacy labels from that era; they do not correspond to any published release under the
> current scheme. The 2026-03-04 history reset remapped these to sequential 0.x tags
> (v0.1.0-v0.6.0) for consistency with the current versioning scheme.

### v0.6.0 (2025-12-29) - formerly v1.16.0

Production-ready containerisation with full GPU support and streamlined developer experience.

#### Added

- Multi-stage Dockerfile with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base image (builder,
  runtime, and dev stages).
- Docker Compose profiles separating production and development workflows (`lem-app`, `lem-dev`).
- VS Code devcontainer configuration with GPU passthrough and Ruff/Pylance extensions.
- Makefile targets for common Docker operations (`make docker-build`, `make experiment`,
  `make datasets`).

#### Changed

- CI workflow reliability improved with concurrency groups preventing parallel releases.
- Dev container runs as root, eliminating permission complexity with virtual environments.

#### Fixed

- Docker CUDA 12.4 base image aligned with host driver requirements.
- Volume permission errors resolved by running dev containers as root.
- Deprecated `torch_dtype` parameter replaced with `dtype` in model loading.
- Removed obsolete `TRANSFORMERS_CACHE` environment variable (superseded by `HF_HOME`).
- CodeCarbon pandas `FutureWarning` suppressed.
- `nvidia-smi` GPU utilisation parsing handles `[N/A]` values gracefully.

---

### v0.5.0 (2025-12-21) - formerly v1.15.0

Comprehensive test coverage ensuring reliability across all components.

#### Added

- End-to-end CLI tests (8 tests) validating complete benchmark workflows.
- Integration tests (47 tests) covering non-GPU workflows.
- Methodology documentation (`docs/methodology.md`) explaining measurement approach.

#### Changed

- Total test count: 416 passing tests (unit + integration + e2e).
- All tests run without GPU access using mocked/simulated data.

#### Removed

- `requirements.txt` (306 frozen packages) - all dependencies now managed via Poetry lockfile.

---

### v0.4.0 (2025-12-21) - formerly v1.13.0

User-friendly command-line interface replacing legacy entry points.

#### Added

- Typer-based CLI (`lem`) with subcommands: `experiment`, `aggregate`, `config validate`,
  `config show`, `results list`, `results show`, `datasets`.
- `ExperimentOrchestrator` with protocol-based dependency injection.
- `ExperimentContext` dataclass for runtime state management.
- Accelerate launcher with configurable retry logic.
- 25 CLI tests and 27 orchestration unit tests.

#### Removed

- Legacy `MAIN_*.py` entry points (6 files).

---

### v0.3.0 (2025-12-20) - formerly v1.10.0

Major architectural refactor establishing clean module boundaries.

#### Breaking Changes

- Package renamed: `llm-bench` to `lem`. All imports now use `llenergymeasure`.

#### Added

- Energy backend plugin registry with automatic CodeCarbon registration.
- `FlopsEstimator` with three-strategy fallback chain (calflops, architecture, parameter
  estimate), each returning a confidence level.
- Results aggregation with temporal overlap detection and GPU attribution verification.
- Export functionality for CSV and JSON formats.
- 296 unit tests covering all new modules.

#### Changed

- Replaced `print()` statements with Loguru structured logging.

---

### v0.2.0 (2025-05-17) - formerly v1.0.0

Research phase complete - stable multi-model benchmarking validated on production hardware.

#### Added

- Multi-model experiment support with scenario-based configuration.
- Experiment suite CSV export with consistent naming conventions.
- Failed experiment detection with cycle tracking and automatic retry.
- Minimum output token enforcement for comparable generation lengths.
- Large model stability improvements (gradient checkpointing, CUDA cache clearing).
- Data wrangling pipelines for experiment result analysis (Pandas-based).
- Plotting functionality for efficiency metrics visualisation.
- FLOPs caching preventing redundant calculations.

---

### v0.1.0 (2025-03-22) - formerly v0.5.0

Core measurement functionality establishing the foundation for all subsequent development.

#### Added

- Distributed results aggregation across multiple GPUs with per-process JSON files.
- FLOPs calculation with quantisation awareness and `calflops` integration.
- Robust process cleanup with signal handlers and distributed barrier synchronisation.
- Optimum benchmark integration for standardised measurements.

#### Changed

- Distributed execution stability improved: proper NCCL initialisation and teardown.
- Major directory restructuring separating config, core, and result handling.


[Unreleased]: https://github.com/henrycgbaker/llenergymeasure/compare/v0.9.0...HEAD
[v0.10.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.10.0
[v0.9.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.9.0
[v0.8.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.8.0
[v0.7.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.7.0

[#22]: https://github.com/henrycgbaker/llenergymeasure/pull/22
[#23]: https://github.com/henrycgbaker/llenergymeasure/pull/23
[#24]: https://github.com/henrycgbaker/llenergymeasure/pull/24
[#26]: https://github.com/henrycgbaker/llenergymeasure/pull/26
[#27]: https://github.com/henrycgbaker/llenergymeasure/pull/27
[#28]: https://github.com/henrycgbaker/llenergymeasure/pull/28
[#113]: https://github.com/henrycgbaker/llenergymeasure/pull/113
[#114]: https://github.com/henrycgbaker/llenergymeasure/pull/114
[#115]: https://github.com/henrycgbaker/llenergymeasure/pull/115
[#116]: https://github.com/henrycgbaker/llenergymeasure/pull/116
[#121]: https://github.com/henrycgbaker/llenergymeasure/pull/121
[#122]: https://github.com/henrycgbaker/llenergymeasure/pull/122
[#124]: https://github.com/henrycgbaker/llenergymeasure/pull/124
[#130]: https://github.com/henrycgbaker/llenergymeasure/pull/130
[#132]: https://github.com/henrycgbaker/llenergymeasure/pull/132
[#133]: https://github.com/henrycgbaker/llenergymeasure/pull/133
[#134]: https://github.com/henrycgbaker/llenergymeasure/pull/134
[#135]: https://github.com/henrycgbaker/llenergymeasure/pull/135
[#137]: https://github.com/henrycgbaker/llenergymeasure/pull/137
[#138]: https://github.com/henrycgbaker/llenergymeasure/pull/138
[#140]: https://github.com/henrycgbaker/llenergymeasure/pull/140
[#141]: https://github.com/henrycgbaker/llenergymeasure/pull/141
[#143]: https://github.com/henrycgbaker/llenergymeasure/pull/143
[#144]: https://github.com/henrycgbaker/llenergymeasure/pull/144
[#145]: https://github.com/henrycgbaker/llenergymeasure/pull/145
[#146]: https://github.com/henrycgbaker/llenergymeasure/pull/146
[#147]: https://github.com/henrycgbaker/llenergymeasure/pull/147
[#152]: https://github.com/henrycgbaker/llenergymeasure/pull/152
[#161]: https://github.com/henrycgbaker/llenergymeasure/pull/161
[#165]: https://github.com/henrycgbaker/llenergymeasure/pull/165
[#171]: https://github.com/henrycgbaker/llenergymeasure/pull/171
[#172]: https://github.com/henrycgbaker/llenergymeasure/pull/172
[#175]: https://github.com/henrycgbaker/llenergymeasure/pull/175
[#176]: https://github.com/henrycgbaker/llenergymeasure/pull/176
[#182]: https://github.com/henrycgbaker/llenergymeasure/pull/182
[#190]: https://github.com/henrycgbaker/llenergymeasure/pull/190
[#195]: https://github.com/henrycgbaker/llenergymeasure/pull/195
[#196]: https://github.com/henrycgbaker/llenergymeasure/pull/196
[#201]: https://github.com/henrycgbaker/llenergymeasure/pull/201
[#203]: https://github.com/henrycgbaker/llenergymeasure/pull/203
[#213]: https://github.com/henrycgbaker/llenergymeasure/pull/213
[#214]: https://github.com/henrycgbaker/llenergymeasure/pull/214
[#242]: https://github.com/henrycgbaker/llenergymeasure/pull/242
[#243]: https://github.com/henrycgbaker/llenergymeasure/pull/243
[#248]: https://github.com/henrycgbaker/llenergymeasure/pull/248
[#250]: https://github.com/henrycgbaker/llenergymeasure/pull/250
[#256]: https://github.com/henrycgbaker/llenergymeasure/pull/256
[#260]: https://github.com/henrycgbaker/llenergymeasure/pull/260
[#261]: https://github.com/henrycgbaker/llenergymeasure/pull/261
[#264]: https://github.com/henrycgbaker/llenergymeasure/pull/264
[#265]: https://github.com/henrycgbaker/llenergymeasure/pull/265
[#266]: https://github.com/henrycgbaker/llenergymeasure/pull/266
[#268]: https://github.com/henrycgbaker/llenergymeasure/pull/268
[#269]: https://github.com/henrycgbaker/llenergymeasure/pull/269
[#270]: https://github.com/henrycgbaker/llenergymeasure/pull/270
[#274]: https://github.com/henrycgbaker/llenergymeasure/pull/274
[#275]: https://github.com/henrycgbaker/llenergymeasure/pull/275
[#276]: https://github.com/henrycgbaker/llenergymeasure/pull/276
[#277]: https://github.com/henrycgbaker/llenergymeasure/pull/277
[#290]: https://github.com/henrycgbaker/llenergymeasure/pull/290
[#291]: https://github.com/henrycgbaker/llenergymeasure/pull/291
[#292]: https://github.com/henrycgbaker/llenergymeasure/pull/292
[#293]: https://github.com/henrycgbaker/llenergymeasure/pull/293
[#375]: https://github.com/henrycgbaker/llenergymeasure/pull/375
[#395]: https://github.com/henrycgbaker/llenergymeasure/pull/395
[#397]: https://github.com/henrycgbaker/llenergymeasure/pull/397
[#410]: https://github.com/henrycgbaker/llenergymeasure/pull/410
[#414]: https://github.com/henrycgbaker/llenergymeasure/pull/414
[#433]: https://github.com/henrycgbaker/llenergymeasure/pull/433
[#434]: https://github.com/henrycgbaker/llenergymeasure/pull/434
[#440]: https://github.com/henrycgbaker/llenergymeasure/pull/440
[#444]: https://github.com/henrycgbaker/llenergymeasure/pull/444
[#447]: https://github.com/henrycgbaker/llenergymeasure/pull/447
[#477]: https://github.com/henrycgbaker/llenergymeasure/pull/477
[#481]: https://github.com/henrycgbaker/llenergymeasure/pull/481
[#482]: https://github.com/henrycgbaker/llenergymeasure/pull/482
[#483]: https://github.com/henrycgbaker/llenergymeasure/pull/483
[#484]: https://github.com/henrycgbaker/llenergymeasure/pull/484
[#485]: https://github.com/henrycgbaker/llenergymeasure/pull/485
[#486]: https://github.com/henrycgbaker/llenergymeasure/pull/486
[#498]: https://github.com/henrycgbaker/llenergymeasure/pull/498
[#509]: https://github.com/henrycgbaker/llenergymeasure/pull/509
[#514]: https://github.com/henrycgbaker/llenergymeasure/pull/514
[#523]: https://github.com/henrycgbaker/llenergymeasure/pull/523
[#529]: https://github.com/henrycgbaker/llenergymeasure/pull/529
[#546]: https://github.com/henrycgbaker/llenergymeasure/pull/546
[#560]: https://github.com/henrycgbaker/llenergymeasure/pull/560
[#566]: https://github.com/henrycgbaker/llenergymeasure/pull/566
[#570]: https://github.com/henrycgbaker/llenergymeasure/pull/570
[#573]: https://github.com/henrycgbaker/llenergymeasure/pull/573
[#575]: https://github.com/henrycgbaker/llenergymeasure/pull/575
