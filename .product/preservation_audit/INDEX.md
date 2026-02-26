# Preservation Audit Index

> Generated 2026-02-19. Systematic cross-comparison of all source modules against all
> `decisions/` and `designs/` documents in Phase 4.5. **47 items total** — 23 from the
> original `PRESERVATION_AUDIT.md` (now expanded with exact code references and deeper
> analysis) and 24 newly discovered by parallel agents reading every source file.
>
> Each item needs a deliberate decision: **Keep — v2.0**, **Keep — Defer vX.X**,
> **Cut** (record why), or **Pending** (human decision needed). "No decision" = silent
> loss during Phase 5.

---

## ⚠ Critical Conflicts (Read These First)

These are not just gaps — they are **direct contradictions** between the code and the plans.
A Phase 5 implementor working only from the design docs would build something wrong.

| # | Conflict | Where |
|---|---------|-------|
| **1** | Planning doc says "CSV not in v2.0"; code has a complete CSV exporter and **no Parquet exporter exists at all** | [N-R03](results/N-R03-csv-export-format.md) |
| **2** | `StudyManifest` is fully designed in `study-resume.md` but has **zero implementation** anywhere | [N-R04](results/N-R04-study-manifest-planned-not-implemented.md) |
| **3** | `UserConfig` uses a **project-local `.lem-config.yaml`**; design specifies a **user-global XDG path** — schemas are entirely different | [N-C06](config/N-C06-user-config-divergence.md) |
| **4** | `designs/reproducibility.md` references `warmup_result.warmup_duration_sec` — **this field does not exist** in `WarmupResult`; would cause a runtime error | [N-C03](config/N-C03-warmup-convergence-result.md) |
| **5** | `EnergyBreakdown` already implements what result-schema defers to v2.1 — the nested model should be moved to v2.0 | [N-C04](config/N-C04-energy-breakdown-model.md) |
| **6** | `FileSystemRepository` uses `raw/` + `aggregated/` directory split; planning doc's flat filename convention is incompatible | [P-01](results/P-01-filesystem-repository.md) |
| **7** | Backend install hints in `backend_detection.py` recommend Docker for vLLM/TensorRT; planning docs say `pip install llenergymeasure[vllm]` | [N-X14](cross-cutting/N-X14-backend-detection.md) |

---

## Risk Summary

| Risk Level | Count | Items |
|-----------|-------|-------|
| **HIGH** | 9 | P-01, P-02, P-03, P-04, P-07, N-R04, N-X04, N-X09, N-X10 |
| **MEDIUM** | 27 | P-05–P-16 (excl. P-06 pending), N-C01–N-C06, N-R01–N-R03, N-X01, N-X03, N-X05–N-X08, N-X11, N-X13 |
| **LOW** | 11 | P-17–P-23, N-X02, N-X12, N-X14 |

**Pending (human decision needed)**: ~~All resolved (2026-02-26)~~

---

## Config & Domain Subsystem (`config/`)

| ID | Feature | Module | Risk | Decision |
|----|---------|--------|------|----------|
| [P-03](config/P-03-config-introspection-ssot.md) | Config Introspection SSOT | `config/introspection.py` | HIGH | Keep — v2.0. Lean on Pydantic v2 `model_json_schema()` (~80% of custom reflection). Preserve domain layers: test value generation, constraint metadata, backend routing. ~300→~250 LOC. Multi-consumer SSOT (tests + docs + CLI) validated by FastAPI pattern. (Revised 2026-02-26) |
| [P-04](config/P-04-unified-quantization.md) | Unified Quantization Abstraction | `config/quantization.py` | HIGH | Cut. Backend-specific params in backend sections (consistent with composition architecture). No translation layer. |
| [P-07](config/P-07-parameter-provenance.md) | Parameter Provenance Tracking | `config/provenance.py` | HIGH | Keep — v2.0 |
| [P-10](config/P-10-unified-speculative-decoding.md) | Unified Speculative Decoding Config | `config/speculative.py` | MEDIUM | Cut. Same policy as P-04 — backend-specific params in backend sections. |
| [P-11](config/P-11-rich-dataset-config.md) | Rich Dataset / Prompt Config | `config/models.py` | MEDIUM | Keep — v2.0 |
| [P-14](config/P-14-warmup-config.md) | Warmup Iteration Config (`WarmupConfig`) | `config/models.py` | MEDIUM | Keep — v2.0. Fixed-count default (`n_warmup: 5`, reduced-output), CV-based opt-in (`convergence_detection: true`). 0/6 peers use CV; ship simple path, keep clever path available. (Revised 2026-02-26) |
| [P-16](config/P-16-naming-alias-registry.md) | Naming / Alias Registry | `config/naming.py` | MEDIUM | Cut. Clean break at v2.0 — no aliases, no deprecation shims. Document renames in migration guide. |
| [N-C01](config/N-C01-environment-metadata.md) | Environment Metadata Snapshot | `domain/environment.py` | MEDIUM | Keep — v2.0 (richer than design doc — **sync design**) |
| [N-C02](config/N-C02-thermal-throttle-detection.md) | Thermal Throttle Detection | `domain/metrics.py` | MEDIUM | Keep — v2.0 |
| [N-C03](config/N-C03-warmup-convergence-result.md) | Warmup Convergence Result | `domain/metrics.py` | LOW | Keep — v2.0 (**design has wrong field name — fix it**) |
| [N-C04](config/N-C04-energy-breakdown-model.md) | Energy Breakdown Model | `domain/metrics.py` | MEDIUM | Keep — v2.0 (**move from deferred v2.1 to v2.0**) |
| [N-C05](config/N-C05-compute-metrics-flops-confidence.md) | Compute Metrics / FLOPs Confidence | `domain/metrics.py` | MEDIUM | Keep — v2.0 (two competing models exist — consolidate) |
| [N-C06](config/N-C06-user-config-divergence.md) | User Config Implementation Divergence | `config/user_config.py` | MEDIUM | Design wins. XDG global `~/.config/llenergymeasure/config.yaml`. Full rewrite of v1.x schema. |

---

## Results Subsystem (`results/`)

| ID | Feature | Module | Risk | Decision |
|----|---------|--------|------|----------|
| [P-01](results/P-01-filesystem-repository.md) | Filesystem Repository / Persistence Layer | `results/repository.py` | HIGH | Keep — v2.0. Unified output: all backends → one `ExperimentResult` → `{name}_{timestamp}/result.json`. PyTorch multi-GPU aggregation is internal to backend, not user-visible. `raw/` + `aggregated/` split → internal `.state/` only. (Revised 2026-02-26) |
| [P-02](results/P-02-study-aggregation-grouping.md) | Study-Level Aggregation with Grouping | `results/aggregation.py` | HIGH | **Cut.** `StudyResult` = manifest + list of `ExperimentResult` paths. No built-in cross-config aggregation — downstream notebook concern. Measurement tool, not statistics tool. (Revised 2026-02-26) |
| [P-05](results/P-05-precision-normalised-metrics.md) | Precision Metadata & Normalised Metrics | `domain/metrics.py` | MEDIUM | **Cut** `NormalisedMetrics` + `precision_factor`. Keep `PrecisionMetadata` as descriptive context only. Normalising hides the efficiency differences the tool exists to measure. (Revised 2026-02-26) |
| [P-06](results/P-06-power-thermal-timeseries-export.md) | Power/Thermal Timeseries Export | `results/timeseries.py` | MEDIUM | Keep — v2.0. Parquet sidecar (`timeseries.parquet`), CSV opt-in via `--export-csv`. Peer research: 0/8 peers use Parquet but data shape demands it; MLPerf uses CSV text, others JSON aggregates only. |
| [P-09](results/P-09-process-completeness-validation.md) | Process Completeness Validation | `results/aggregation.py` | MEDIUM | Keep — v2.0 |
| [P-12](results/P-12-extended-efficiency-metrics.md) | Extended Efficiency Metrics (6 sub-models) | `domain/metrics.py` | MEDIUM | Keep — v2.0 |
| [P-17](results/P-17-bootstrap-ci.md) | Bootstrap CI Infrastructure | `results/bootstrap.py` | LOW | Keep — Defer v2.1 (`BootstrapResult` conflicts with planned `MetricCI` — resolve at v2.1) |
| [P-18](results/P-18-latency-trimmed-itl.md) | Latency — Trimmed vs Full ITL | `results/aggregation.py` | LOW | Keep — v2.0 |
| [P-20](results/P-20-temporal-overlap-attribution.md) | Temporal Overlap & GPU Attribution Verification | `results/aggregation.py` | LOW | Keep — v2.0 |
| [P-22](results/P-22-config-warnings-results.md) | Config Warnings Embedded in Results | `results/aggregation.py` | LOW | Keep — v2.0 |
| [N-R01](results/N-R01-extended-metrics-aggregation.md) | Extended Metrics Late Aggregation Pattern | `results/aggregation.py` | MEDIUM | Keep — v2.0 (`precision_factor` placeholder bug must be fixed) |
| [N-R02](results/N-R02-energy-thermal-aggregation-logic.md) | Energy & Thermal Throttle Aggregation Logic | `results/aggregation.py` | MEDIUM | Keep — v2.0 (note: shared-baseline assumption breaks at v2.2 Docker multi-host) |
| [N-R03](results/N-R03-csv-export-format.md) | CSV Export Format | `results/exporters.py` | MEDIUM | Keep — v2.0 (**⚠ direct conflict** — planning rejects CSV, but it's the only exporter that exists) |
| [N-R04](results/N-R04-study-manifest-planned-not-implemented.md) | StudyManifest — Designed But Not Implemented | *(planned: study/manifest.py)* | HIGH | Keep — v2.0 (**build it** — design is complete in `study-resume.md`) |

---

## Core Measurement Subsystem (`core/`)

| ID | Feature | Module | Risk | Decision |
|----|---------|--------|------|----------|
| [P-08](core/P-08-traffic-simulation.md) | Traffic Simulation (Poisson Arrival) | `core/traffic.py` | MEDIUM | Keep — v2.0 (ExperimentConfig as-is; Poisson/constant-rate traffic sim, MLPerf standard. Later milestone within v2.0.) |
| [P-19](core/P-19-gpu-utilisation-sampling.md) | GPU Utilisation Sampling | `core/gpu_utilisation.py` | MEDIUM | Keep — v2.0 (concurrent NVML access with energy backends needs validation) |
| [P-21](core/P-21-mig-detection.md) | MIG Instance Detection & Warning | `results/aggregation.py` | LOW | Keep — v2.0 |
| [P-23](core/P-23-streaming-latency-ttft-itl.md) | Streaming Latency (TTFT / ITL) Collection | `domain/metrics.py` | LOW | Keep — v2.0 (`LatencyMeasurementMode` enum clarifies which backends produce true streaming) |
| [N-X01](core/N-X01-baseline-power-measurement.md) | Baseline Power Measurement & Cache | `core/baseline.py` | MEDIUM | Keep — v2.0 |
| [N-X02](core/N-X02-inference-metrics-computation.md) | Inference Metrics Computation | `core/inference.py` | LOW | Keep — v2.0 (note: `latency_per_token_ms` here ≠ measured ITL — different metrics) |

---

## CLI & UX (`cli/`)

| ID | Feature | Module | Risk | Decision |
|----|---------|--------|------|----------|
| [P-13](cli/P-13-webhook-notifications.md) | Webhook Notifications | `notifications/webhook.py` | MEDIUM | Keep — v2.0 as optional extra. `llenergymeasure[webhooks]` with `httpx` dependency. Useful error if `httpx` not installed. Not core download. |
| [P-15](cli/P-15-single-experiment-resume.md) | Single-Experiment Resume (`--resume`) | `cli/resume.py` | MEDIUM | Keep state machine (refactor 6→3 states during impl), cut CLI `--resume` command. Study resume handles recovery. |
| [N-X03](cli/N-X03-progress-reporting-architecture.md) | Progress Reporting Architecture | `progress.py` | MEDIUM | Keep — v2.0 (tqdm `ProgressTracker` cannot run inside Rich `Live` context — note ordering) |
| [N-X04](cli/N-X04-display-architecture.md) | Display Architecture | `cli/display/` | HIGH | **Simplify drastically.** 950 LOC → ~200 LOC. Plain output (vLLM pattern: key:value + ASCII separators). tqdm for progress. No Rich dependency required. 0/5 peers use Rich for results display. (Revised 2026-02-26) |
| [N-X05](cli/N-X05-subprocess-lifecycle-management.md) | Subprocess Lifecycle Management | `cli/lifecycle.py` | MEDIUM | Keep — v2.0 (vLLM env var `VLLM_ENABLE_V1_MULTIPROCESSING=0` must migrate to new exec model) |
| [N-X06](cli/N-X06-batch-processing-command.md) | Batch Processing Command | `cli/batch.py` | MEDIUM | Cut command, keep validation pattern. `llem run study.yaml` with `experiments:` list covers the use case. Peer research: 0/8 peers have a `batch` subcommand; 0/8 accept multiple config file paths. Upfront validation reused in `--dry-run`. |
| [N-X07](cli/N-X07-preset-registry-constants.md) | Built-In Preset Registry (10 presets) | `constants.py` | MEDIUM | **Drop `PRESETS` registry.** Replace with opinionated Pydantic defaults on `ExperimentConfig`. 0/8 peers use named preset registries. Domain knowledge (vLLM tuning, etc.) → backend section defaults + example YAMLs in `configs/examples/`. Keep all operational constants (timeouts, schema version, warmup counts). (Revised 2026-02-26) |
| [N-X08](cli/N-X08-security-path-sanitization.md) | Path Sanitisation & Security | *(security module)* | LOW | Keep — v2.0 |

---

## Cross-Cutting (`cross-cutting/`)

| ID | Feature | Module | Risk | Decision |
|----|---------|--------|------|----------|
| [N-X09](cross-cutting/N-X09-protocols-di-interfaces.md) | Protocols & DI Interfaces (5 protocols) | `protocols.py` | HIGH | Keep — v2.0 |
| [N-X10](cross-cutting/N-X10-experiment-state-machine.md) | Experiment State Machine (6 states) | *(state module)* | HIGH | Keep — v2.0 (planning targets 3 states — expand or reconcile) |
| [N-X11](cross-cutting/N-X11-resilience-retry-logic.md) | Resilience & Retry Logic | `resilience.py` | LOW | Keep — v2.0 |
| [N-X12](cross-cutting/N-X12-domain-model-flopsresult.md) | FlopsResult + PrecisionMetadata domain models | `domain/metrics.py` | LOW | Keep — v2.0 |
| [N-X13](cross-cutting/N-X13-config-loader-yaml-resolution.md) | Config Loader / YAML Resolution Chain | `config/loader.py` | MEDIUM | Keep — v2.0 (`_extends` inheritance with cycle detection not mentioned anywhere in plans) |
| [N-X14](cross-cutting/N-X14-backend-detection.md) | Backend Auto-Detection Strategy | `config/backend_detection.py` | LOW | Keep — v2.0 (**⚠ Docker hint vs pip install conflict with planning docs**) |

---

## Decisions Required Before Phase 5 Coding — ALL RESOLVED (2026-02-26)

| # | Item | Decision | Date |
|---|------|----------|------|
| 1 | P-04 + P-10 | **Cut both.** Backend-specific params in backend sections (consistent with composition architecture). No translation layer. | 2026-02-26 |
| 2 | N-C06 | **Design wins.** XDG global `~/.config/llenergymeasure/config.yaml`. Full rewrite of v1.x schema. | 2026-02-26 |
| 3 | N-R03 | **Keep both CSV + Parquet.** CSV for accessibility (Excel, R, `cat`). Parquet for study-scale analysis. | 2026-02-26 |
| 4 | P-06 | **Keep — v2.0.** Time-series decided in session decisions #47/#55/#57. Sampling logic carries forward; export format changes to Parquet sidecar. | 2026-02-26 |
| 5 | P-08 | **Keep — v2.0.** Poisson/constant-rate traffic simulation. MLPerf-standard. Later milestone within v2.0. | 2026-02-26 |
| 6 | P-13 | **Keep — v2.0 as optional extra.** `llenergymeasure[webhooks]` with `httpx` dependency. Useful error if `httpx` not installed. Not core download. | 2026-02-26 |
| 7 | P-15 | **Keep state machine (refactor 6→3 states), cut CLI `--resume` command.** Study resume handles recovery. | 2026-02-26 |
| 8 | P-16 | **Cut.** Clean break at v2.0 — no aliases, no deprecation shims. Document renames in migration guide. | 2026-02-26 |
| 9 | N-X06 | **Cut command, keep validation pattern.** `llem run study.yaml` with `experiments:` list covers the use case. Upfront validation reused in `--dry-run`. | 2026-02-26 |
| 10 | P-05 | **Cut `NormalisedMetrics` + `precision_factor`.** Keep `PrecisionMetadata` as descriptive context only. Quantisation efficiency is what the tool measures — normalising hides it. | 2026-02-26 |
| 11 | P-02 | **Cut study-level aggregation.** `StudyResult` = manifest + list of results. No built-in cross-config stats. Downstream notebook concern. | 2026-02-26 |
| 12 | P-01 | **Unified output layout.** All backends → one `ExperimentResult` → `{name}_{timestamp}/result.json`. PyTorch multi-GPU raw files are internal (`.state/`), not user-visible. | 2026-02-26 |
| 13 | P-14 | **Fixed-count default, CV opt-in.** `n_warmup: 5` reduced-output default. `convergence_detection: true` for CV-based. 0/6 peers use CV. | 2026-02-26 |
| 14 | N-X04 | **Simplify display to ~200 LOC.** Plain output (vLLM pattern), tqdm for progress. No Rich dependency. 0/5 peers use Rich. | 2026-02-26 |
| 15 | N-X10 | **3-state confirmed.** INITIALISING, MEASURING, DONE + `failed: bool`. Peer-validated. Keep `StateManager`, `find_by_config_hash()`, `cleanup_stale()`, atomic writes. Drop `ProcessProgress`. | 2026-02-26 |
| 16 | P-17 | **Defer v2.1 confirmed.** Raw measurement is primary; CIs are downstream analysis. Infrastructure works, wire up at v2.1. | 2026-02-26 |
| 17 | N-C05 | **Keep both `method` and `confidence` on `FlopsResult`.** Consolidate `ComputeMetrics` → nested `FlopsResult`. | 2026-02-26 |
| 18 | N-X07 | **Drop `PRESETS` registry.** Opinionated Pydantic defaults replace named presets (0/8 peers use registries). Domain knowledge → backend defaults + example YAMLs. Keep operational constants. | 2026-02-26 |
| 19 | P-03 | **Keep introspection, lean on Pydantic v2.** `model_json_schema()` replaces custom reflection. Preserve domain-specific layers (test values, constraints, backend routing). ~300→~250 LOC. | 2026-02-26 |

---

## Immediately Actionable (No Decision Needed)

These should be fixed in planning docs before Phase 5 begins — no human decision required:

1. **Fix `designs/reproducibility.md`** — remove reference to `warmup_result.warmup_duration_sec` (field doesn't exist; actual field is `warmup_duration_s` or similar — check N-C03)
2. **Add N-R04 to Phase 5 task list** — `StudyManifest` + `ManifestWriter` must be built (design is complete in `study-resume.md`)
3. **Add `EnergyBreakdown` to v2.0 result schema** — already implemented; move from deferred v2.1
4. ~~**Add 10 built-in presets to CLI design doc**~~ — **DONE (2026-02-26)**: `PRESETS` registry dropped. Opinionated Pydantic defaults replace named presets. Domain knowledge → backend section defaults + example YAMLs.
5. ~~**Add `_extends` config inheritance to config-architecture decision doc**~~ — **DONE (2026-02-26)**: explicitly cut from v2.0 in `config-architecture.md`
6. **Add `protocols.py` module to architecture design** — 5 DI protocols form the backbone of the system

---

## See Also

- **PRESERVATION_AUDIT.md** — original summary-level audit (P-01 to P-23) — file removed during migration
- **[NEEDS_ADDRESSING.md](../NEEDS_ADDRESSING.md)** — open design questions and inconsistencies
