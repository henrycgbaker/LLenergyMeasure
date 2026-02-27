# Requirements: LLenergyMeasure M2 — Study / Sweep

**Defined:** 2026-02-27
**Core Value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Source of truth:** `.product/REQUIREMENTS.md` — this file adds M2 scope adjustments and phase traceability.
**Research:** `.planning/research/SUMMARY.md` (M2 domain research)

---

## M2 Requirements

Requirements for M2 (Study/Sweep Execution). 24 requirements total (22 from `.product/REQUIREMENTS.md` M2 tags + 2 new).
Full definitions, context, and design references are in `.product/REQUIREMENTS.md`.

### Config / Sweep (6 requirements)

| ID | Requirement |
|----|-------------|
| CFG-11 | `StudyConfig` = resolved container: `list[ExperimentConfig]` + `ExecutionConfig` |
| CFG-12 | Sweep resolution at YAML parse time, before Pydantic validation |
| CFG-13 | Dotted notation sweep keys: `pytorch.batch_size: [1, 8]` — backend-scoped grid |
| CFG-14 | Three modes: grid sweep (Cartesian), explicit `experiments:` list, combined |
| CFG-15 | `ExecutionConfig`: `n_cycles` (Pydantic default=1, CLI effective default=3), `cycle_order`, `config_gap_seconds`, `cycle_gap_seconds` |
| CFG-16 | `study_design_hash` = SHA-256[:16] of sweep+experiments only (execution block excluded) |

### Study Execution (7 requirements)

| ID | Requirement |
|----|-------------|
| STU-01 | `StudyRunner` iterates experiments, spawns subprocesses via `multiprocessing.get_context("spawn")` |
| STU-02 | IPC via `multiprocessing.Pipe` for result return (Pipe-only; file fallback dropped) |
| STU-03 | `daemon=False` on subprocess (clean CUDA teardown) |
| STU-04 | Timeout via `p.join(timeout=...)` + SIGKILL on timeout |
| STU-06 | Config gap between experiments from user config (machine-local) |
| STU-07 | `cycle_order: sequential \| interleaved` — interleaved round-robins across configs |
| STU-NEW-01 | `_run()` body implemented — dispatches single vs study, returns `StudyResult` |

### Manifest (2 requirements)

| ID | Requirement |
|----|-------------|
| STU-08 | `StudyManifest` written after each experiment completes (checkpoint pattern) |
| STU-09 | `ManifestWriter`: `mark_running()`, `mark_completed()`, `mark_failed()` — atomic writes via `os.replace()` |

### Results (4 requirements)

| ID | Requirement |
|----|-------------|
| RES-13 | `StudyResult`: `study_design_hash`, `measurement_protocol`, `result_files: list[str]`, `summary: StudySummary` |
| RES-14 | `StudyManifest`: in-progress checkpoint, distinct from `StudyResult` |
| RES-15 | `result_files` contains paths, not embedded results |
| RES-NEW-01 | Study output layout: `{study_name}_{timestamp}/` containing per-experiment subdirs + `manifest.json` |

### CLI (2 requirements)

| ID | Requirement |
|----|-------------|
| CLI-05 | Study-mode flags: `--cycles`, `--no-gaps`, `--order` (`--resume` deferred to M4) |
| CLI-11 | Thermal gap countdown display during inter-experiment pauses |

### Core Measurement (1 requirement)

| ID | Requirement |
|----|-------------|
| CM-10 | Multi-backend study without Docker → hard error at pre-flight. Single-backend-only in M2 (explicit). |

### Library API (2 requirements)

| ID | Requirement |
|----|-------------|
| LA-02 | `run_study(config: str \| Path \| StudyConfig) -> StudyResult` |
| LA-05 | `run_study()` always writes manifest to disk (documented exception to side-effect-free) |

---

## Scope Adjustments

| Adjustment | Rationale |
|------------|-----------|
| `--resume` deferred to M4 (STU-10, STU-11) | Manifest is always-on but resume logic adds complexity; ship checkpoint first |
| IPC file-based fallback dropped from STU-02 | `ExperimentResult` fits in Pipe buffer; 1MB fallback is over-engineering for M2 |
| CM-10 single-backend-only explicit | Docker multi-backend is M3; M2 validates and hard-errors |
| STU-NEW-01 added: `_run()` body | M1 left `_run()` as `NotImplementedError`; M2 implements the dispatcher |
| RES-NEW-01 added: study output layout | Needed for manifest placement and per-experiment result organisation |

---

## Deferred

| ID | What | Deferred To |
|----|------|-------------|
| STU-10 | `--resume` flag implementation | M4 |
| STU-11 | Resume identity via `measurement_config_hash` | M4 |

## Out of Scope

| Feature | Reason |
|---------|--------|
| Docker multi-backend execution | M3 — requires container lifecycle, GPU passthrough |
| `--resume` flag | M4 — manifest always-on in M2 enables this later |
| IPC file-based fallback | Dropped — Pipe sufficient for ExperimentResult sizes |
| vLLM / TensorRT-LLM backends | M3 — PyTorch single-backend only in M2 |
| Bootstrap CIs | v2.1 — raw measurement primary |
| lm-eval integration | v3.0 — separate version |

---

## Traceability

### M1 Gap Closure (from M1-MILESTONE-AUDIT.md)

| Requirement | Phase | Status | Gap Type |
|-------------|-------|--------|----------|
| RES-06 | Phase 8.1 | Complete | Partial — baseline fields never populated |
| RES-16 | Phase 8.1 | Complete | Partial — timeseries path divergence + effective_config empty |
| CM-16 | Phase 8.1 | Complete | Partial — timeseries field name mismatch |
| CFG-01–10 | Phase 8.2 | Complete | Verified — 02-VERIFICATION.md created with 11/11 truths |
| CFG-18–26 | Phase 8.2 | Complete | Verified — 02-VERIFICATION.md created with 11/11 truths |

### M2 Requirements

*Roadmap created 2026-02-27 — all 24 M2 requirements mapped to phases.*

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-11 | Phase 9 | Complete |
| CFG-12 | Phase 9 | Complete |
| CFG-13 | Phase 9 | Complete |
| CFG-14 | Phase 9 | Complete |
| CFG-15 | Phase 9 | Complete |
| CFG-16 | Phase 9 | Complete |
| STU-08 | Phase 10 | Complete |
| STU-09 | Phase 10 | Complete |
| RES-14 | Phase 10 | Complete |
| RES-NEW-01 | Phase 10 | Complete |
| STU-01 | Phase 11 | Complete |
| STU-02 | Phase 11 | Complete |
| STU-03 | Phase 11 | Complete |
| STU-04 | Phase 11 | Complete |
| STU-06 | Phase 11 | Pending |
| STU-07 | Phase 11 | Complete |
| LA-02 | Phase 12 | Pending |
| LA-05 | Phase 12 | Pending |
| STU-NEW-01 | Phase 12 | Pending |
| RES-13 | Phase 12 | Pending |
| RES-15 | Phase 12 | Pending |
| CLI-05 | Phase 12 | Pending |
| CLI-11 | Phase 12 | Pending |
| CM-10 | Phase 12 | Pending |

**Coverage:**
- M1 gap closure: 3 requirement gaps + 19 verification gaps → Phases 8.1, 8.2
- M2 requirements: 24 total, mapped to phases: 24, unmapped: 0

---
*Requirements defined: 2026-02-27*
*Traceability filled: 2026-02-27*
*M1 gap closure added: 2026-02-27*
