# Requirements: LLenergyMeasure M1

**Defined:** 2026-02-26
**Core Value:** Researchers can run broad parameter sweeps and produce publishable measurements showing which implementation choices matter most for LLM energy efficiency.
**Source of truth:** `.product/REQUIREMENTS.md` — this file adds phase traceability only.

## M1 Requirements

Requirements for M1 (Core Single-Experiment). Each maps to roadmap phases below.
Full definitions, context, and design references are in `.product/REQUIREMENTS.md`.

### Library API (9 requirements)

LA-01, LA-03, LA-04, LA-06, LA-07, LA-08, LA-09, LA-10, CFG-17
- `run_experiment()` public API and internal `_run(StudyConfig)`
- `__init__.py` exports, stability guarantees, no union return types
- CFG-17 (single run = degenerate StudyConfig) co-located here — requires _run(StudyConfig) defined in this phase
- Note: LA-02 is M2 (run_study), LA-05 is M2 (study manifest writes)

### Config (19 requirements)

CFG-01 through CFG-10, CFG-18 through CFG-26
- `ExperimentConfig` composition model (Pydantic v2, backend sections, validators)
- YAML loading, user config (`~/.config/llenergymeasure/config.yaml`)
- Config introspection (SSOT for tests, docs, CLI)
- Note: CFG-11 through CFG-16 are M2 (StudyConfig, sweep grammar)
- Note: CFG-17 is Phase 3 (single run = degenerate StudyConfig; requires StudyConfig)

### Core Measurement (28 requirements)

CM-01, CM-04 through CM-06, CM-11 through CM-34
- PyTorch inference backend + P0 bug fix
- Energy backends: NVML (base), Zeus (optional), CodeCarbon (optional)
- Baseline power, warmup (fixed + CV-based), FLOPs estimation
- Pre-flight checks, environment snapshot, thermal detection
- Note: CM-02, CM-03, CM-07 through CM-09 are M3 (vLLM, TRT, Docker fixes)

### Results (18 requirements)

RES-01 through RES-12, RES-16 through RES-21
- `ExperimentResult` schema (all v2.0 fields)
- Persistence: JSON primary, Parquet timeseries, CSV opt-in
- Late aggregation, output layout, collision handling
- Note: RES-13 through RES-15 are M2 (StudyResult, manifest)

### CLI (13 requirements)

CLI-01 through CLI-04, CLI-06 through CLI-14
- `llem run [CONFIG] [OPTIONS]`, `llem config [--verbose]`, `llem --version`
- Plain text output (~200 LOC, tqdm), exit codes
- `LLEMError` hierarchy, `--dry-run`, `--quiet`, `--verbose`
- Note: CLI-05, CLI-11 are M2 (study-mode flags, thermal countdown)

### Infrastructure (16 requirements)

STU-05, INF-01 through INF-12, INF-18 through INF-20
- `pyproject.toml`, `src/` layout, hatchling build, extras
- Protocols (5 DI interfaces), state machine (3 states)
- Testing: unit + integration tiers, protocol mocks
- Resilience/retry, subprocess lifecycle, path sanitisation
- Note: INF-13 through INF-17 are M3 (Docker infrastructure)

## Future Requirements (M2–M4)

Tracked in `.product/REQUIREMENTS.md`. Not in current scope.

- **M2 (Study/Sweep):** CFG-11–16, LA-02, LA-05, RES-13–15, CLI-05, CLI-11, STU-01–09, CM-10
- **M3 (Docker Multi-Backend):** CM-02, CM-03, CM-07–09, INF-13–17. Note: repeat the parameter audit (M1 Phase 4.1) for vLLM and TensorRT-LLM backends when implemented.
- **M4 (Advanced Features):** STU-10, STU-11

## Out of Scope

| Feature | Reason |
|---------|--------|
| Study/sweep execution | M2 — needs single experiment working first |
| Docker multi-backend | M3 — PyTorch local only in M1 |
| Traffic simulation, streaming latency | M4 — advanced features |
| Bootstrap CIs | v2.1 — raw measurement primary |
| lm-eval integration | v3.0 — separate version |
| Web platform | v4.0 — separate product |
| Singularity/Apptainer | v2.1+ — NotImplementedError in v2.0 |
| Shareability/upload | Post-v2.0 — trust model unresolved |

## Traceability

Maps each M1 requirement to exactly one phase.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INF-01 | Phase 1 | Complete |
| INF-02 | Phase 1 | Complete |
| INF-03 | Phase 1 | Complete |
| INF-04 | Phase 1 | Complete |
| INF-05 | Phase 1 | Complete |
| INF-06 | Phase 1 | Complete |
| INF-07 | Phase 1 | Complete |
| INF-08 | Phase 1 | Complete |
| INF-18 | Phase 1 | Complete |
| INF-19 | Phase 1 | Complete |
| INF-20 | Phase 1 | Complete |
| CFG-01 | Phase 2 | Complete |
| CFG-02 | Phase 2 | Complete |
| CFG-03 | Phase 2 | Complete |
| CFG-04 | Phase 2 | Complete |
| CFG-05 | Phase 2 | Complete |
| CFG-06 | Phase 2 | Complete |
| CFG-07 | Phase 2 | Complete |
| CFG-08 | Phase 2 | Complete |
| CFG-09 | Phase 2 | Complete |
| CFG-10 | Phase 2 | Complete |
| CFG-18 | Phase 2 | Complete |
| CFG-19 | Phase 2 | Complete |
| CFG-20 | Phase 2 | Complete |
| CFG-21 | Phase 2 | Complete |
| CFG-22 | Phase 2 | Complete |
| CFG-23 | Phase 2 | Complete |
| CFG-24 | Phase 2 | Complete |
| CFG-25 | Phase 2 | Complete |
| CFG-26 | Phase 2 | Complete |
| LA-01 | Phase 3 | Complete |
| LA-03 | Phase 3 | Complete |
| LA-04 | Phase 3 | Complete |
| LA-06 | Phase 3 | Complete |
| LA-07 | Phase 3 | Complete |
| LA-08 | Phase 3 | Complete |
| LA-09 | Phase 3 | Complete |
| LA-10 | Phase 3 | Complete |
| CFG-17 | Phase 3 | Complete |
| CM-01 | Phase 4 | Pending |
| CM-04 | Phase 4 | Pending |
| CM-05 | Phase 4 | Pending |
| CM-06 | Phase 4 | Pending |
| CM-29 | Phase 4 | Pending |
| CM-30 | Phase 4 | Pending |
| CM-31 | Phase 4 | Pending |
| CM-32 | Phase 4 | Pending |
| CM-33 | Phase 4 | Pending |
| CM-34 | Phase 4 | Pending |
| CM-11 | Phase 5 | Pending |
| CM-12 | Phase 5 | Pending |
| CM-13 | Phase 5 | Pending |
| CM-14 | Phase 5 | Pending |
| CM-15 | Phase 5 | Pending |
| CM-16 | Phase 5 | Pending |
| CM-17 | Phase 5 | Pending |
| CM-18 | Phase 5 | Pending |
| CM-19 | Phase 5 | Pending |
| CM-20 | Phase 5 | Pending |
| CM-21 | Phase 5 | Pending |
| CM-22 | Phase 5 | Pending |
| CM-23 | Phase 5 | Pending |
| CM-24 | Phase 5 | Pending |
| CM-25 | Phase 5 | Pending |
| CM-26 | Phase 5 | Pending |
| CM-27 | Phase 5 | Pending |
| CM-28 | Phase 5 | Pending |
| RES-01 | Phase 6 | Pending |
| RES-02 | Phase 6 | Pending |
| RES-03 | Phase 6 | Pending |
| RES-04 | Phase 6 | Pending |
| RES-05 | Phase 6 | Pending |
| RES-06 | Phase 6 | Pending |
| RES-07 | Phase 6 | Pending |
| RES-08 | Phase 6 | Pending |
| RES-09 | Phase 6 | Pending |
| RES-10 | Phase 6 | Pending |
| RES-11 | Phase 6 | Pending |
| RES-12 | Phase 6 | Pending |
| RES-16 | Phase 6 | Pending |
| RES-17 | Phase 6 | Pending |
| RES-18 | Phase 6 | Pending |
| RES-19 | Phase 6 | Pending |
| RES-20 | Phase 6 | Pending |
| RES-21 | Phase 6 | Pending |
| CLI-01 | Phase 7 | Pending |
| CLI-02 | Phase 7 | Pending |
| CLI-03 | Phase 7 | Pending |
| CLI-04 | Phase 7 | Pending |
| CLI-06 | Phase 7 | Pending |
| CLI-07 | Phase 7 | Pending |
| CLI-08 | Phase 7 | Pending |
| CLI-09 | Phase 7 | Pending |
| CLI-10 | Phase 7 | Pending |
| CLI-12 | Phase 7 | Pending |
| CLI-13 | Phase 7 | Pending |
| CLI-14 | Phase 7 | Pending |
| STU-05 | Phase 8 | Pending |
| INF-09 | Phase 8 | Pending |
| INF-10 | Phase 8 | Pending |
| INF-11 | Phase 8 | Pending |
| INF-12 | Phase 8 | Pending |

**Coverage:**
- M1 requirements: 103 total
- Mapped to phases: 103
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 — traceability populated by roadmapper*
