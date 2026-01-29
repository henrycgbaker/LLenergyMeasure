# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Accurate, comprehensive measurement of the true cost of LLM inference — energy, compute, and quality tradeoffs — with research-grade rigour.
**Current focus:** Phase 1 Complete - Ready for Phase 2

## Current Position

Phase: 1 of 4 (Measurement Foundations)
Plan: 6 of 6 complete
Status: Phase complete
Last activity: 2026-01-29 — Completed 01-06-PLAN.md (unit tests & UAT round 1)

Progress: [██████░░░░] 100% Phase 1 (6/6)

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 7 min
- Total execution time: 48 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-measurement-foundations | 6/6 | 48 min | 7 min |

**Recent Trend:**
- 01-01: 7 min (2 tasks, domain models + config extensions)
- 01-03: 4 min (1 task, warmup convergence module)
- 01-04: 4 min (2 tasks, CSV export + timeseries)
- 01-02: 11 min (2 tasks, NVML measurement primitives)
- 01-05: 7 min (2 tasks, orchestrator integration)
- 01-06: 15 min (2 tasks, unit tests + UAT checkpoint)
- Trend: Consistent ~4-7 min; 01-02/01-06 longer (annotation resolution / UAT wait)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Split v2 into 4 sub-releases (v1.19-v1.22): Each independently shippable, reduces risk, enables early UAT
- Measurement foundations before parameter work: Fixes systematic 15-30% energy bias; accurate data before more features
- Campaign orchestrator as own release: Highest-risk item (Docker exec model), shouldn't block measurement improvements
- Long-running containers with `docker compose exec`: Avoids per-experiment container startup overhead; containers stay warm, shared volumes for results
- Warmup convergence (CV-based) over fixed count: Scientifically more robust; existing CycleStatistics already tracks CV
- [01-01] Removed `from __future__ import annotations` from domain models: Pydantic v2 incompatibility with deferred annotations in nested models
- [01-01] Runtime imports for new domain types in experiment.py: Pydantic needs resolved types at class definition time
- [01-04] Renamed CSV column `total_energy_j` to `energy_raw_j`: Grouped-prefix convention for CSV readability; Pydantic model field unchanged
- [01-02] Used `typing.Any` for pynvml module/handle params: mypy cannot type-check dynamically imported modules; `from __future__ import annotations` causes ruff to remove type imports as unused
- [01-06 UAT] Warmup skips gracefully for backend-managed models (BackendModelLoaderAdapter returns None)
- [01-06 UAT] Energy breakdown uses experiment timestamps when CodeCarbon reports duration_sec=0.0

### Pending Todos

None yet.

### Blockers/Concerns

None. Phase 1 validated end-to-end on A100 GPU.

## Session Continuity

Last session: 2026-01-29
Stopped at: Completed 01-06-PLAN.md (unit tests & UAT round 1) — Phase 1 complete
Resume file: None
