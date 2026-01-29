# Roadmap: LLenergyMeasure v2.0.0

## Overview

The v2.0.0 milestone transforms LLenergyMeasure from a functional research tool into a production-grade CLI with measurement accuracy and campaign orchestration capabilities. Four phased releases address systematic energy measurement errors (15-30% idle power contamination), enable multi-backend campaign workflows via long-running Docker containers, achieve 90%+ backend parameter coverage, and complete comprehensive user acceptance testing. This roadmap delivers on the core value: accurate, comprehensive measurement of the true cost of LLM inference with research-grade rigour.

## Milestones

- ✅ **v1.19.0 Measurement Foundations** - Phase 1 (complete)
- 📋 **v1.20.0 Campaign Orchestrator** - Phase 2 (planned)
- 📋 **v1.21.0 Parameter Completeness** - Phase 3 (planned)
- 📋 **v1.22.0 Polish + UAT** - Phase 4 (planned)
- 🎯 **v2.0.0 Release** - Fully Working CLI milestone

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Measurement Foundations** - Fix systematic energy errors, capture environment metadata, enable warmup convergence
- [ ] **Phase 2: Campaign Orchestrator** - Long-running containers, backend-aware grid generation, manifest tracking
- [ ] **Phase 3: Parameter Completeness** - 90%+ backend coverage, version pinning, SSOT introspection updates
- [ ] **Phase 4: Polish + UAT** - Cleanup, architectural refactor, documentation refresh, full workflow validation

## Phase Details

### Phase 1: Measurement Foundations (v1.19.0)

**Goal**: Users measure inference energy with research-grade accuracy — baseline power subtracted, thermal throttling detected, environment fully documented, warmup convergence automatic

**Depends on**: Nothing (first phase)

**Requirements**: MEAS-01, MEAS-02, MEAS-03, MEAS-04, MEAS-05, MEAS-06, MEAS-07, MEAS-09

**Success Criteria** (what must be TRUE):
1. User runs experiment and receives both raw energy (backwards compatible) and baseline-adjusted energy (corrects 15-30% systematic overestimation)
2. User inspects results metadata and sees comprehensive environment details (GPU model, CUDA version, driver, thermal state, power limits, CPU governor, container detection)
3. User views time-series power/memory/utilisation data at configurable sampling rates (1-10Hz) for any experiment
4. User receives automatic flag in results when thermal throttling detected during experiment
5. User configures warmup convergence detection and warmup continues until CV stabilises (not fixed prompt count)
6. User exports extended efficiency metrics to CSV (memory, GPU utilisation, request latency, batch size, KV cache metrics — currently JSON-only)
7. Fresh clone installation succeeds, user runs one experiment following quickstart, pain points documented
8. All config extensions (WarmupConfig, BaselineConfig, TimeSeriesConfig) threaded through SSOT introspection — appear in CLI outputs, results JSON/CSV, runtime validation tests, and generated docs

**Context**: See `phases/01-CONTEXT.md` for detailed implementation decisions

**Plans**: 6 plans in 4 waves

Plans:
- [x] 01-01-PLAN.md — Schema v3 domain models + configuration extensions (Wave 1)
- [x] 01-02-PLAN.md — PowerThermalSampler + environment metadata + baseline power (Wave 2)
- [x] 01-03-PLAN.md — Warmup convergence detection (Wave 2)
- [x] 01-04-PLAN.md — CSV export extensions + time-series export (Wave 2)
- [x] 01-05-PLAN.md — Orchestrator integration (Wave 3)
- [x] 01-06-PLAN.md — Unit tests + UAT round 1 (Wave 4)

---

### Phase 2: Campaign Orchestrator (v1.20.0)

**Goal**: Users orchestrate multi-backend campaigns with long-running containers, backend-aware grid generation, and persistent manifest tracking — no per-experiment container startup overhead. Campaigns own cycles (repetitions of the full experiment set for statistical robustness).

**Depends on**: Phase 1 (needs accurate measurements for campaign validation)

**Requirements**: CAMP-01, CAMP-02, CAMP-03, CAMP-04, CAMP-05, CAMP-06, CAMP-07, CAMP-08, MEAS-08

**Architectural model** (campaign-cycle hierarchy):
```
Campaign (user interaction level)
  └── Cycle N (repetition of entire campaign for statistical robustness)
        ├── Experiment A (atomic unit — one config, one run)
        ├── Experiment B
        └── Experiment C
```
Experiments are atomic. Cycles repeat the full experiment set. Campaigns orchestrate everything.

**Success Criteria** (what must be TRUE):
1. User starts campaign and backend containers stay running across experiments (no `docker compose run --rm` overhead), receiving commands via `docker compose exec`
2. User defines campaign with backend-aware grid generation (cartesian product respects per-backend param validity — no invalid backend × param combinations)
3. User inspects campaign manifest and sees exp_id → config → backend → container → status → result_path tracking for all experiments
4. User configures daemon mode and experiments run at scheduled times (not just sequential with thermal gaps)
5. User sets `force_cold_start: true` and model unloads between experiments for cold-start benchmarking (warmup fairness is default)
6. Containers undergo health checks, start only needed backends, graceful teardown after campaign completion
7. Existing campaign features retained: randomisation within cycles, interleaved/shuffled/grouped structures, thermal gaps, cycle-as-highest-organisational-principle
8. Cross-backend campaign runs successfully (PyTorch + vLLM + TensorRT experiments dispatched correctly), UAT validates orchestrator behaviour
9. User runs multi-cycle campaign and receives 95% confidence intervals via bootstrap resampling (moved from Phase 1 — requires cycle-level data)
10. All config extensions added in this phase threaded through SSOT introspection — appear in CLI outputs, results JSON/CSV, runtime validation tests, and generated docs (introspection.py auto-discovers new Pydantic fields)

**Discussion points** (to address during planning):
- IO path configuration: currently only `io.results_dir` exists. Campaign YAML should configure paths for configs, state, and shared volumes (especially for Docker volume mapping). Consider a broader `io` section or campaign-level path config with sensible defaults.

**Plans**: TBD

Plans:
- [ ] 02-01: TBD during planning
- [ ] 02-02: TBD during planning
- [ ] 02-03: TBD during planning

---

### Phase 3: Parameter Completeness (v1.21.0)

**Goal**: Users configure 90%+ of energy/throughput-impactful parameters for each backend with version pinning, SSOT auto-discovery, and escape hatch for niche cases

**Depends on**: Phase 2 (uses campaign orchestrator for cross-backend parameter audit validation)

**Requirements**: PARAM-01, PARAM-02, PARAM-03, PARAM-04, PARAM-05

**Success Criteria** (what must be TRUE):
1. PyTorch backend supports 95%+ of energy-impactful parameters (up from 93.8%), vLLM 90%+ (up from 81.9%), TensorRT 95%+ (up from 93.8%)
2. User reviews systematic parameter audit results showing which kwargs affect energy/throughput for each backend
3. User passes undocumented/niche parameter via `extra:` escape hatch and it reaches backend successfully
4. SSOT introspection auto-discovers new parameters added to Pydantic models (no manual parameter list maintenance)
5. Cross-backend parameter audit campaign runs successfully via orchestrator (UAT round 3), results verify parameter coverage
6. All config extensions added in this phase threaded through SSOT introspection — appear in CLI outputs, results JSON/CSV, runtime validation tests, and generated docs (introspection.py auto-discovers new Pydantic fields)

**Discussion points** (to address during planning):
- Audit `parameter-support-matrix.md` vs `config-reference.md` divergence: matrix is generated from runtime test results (GPU-tested params only), while config-reference covers all Pydantic fields. Verify `test_all_params.py --discover` mode correctly picks up all params from SSOT (including Phase 1 baseline/warmup/timeseries). Consider making `--discover` the default mode for robustness. Regenerate matrix with full SSOT coverage.

**Plans**: TBD

Plans:
- [ ] 03-01: TBD during planning
- [ ] 03-02: TBD during planning

---

### Phase 4: Polish + UAT (v1.22.0)

**Goal**: v2.0.0 release-ready — codebase cleaned via Occam's razor, architecture refactored if pain points found, documentation refreshed, Docker images validated, full workflow UAT passed

**Depends on**: Phase 3 (needs all features complete before final validation)

**Requirements**: UAT-01, UAT-02, UAT-03, UAT-04, UAT-05

**Success Criteria** (what must be TRUE):
1. Codebase cleanup pass complete: dead code removed, naming consistency improved, over-engineered areas simplified per Occam's razor
2. Architectural refactor implemented if cleanup revealed structural pain points (or skipped if architecture sound)
3. Full campaign workflow UAT passed: fresh VM, multi-backend grid, cycle execution, result export — all backends, end-to-end
4. Documentation refreshed: README accurate, quickstart validated, deployment guides current, troubleshooting updated
5. Docker images rebuilt from v2.0.0 tag and validated: PyTorch, vLLM, TensorRT services start cleanly, run experiments successfully
6. All config extensions from all phases verified end-to-end through SSOT introspection — CLI outputs, results JSON/CSV, runtime validation tests, and generated docs all reflect the complete parameter surface

**Plans**: TBD

Plans:
- [ ] 04-01: TBD during planning
- [ ] 04-02: TBD during planning

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Measurement Foundations | 6/6 | Complete | 2026-01-29 |
| 2. Campaign Orchestrator | 0/? | Not started | - |
| 3. Parameter Completeness | 0/? | Not started | - |
| 4. Polish + UAT | 0/? | Not started | - |

---

**Total Requirements:** 27 v1 requirements mapped across 4 phases
**Coverage:** 27/27 (100%)

**Next:** `/gsd:plan-phase 2` to begin Phase 2 planning
