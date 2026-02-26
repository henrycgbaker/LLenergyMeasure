# Roadmap: LLenergyMeasure


TODO: this is now degraded - it needs updating based on new product decisions and desings (see phase 4.5)

## Overview

LLenergyMeasure is a library-first LLM inference efficiency measurement framework. The v2.x series builds a solid, research-grade CLI and library. v3.0 adds quality-alongside-efficiency via lm-eval integration. v4.0 is a separate web platform product.

**Core value:** The only tool that systematically quantifies how deployment configuration (batch size, quantisation, backend, parallelism) affects LLM energy efficiency — with research-grade rigour.

## Milestones

- **v2.0 — Clean Foundation** — Phase 5 (current milestone)
- **v2.1 — Measurement Depth** — Phase 6 (planned)
- **v2.2 — Scale** — Phase 7 (planned)
- **v2.3 — Parameter Completeness** — Phase 8 (planned)
- **v2.4 — Shareability** — Phase 9 (planned)
- **v3.0 — Quality + Efficiency** — Phase 10 (planned)
- **v4.0 — Web Platform** — Separate product, separate repo

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4, ...): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Measurement Foundations** - Fix systematic energy errors, capture environment metadata, enable warmup convergence
- [x] **Phase 2: Campaign Orchestrator** - Long-running containers, backend-aware grid generation, manifest tracking
- [x] **Phase 2.1: Zero-Config Install Experience** - Auto-detect Docker, auto-generate .env, PyPI-ready packaging (INSERTED)
- [x] **Phase 2.2: Campaign Execution Model** - Fix container routing, cycle context, dual container strategy (INSERTED)
- [x] **Phase 2.3: Campaign State & Resume** - State persistence, `lem resume`, user preferences (INSERTED)
- [x] **Phase 2.4: CLI Polish & Testing Infrastructure** - Aggregation group_by, CLI UX, example configs, smoke tests (INSERTED)
- [x] **Phase 3: GPU Routing Fix** - Fix CUDA_VISIBLE_DEVICES propagation, fail-fast config validation
- [x] **Phase 4: Codebase Audit** - Identify stubs, dead code, unimplemented features, verify plans vs implementation
- [x] **Phase 4.5: Strategic Reset** - Product vision settled, versioning roadmap confirmed, architecture decisions recorded (INSERTED)
- [ ] **Phase 5: Clean Foundation (v2.0)** - Library-first restructure, `llem` rename, P0 bug fixes, dead code removal, CLI 15→9, state machine 6→3
- [ ] **Phase 6: Measurement Depth (v2.1)** - Zeus energy backend, baseline power, thermal time-series, warmup convergence, env metadata, schema v3
- [ ] **Phase 7: Scale (v2.2)** - Campaign as separate module, Docker multi-backend, grid generation
- [ ] **Phase 8: Parameter Completeness (v2.3)** - PyTorch/vLLM/TRT to 90%+, prefill/decode phase split
- [ ] **Phase 9: Shareability (v2.4)** - `llem results push`, opt-in central DB, HuggingFace Datasets export
- [ ] **Phase 10: Quality + Efficiency (v3.0)** - lm-eval integration, accuracy vs efficiency tradeoff metrics

---

## Phase Details

### Phase 1: Measurement Foundations (complete)

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

### Phase 2: Campaign Orchestrator (complete)

**Goal**: Users orchestrate multi-backend campaigns with long-running containers, backend-aware grid generation, and persistent manifest tracking — no per-experiment container startup overhead. Campaigns own cycles (repetitions of the full experiment set for statistical robustness).

**Depends on**: Phase 1 (needs accurate measurements for campaign validation)

**Requirements**: CAMP-01, CAMP-02, CAMP-03, CAMP-04, CAMP-05, CAMP-06, CAMP-07, CAMP-08, MEAS-08

**Plans**: 8 plans in 4 waves

Plans:
- [x] 02-01-PLAN.md — Campaign config extensions (grid, manifest, cold start, health check, daemon, IO models) (Wave 1)
- [x] 02-02-PLAN.md — Docker dispatch via `docker compose run --rm` (Wave 1)
- [x] 02-03-PLAN.md — Bootstrap CI + campaign-level aggregation (Wave 1)
- [x] 02-04-PLAN.md — Campaign manifest persistence + resume (Wave 2)
- [x] 02-05-PLAN.md — Backend-aware grid expansion + validation (Wave 2)
- [x] 02-06-PLAN.md — Campaign orchestrator integration (CLI + CampaignRunner wiring) (Wave 3)
- [x] 02-07-PLAN.md — SSOT introspection threading for campaign config (Wave 3)
- [x] 02-08-PLAN.md — Unit tests + UAT round 2 (Wave 4)

---

### Phase 2.1: Zero-Config Install Experience (INSERTED, complete)

**Goal**: Users install via `pip install llenergymeasure` (or `pip install -e .`) and everything works out of the box.

**Plans**: 6 plans in 4 waves — all complete.

---

### Phase 2.2: Campaign Execution Model (INSERTED, complete)

**Goal**: Fix container routing bugs, propagate campaign context to experiments, implement dual container strategy.

**Plans**: 4 plans in 3 waves — all complete.

---

### Phase 2.3: Campaign State & Resume (INSERTED, complete)

**Goal**: Interactive campaign resume via `lem resume`, guided project setup via `lem init`, and webhook notification system.

**Plans**: 4 plans in 2 waves — all complete.

---

### Phase 2.4: CLI Polish & Testing Infrastructure (INSERTED, complete)

**Goal**: Clean up CLI UX issues, improve developer experience with systematic smoke tests, update all example configs to schema v3.0.0.

**Plans**: 6 plans in 2 waves — all complete.

---

### Phase 3: GPU Routing Fix (complete)

**Goal**: Fix CUDA_VISIBLE_DEVICES propagation to Docker containers, add fail-fast parallelism validation.

**Plans**: 3 plans in 2 waves — all complete.

---

### Phase 4: Codebase Audit (complete)

**Goal**: Thoroughly audit the codebase — stubs, dead code, unimplemented features, over-engineering, planning vs implementation gaps.

**Plans**: 6 plans in 2 waves — all complete.

---

### Phase 4.5: Strategic Reset (INSERTED, complete)

**Goal**: Settle product vision, confirm versioning roadmap, record all architecture decisions before beginning Phase 5 work.

**Key outputs**:
- Product vision: Dual-product (CLI + web platform), deployment efficiency positioning (Option A primary)
- Versioning: v2.0 Clean Foundation → v2.x series → v3.0 Quality → v4.0 Web Platform
- Architecture: Library-first restructure at v2.0; study as separate module (v2.2); Zeus as energy backend (v2.1); lm-eval at v3.0
- Installation: `llem` rename at v2.0, no default backend, explicit extras per backend
- Decision records: `decisions/` directory with versioning-roadmap.md, product-vision.md, architecture.md, installation.md
- Design specs: `design/` directory with package-structure.md, cli-commands.md

**Plans**: Research and discussion — no GSD plan files (discussion-format phase)

---

### Phase 5: Clean Foundation (v2.0) — CURRENT MILESTONE

**Goal**: Library-first restructure, `llem` rename, P0 bug fixes, dead code removal, CLI 15→3, state machine 6→3.

**Depends on**: Phase 4.5 (vision and architecture decisions confirmed)

**Success Criteria** (what must be TRUE):
1. `import llenergymeasure` works as a library — `__init__.py` exports Tier 1 + Tier 2 public API
2. CLI renamed `lem` → `llem` throughout (pyproject.toml, docs, examples, tests)
3. All 4 P0 bugs fixed (PyTorch model_kwargs L375, vLLM no native streaming, Docker broken, vLLM shm-size missing)
4. 1,524 lines of confirmed dead code removed
5. CLI reduced from 15 to 3 commands (experiment, study, status)
6. State machine reduced from 6 to 3 states
7. No default backend at base install — `pip install llenergymeasure` installs no GPU backend
8. All tests passing at v2.0 library boundary

**Plans**: TBD (run `/gsd:plan-phase 5` to break down)

Plans:
- [ ] TBD

---

### Phase 6: Measurement Depth (v2.1) — planned

**Goal**: Zeus energy backend, baseline power, thermal time-series, warmup convergence, environment metadata, bootstrap CI, schema v3.

**Depends on**: Phase 5 (clean foundation required before extending measurement system)

**Key items**:
- Zeus as optional energy backend (`llenergymeasure[zeus]`) — NVML-direct, ~5% accuracy vs CodeCarbon ~10-15%
- Baseline power measurement before experiments
- Thermal time-series at configurable intervals
- Warmup convergence detection (CV-based, not fixed count)
- Environment metadata per experiment
- Bootstrap resampling for 95% confidence intervals
- Schema v3 with migration path from v2

**Plans**: TBD

---

### Phase 7: Scale (v2.2) — planned

**Goal**: Campaign as separate importable module, Docker multi-backend orchestration, programmatic grid generation.

**Depends on**: Phase 6 (measurement depth complete before scale work)

**Key items**:
- `study` module extracted from CLI — importable, not CLI-only
- `study.runner` + `study.grid` as separate package
- Docker multi-backend dispatch
- Backend-aware grid generation retained

**Plans**: TBD

---

### Phase 8: Parameter Completeness (v2.3) — planned

**Goal**: 90%+ energy/throughput-impactful parameter coverage for each backend. Prefill/decode phase split.

**Depends on**: Phase 7

**Key items**:
- PyTorch 93.8% → 95%+, vLLM 81.9% → 90%+, TensorRT 93.8% → 95%+
- Prefill/decode phase split (separate timing, energy, throughput per phase)
- SSOT introspection auto-discovers new parameters
- `extra:` escape hatch validated end-to-end

**Plans**: TBD

---

### Phase 9: Shareability (v2.4) — planned

**Goal**: Results sharing infrastructure. `llem results push`, opt-in central database, HuggingFace Datasets export.

**Depends on**: Phase 8

**Key items**:
- `llem results push <id>` — upload result to central DB
- Opt-in central archive (growing evidence base for policy advocacy)
- HuggingFace Datasets export format
- Privacy: opt-in only, no automatic uploads

**Plans**: TBD

---

### Phase 10: Quality + Efficiency (v3.0) — planned

**Goal**: lm-eval integration — quality-alongside-efficiency, the unique differentiator no other tool provides.

**Depends on**: v2.x complete (solid CLI foundation required)

**Key items**:
- `llenergymeasure[lm-eval]` optional extra
- Run accuracy benchmarks alongside efficiency measurement
- Quality vs efficiency tradeoff metrics and Pareto frontier analysis
- Unique positioning: the only tool combining inference efficiency + accuracy benchmarking

**Plans**: TBD

---

### v4.0: Web Platform — separate product

**Goal**: Browser-based results explorer, live dashboard, public leaderboard for policy advocacy.

**Scope**: Separate repo, own lifecycle, shares library API but not server.

**Key items**:
- Static JSON MVP first (no backend required)
- Dynamic API second (FastAPI)
- Live features third (real-time dashboard, leaderboard)
- Deployment: self-hosted GPU workers + lightweight central server (no GPUs)
- Users: policy makers, decision makers, public — no GPU required

**Plans**: TBD (separate product planning)

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 2.1 → 2.2 → 2.3 → 2.4 → 3 → 4 → 4.5 → 5 → 6 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Measurement Foundations | 6/6 | Complete | 2026-01-29 |
| 2. Campaign Orchestrator | 8/8 | Complete | 2026-01-30 |
| 2.1 Zero-Config Install (INSERTED) | 6/6 | Complete | 2026-01-31 |
| 2.2 Campaign Execution Model (INSERTED) | 4/4 | Complete | 2026-02-03 |
| 2.3 Campaign State & Resume (INSERTED) | 4/4 | Complete | 2026-02-04 |
| 2.4 CLI Polish & Testing Infrastructure (INSERTED) | 6/6 | Complete | 2026-02-04 |
| 3. GPU Routing Fix | 3/3 | Complete | 2026-02-04 |
| 4. Codebase Audit | 6/6 | Complete | 2026-02-05 |
| 4.5 Strategic Reset (INSERTED) | — | Complete | 2026-02-17 |
| 5. Clean Foundation (v2.0) | 0/? | Planned | - |
| 6. Measurement Depth (v2.1) | 0/? | Planned | - |
| 7. Scale (v2.2) | 0/? | Planned | - |
| 8. Parameter Completeness (v2.3) | 0/? | Planned | - |
| 9. Shareability (v2.4) | 0/? | Planned | - |
| 10. Quality + Efficiency (v3.0) | 0/? | Planned | - |

---

**Next:** Plan Phase 5 with `/gsd:plan-phase 5`
