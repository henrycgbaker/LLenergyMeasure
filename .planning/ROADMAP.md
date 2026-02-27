# Roadmap: LLenergyMeasure

## Milestones

- [x] **v1.x Foundation & Planning** — Phases 1–4.5 (shipped 2026-02-26)
- [x] **v1.17.0 M1 — Core Single-Experiment** — Phases 1–8.2 (shipped 2026-02-27)
- [x] **v1.18.0 M2 — Study / Sweep** — Phases 9–15 (shipped 2026-02-27)
- [ ] **M3 — Docker Multi-Backend** — TBD

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...): Planned milestone work
- Decimal phases (9.1, 10.1...): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v1.17.0 M1 — Core Single-Experiment (Phases 1–8.2) — SHIPPED 2026-02-27</summary>

- [x] **Phase 1: Package Foundation** - Dead code removal, src/ layout, pyproject.toml, protocols, state machine, resilience carry-forwards (completed 2026-02-26)
- [x] **Phase 2: Config System** - ExperimentConfig composition model, YAML loader, user config, SSOT introspection (completed 2026-02-26)
- [x] **Phase 3: Library API** - `__init__.py` public API, `run_experiment()`, internal `_run(StudyConfig)`, API stability contract (completed 2026-02-26)
- [x] **Phase 4: PyTorch Backend and Pre-flight** - PyTorch inference backend (P0 bug fix), InferenceBackend protocol, pre-flight checks, environment snapshot (completed 2026-02-26)
- [x] **Phase 4.1: PyTorch Parameter Audit** - INSERTED — Audit PyTorchConfig fields against upstream `transformers`/`torch` APIs (completed 2026-02-26)
- [x] **Phase 5: Energy Measurement** - NVML poller, Zeus optional, CodeCarbon optional, baseline power, warmup, FLOPs estimation, timeseries (completed 2026-02-26)
- [x] **Phase 6: Results Schema and Persistence** - ExperimentResult schema, EnergyBreakdown, persistence API, late aggregation, output layout (completed 2026-02-26)
- [x] **Phase 7: CLI** - `llem run`, `llem config`, `llem --version`, plain text display, exit codes, error hierarchy (completed 2026-02-27)
- [x] **Phase 8: Testing and Integration** - Unit + integration test tiers, protocol mocks, GPU CI workflow, UAT against M1 exit criteria (completed 2026-02-27)
- [x] **Phase 8.1: PyTorch Result Wiring Fixes** - INSERTED — Fix `_build_result()` field wiring: timeseries, effective_config, baseline fields. Add `extra="forbid"`. Gap closure. (completed 2026-02-27)
- [x] **Phase 8.2: M1 Tech Debt Cleanup** - INSERTED — Phase 2 VERIFICATION.md, REQUIREMENTS.md status drift, v1.x import breakages, orphaned exports cleanup. (completed 2026-02-27)

Full details: `milestones/v1.17.0-ROADMAP.md`

</details>

<details>
<summary>✅ v1.18.0 M2 — Study / Sweep (Phases 9–15) — SHIPPED 2026-02-27</summary>

- [x] **Phase 9: Grid Expansion and StudyConfig** - Sweep YAML grammar, `StudyConfig` + `ExecutionConfig` models, Cartesian grid expander, cycle ordering, pre-flight count display (completed 2026-02-27)
- [x] **Phase 10: Manifest Writer** - `StudyManifest` checkpoint model, `ManifestWriter` with atomic writes, study output directory layout (completed 2026-02-27)
- [x] **Phase 11: Subprocess Isolation and StudyRunner** - Subprocess dispatch via `spawn`, `Pipe`/`Queue` IPC, timeout handling, SIGINT, skip-and-continue, thermal gaps (completed 2026-02-27)
- [x] **Phase 12: Integration** - `StudyRunner.run()`, `run_study()` public API, `_run()` body, CLI study flags, study progress display, `StudyResult` assembly, multi-backend hard error (completed 2026-02-27)
- [x] **Phase 14: Multi-Cycle Execution Fixes** - Fix double `apply_cycles()`, cycle tracking, manifest completion status. Gap closure. (completed 2026-02-27)
- [x] **Phase 15: M2 Tech Debt and Progress Wiring** - Wire progress display, fix phantom field, ROADMAP/SUMMARY drift. (completed 2026-02-27)

Full details: `milestones/v1.18.0-ROADMAP.md`

</details>

### M3 — Docker Multi-Backend (Next)

**Milestone Goal:** TBD — Docker container lifecycle, GPU passthrough, vLLM and TensorRT-LLM backends, multi-backend study execution.

Phases: TBD (run `/gsd:new-milestone` to define)

### Phase 13: Documentation — M1/M2/M3 backfill
**Goal:** [To be planned — deferred to end of M3 when all backends are complete]
**Requirements**: TBD
**Depends on:** M3 completion
**Plans:** 0/TBD

Plans:
- [ ] TBD (run /gsd:plan-phase 13 to break down)

---

*M1 roadmap created: 2026-02-26*
*M2 roadmap appended: 2026-02-27*
*M1 shipped: 2026-02-27 (v1.17.0)*
*M2 shipped: 2026-02-27 (v1.18.0)*
*Phase 13 deferred to M3: 2026-02-27*
