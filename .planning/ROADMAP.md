# Roadmap: LLenergyMeasure

## Milestones

- [x] **v1.x Foundation & Planning** — Phases 1–4.5 (shipped 2026-02-26)
- [x] **M1 — Core Single-Experiment** — Phases 1–8.2 (completed 2026-02-27)
- [ ] **M2 — Study / Sweep** — Phases 9–13

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...): Planned milestone work
- Decimal phases (9.1, 10.1...): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ M1 — Core Single-Experiment (Phases 1–8.2) — COMPLETED 2026-02-27</summary>

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

</details>

### M1 Gap Closure

### Phase 8.1: PyTorch Result Wiring Fixes
**Goal:** Fix the 4 broken E2E flows caused by `PyTorchBackend._build_result()` not wiring 3 fields correctly — restoring timeseries round-trip, output directory naming, baseline display, and sidecar discovery.
**Depends on**: Phase 8 (M1 shipped — fixes integration bugs found by audit)
**Requirements**: RES-06, RES-16, CM-16
**Gap Closure:** Closes 3 requirement gaps, 4 integration gaps, 4 broken flows from M1 audit
**Success Criteria** (what must be TRUE):
  1. `ExperimentResult.timeseries` is a `Path` (not None) after `PyTorchBackend.run()` — field name mismatch fixed
  2. `ExperimentResult.effective_config` is populated with the experiment's config dict — output dirs named `{model}_{backend}_{ts}`
  3. `ExperimentResult.baseline_power_w` and `energy_adjusted_j` are populated from `EnergyBreakdown` data
  4. `ExperimentResult.model_config` has `extra="forbid"` — unrecognised kwargs raise `ValidationError`
  5. Timeseries parquet is co-located with `result.json` in the same subdirectory
**Plans**: 1 plan (Wave 1)

Plans:
- [x] 08.1-01: Fix _build_result() wiring (timeseries, effective_config, baseline), add extra="forbid", wire CLI timeseries co-location, add tests [Wave 1]

---

### Phase 8.2: M1 Tech Debt Cleanup
**Goal:** Address all tech debt identified by the M1 audit — formal verification for Phase 2, requirements status alignment, broken import chain cleanup, orphaned export removal, and CLAUDE.md corrections.
**Depends on**: Phase 8.1 (wiring fixes must land first so verification can reference working code)
**Requirements**: CFG-01 through CFG-10, CFG-18–CFG-26 (verification gap closure)
**Gap Closure:** Closes 19 verification gaps, 12 tech debt items from M1 audit
**Success Criteria** (what must be TRUE):
  1. Phase 2 has a VERIFICATION.md confirming all 19 CFG requirements
  2. REQUIREMENTS.md traceability: CFG-01–10, CFG-18–26 rows updated from Pending to Complete
  3. `cli/experiment.py` and `cli/utils.py` broken v1.x imports removed or fixed
  4. `cli/CLAUDE.md` no longer references deleted v1.x commands
  5. Orphaned exports removed: `export_aggregated_to_csv()`, `SubprocessRunner`. Kept (active callers): `aggregate_results()`, `FlopsEstimator`, `StateManager`
**Plans**: 2 plans (Wave 1 — parallel)

Plans:
- [x] 08.2-01: Documentation cleanup — Phase 2 VERIFICATION.md, REQUIREMENTS.md traceability update, cli/CLAUDE.md correction [Wave 1]
- [x] 08.2-02: Code cleanup — broken imports in cli/ modules, orphaned exports removal, test verification [Wave 1]

---

### M2 — Study / Sweep (Current)

**Milestone Goal:** `llem run study.yaml` runs a multi-experiment sweep with subprocess isolation, cycle ordering, thermal gaps, and a checkpoint manifest — producing per-experiment `ExperimentResult` files and a `StudyResult` summary.

- [ ] **Phase 9: Grid Expansion and StudyConfig** - Sweep YAML grammar, `StudyConfig` + `ExecutionConfig` models, Cartesian grid expander, cycle ordering, pre-flight count display
- [x] **Phase 10: Manifest Writer** - `StudyManifest` checkpoint model, `ManifestWriter` with atomic writes, study output directory layout (completed 2026-02-27)
- [x] **Phase 11: Subprocess Isolation and StudyRunner** - Subprocess dispatch via `spawn`, `Pipe`/`Queue` IPC, timeout handling, SIGINT, skip-and-continue, thermal gaps (completed 2026-02-27)
- [x] **Phase 12: Integration** - `StudyRunner.run()`, `run_study()` public API, `_run()` body, CLI study flags, study progress display, `StudyResult` assembly, multi-backend hard error (completed 2026-02-27)

## Phase Details

### Phase 9: Grid Expansion and StudyConfig
**Goal**: Researchers can express a sweep configuration in YAML and have it resolve to a complete, ordered list of `ExperimentConfig` objects before any subprocess is spawned — with a pre-flight count display that prevents combinatorial surprises.
**Depends on**: Phase 8 (M1 complete — `ExperimentConfig` and `ExecutionConfig` models must exist)
**Requirements**: CFG-11, CFG-12, CFG-13, CFG-14, CFG-15, CFG-16
**Success Criteria** (what must be TRUE):
  1. A sweep YAML with `pytorch.batch_size: [1, 8]` and `pytorch.precision: [fp32, fp16]` resolves to exactly 4 `ExperimentConfig` objects via `expand_grid()`
  2. The `experiments:` explicit list mode and combined mode (sweep + explicit) both resolve correctly into a flat `list[ExperimentConfig]`
  3. `n_cycles=3` with `cycle_order=interleaved` produces an experiment list that round-robins across configs rather than repeating each config consecutively
  4. `study_design_hash` is a 16-char hex string that changes when sweep dimensions change and stays the same when only `execution:` block values change
  5. A sweep producing more than a configurable cap of experiments displays the count and estimated wall time before proceeding
**Plans**: 2 plans (Wave 1 → Wave 2)

Plans:
- [ ] 09-01: ExecutionConfig + StudyConfig models, study/grid.py (expand_grid, apply_cycles, hash, SkippedConfig), unit tests (TDD) [Wave 1]
- [ ] 09-02: load_study_config() in loader.py, format_preflight_summary(), integration tests [Wave 2]

---

### Phase 10: Manifest Writer
**Goal**: Every study run produces an atomic, corruption-proof checkpoint file that records the state of every experiment — written after each state transition so an interrupted study leaves a readable manifest.
**Depends on**: Phase 9 (study output layout requires `StudyConfig` to provide name and hash)
**Requirements**: STU-08, STU-09, RES-14, RES-NEW-01
**Success Criteria** (what must be TRUE):
  1. `ManifestWriter.mark_running()`, `.mark_completed()`, and `.mark_failed()` each produce a valid `manifest.json` in the study output directory via atomic `os.replace()`
  2. Simulating an interruption mid-write (e.g., killing the process after `write_text()` but before `os.replace()`) leaves the previous `manifest.json` intact and parseable
  3. The study output directory follows the `{study_name}_{timestamp}/` layout, with flat per-experiment result files and `manifest.json` at the top level
  4. `StudyManifest` and `StudyResult` are distinct types — manifest is the checkpoint, result is the final return value
**Plans**: 1 plan (Wave 1)

Plans:
- [ ] 10-01: StudyManifest model, ManifestWriter with atomic writes, study output directory helpers, TDD tests [Wave 1]

---

### Phase 11: Subprocess Isolation and StudyRunner
**Goal**: Each experiment in a study runs in a freshly spawned subprocess with a clean CUDA state, results are returned via `Pipe`, progress events flow via `Queue`, and the study survives experiment failures, timeouts, and SIGINT without data corruption.
**Depends on**: Phase 10 (manifest must exist before runner can checkpoint state)
**Requirements**: STU-01, STU-02, STU-03, STU-04, STU-06, STU-07
**Success Criteria** (what must be TRUE):
  1. Each experiment subprocess is started with `multiprocessing.get_context("spawn")` — verified by asserting `p.start_method == "spawn"` in tests
  2. A subprocess that raises an unhandled exception produces a structured failure result (type, message, truncated traceback) via the `Pipe`, and the study continues with the next experiment rather than aborting
  3. A subprocess that exceeds `experiment_timeout_seconds` is killed via SIGKILL, not SIGTERM, and the manifest marks the experiment as `"failed"`
  4. Pressing Ctrl+C during a study kills the active subprocess, marks the manifest as interrupted, and exits with code 130 — `manifest.json` is left in a readable state
  5. Config gap (`config_gap_seconds`) between experiments and cycle gap (`cycle_gap_seconds`) between cycles are both honoured, with a visible countdown in the terminal during each pause
**Plans**: TBD

Plans:
- [ ] TBD

---

### Phase 12: Integration
**Goal**: `llem run study.yaml` is fully wired end-to-end — the CLI detects study mode, routes to `StudyRunner`, the `run_study()` public API returns a `StudyResult`, and study-specific flags (`--cycles`, `--no-gaps`, `--order`) all work correctly with a single-backend-only constraint enforced at pre-flight.
**Depends on**: Phase 11 (all subprocess and manifest components must exist before wiring)
**Requirements**: LA-02, LA-05, STU-NEW-01, RES-13, RES-15, CLI-05, CLI-11, CM-10
**Success Criteria** (what must be TRUE):
  1. `llem run study.yaml` runs a 2-config × 2-cycle sweep end-to-end on a GPU machine, producing per-experiment `result.json` files and a final `StudyResult` summary
  2. `run_study(config)` is importable from `llenergymeasure` and returns exactly `StudyResult` — no union types, always writes `manifest.json` to disk as documented
  3. `llem run study.yaml --cycles 5 --order interleaved --no-gaps` overrides `execution:` block values from YAML
  4. `llem run study.yaml` with `backend: [pytorch, vllm]` raises `PreFlightError` with a message directing the user to the Docker runner (M3) — exits with code 1
  5. The terminal displays a per-experiment progress line and a visible thermal gap countdown during inter-experiment pauses
**Plans**: TBD

Plans:
- [ ] TBD

---

## Progress

**Execution Order:**
8.1 → 8.2 → 9 → 10 → 11 → 12 → 13

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Package Foundation | 5/5 | Complete | 2026-02-26 |
| 2. Config System | 4/4 | Complete | 2026-02-26 |
| 3. Library API | 2/2 | Complete | 2026-02-26 |
| 4. PyTorch Backend and Pre-flight | 3/3 | Complete | 2026-02-26 |
| 4.1. PyTorch Parameter Audit | 3/3 | Complete | 2026-02-26 |
| 5. Energy Measurement | 3/3 | Complete | 2026-02-26 |
| 6. Results Schema and Persistence | 3/3 | Complete | 2026-02-26 |
| 7. CLI | 3/3 | Complete | 2026-02-27 |
| 8. Testing and Integration | 3/3 | Complete | 2026-02-27 |
| 8.1. PyTorch Result Wiring Fixes | 1/1 | Complete | 2026-02-27 |
| 8.2. M1 Tech Debt Cleanup | 2/2 | Complete   | 2026-02-27 |
| 9. Grid Expansion and StudyConfig | 1/2 | In Progress|  |
| 10. Manifest Writer | 1/1 | Complete    | 2026-02-27 |
| 11. Subprocess Isolation and StudyRunner | 2/2 | Complete    | 2026-02-27 |
| 12. Integration | 3/3 | Complete    | 2026-02-27 |
| 13. Documentation — M1 backfill and M2 updates | 0/TBD | Not started | - |

### Phase 13: Documentation — M1 backfill and M2 updates

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 12
**Plans:** 3/3 plans complete

Plans:
- [ ] TBD (run /gsd:plan-phase 13 to break down)

---

*M1 roadmap created: 2026-02-26*
*M2 roadmap appended: 2026-02-27*
