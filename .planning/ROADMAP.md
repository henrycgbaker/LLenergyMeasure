# Roadmap: LLenergyMeasure v2.0.0

## Overview

The v2.0.0 milestone transforms LLenergyMeasure from a functional research tool into a production-grade CLI with measurement accuracy and campaign orchestration capabilities. Four phased releases address systematic energy measurement errors (15-30% idle power contamination), enable multi-backend campaign workflows via long-running Docker containers, achieve 90%+ backend parameter coverage, and complete comprehensive user acceptance testing. This roadmap delivers on the core value: accurate, comprehensive measurement of the true cost of LLM inference with research-grade rigour.

## Milestones

- **v1.19.0 Measurement Foundations** - Phase 1 (complete)
- **v1.20.0 Campaign Orchestrator** - Phase 2 (complete)
- **v1.21.0 GPU Routing + Audit + Refactor** - Phases 3-5 (planned)
- **v1.22.0 Parameter Completeness** - Phase 6 (planned)
- **v1.23.0 Polish + UAT** - Phase 7 (planned)
- **v2.0.0 Release** - Fully Working CLI milestone

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
- [ ] **Phase 3: GPU Routing Fix** - Fix CUDA_VISIBLE_DEVICES propagation, fail-fast config validation
- [ ] **Phase 4: Codebase Audit** - Identify stubs, dead code, unimplemented features, verify plans vs implementation
- [ ] **Phase 5: Refactor & Simplify** - Remove dead code, simplify workflows, rename lem->llem
- [ ] **Phase 6: Parameter Completeness** - 90%+ backend coverage, version pinning, SSOT introspection updates
- [ ] **Phase 7: Polish + UAT** - Cleanup, architectural refactor, documentation refresh, full workflow validation

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
1. User starts campaign and each experiment runs in an ephemeral container via `docker compose run --rm` — simple lifecycle, no health-check complexity, correct for workloads without cross-experiment model persistence
2. User defines campaign with backend-aware grid generation (cartesian product respects per-backend param validity — no invalid backend × param combinations)
3. User inspects campaign manifest and sees exp_id -> config -> backend -> container -> status -> result_path tracking for all experiments
4. User configures daemon mode and experiments run at scheduled times (not just sequential with thermal gaps)
5. User sets `force_cold_start: true` and model unloads between experiments for cold-start benchmarking (warmup fairness is default)
6. Campaign dispatches to correct backend containers, checks images are built before execution, ephemeral containers auto-clean up via `--rm`
7. Existing campaign features retained: randomisation within cycles, interleaved/shuffled/grouped structures, thermal gaps, cycle-as-highest-organisational-principle
8. Cross-backend campaign runs successfully (PyTorch + vLLM + TensorRT experiments dispatched correctly), UAT validates orchestrator behaviour
9. User runs multi-cycle campaign and receives 95% confidence intervals via bootstrap resampling (moved from Phase 1 — requires cycle-level data)
10. All config extensions added in this phase threaded through SSOT introspection — appear in CLI outputs, results JSON/CSV, runtime validation tests, and generated docs (introspection.py auto-discovers new Pydantic fields)

**Context**: See `phases/02-CONTEXT.md` for detailed implementation decisions

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

### Phase 2.1: Zero-Config Install Experience (INSERTED)

**Goal**: Users install via `pip install llenergymeasure` (or `pip install -e .`) and everything works out of the box — local-first with PyTorch default, Docker auto-detected for multi-backend campaigns, `.env` auto-generated on first Docker use. No manual `setup.sh` required.

**Depends on**: Phase 2 (campaign orchestrator must exist before install experience can be validated end-to-end)

**Requirements**: Derived from UAT pain points discovered during Phase 2 execution

**Success Criteria** (what must be TRUE):
1. User runs `pip install -e .` followed by `lem campaign config.yaml` and it works — no manual setup.sh, no missing .env errors
2. Docker auto-detection: if Docker is available, .env is auto-generated with PUID/PGID; if not, falls back to local execution seamlessly
3. Post-install hooks or first-run detection handles all configuration that setup.sh currently does manually
4. Package is PyPI-publishable: `pip install llenergymeasure` from PyPI network works out of the box
5. Both install paths (pip install + setup.sh Docker-first) produce identical working setups
6. Local execution (conda, venv, poetry) correctly detected — no false Docker routing

**Context**: Discovered during Phase 2 UAT — `_should_use_docker()` misidentified conda installs as needing Docker, `.env` with PUID/PGID required manual generation, `setup.sh` not triggered by pip install.

**Plans**: 6 plans in 4 waves

Plans:
- [x] 02.1-01-PLAN.md — Detection modules: docker_detection, backend_detection, env_setup (Wave 1)
- [x] 02.1-02-PLAN.md — CLI wiring: `lem doctor` unified diagnostic command (Wave 2)
- [x] 02.1-06-PLAN.md — Campaign refactor: _should_use_docker + ensure_env_file wiring (Wave 2)
- [x] 02.1-03-PLAN.md — Packaging: PyTorch default + Makefile + delete setup.sh + PyPI validation (Wave 3)
- [x] 02.1-04-PLAN.md — Documentation refresh: README, quickstart, backends, deployment (Wave 3)
- [x] 02.1-05-PLAN.md — Unit tests + UAT verification checkpoint (Wave 4)

---

### Phase 2.2: Campaign Execution Model (INSERTED)

**Goal**: Fix container routing bugs, propagate campaign context to experiments, implement dual container strategy (ephemeral vs persistent), ensure CI computed at campaign level only.

**Depends on**: Phase 2.1 (needs working Docker detection and dispatch)

**Requirements**: Derived from UAT bugs discovered during Phase 2/2.1 testing

**Success Criteria** (what must be TRUE):
1. TensorRT experiments route to `tensorrt` container (not `base`)
2. Experiments running as part of campaign display "Part of campaign X, cycle Y/Z" instead of "single cycle" warning
3. User can configure `docker.strategy: persistent` in `.lem-config.yaml` to use `up + exec` pattern
4. Campaign CLI flag `--container-strategy [ephemeral|persistent]` overrides config
5. CI warning suppressed for experiments within multi-cycle campaigns
6. Thermal gap defaults configurable via user preferences

**Context**: See `.planning/ARCHITECTURE-DISCUSSION.md` for decisions

**Plans**: 4 plans in 3 waves

Plans:
- [x] 02.2-01-PLAN.md — Campaign context propagation + CI warning suppression + remove --cycles from experiment (Wave 1)
- [x] 02.2-02-PLAN.md — Container routing fix + minimal user config loading for thermal gaps (Wave 1)
- [x] 02.2-03-PLAN.md — Dual container strategy (ephemeral default + persistent option) (Wave 2)
- [x] 02.2-04-PLAN.md — Unit tests + UAT verification checkpoint (Wave 3)

---

### Phase 2.3: Campaign State & Resume (INSERTED)

**Goal**: Interactive campaign resume via `lem resume`, guided project setup via `lem init`, and webhook notification system for experiment completion/failure.

**Depends on**: Phase 2.2 (needs correct execution model before adding state management)

**Requirements**: Derived from feature requests and UAT feedback

**Success Criteria** (what must be TRUE):
1. `lem resume` discovers interrupted campaigns in `.state/` and presents interactive menu
2. `lem resume --dry-run` shows what would be resumed without executing
3. `lem resume --wipe` clears all state after confirmation
4. `lem init` walks user through guided setup wizard with environment detection
5. `lem init --non-interactive` creates config with defaults or CLI-provided values
6. User preferences file `.lem-config.yaml` supports: results_dir, thermal_gaps, docker preferences, notification webhooks (default_backend removed — must be explicit in config)
7. Webhook notifications POST to configured URL on experiment completion/failure

**Context**: See `phases/02.3-CONTEXT.md` for detailed implementation decisions

**Plans**: 4 plans in 2 waves

Plans:
- [x] 02.3-01-PLAN.md — `lem resume` command + UserConfig notifications + webhook sender (Wave 1)
- [x] 02.3-02-PLAN.md — `lem init` interactive wizard (Wave 1)
- [x] 02.3-03-PLAN.md — Unit tests for resume and init commands (Wave 2)
- [x] 02.3-04-PLAN.md — Verification checkpoint (Wave 2)

---

### Phase 2.4: CLI Polish & Testing Infrastructure (INSERTED)

**Goal**: Clean up CLI UX issues, improve developer experience with systematic smoke tests, and ensure all configurations and examples are up-to-date with schema v3.0.0.

**Depends on**: Phase 2.3 (builds on complete campaign/resume functionality)

**Requirements**: Derived from accumulated todos and UAT observations

**Success Criteria** (what must be TRUE):
1. Campaign aggregation supports configurable `group_by` — users specify how to group experiments for summary statistics (by model, by backend, by batch_size, etc.) with results printed in CLI and recorded in campaign metadata
2. CLI commands follow Typer best practices — required inputs are positional arguments, optional modifiers are flags
3. CLI output has three-tier verbosity (quiet/standard/verbose) with backend-specific noise filtered appropriately
4. All example configs use schema v3.0.0, maximize each backend's functionality, and test_configs directory is complete
5. `lem config list` command exists for discoverability of available configurations
6. pyproject.toml and Docker dependencies follow consistent SSOT pattern
7. Systematic inference smoke tests run actual inference across parameter combinations, capturing and reporting warnings/errors (e.g., flash_attention_2 fallback)

**Context**: See `phases/02.4-CONTEXT.md` for implementation decisions, `phases/02.4-RESEARCH.md` for research findings

**Plans**: 6 plans in 2 waves

Plans:
- [x] 02.4-01-PLAN.md — `lem config list` command + campaign `--group-by` flag (Wave 1)
- [x] 02.4-02-PLAN.md — Example configs schema v3.0.0 + schema_version validation (Wave 1)
- [x] 02.4-03-PLAN.md — Smoke test suite with strict warning capture (Wave 1)
- [x] 02.4-05-PLAN.md — Backend noise filtering + log capture + --json output (Wave 1)
- [x] 02.4-06-PLAN.md — Docker lifecycle output (strategy display, build progress, dispatch status) (Wave 1)
- [x] 02.4-04-PLAN.md — Unit tests + verification checkpoint (Wave 2)

---

### Phase 3: GPU Routing Fix

**Goal**: Fix the critical bug where CUDA_VISIBLE_DEVICES is not properly propagated to Docker containers, causing GPU initialization failures. Add fail-fast config validation for parallelism constraints.

**Depends on**: Phase 2.4 (builds on complete Docker dispatch infrastructure)

**Requirements**: Derived from UAT failures — vLLM/TensorRT tensor parallelism fails due to CUDA not being visible in containers

**Success Criteria** (what must be TRUE):
1. User-provided `gpus` field from config becomes SSOT for available devices — propagated to containers via NVIDIA_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES
2. Docker containers receive GPU access correctly — vLLM tensor_parallel_size=2 with gpus=[0,1,2,3] initializes successfully
3. TensorRT experiments route to `tensorrt` container (not `base`) and have GPU access
4. Config validation fails fast for invalid parallelism configs (e.g., tensor_parallel_size > available GPUs, attention heads not divisible by TP size)
5. Runtime GPU detection removed from experiment path — config.gpus is authoritative
6. Clear error messages when parallelism constraints violated (before container launch, not deep in backend initialization)

**Context**: Discovered during vLLM/TensorRT UAT — "CUDA driver initialization failed" errors from worker processes, incorrect container routing for tensorrt backend.

**Research**: See `phases/03-gpu-routing-fix/03-RESEARCH.md` for NVIDIA Container Toolkit details and GPU routing patterns.

**Plans**: 3 plans in 2 waves

Plans:
- [ ] 03-01-PLAN.md — GPU env propagation to Docker (NVIDIA_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES remapping) (Wave 1)
- [ ] 03-02-PLAN.md — Fail-fast parallelism validation (tensor_parallel_size vs gpus) (Wave 1)
- [ ] 03-03-PLAN.md — Unit tests + manual verification checkpoint (Wave 2)

---

### Phase 4: Codebase Audit

**Goal**: Thoroughly audit the codebase to identify stubs, dead code, unimplemented features, over-engineering, and gaps between planning documents and actual implementation.

**Depends on**: Phase 3 (GPU routing must work before comprehensive audit)

**Requirements**: Derived from accumulated technical debt and UAT observations

**Success Criteria** (what must be TRUE):
1. Audit report identifies all stubs (code that exists but doesn't function)
2. Audit report identifies all dead code (unused modules, functions, config fields)
3. Audit report identifies all unimplemented features from Phase 1-2 planning docs
4. Audit report identifies over-engineered patterns that can be simplified
5. Audit report identifies config fields that exist in schema but aren't wired up (e.g., `timeseries`)
6. Audit report identifies duplicated functionality/code paths
7. Audit report identifies misleading output (e.g., parallelism "appeared" to work but wasn't)
8. All findings documented with specific file:line references and recommended actions

**Context**: User feedback indicates codebase has accumulated complexity beyond what the core tool requires. Need systematic review before refactoring.

**Plans**: 6 plans in 2 waves

Plans:
- [ ] 04-01-PLAN.md — Automated analysis (vulture, deadcode, ruff) + industry CLI comparison (Wave 1)
- [ ] 04-02-PLAN.md — CLI surface and configuration system audit (Wave 1)
- [ ] 04-03-PLAN.md — Core engine audit (backends, metrics, measurement) (Wave 1)
- [ ] 04-04-PLAN.md — Orchestration, campaign, results, state, and domain audit (Wave 1)
- [ ] 04-05-PLAN.md — Docker infrastructure, detection systems, tests, and planning cross-reference (Wave 1)
- [ ] 04-06-PLAN.md — Report assembly + user review checkpoint (Wave 2)

---

### Phase 5: Refactor & Simplify

**Goal**: Act on audit findings to simplify the codebase, remove dead code, fix identified issues, and rename the CLI from `lem` to `llem`.

**Depends on**: Phase 4 (needs audit findings before refactoring)

**Requirements**: Derived from Phase 4 audit results

**Success Criteria** (what must be TRUE):
1. All dead code identified in audit removed
2. All stubs either implemented or removed (with documentation if deferred)
3. Over-engineered patterns simplified per Occam's razor
4. Config fields wired up end-to-end or removed from schema
5. CLI renamed from `lem` to `llem` (all references updated: pyproject.toml, docs, examples, tests)
6. Backend configs pass directly to inference backends without unnecessary parsing/transformation
7. Test suite updated to catch "silent failures" (things that appear to work but don't)
8. SSOT principles enforced throughout codebase

**Context**: Simplification pass to reduce maintenance burden and improve reliability before adding more features.

**Plans**: TBD (run /gsd:plan-phase 5 to break down)

Plans:
- [ ] TBD

---

### Phase 6: Parameter Completeness (v1.22.0)

**Goal**: Users configure 90%+ of energy/throughput-impactful parameters for each backend with SSOT auto-discovery and escape hatch for niche cases

**Depends on**: Phase 5 (needs clean codebase before adding parameters)

**Requirements**: PARAM-01, PARAM-02, PARAM-03, PARAM-04, PARAM-05

**Success Criteria** (what must be TRUE):
1. PyTorch backend supports 95%+ of energy-impactful parameters (up from 93.8%), vLLM 90%+ (up from 81.9%), TensorRT 95%+ (up from 93.8%)
2. User reviews systematic parameter audit results showing which kwargs affect energy/throughput for each backend
3. User passes undocumented/niche parameter via `extra:` escape hatch and it reaches backend successfully
4. SSOT introspection auto-discovers new parameters added to Pydantic models (no manual parameter list maintenance)
5. Cross-backend parameter audit campaign runs successfully via orchestrator (UAT round 3), results verify parameter coverage
6. All config extensions added in this phase threaded through SSOT introspection — appear in CLI outputs, results JSON/CSV, runtime validation tests, and generated docs (introspection.py auto-discovers new Pydantic fields)

**Context**: See `phases/06-parameter-completeness/` for implementation decisions and research findings

**Plans**: 6 plans in 4 waves

Plans:
- [ ] 06-01-PLAN.md — PyTorch parameter expansion (8 new generation params) (Wave 1)
- [ ] 06-02-PLAN.md — vLLM parameter expansion (9 new engine params) (Wave 1)
- [ ] 06-03-PLAN.md — TensorRT parameter expansion (10 new build/runtime params) (Wave 1)
- [ ] 06-04-PLAN.md — SSOT introspection updates + documentation regeneration (Wave 2)
- [ ] 06-05-PLAN.md — Comprehensive example configs for all backends (Wave 3)
- [ ] 06-06-PLAN.md — Unit tests + UAT verification (Wave 4)

---

### Phase 7: Polish + UAT (v1.23.0)

**Goal**: v2.0.0 release-ready — codebase cleaned via Occam's razor, architecture refactored if pain points found, documentation refreshed, Docker images validated, full workflow UAT passed

**Depends on**: Phase 6 (needs all features complete before final validation)

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
- [ ] 07-01: TBD during planning
- [ ] 07-02: TBD during planning

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Measurement Foundations | 6/6 | Complete | 2026-01-29 |
| 2. Campaign Orchestrator | 8/8 | Complete | 2026-01-30 |
| 2.1 Zero-Config Install (INSERTED) | 6/6 | Complete | 2026-01-31 |
| 2.2 Campaign Execution Model (INSERTED) | 4/4 | Complete | 2026-02-03 |
| 2.3 Campaign State & Resume (INSERTED) | 4/4 | Complete | 2026-02-04 |
| 2.4 CLI Polish & Testing Infrastructure (INSERTED) | 6/6 | Complete | 2026-02-04 |
| 3. GPU Routing Fix | 3/3 | Complete | 2026-02-04 |
| 4. Codebase Audit | 0/6 | Planned | - |
| 5. Refactor & Simplify | 0/? | Planned | - |
| 6. Parameter Completeness | 0/6 | Planned | - |
| 7. Polish + UAT | 0/? | Not started | - |

---

**Total Requirements:** 27 v1 requirements mapped across 7 phases + 4 inserted phases for architecture fixes
**Coverage:** 27/27 (100%) + additional UAT-derived requirements

**Next:** Plan Phase 4 with `/gsd:plan-phase 4`
