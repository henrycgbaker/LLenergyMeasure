# Milestones: LLenergyMeasure

## Completed

### v1.x — Foundation & Planning (Phases 1–4.5)

**Shipped:** 2026-01-29 to 2026-02-26
**Phases:** 1, 2, 2.1, 2.2, 2.3, 2.4, 3, 4, 4.5
**Plans completed:** 46

What shipped:
- Measurement foundations (baseline power, thermal sampling, warmup convergence, env metadata)
- Campaign orchestrator (Docker dispatch, grid generation, manifest tracking, cycle management)
- Zero-config install experience (auto Docker detect, auto .env, PyPI-ready packaging)
- Campaign execution model (container routing, cycle context, dual container strategy)
- Campaign state & resume (state persistence, `lem resume`, user preferences, init wizard)
- CLI polish & testing (aggregation group_by, backend noise filtering, smoke tests, example configs)
- GPU routing fix (CUDA_VISIBLE_DEVICES propagation, fail-fast validation, HPC compat)
- Codebase audit (4 P0 bugs, 1,524 lines dead code, over-engineering identified)
- Strategic reset (product vision, versioning roadmap, all architecture decisions recorded)
- Full product planning (120+ requirements, 4 milestones M1–M4, ADRs, design specs)

---

### v1.17.0 — M1: Core Single-Experiment (Phases 1–8.2)

**Shipped:** 2026-02-27
**Phases:** 1, 2, 3, 4, 4.1, 5, 6, 7, 8, 8.1, 8.2 (11 phases)
**Plans completed:** 32
**Timeline:** 2026-02-26 → 2026-02-27
**Git range:** 23e1f61..2003c6f (194 files changed, 12,566 insertions, 31,501 deletions)

**Delivered:** `llem run --model meta-llama/Llama-3.1-8B` produces a complete ExperimentResult with energy measurement, baseline correction, warmup, FLOPs, environment snapshot, JSON + Parquet output.

What shipped:
- Package Foundation — Hatchling build, 21 dead code files deleted, `llem` entry point, zero-dep base install
- Config System — ExperimentConfig composition model with `extra="forbid"`, YAML loader, SSOT introspection
- Library API — `run_experiment()` public API, `_run(StudyConfig)` internal dispatcher, API stability contract
- PyTorch Backend — Pre-flight checks, EnvironmentSnapshot (CM-32/33), InferenceBackend protocol, P0 bug fix
- PyTorch Parameter Audit — Backend config fields validated against upstream transformers/torch APIs
- Energy Measurement — NVML poller, Zeus/CodeCarbon optional, baseline power correction, warmup convergence, FLOPs estimation, timeseries
- Results Schema — ExperimentResult with EnergyBreakdown, JSON + Parquet persistence, late aggregation, output layout
- CLI — `llem run`, `llem config`, `--version`, plain text display (no Rich), exit codes, error hierarchy
- Testing — 258+ GPU-free unit tests, protocol injection fakes, test factories
- Gap Closure — Result wiring fixes (timeseries, effective_config, baseline), `extra="forbid"`, Phase 2 verification

**Archive:** `milestones/v1.17.0-ROADMAP.md`, `milestones/v1.17.0-REQUIREMENTS.md`

---

### v1.18.0 — M2: Study / Sweep (Phases 9–15)

**Shipped:** 2026-02-27
**Phases:** 9, 10, 11, 12, 14, 15 (6 phases + 2 gap closure)
**Plans completed:** 11
**Timeline:** 2026-02-27 (single day)
**Git range:** 2003c6f..HEAD (29 files changed, 3,011 insertions, 105 deletions)

**Delivered:** `llem run study.yaml` runs a multi-experiment sweep with subprocess isolation, cycle ordering, thermal gaps, and a checkpoint manifest — producing per-experiment ExperimentResult files and a StudyResult summary.

What shipped:
- Grid Expansion — Sweep YAML grammar with Cartesian grid, dotted notation (`pytorch.batch_size: [1, 8]`), study_design_hash, cycle ordering
- Manifest Writer — StudyManifest checkpoint with atomic `os.replace()` writes after every state transition
- Subprocess Isolation — `multiprocessing.get_context("spawn")`, Pipe IPC, SIGKILL timeout, structured failure results
- SIGINT Handling — Two-stage interrupt, manifest preservation, thermal gap countdown with Enter-to-skip, exit 130
- Integration — `run_study()` public API, `_run()` dispatcher, CLI study flags (`--cycles`, `--order`, `--no-gaps`), StudyResult assembly
- Gap Closure — Fixed double `apply_cycles()`, per-config cycle tracking, manifest completion status, progress display wiring

**Phase 13 (Documentation):** Deferred to end of M3 — docs will be written once all backends are complete.

**Archive:** `milestones/v1.18.0-ROADMAP.md`, `milestones/v1.18.0-REQUIREMENTS.md`

---

## Current

### M3 — Docker Multi-Backend

**Goal:** TBD — Docker container lifecycle, GPU passthrough, multi-backend execution.

---

## Planned

### M4 — Advanced Features

See `.product/ROADMAP.md` for full milestone definitions.
