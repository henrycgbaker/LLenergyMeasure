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

Last phase number: 4.5 (but renumbering from Phase 1 for M1)

---

## Current

### M1 — Core Single-Experiment (Phase 1+)

**Started:** 2026-02-26
**Goal:** `llem run --model meta-llama/Llama-3.1-8B` produces a complete ExperimentResult with energy measurement, baseline correction, warmup, FLOPs, environment snapshot, JSON + Parquet output.
**Source:** `.product/ROADMAP.md` M1 scope, `.product/REQUIREMENTS.md` M1 requirements

---

## Planned

### M2 — Study / Sweep
### M3 — Docker Multi-Backend
### M4 — Advanced Features

See `.product/ROADMAP.md` for full milestone definitions.
