# Product Redesign Planning

**This directory is the single source of truth for LLenergyMeasure product decisions.**

It supersedes all other planning documents in this repository (esp: `./planning`).

---

## What This Is

This is the output of Phase 4.5 (Strategic Reset) — a complete product vision and
architecture redesign for LLenergyMeasure. It is NOT just pre-work for Phase 5.

The existing codebase (v1.x) accumulated significant technical debt and misaligned
product direction over ~10 development phases. This redesign:

- Establishes the correct product vision (CLI library + future web platform)
- Redefines the CLI surface from 15 commands to 3 (`llem run`, `llem study`, `llem config`)
- Specifies a library-first architecture (`run_experiment()`, `run_study()`)
- Defines a research-grade result schema and measurement methodology
- Resolves all major architectural ambiguities before implementation begins

Everything here has been designed against peer codebases (lm-eval, optimum-benchmark,
Zeus, CodeCarbon, AIEnergyScore, vLLM, MLflow, Hydra, W&B, etc.) and validated through
extended facilitated decision sessions.

---

## Structure

```
decisions/       Architecture, API, CLI, and product decisions (canonical)
designs/         Detailed design specs — config formats, schemas, CLI commands
research/        Peer codebase research that informed decisions
NEEDS_ADDRESSING.md   Known gaps, TODOs, items requiring further resolution
RESEARCH-LANDSCAPE.md Overview of the peer tool ecosystem
TODO.md          Work plan: what needs to happen before Phase 5 can be planned
README.md        This file
CLAUDE.md        Instructions for Claude Code when working in this directory
```

---

## Source of Truth Hierarchy

1. **This directory** — highest priority, all confirmed decisions
2. **Codebase** — current v1.x implementation (being superseded)
3. **`.planning/phases/`** — historical records of prior phases (do not use for planning)
4. **`.planning/PROJECT.md`, `.planning/ROADMAP.md`** — STALE, do not use

---

## Status

**In progress** — see `TODO.md` for what remains before Phase 5 planning can begin.

Specifically: manual inspection, consistency review, further decision areas, and eventually
full rewrites of `PROJECT.md`, `ROADMAP.md`, and root `CLAUDE.md` to reflect this redesign.
