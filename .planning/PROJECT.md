# LLenergyMeasure

## What This Is

A measurement tool that maps the LLM deployment parameter space — batch size, quantisation, precision, backend, parallelism, KV cache settings, and other implementation choices — and quantifies how each affects energy consumption, throughput, latency, and FLOPs. Holds the model and workload constant; varies only the deployment configuration. The core insight: a single parameter change (e.g. batch size) can produce a 7.5x energy difference on the same model, yet no tool systematically measures this space.

Dual-product vision: CLI tool for the research community (v2.0), web platform for policy makers and the public (deferred). The tool serves research and policy advocacy — demonstrating that implementation choices, not just model selection, drive energy efficiency.

## Core Value

Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.

## Current Milestone: M2 — Study / Sweep

**Goal:** `llem run study.yaml` runs a multi-experiment sweep with subprocess isolation, cycle ordering, thermal gaps, and a checkpoint manifest — producing per-experiment `ExperimentResult` files and a `StudyResult` summary. Single backend (PyTorch) only in M2.

**Target features:**
- `StudyConfig` + `ExecutionConfig` resolved from sweep YAML
- Sweep grammar: dotted notation `pytorch.batch_size: [1, 8]` grid expansion
- `StudyRunner` with `multiprocessing.get_context("spawn")` subprocess isolation
- IPC via `Pipe` for result return + `Queue` for progress events
- `StudyManifest` always-on checkpoint after each experiment
- Cycle ordering (`sequential` | `interleaved`), thermal gap management
- `StudyResult` + `run_study()` public API
- CLI study flags (`--cycles`, `--no-gaps`, `--order`)
- Study output layout: `{name}_{timestamp}/` with per-experiment subdirs + manifest

## Requirements

### Validated

<!-- Shipped and confirmed valuable — capabilities proven in v1.x. -->

- ✓ Single experiment execution across 3 inference backends (PyTorch, vLLM, TensorRT-LLM)
- ✓ Docker container execution for backend isolation
- ✓ Campaign/multi-config execution with scheduling strategies
- ✓ Energy tracking via CodeCarbon
- ✓ Result persistence (JSON) with raw per-process and aggregated results
- ✓ State management with resume capability
- ✓ Configuration loading/validation via Pydantic with backend-specific configs
- ✓ GPU routing and multi-GPU detection
- ✓ Basic metrics: latency (TTFT, TPOT), throughput (tokens/sec), energy (Wh, CO2)
- ✓ Extended metrics: GPU utilisation sampling, compute metrics
- ✓ CLI via Typer with Rich output
- ✓ Product planning complete — 120+ requirements, 4 milestones, all ADRs finalised

### Active

<!-- M2 scope — Study/Sweep execution. Full list in .planning/REQUIREMENTS.md -->

See `.planning/REQUIREMENTS.md` for authoritative M2 requirement list (24 requirements).

**Config/Sweep:** CFG-11 through CFG-16
**Study Execution:** STU-01 through STU-04, STU-06, STU-07, STU-NEW-01
**Manifest:** STU-08, STU-09
**Results:** RES-13 through RES-15, RES-NEW-01
**CLI:** CLI-05, CLI-11
**Core Measurement:** CM-10
**Library API:** LA-02, LA-05

### Out of Scope

<!-- Explicit boundaries for M2. -->

- Docker multi-backend (M3) — single backend only in M2
- `--resume` flag (M4) — manifest is always-on but resume is deferred
- IPC file-based fallback — Pipe-only; dropped from scope
- Traffic simulation, streaming latency (M4) — advanced features
- lm-eval integration (v3.0) — quality metrics not v2.0 scope
- Web platform (v4.0) — separate product, separate repo

## Context

**Existing codebase**: ~22,000 lines of Python. Modular Typer CLI with layered architecture (CLI → Orchestration → Core → Domain). Three inference backends. Campaign execution with Docker dispatch. Known issues: 4 P0 bugs, ~1,524 lines confirmed dead code. Codebase audit at `.planning/phases/04-codebase-audit/AUDIT-REPORT.md`.

**Product design (SSOT)**: `.product/` directory — 20+ decision files, 16+ research files, detailed designs, 47-feature preservation audit. All decisions finalised. `.product/REQUIREMENTS.md` and `.product/ROADMAP.md` are authoritative.

**Primary audience**: Research community — academics and ML engineers studying LLM inference efficiency.

**Key use case**: Broad parameter sweeps across deployment configurations. Results used for research papers and policy advocacy.

## Constraints

- **Platform**: Linux required for vLLM and TensorRT-LLM backends. NVIDIA GPU with CUDA required.
- **Python**: 3.10+ (TensorRT-LLM compatibility requirement)
- **Backend isolation**: vLLM and TensorRT-LLM cannot coexist in the same process — subprocess or Docker isolation is a correctness requirement.
- **Methodology**: Measurements must be scientifically sound — warmup, thermal stabilisation, statistical confidence. A researcher must be able to cite results in a paper.
- **Dependency management**: Zero backend deps at base install. Backends as optional extras (`[pytorch]`, `[vllm]`, `[tensorrt]`).

## Key Decisions

<!-- Full decision log in .product/decisions/. All finalised after deep peer research. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v2.0 delivered across 4 milestones (M1–M4) | Each milestone ships usable product; no separate v2.x versions | ✓ Good |
| Library-first (`import llenergymeasure`) | Industry norm (lm-eval, MLflow, CodeCarbon); CLI is thin wrapper | ✓ Good |
| 2 commands + 1 flag (`llem run`, `llem config`, `--version`) | Peer research: simplest viable CLI | ✓ Good |
| Composition config (not inheritance) | `ExperimentConfig` with optional backend sections; cleaner than hierarchy | ✓ Good |
| Option C architecture (experiment-study) | `_run(StudyConfig)` always; public API wraps/unwraps | ✓ Good |
| No union return types | 0 peer tools use union returns; Python typing guidance forbids it | ✓ Good |
| `llem` rename (no `lem` shim) | Clean break at v2.0 | ✓ Good |
| No default backend at base install | Each backend is explicit extra | ✓ Good |
| All product decisions finalised | `.product/` SSOT complete; ready for implementation | ✓ Good |
| Web platform deferred to v4.0 | Separate product, separate repo | ✓ Good |
| lm-eval deferred to v3.0 | Quality metrics not essential for v2.0 scope | ✓ Good |

---
*Last updated: 2026-02-27 after M2 milestone initialisation*
