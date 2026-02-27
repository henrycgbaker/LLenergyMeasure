# LLenergyMeasure

## What This Is

A measurement tool that maps the LLM deployment parameter space — batch size, quantisation, precision, backend, parallelism, KV cache settings, and other implementation choices — and quantifies how each affects energy consumption, throughput, latency, and FLOPs. Holds the model and workload constant; varies only the deployment configuration. The core insight: a single parameter change (e.g. batch size) can produce a 7.5x energy difference on the same model, yet no tool systematically measures this space.

Dual-product vision: CLI tool for the research community (v2.0), web platform for policy makers and the public (deferred). The tool serves research and policy advocacy — demonstrating that implementation choices, not just model selection, drive energy efficiency.

## Core Value

Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.

## Current Milestone: M1 — Core Single-Experiment

**Goal:** `llem run --model meta-llama/Llama-3.1-8B` produces a complete `ExperimentResult` with energy measurement, baseline correction, warmup, FLOPs, environment snapshot, JSON + Parquet output. PyTorch backend, local execution only.

**Target features:**
- Library-first restructure (`import llenergymeasure`, `__init__.py` public API)
- `ExperimentConfig` composition model (Pydantic v2, backend sections, field renames)
- PyTorch inference backend (with P0 model_kwargs bug fix)
- Energy measurement (NVML base, Zeus optional, CodeCarbon optional)
- Baseline power + warmup + FLOPs estimation
- `ExperimentResult` schema (all v2.0 fields)
- `llem run` + `llem config` + `llem --version` CLI
- Plain text output, pre-flight checks, error hierarchy
- Dead code removal (1,524 lines), test infrastructure

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

<!-- M1 scope — Core Single-Experiment, PyTorch local. Full list in .product/REQUIREMENTS.md -->

See `.product/REQUIREMENTS.md` for authoritative M1 requirement list (~90 requirements tagged M1).

**Library API:** LA-01 through LA-10
**Config:** CFG-01 through CFG-10, CFG-17 through CFG-26
**Core Measurement:** CM-01, CM-04 through CM-06, CM-11 through CM-34
**Results:** RES-01 through RES-12, RES-16 through RES-21
**CLI:** CLI-01 through CLI-04, CLI-06 through CLI-14
**Infrastructure:** STU-05, INF-01 through INF-12, INF-18 through INF-20

### Out of Scope

<!-- Explicit boundaries for M1. -->

- Study/sweep execution (M2) — needs single experiment working first
- Docker multi-backend (M3) — PyTorch local only in M1
- Traffic simulation, streaming latency, study resume (M4) — advanced features
- lm-eval integration (v3.0) — quality metrics not v2.0 scope
- Web platform (v4.0) — separate product, separate repo
- Bootstrap CIs (v2.1) — raw measurement is primary
- Singularity/Apptainer runner (v2.1+) — NotImplementedError in v2.0
- Shareability/upload (post-v2.0) — trust model unresolved

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
*Last updated: 2026-02-26 after M1 milestone initialisation*
