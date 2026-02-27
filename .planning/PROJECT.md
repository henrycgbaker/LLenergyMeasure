# LLenergyMeasure

## What This Is

A measurement tool that maps the LLM deployment parameter space — batch size, quantisation, precision, backend, parallelism, KV cache settings, and other implementation choices — and quantifies how each affects energy consumption, throughput, latency, and FLOPs. Holds the model and workload constant; varies only the deployment configuration. The core insight: a single parameter change (e.g. batch size) can produce a 7.5x energy difference on the same model, yet no tool systematically measures this space.

Dual-product vision: CLI tool for the research community (v2.0), web platform for policy makers and the public (deferred). The tool serves research and policy advocacy — demonstrating that implementation choices, not just model selection, drive energy efficiency.

## Core Value

Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.

## Current Milestone: v1.19.0 — M3: Docker + vLLM

**Goal:** Docker container infrastructure with ephemeral per-experiment lifecycle, vLLM backend activation, Docker pre-flight validation, GPU memory cleanup, and full user documentation including Docker setup guide.

## Requirements

### Validated

<!-- Shipped and confirmed valuable — capabilities proven across M1 + M2. -->

**M1 — Core Single-Experiment (v1.17.0):**
- ✓ Package foundation with zero-dep base install, `llem` entry point — v1.17.0
- ✓ ExperimentConfig composition model with `extra="forbid"`, YAML loader, SSOT introspection — v1.17.0
- ✓ `run_experiment()` public API, `_run(StudyConfig)` internal dispatcher, API stability contract — v1.17.0
- ✓ PyTorch backend with pre-flight checks, EnvironmentSnapshot, InferenceBackend protocol — v1.17.0
- ✓ NVML energy poller, baseline power correction, warmup convergence, FLOPs estimation — v1.17.0
- ✓ ExperimentResult with EnergyBreakdown, JSON + Parquet persistence, late aggregation — v1.17.0
- ✓ CLI: `llem run`, `llem config`, `--version`, plain text display, error hierarchy — v1.17.0
- ✓ 258+ GPU-free unit tests with protocol injection fakes — v1.17.0

**M2 — Study / Sweep (v1.18.0):**
- ✓ StudyConfig + ExecutionConfig with sweep YAML grammar and Cartesian grid expansion — v1.18.0
- ✓ StudyManifest checkpoint with atomic writes after every state transition — v1.18.0
- ✓ Subprocess isolation via `multiprocessing.spawn`, Pipe IPC, SIGKILL timeout — v1.18.0
- ✓ SIGINT handling with manifest preservation and exit 130 — v1.18.0
- ✓ `run_study()` public API returning `StudyResult`, `_run()` dispatcher — v1.18.0
- ✓ CLI study flags: `--cycles`, `--order`, `--no-gaps` — v1.18.0
- ✓ Cycle ordering (sequential/interleaved), thermal gap management — v1.18.0
- ✓ Multi-backend study without Docker → hard error at pre-flight — v1.18.0

### Active

<!-- M3 scope — Docker + vLLM. Defined 2026-02-27. -->

- [ ] Docker ephemeral runner (`docker run --rm` per experiment, config via env/volume, result via shared volume)
- [ ] Docker pre-flight checks (NVIDIA Container Toolkit, GPU visibility inside container, CUDA/driver compat)
- [ ] vLLM inference backend activation (P0 fixes: streaming, shm-size)
- [ ] GPU memory cleanup between experiments (NVML check in both local and Docker paths)
- [ ] Docker images for vLLM backend (CI publish on release tag)
- [ ] Full user documentation including Docker setup guide
- [ ] `aienergyscore.jsonl` built-in dataset file (carried from M1)
- [ ] Confirm `peak_memory_mb` measurement semantics (carried from M1)

### Out of Scope

<!-- Explicit boundaries. -->

- TensorRT-LLM backend (M4/v1.20.0) — Docker infra reused, backend-specific activation
- SGLang backend (M5/v1.21.0) — RadixAttention energy profiles, Docker-isolated
- `--resume` flag — manifest is always-on but resume is deferred
- Traffic simulation, streaming latency — advanced features
- Persistent Docker containers — ephemeral only in v1.19.0
- `llem compile-engines` pre-compilation command — deferred
- lm-eval integration (v3.0) — quality metrics not v2.0 scope
- Web platform (v4.0) — separate product, separate repo

## Context

**Codebase**: ~24,674 lines of Python across `src/llenergymeasure/`. Modular architecture: CLI → API → Study → Core → Domain. Three inference backends (PyTorch active; vLLM and TensorRT-LLM pending Docker isolation in M3). 538 tests.

**Product design (SSOT)**: `.product/` directory — 20+ decision files, 16+ research files, detailed designs, 47-feature preservation audit. All decisions finalised.

**Primary audience**: Research community — academics and ML engineers studying LLM inference efficiency.

**Key use case**: Broad parameter sweeps across deployment configurations. Results used for research papers and policy advocacy.

**Shipped milestones**: v1.17.0 (M1 — single experiment), v1.18.0 (M2 — study/sweep). Full history in `.planning/MILESTONES.md`.

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
| Spawn-only subprocess isolation | Never `set_start_method()` globally; fork causes silent CUDA corruption | ✓ Good (M2) |
| Pipe-only IPC (file fallback dropped) | ExperimentResult fits in Pipe buffer; 1MB fallback over-engineering | ✓ Good (M2) |
| Manifest always-on, resume deferred | Ship checkpoint pattern in M2; resume logic is M4 | ✓ Good (M2) |
| Phase 13 docs deferred to M3 | Write docs once against final backend story | — Pending |
| One backend per milestone (M3–M5) | Each milestone activates one backend on shared Docker infra; reduces risk | — Pending |
| GPU memory cleanup between experiments | AIEnergyScore pattern; defensive measure for measurement purity | — Pending |

---
*Last updated: 2026-02-27 after M3 milestone setup*
