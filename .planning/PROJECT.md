# LLenergyMeasure

## What This Is

A measurement tool that maps the LLM deployment parameter space — batch size, quantisation, precision, backend, parallelism, KV cache settings, and other implementation choices — and quantifies how each affects energy consumption, throughput, latency, and FLOPs. Holds the model and workload constant; varies only the deployment configuration. The core insight: a single parameter change (e.g. batch size) can produce a 7.5x energy difference on the same model, yet no tool systematically measures this space.

Dual-product vision: CLI tool for the research community (v2.0), web platform for policy makers and the public (deferred). The tool serves research and policy advocacy — demonstrating that implementation choices, not just model selection, drive energy efficiency.

## Core Value

Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.

## Requirements

### Validated

<!-- Shipped and confirmed valuable — capabilities that exist in the v1.x codebase. -->

- ✓ Single experiment execution across 3 inference backends (PyTorch, vLLM, TensorRT-LLM) — existing
- ✓ Docker container execution for backend isolation — existing (but buggy — P0 fixes needed)
- ✓ Campaign/multi-config execution with scheduling strategies (interleaved, shuffled, grouped) — existing
- ✓ Energy tracking via CodeCarbon — existing
- ✓ Result persistence (JSON) with raw per-process and aggregated results — existing
- ✓ State management with resume capability — existing
- ✓ Configuration loading/validation via Pydantic with backend-specific configs — existing
- ✓ GPU routing and multi-GPU detection — existing
- ✓ Basic metrics: latency (TTFT, TPOT), throughput (tokens/sec), energy (Wh, CO2) — existing
- ✓ Extended metrics: GPU utilisation sampling, compute metrics — existing
- ✓ CLI via Typer with Rich output — existing (but needs restructuring)

### Active

<!-- v2.0 scope. All subject to revision during product harmonisation phase. -->

**Product Design (pre-implementation)**
- [ ] Fresh codebase map — current `.planning/codebase/` is stale
- [ ] Deep peer audit of all `.product/` decisions against 5-8 peer tools (lm-eval, MLflow, Zeus, CodeCarbon, Optimum-Benchmark, etc.)
- [ ] Harmonise `.product/` decisions and designs based on peer research — resolve inconsistencies, revisit all architectural choices (nothing is sacred)
- [ ] Fresh codebase audit — determine what v1.x code is salvageable vs. needs rewriting

**Architecture**
- [ ] Library-first restructure — `import llenergymeasure` as primary interface; CLI is a thin wrapper
- [ ] CLI surface redesign — peer research decides (unified `llem run` vs. separate commands)
- [ ] Config model redesign — peer research validates (three-layer vs. simpler model)
- [ ] Module structure — single `src/llenergymeasure/` package with CLI and study modules inside
- [ ] Stable public API via `__init__.py` exports only

**Core Measurement**
- [ ] Subprocess isolation — each experiment in a fresh `multiprocessing.Process` for clean GPU state
- [ ] Parameter sweep / study capability — broad parameter ranges across many params; the primary research use case
- [ ] Warmup strategy — methodology for thermal stabilisation before measurement
- [ ] Baseline power measurement — idle power subtraction for accurate energy attribution
- [ ] Bootstrap confidence intervals — statistical rigour for publishable results
- [ ] Thermal time-series — power/temperature data over time
- [ ] FLOPs estimation — analytical computation of operations per inference
- [ ] Measurement methodology metadata — stored with results for reproducibility

**Backends & Parameters**
- [ ] Docker multi-backend orchestration — cross-backend studies (vLLM, TRT-LLM in containers)
- [ ] Parameter completeness — PyTorch/vLLM/TRT configs to 90%+ of available parameters
- [ ] Zeus energy backend — more accurate NVML energy measurement as optional extra
- [ ] Prefill/decode phase split — separate metrics for prompt processing vs. generation

**Results & Output**
- [ ] Result schema redesign — `ExperimentResult` with config hash, methodology metadata, steady-state window
- [ ] Study results — `StudyResult` aggregating multiple experiments with study design metadata
- [ ] Visual exploration of results — charts/plots for parameter space inspection
- [ ] Human-readable output filenames and directory structure

**Quality**
- [ ] P0 bug fixes (4 bugs from v1.x audit)
- [ ] Dead code removal (~1,524 lines confirmed removable)
- [ ] Code quality audit milestones — checkpoints during implementation to verify correctness
- [ ] Test coverage at library API boundary

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Web platform — deferred post-v2.0; requires validated CLI results and resolved GPU provisioning model
- lm-eval integration (quality + efficiency tradeoff) — deferred to v3.0; v2.0 measures energy/throughput/latency, not model quality
- Shareability (results upload, central DB, HuggingFace export) — deferred post-v2.0; trust model unresolved
- Statistical sensitivity analysis (ANOVA, variance attribution) — deferred post-v2.0; visual exploration first, statistical tools later
- HPC/SLURM/Apptainer support — deferred post-v2.0; complexity not warranted until core is stable
- SGLang backend — deferred; focus on existing 3 backends first
- Model comparison leaderboard — not our positioning; ML.ENERGY and HF leaderboard already serve this

## Context

**Existing codebase**: ~22,000 lines of Python. Modular Typer CLI with layered architecture (CLI → Orchestration → Core → Domain). Three inference backends. Campaign execution with Docker dispatch. Known issues: 4 P0 bugs, ~1,524 lines confirmed dead code. Codebase audit at `.planning/phases/04-codebase-audit/AUDIT-REPORT.md`.

**Product design**: Extensive ADR-format documentation in `.product/` (20+ decision files, 16+ research files, detailed designs, 47-feature preservation audit). Developed across 6+ sessions. Contains inconsistencies and draft decisions that need harmonisation. **Nothing in `.product/` is sacred** — all decisions are revisitable based on peer research evidence.

**Stale artefacts**: `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/codebase/` map — all based on the old product model and need replacement.

**Primary audience**: Research community — academics and ML engineers studying LLM inference efficiency. The tool must produce publishable, methodology-sound results.

**Key use case**: Broad parameter sweeps across deployment configurations. Not "does batch size matter?" but "across ALL implementation parameters, which ones matter most, and how much variance does each contribute?" Results used for research papers and policy advocacy.

**Empirical anchor**: ML.ENERGY's data shows a 7.5x energy difference from batch size alone. No existing tool systematically quantifies this parameter space.

## Constraints

- **Platform**: Linux required for vLLM and TensorRT-LLM backends. NVIDIA GPU with CUDA required.
- **Python**: 3.10+ (TensorRT-LLM compatibility requirement)
- **Backend isolation**: vLLM and TensorRT-LLM cannot coexist in the same process (CUDA/driver conflicts) — subprocess or Docker isolation is a correctness requirement, not a preference.
- **Methodology**: Measurements must be scientifically sound — warmup, thermal stabilisation, statistical confidence. A researcher must be able to cite results in a paper.
- **Dependency management**: Zero backend deps at base install. Backends as optional extras (`[pytorch]`, `[vllm]`, `[tensorrt]`).

## Key Decisions

<!-- Decisions from project initialisation. Full decision log in .product/decisions/ — all subject to peer review during harmonisation. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v2.0 = complete CLI research tool | No micro-versioning (the old v2.0–v2.4 split was wrong); ship the full tool as one release with audit milestones along the way | — Pending |
| Everything revisitable via peer research | No decision from `.product/` is sacred; deep peer audit against 5-8 tools decides architecture | — Pending |
| Product harmonisation before implementation | `.product/` has inconsistencies and draft decisions that must be resolved before coding starts | — Pending |
| Fresh codebase map + audit | Old `.planning/codebase/` is stale; new audit determines what's salvageable vs. rewrite | — Pending |
| Web platform deferred | Dual-product vision confirmed, but web is post-v2.0 | — Pending |
| lm-eval deferred to v3.0 | Quality metrics not essential for v2.0; energy/throughput/latency is the v2.0 scope | — Pending |
| Shareability deferred post-v2.0 | Results upload, central DB, HF export — trust model unresolved | — Pending |
| Visual exploration in v2.0, statistical analysis later | Charts/plots for parameter space inspection are v2.0; ANOVA/sensitivity analysis is post-v2.0 | — Pending |
| Audit milestones = code quality checks | Milestones between implementation phases check correctness, not revisit product direction | — Pending |

---
*Last updated: 2026-02-25 after project reinitialisation*
