# LLenergyMeasure

## What This Is

LLM inference efficiency measurement framework for benchmarking energy consumption, throughput, and FLOPs across HuggingFace models. CLI-driven tool with three backend engines (PyTorch/Transformers, vLLM, TensorRT-LLM), Docker-based isolation, and multi-cycle statistical experiments.

**Core Value:** Accurate, comprehensive measurement of the true cost of LLM inference — energy, compute, and quality tradeoffs — with research-grade rigour.

## Who It's For

- ML researchers benchmarking inference efficiency
- Practitioners optimising deployment configurations
- The open-source ML community (future: public leaderboard)

## Current State (Brownfield)

Substantial CLI tool with layered architecture: CLI → Orchestration → Core → Domain.

**Architecture patterns:** Dependency injection via protocols, late aggregation (per-process results saved separately), SSOT config (parameter metadata derived from Pydantic models), distributed execution via Accelerate/torchrun.

**Stack:** Python 3.10+, Typer CLI, Pydantic 2.0+, PyTorch 2.5, CodeCarbon (energy), Docker (backend isolation), Poetry.

## Requirements

### Validated

- ✓ Multi-backend inference (PyTorch/Transformers, vLLM, TensorRT-LLM) — existing
- ✓ Backend-specific configs via Pydantic models (PyTorchConfig, VLLMConfig, TensorRTConfig) — existing
- ✓ YAML-based experiment configuration with preset inheritance — existing
- ✓ Energy measurement via CodeCarbon (GPU/CPU/RAM breakdown, CO₂) — existing
- ✓ Throughput metrics (tokens/sec), latency distributions — existing
- ✓ FLOPs estimation with confidence-tracked fallback chain (calflops → architecture → parameter) — existing
- ✓ Streaming latency metrics (TTFT, ITL) for all backends — existing
- ✓ Multi-cycle experiments with statistical robustness (mean, std, 95% CI, CV) — existing
- ✓ Campaign orchestration (multiple configs, cycles, randomisation, thermal gaps) — existing
- ✓ Docker containerisation per backend (conflicting deps require isolation) — existing
- ✓ Tensor parallelism and pipeline parallelism support — existing
- ✓ MIG GPU topology detection — existing
- ✓ LoRA adapter loading — existing
- ✓ Decoder sampling presets (deterministic, standard, creative, factual) — existing
- ✓ Industry-standard batching strategies (static, dynamic, sorted) — existing
- ✓ Traffic simulation (Poisson + constant arrival) — existing
- ✓ Grid config generation (`lem config generate-grid`) — existing
- ✓ Late aggregation pattern (per-process raw results, on-demand aggregation) — existing
- ✓ SSOT introspection (auto-discover params from Pydantic models) — existing
- ✓ Parameter provenance tracking — existing
- ✓ Experiment resumption for interrupted runs — existing
- ✓ 416+ tests (unit, integration, e2e) — existing

### Active

#### v1.19.0 — Measurement Foundations

- [ ] **MEAS-01**: Capture environment metadata per experiment (CUDA version, driver version, GPU thermal state, power limits, CPU governor, container detection)
- [ ] **MEAS-02**: Measure idle GPU baseline power before experiments; report both raw and baseline-adjusted energy readings (raw preserved for backwards compatibility, adjusted fixes 15-30% systematic overestimation)
- [ ] **MEAS-03**: Detect and flag thermal throttling during experiments via NVML performance state monitoring
- [ ] **MEAS-04**: Collect time-series power/memory/utilisation samples at configurable intervals (extend existing GPUUtilisationSampler)
- [ ] **MEAS-05**: Implement warmup convergence detection (continue until CV stabilises, not fixed prompt count)
- [ ] **MEAS-06**: Update results schema to v3 with migration path from v2 results
- [ ] **MEAS-07**: Add extended efficiency metrics to CSV export (currently JSON-only; memory, GPU utilisation, request latency, batch, KV cache metrics)
- [ ] **MEAS-08**: Implement bootstrap resampling for 95% confidence intervals (enhance existing multi-cycle statistics)
- [ ] **MEAS-09**: UAT round 1 (fresh clone, install, run one experiment, document pain points)

#### v1.20.0 — Campaign Orchestrator

- [ ] **CAMP-01**: Switch from `docker compose run --rm` to long-running containers with `docker compose exec` dispatch
- [ ] **CAMP-02**: Backend-aware programmatic grid generation (cartesian product respecting per-backend param validity)
- [ ] **CAMP-03**: Campaign manifest tracking (exp_id → config → backend → container → status → result_path)
- [ ] **CAMP-04**: Daemon mode for scheduled cycles (run at specific times, not just sequential with delays)
- [ ] **CAMP-05**: Configurable `force_cold_start` option (unload model between experiments for cold-start benchmarking; default: warmup ensures fairness)
- [ ] **CAMP-06**: Container health checks and lifecycle management (start only needed backends, graceful teardown)
- [ ] **CAMP-07**: Retain existing campaign features (randomisation within cycles, interleaved/shuffled/grouped structures, thermal gaps, cycle-as-highest-organisational-principle)
- [ ] **CAMP-08**: UAT round 2 — validate orchestrator dispatches cross-backend campaigns correctly

#### v1.21.0 — Parameter Completeness

- [ ] **PARAM-01**: Close known parameter gaps (PyTorch 93.8% → 95%+, vLLM 81.9% → 90%+, TensorRT 93.8% → 95%+)
- [ ] **PARAM-02**: Systematic audit of energy/throughput-impactful kwargs for each backend
- [ ] **PARAM-03**: Validate `extra:` escape hatch works for undocumented/niche parameters
- [ ] **PARAM-04**: Update SSOT introspection to auto-discover any new parameters
- [ ] **PARAM-05**: UAT round 3 — use campaign orchestrator to run parameter audit configs across all backends, verify results

#### v1.22.0 — Polish + UAT

- [ ] **UAT-01**: Occam's razor cleanup pass (dead code, naming consistency, simplify over-engineered areas)
- [ ] **UAT-02**: Architectural refactor if pain points emerge during cleanup
- [ ] **UAT-03**: UAT round 2 (full campaign workflow across backends, end-to-end)
- [ ] **UAT-04**: Documentation refresh (README, quickstart, deployment guides)
- [ ] **UAT-05**: Docker image rebuild and validation

### v2.0.0 — Fully Working CLI (Release Milestone)

v1.19–v1.22 culminate here. Tag, changelog, Docker images built from tag.

### v2.1.0 — Precision Metrics

- [ ] **PREC-01**: Prefill/decode phase split (separate timing, energy, throughput per phase — PyTorch first, other backends best-effort)
- [ ] **PREC-02**: Profiler integration (optional `torch.profiler` mode for ground-truth FLOPs — PyTorch only)

### v3.0.0 — Web Platform + Analysis (Separate Product)

- [ ] Web interface for running experiments and viewing results
- [ ] Analysis module: quality metrics (perplexity), causal analysis of efficiency drivers, tradeoff visualisation, Pareto frontier analysis
- [ ] Stack TBD closer to the time (likely FastAPI + React)
- [ ] Separate repo, own lifecycle

### Out of Scope

- MLflow/W&B integrations — revisit after v2.0.0, low priority
- SSM/hybrid architecture support (Mamba, RWKV, Jamba) — future, dependent on research community adoption
- 100% exhaustive kwargs coverage — 90%+ of energy-impactful params with `extra:` escape hatch for the long tail
- PyPI distribution — install from git is sufficient for research tools

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Split v2 into 4 sub-releases (v1.19–v1.22) | Each independently shippable, reduces risk, enables early UAT | Adopted |
| Measurement foundations before parameter work | Fixes systematic 15-30% energy bias; accurate data before more features | Adopted |
| Campaign orchestrator as own release | Highest-risk item (Docker exec model), shouldn't block measurement improvements | Adopted |
| Long-running containers with `docker compose exec` | Avoids per-experiment container startup overhead; containers stay warm, shared volumes for results | Adopted |
| Backend-aware grid generation | SSOT already knows which params belong to which backend; grid respects validity | Adopted |
| Warmup convergence (CV-based) over fixed count | Scientifically more robust; existing CycleStatistics already tracks CV | Adopted |
| Thermal throttling detection via NVML | Cheap to implement, high diagnostic value, flags measurement anomalies | Adopted |
| 90%+ targeted param coverage (not 100%) | "ALL kwargs" is infinite scope; `extra:` escape hatch handles edge cases | Adopted |
| Web platform as separate product (v3.0.0) | Different tech stack, different lifecycle; shares Pydantic models but otherwise independent | Adopted |
| Option B versioning (relabel existing v2.0.0 → v1.17.0) | Clean semver: v1.0.0 = thesis, v1.x = build phase, v2.0.0 = fully working CLI | Adopted |

## Versioning

```
v0.5.0  (Mar 2025)   Core measurement foundation
v1.0.0  (May 2025)   Thesis submission (tag: 7c9904b)
v1.10–v1.16          Post-thesis rebuild (arch refactor, CLI, tests, Docker)
v1.17.0              Relabelled from existing v2.0.0 (refactored CLI with DI, late aggregation)
v1.18.0              Tag current state (3 backends, campaigns, SSOT, streaming, extended metrics)
v1.19.0              Measurement Foundations
v1.20.0              Campaign Orchestrator
v1.21.0              Parameter Completeness
v1.22.0              Polish + UAT
─── v2.0.0 ───       Fully Working CLI release milestone
v2.1.0               Precision Metrics (prefill/decode, profiler)
─── v3.0.0 ───       Web Platform + Analysis (separate product)
```

## Constraints

- **Backend isolation**: vLLM and TensorRT-LLM have conflicting dependencies — must use separate Docker containers
- **GPU requirement**: NVIDIA GPU required for any inference; TensorRT requires Ampere+ (compute capability >= 8.0)
- **Simplicity first**: Occam's razor — minimum viable solution, no over-engineering, no premature abstractions
- **Research rigour**: Statistical validity, reproducibility, clear methodology documentation
- **Minimal dependencies**: Don't bloat the core tool

## Campaign Orchestrator Design (v1.21.0)

### Container Lifecycle: Long-Running Services

```
Host (orchestrator)
    |
    +-- docker compose up -d pytorch vllm    <-- start only needed backends
    |
    +-- Experiment 0001 (pytorch config)
    |   +-- docker compose exec pytorch lem experiment <config>
    |
    +-- Experiment 0002 (vllm config)
    |   +-- docker compose exec vllm lem experiment <config>
    |
    +-- Experiment 0003 (pytorch config)
    |   +-- docker compose exec pytorch lem experiment <config>
    |
    +-- docker compose down                  <-- teardown after campaign
```

### Campaign Definition: Two Modes

**Mode 1 — Explicit config list** (existing, retained):
```yaml
campaign_name: "cross-backend-comparison"
configs:
  - configs/pytorch_fp16.yaml
  - configs/vllm_fp16.yaml
  - configs/tensorrt_fp16.yaml
```

**Mode 2 — Programmatic grid** (new, backend-aware):
```yaml
campaign_name: "full-backend-sweep"
base_config:
  model:
    name: "meta-llama/Llama-3.1-8B"
    fp_precision: fp16
  max_output_tokens: 256
vary:
  backend: [pytorch, vllm]
  batching:
    batch_size: [1, 4, 8]
# Generates 2 x 3 = 6 experiments, routed by backend
```

### Shared Volume Architecture

```
Host filesystem (mounted into ALL containers):
+-- results/     --> /app/results
+-- configs/     --> /app/configs
+-- .state/      --> /app/.state (experiment ID counter)
```

### Warmup Fairness

Default: warmup prompts before each experiment ensure steady-state GPU before measurement. Warmup time is not measured — both first and subsequent experiments start measurement in the same warm state.

Configurable: `force_cold_start: true` unloads model between experiments for cold-start benchmarking.

## To Discuss

Items from legacy planning docs. Triaged and assigned to releases or deferred.

### Streaming — split across releases

**v1.22.0 (Polish):**
- Document how streaming works internally (TTFT vs ITL, token-by-token generation)
- Clarify what TTFT samples vs ITL samples represent
- Investigate why streaming results have different structure (latency_measurements)

**v2.1.0 (Precision):**
- TTFT component breakdown (tokenisation/prefill/decode phases)
- Energy-per-token correlation (TTFT energy vs decode energy)

**v3.0.0 (Web Platform):**
- Real-time CLI/web UI token display during inference

**Deferred (revisit when relevant):**
- vLLM queue time separation (isolate scheduling delay from compute)
- Speculative decoding acceptance rate metrics
- KV cache hit rate metrics — already partially in ExtendedEfficiencyMetrics

### Quality Metrics — v3.0.0 analysis module

- Perplexity computation (on generated outputs + held-out eval sets)
- Pareto frontier analysis (quality vs efficiency tradeoffs)
- External quality score integration (BLEU, ROUGE, accuracy via user-provided files)

### Analysis Features — v3.0.0 analysis module

- Correlation analysis: parameter values → energy/efficiency metrics
- Multi-experiment comparison dashboard
- Automated parameter sweep with early stopping and Pareto identification
- NormalisedMetrics population (precision-adjusted FLOPs, cross-backend normalisation — needs PrecisionMetadata collection from backends)

### Ecosystem Integration — deferred (post-v3)

- MLflow integration (log params, metrics, artifacts)
- Weights & Biases integration (W&B Tables for structured results)

### Web Platform Details — v3.0.0 planning input

- Framework comparison: Streamlit vs Gradio vs Dash vs FastAPI+React (decision: FastAPI+React, but consider Streamlit prototype first)
- Visualisation requirements: efficiency scatter, comparison bar charts, multi-cycle box plots, configuration sensitivity heatmap, hardware scaling curves
- Libraries: Plotly.js (primary), Recharts, Tanstack Table
- Worker architecture: Celery + Redis job queue, self-hosted GPU workers connecting outbound
- Data layer: PostgreSQL + S3/MinIO + Redis
- Deployment: Railway/Render (API) + Vercel (frontend)
- Existing inspiration: LLM-Perf Leaderboard, Open LLM Leaderboard, Artificial Analysis Leaderboard
- Open questions: auth provider, monorepo vs split, result validation, hardware standardisation, pricing

### Future — deferred

- SSM/hybrid architecture support (Mamba, RWKV, Jamba, StripedHyena, Griffin)
- Cross-backend comparison CLI: `lem compare --backends pytorch,vllm`

### Discarded

- ~~Parallelism Refactor (unified config block, capability matrix)~~ — each backend handles parallelism natively; unified abstraction adds complexity without clear value

---
*Last updated: 2026-01-29 after initialization*
