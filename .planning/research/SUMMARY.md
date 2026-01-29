# Project Research Summary

**Project:** LLenergyMeasure Milestone — Precision Enhancement
**Domain:** LLM inference efficiency measurement and benchmarking
**Researched:** 2026-01-29
**Confidence:** HIGH

## Executive Summary

LLenergyMeasure's v2.0 CLI milestone focuses on precision enhancement through three core areas: energy measurement accuracy, comprehensive system metadata capture, and backend parameter completeness. Research reveals that whilst energy measurement is a strong differentiator (only 4 tools in the market measure energy), LLenergyMeasure has critical table stakes gaps that undermine credibility: missing baseline power subtraction leads to 15-30% systematic energy overestimation, inadequate environment metadata prevents reproducibility, and backend parameter coverage is incomplete.

The recommended approach involves foundational measurement improvements first (baseline power, time-series sampling, thermal monitoring), followed by systematic parameter auditing with version pinning, then campaign orchestration redesign using long-running Docker containers. The architecture should favour host-side Python orchestration with Docker SDK (python-on-whales) managing backend containers via `docker exec` dispatch, avoiding container recreation overhead whilst maintaining clean experiment isolation.

Critical risks include: idle power contamination invalidating energy comparisons, NVML sampling blind spots causing high variance on short experiments, thermal throttling masking true performance, vLLM/TensorRT API instability breaking parameter configs, and container state leakage accumulating memory across campaigns. These are mitigated through baseline subtraction, multi-cycle statistical aggregation, thermal state monitoring, exact version pinning, and health-checked long-running containers with explicit cleanup.

## Key Findings

### Recommended Stack

The technology stack prioritises official, actively maintained libraries with proven stability. Core monitoring uses nvidia-ml-py (v13.590.48) for NVML GPU monitoring — the official NVIDIA bindings that supersede deprecated pynvml. Campaign orchestration uses python-on-whales (0.70+) over docker-py due to superior Docker Compose exec support, CLI feature parity, and thread-safe stateless design endorsed by Docker's official blog.

**Core technologies:**
- **nvidia-ml-py (13.590.48)** — GPU power, thermal, memory monitoring via NVML — Official NVIDIA bindings, actively maintained, Jan 2026 release
- **python-on-whales (0.70+)** — Docker Compose orchestration — CLI parity, native `docker compose exec`, thread-safe, Docker-endorsed
- **transformers (5.0+)** — PyTorch `generate()` parameter introspection — Official HuggingFace library, ~50+ parameters
- **vllm (0.8.1+)** — vLLM backend parameters — Official vLLM library, 25+ constructor parameters
- **tensorrt-llm (0.12.0+)** — TensorRT backend parameters — Official NVIDIA library, 60+ constructor parameters

**Version compatibility:** Exact version pinning required for vLLM/TensorRT (pre-1.0 API stability). nvidia-ml-py ≥13.590.48 for latest NVML functions. Python 3.10+, CUDA 12.x, Docker 20.10+ with Compose V2.

### Expected Features

The LLM benchmarking landscape has matured into three tiers: industry standards (MLPerf), platform benchmarking (Optimum-Benchmark, vLLM tools), and research tools (LLMPerf, LLenergyMeasure). Energy measurement remains rare (only MLPerf Power, TokenPowerBench, ML.ENERGY, LLenergyMeasure), making it a core differentiator. However, competitive positioning requires addressing table stakes gaps.

**Must have (table stakes):**
- Performance metrics (TTFT, ITL, throughput) — ✅ Has
- Configuration control (batch size, sampling, quantization) — ✅ Has
- Result persistence and export — ✅ Has
- Reproducibility controls (seed, warmup, deterministic mode) — ✅ Has
- Dataset support (standard + custom) — ✅ Has
- **System metadata capture** — ⚠️ **CRITICAL GAP**: Missing hardware details (GPU model, VRAM, compute capability), software versions (CUDA, driver), system state (temperature, power limits), dataset characteristics (ISL/OSL distribution)
- **Dataset characteristics reporting** — ⚠️ **GAP**: Actual input/output length distributions should be reported in results

**Should have (competitive differentiators):**
- **Energy measurement** — ✅ Core differentiator, but needs enhancement
- **Baseline power subtraction** — ❌ **CRITICAL GAP**: Idle power contamination causes 15-30% overestimation
- **Time-series power data** — ❌ **OPPORTUNITY**: TokenPowerBench sets standard with prefill/decode phase analysis
- **Statistical rigor (confidence intervals)** — ❌ **OPPORTUNITY**: Multi-cycle exists, but missing 95% CI (would significantly increase research credibility)
- Campaign orchestration — ✅ Has (grid generation + execution), needs enhancement for multi-backend dispatch

**Defer (v2+):**
- Perfect reproducibility guarantees (impossible with BF16, document limitations instead)
- LLM-as-judge quality scoring (scope creep, orthogonal to efficiency)
- Web UI/dashboard (CLI-first milestone, web platform is separate)
- Closed-loop optimisation (too complex, grid search sufficient)
- Training benchmarks (inference focus only)

### Architecture Approach

Host-based Python orchestrator managing long-running Docker containers via `docker exec` dispatch, not `docker compose run` per experiment. This avoids 10-30 second container startup overhead, prevents GPU memory fragmentation, and enables warmup state persistence across experiments. Backend containers (PyTorch, vLLM, TensorRT) run in detached mode with health checks, receiving experiment commands via exec calls.

**Major components:**
1. **CampaignOrchestrator** — Campaign manifest execution, container dispatch, cycle coordination, integrates HealthMonitor, PowerSampler, ResultsAggregator
2. **ContainerManager** — Container lifecycle (start/stop/health), exec dispatch via python-on-whales SDK
3. **HealthMonitor** — Periodic health checks (application-level: CUDA availability), auto-recovery on failure
4. **PowerSampler** — Host-side NVML time-series sampling (10Hz) parallel to inference, saves per-experiment timeseries
5. **CampaignManifest** — Persistent experiment queue with backend routing, cycle tracking, atomic state updates
6. **StateManager** — Experiment/campaign state persistence with atomic writes, crash recovery

**Data flow:** Campaign manifest → Backend router → ContainerManager.exec(container, cmd) → Shared volume results → Multi-cycle aggregation. Power sampling and health monitoring run as parallel daemon threads.

**Key patterns:**
- Long-running containers with exec dispatch (Pattern 1) — Avoids startup overhead, preserves warmup
- Health check daemon with auto-recovery (Pattern 2) — Application-aware validation, auto-restart on failure
- Time-series power sampling (Pattern 3) — Host-side NVML for accuracy, configurable 1-10Hz
- Campaign manifest with state persistence (Pattern 4) — Atomic writes, crash recovery, resume capability
- Warmup convergence detection (Pattern 5) — Coefficient of variation <5% across rolling window

### Critical Pitfalls

1. **Idle Power Contamination (CRITICAL)** — Energy measurements include baseline GPU power (idle state), causing 15-30% systematic overestimation. Longer experiments penalised disproportionately, invalidating cross-configuration comparisons. **Mitigation:** Measure idle baseline before experiments, subtract: `inference_energy = total - (baseline_watts × duration)`. Record both raw and corrected values.

2. **NVML Sampling Blind Spots (CRITICAL)** — NVML samples at max 66.7Hz but only 25% of runtime sampled on A100/H100. Short experiments (<1 min) show high variance, up to 73% average error, 300% maximum. **Mitigation:** Increase sampling frequency (100ms for <5min experiments), multi-cycle execution, report CV to flag high-variance measurements.

3. **Docker Container State Leakage (CRITICAL)** — Using `docker run --rm` per experiment wastes 30-60s startup time; long-running containers without cleanup accumulate GPU memory leaks. vLLM/TensorRT with `ipc: host` can conflict. **Mitigation:** Long-running containers with `docker exec`, explicit model unload when `force_cold_start=true`, health checks monitor GPU memory, graceful teardown between experiments.

4. **Backend API Instability (CRITICAL)** — vLLM/TensorRT undergo frequent breaking changes (pre-1.0 API stability). Parameter names change, defaults flip, TensorRT switched default backend C++ → PyTorch. Parameter audit becomes stale within 3 months. **Mitigation:** Pin exact versions in Docker (e.g., `vllm==0.6.3.post1`), runtime version detection warns on mismatch, quarterly parameter audit cycle, defensive kwargs filtering.

5. **Thermal Throttling Blind Spots (CRITICAL)** — GPU throttling silently degrades performance and inflates energy during long campaigns. Sequential runs show degrading performance (thermal accumulation). First run fast, tenth run slow. **Mitigation:** Monitor NVML performance state and clock speeds, record thermal state in metadata, flag results when throttling detected, enforce thermal gaps between experiments.

## Implications for Roadmap

Based on research, suggested phase structure follows dependency order: measurement foundations → parameter completeness → orchestration → validation.

### Phase 1: Measurement Foundations (v1.19.0)
**Rationale:** Foundational for all subsequent work. Baseline power subtraction and thermal monitoring address critical measurement accuracy gaps. Environment metadata enables reproducibility. Must come first as later phases depend on accurate measurements.

**Delivers:**
- Baseline power measurement and subtraction (idle vs active)
- Time-series power sampling (host-side NVML, configurable 1-10Hz)
- Thermal throttling detection and flagging
- Comprehensive environment metadata capture (GPU, CUDA, driver, CPU, container detection)
- Dataset characteristics reporting (ISL/OSL distributions)
- Warmup convergence detection (CV-based)
- Results schema v3 with metadata fields

**Addresses features:**
- System metadata capture (table stakes gap)
- Baseline power subtraction (critical differentiator gap)
- Time-series power data (competitive opportunity)

**Avoids pitfalls:**
- Pitfall #1: Idle power contamination
- Pitfall #3: NVML sampling blind spots
- Pitfall #7: Thermal throttling blind spots
- Pitfall #8: Environment metadata omission

**Implementation notes:**
- Extend existing `GPUUtilisationSampler` pattern for power sampling
- Use nvidia-ml-py ≥13.590.48 (already dependency, just bump version)
- Additive schema changes for backwards compatibility
- Conservative baseline subtraction to avoid negative energy readings

### Phase 2: Parameter Completeness (v1.20.0)
**Rationale:** Systematic parameter audit ensures backend coverage is comprehensive, addressing user confusion and incomplete configurations. SSOT introspection already exists, making this extension rather than new architecture. Must follow measurement foundations to validate parameters with accurate energy data.

**Delivers:**
- 90%+ coverage of energy/throughput-impactful parameters across all backends
- Backend-specific parameter introspection (PyTorch/HF, vLLM, TensorRT)
- Version pinning and compatibility matrix
- Mutual exclusion constraints (e.g., `load_in_4bit` × `load_in_8bit`)
- Streaming constraints documentation
- Parameter test value derivation from Pydantic models
- Runtime version detection with mismatch warnings

**Uses stack:**
- transformers 5.0+ for `generate()` introspection
- vllm 0.8.1+ for `LLM()` introspection
- tensorrt-llm 0.12.0+ for `LLM()` introspection
- Existing SSOT introspection module

**Implements architecture:**
- Extends existing `introspection.py` SSOT module
- Adds version detection layer
- Quarterly audit maintenance process

**Avoids pitfalls:**
- Pitfall #5: Backend API instability (via version pinning)
- Pitfall #6: Parameter audit scope creep (90% target, not 100%)

**Implementation notes:**
- Hard limit: 90%+ of energy-impactful params, document exclusions
- Pin exact versions in Docker images
- Use `get_backend_params()` from SSOT introspection
- Defensive kwargs filtering for unknown parameters

### Phase 3: Campaign Orchestration Redesign (v1.21.0)
**Rationale:** Campaign execution is key workflow bottleneck. Current manual/sequential approach doesn't scale to multi-backend grids. Long-running container pattern eliminates startup overhead whilst health monitoring provides robustness. Depends on measurement foundations (Phase 1) for accurate campaign results and parameter completeness (Phase 2) for backend-aware grid generation.

**Delivers:**
- CampaignOrchestrator with manifest-based execution
- Long-running container lifecycle management (start once, exec many)
- Health check daemon with auto-recovery
- Backend-aware grid generation (avoid invalid backend × param combinations)
- Campaign manifest with persistent state (crash recovery)
- Multi-backend dispatch routing
- Thermal gap enforcement between experiments

**Uses stack:**
- python-on-whales 0.70+ for Docker orchestration
- Existing StateManager pattern for persistence
- Docker Compose health checks

**Implements architecture:**
- CampaignOrchestrator component
- ContainerManager component
- HealthMonitor component
- CampaignManifest with StateManager
- BackendRouter for dispatch

**Avoids pitfalls:**
- Pitfall #4: Docker container state leakage (via long-running + cleanup)
- Pitfall #11: Grid generation cartesian explosion (backend-aware filtering)
- Pitfall #12: Configuration drift without detection (manifest persistence)

**Implementation notes:**
- Start simple: sequential dispatch, no parallelism initially
- Explicit model unload when `force_cold_start=true`
- Health checks: `python -c "import torch; assert torch.cuda.is_available()"`
- Atomic state writes with temp-then-rename pattern

### Phase 4: Statistical Rigor Enhancement (v2.1.0)
**Rationale:** Multi-cycle execution already exists, confidence intervals are natural extension. Adds research credibility without architectural complexity. Can be developed independently after core measurement/orchestration solid.

**Delivers:**
- 95% confidence intervals for multi-cycle results
- Bootstrap resampling (percentile method, 1000 iterations)
- Enhanced aggregation statistics (mean, median, std dev, P50/P95/P99, CI)
- Outlier detection and flagging
- Statistical comparison utilities

**Addresses features:**
- Statistical rigor (competitive differentiator)

**Implementation notes:**
- Use scipy.stats or numpy percentile bootstrapping
- Follow Benchmark2 pattern (FAQ paper 2026)
- Frequentist coverage guarantees

### Phase 5: User Acceptance Testing & Refinement (v1.22.0)
**Rationale:** Final validation before release. Surfaces "works on my machine" issues, error message clarity, documentation gaps. Structured UAT prevents rushed testing that skips critical scenarios.

**Delivers:**
- UAT round 1: Fresh VM, follow quickstart from scratch (1 hour session)
- UAT round 2: Full campaign with multi-backend grid (2 hours session)
- Error message improvements based on feedback
- Documentation refresh (quickstart, troubleshooting, prerequisites)
- Validation script (`make check-prereqs`)
- UAT acceptance criteria with automated validation
- Cleanup pass (remove dead code, consolidate docs)

**Addresses features:**
- Documentation completeness (table stakes)
- User onboarding experience

**Avoids pitfalls:**
- Pitfall #16: "Works on my machine" trap
- Pitfall #17: Error message archaeology
- Pitfall #18: Incomplete quickstart assumptions
- Pitfall #19: Pass/fail criteria ambiguity
- Pitfall #20: Time pressure skipping testing

**Implementation notes:**
- External tester or fresh VM (avoid confirmation bias)
- Time-box sessions: 1hr round 1, 2hr round 2
- Asynchronous UAT option for busy users
- Provide pre-configured test scenarios

### Phase Ordering Rationale

**Critical path:** Phase 1 (Measurement) → Phase 2 (Parameters) → Phase 3 (Orchestration) → Phase 4 (Statistics, parallel) → Phase 5 (UAT)

- **Phase 1 first:** Measurement accuracy is foundational. All later work depends on accurate energy/performance data. Baseline power subtraction affects every experiment. Environmental metadata enables reproducibility.
- **Phase 2 after Phase 1:** Parameter validation requires accurate measurements to test parameter impacts. Version pinning prevents API instability issues during orchestration development.
- **Phase 3 after Phase 1+2:** Campaign orchestration needs accurate measurements and complete parameters. Long-running containers benefit from convergence detection (Phase 1). Backend routing uses parameter introspection (Phase 2).
- **Phase 4 independent:** Statistical enhancements can run parallel to Phase 5 or after orchestration solid. Extends existing multi-cycle without architectural changes.
- **Phase 5 last:** UAT validates complete v2.0 CLI system. All features must be implemented before user testing.

**Dependency arrows:**
```
Phase 1 (Measurement) ────┬──→ Phase 3 (Orchestration) ──→ Phase 5 (UAT)
                          │
Phase 2 (Parameters) ─────┘

Phase 4 (Statistics) ────────────────────────────────────→ Phase 5 (UAT)
```

**Groupings based on architecture:**
- Phases 1+2: SSOT enhancement (measurement + parameter metadata)
- Phase 3: Orchestration layer (new component)
- Phase 4: Results enhancement (statistical layer)
- Phase 5: Validation (QA layer)

**Avoids pitfalls:**
- Early measurement accuracy prevents contaminated data throughout development
- Parameter pinning before orchestration prevents API breakage mid-campaign development
- Structured UAT prevents rushed testing that misses critical bugs

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 3 (Campaign Orchestration):** Complex integration between Docker SDK, NVML monitoring, and state persistence. Container lifecycle edge cases need validation. Health check recovery strategies need testing with actual GPU failures.
- **Phase 4 (Statistical Rigor):** Bootstrap method selection (percentile vs BCa). Appropriate significance levels for energy measurements (frequentist coverage guarantees).

Phases with standard patterns (skip research-phase):

- **Phase 1 (Measurement):** NVML patterns well-documented, existing `GPUUtilisationSampler` provides reference implementation. Baseline power methodology from ML.ENERGY best practices.
- **Phase 2 (Parameters):** Introspection pattern exists in codebase, official library documentation available. Straightforward extension of SSOT module.
- **Phase 5 (UAT):** Standard UAT practices, no domain-specific research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official NVIDIA packages (nvidia-ml-py), Docker-endorsed python-on-whales, official HF/vLLM/TensorRT libraries. Jan 2026 versions verified. |
| Features | HIGH | Verified against 6 major tools (MLPerf, TokenPowerBench, Optimum-Benchmark, LLMPerf, vLLM, TensorRT) and 10+ recent papers (2025-2026). Clear table stakes vs differentiator classification. |
| Architecture | HIGH | Docker orchestration patterns from official documentation, health check/exec dispatch verified. Host-side NVML sampling from research literature. Manifest persistence extends existing StateManager. |
| Pitfalls | HIGH | Energy measurement pitfalls from 6 research papers + NVML docs. Docker gotchas from NVIDIA forums + GitHub issues. Backend API instability from release notes. Prefill/decode from recent papers. UAT from general best practices. |

**Overall confidence:** HIGH

Research based on:
- Official documentation (NVIDIA NVML API, Docker SDK, HuggingFace/vLLM/TensorRT APIs)
- Recent research papers (2025-2026: TokenPowerBench, ML.ENERGY, MLPerf Power, FAQ evaluation, DABench-LLM)
- Community knowledge (NVIDIA forums, GitHub issues, Docker blogs)
- Existing codebase analysis (SSOT introspection, StateManager, GPUUtilisationSampler patterns)

### Gaps to Address

**Baseline power methodology:** Community best practices converge on conservative subtraction (idle_watts × duration), but no official NVIDIA specification found. Confidence MEDIUM. **Mitigation:** Implement conservative approach, document methodology, validate against external power meters during UAT.

**vLLM ITL measurement:** Non-streaming vLLM doesn't provide per-token timestamps. Tool estimates ITL via total_decode_time / token_count, masking variance. **Mitigation:** Recommend streaming mode for latency benchmarks, flag estimated vs measured ITL in results schema.

**MIG energy isolation:** NVML reports parent GPU power, cannot isolate per-MIG-instance energy. **Mitigation:** Document limitation clearly, flag in results metadata when MIG detected, accept parent GPU power as best available measurement.

**Warmup convergence thresholds:** Coefficient of variation <5% is domain inference, not research-validated threshold. **Mitigation:** Make configurable, document default rationale, allow user override.

**Time-series database selection:** InfluxDB/TimescaleDB appropriate for power timeseries but adds dependency complexity. **Mitigation:** Start with JSON file storage (Phase 1), defer time-series DB to post-v2.0 if needed.

## Sources

### Primary (HIGH confidence)

**Technology stack:**
- [nvidia-ml-py PyPI](https://pypi.org/project/nvidia-ml-py/) — Official NVIDIA package, v13.590.48 verified Jan 2026
- [NVML API Reference](https://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf) — Power/thermal monitoring functions
- [python-on-whales GitHub](https://github.com/gabrieldemarmiesse/python-on-whales) — Docker orchestration, 696 stars, MIT license
- [Docker Blog: Python-on-whales](https://www.docker.com/blog/guest-post-calling-the-docker-cli-from-python-with-python-on-whales/) — Official endorsement
- [HuggingFace Transformers: Text Generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation) — `generate()` parameters
- [vLLM LLM Class API](https://docs.vllm.ai/en/v0.8.1/api/offline_inference/llm.html) — Constructor parameters
- [TensorRT-LLM API Reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html) — LLM class parameters

**Feature landscape:**
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — Industry standard benchmarks
- [TokenPowerBench (Dec 2025)](https://arxiv.org/html/2512.03024v1) — LLM power benchmarking state-of-art
- [ML.ENERGY Benchmark](https://arxiv.org/html/2505.06371v1) — Automated energy measurement
- [Optimum-Benchmark GitHub](https://github.com/huggingface/optimum-benchmark) — Multi-backend benchmarking
- [LLMPerf GitHub](https://github.com/ray-project/llmperf) — API benchmarking
- [NVIDIA LLM Benchmarking Fundamentals](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) — Best practices
- [FAQ: Efficient LLM Evaluation (Jan 2026)](https://arxiv.org/abs/2601.20251) — Statistical guarantees

**Architecture:**
- [Docker SDK for Python](https://docker-py.readthedocs.io/en/stable/containers.html) — Container lifecycle
- [Docker Health Checks](https://lumigo.io/container-monitoring/docker-health-check-a-practical-guide/) — Health monitoring patterns
- [Docker Restart Policies](https://www.cloudbees.com/blog/ensuring-containers-are-always-running-dockers-restart-policy) — Crash recovery

**Pitfalls:**
- [Part-time Power Measurements (arXiv)](https://arxiv.org/html/2312.02741v2) — NVML sampling issues, 73% error rates
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/) — Baseline power methodology
- [Per-query energy consumption (2026)](https://muxup.com/2026q1/per-query-energy-consumption-of-llms) — Energy measurement analysis
- [vLLM memory leaks](https://github.com/vllm-project/vllm/issues/15294) — V1 engine 200GB RAM leaks
- [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html) — Breaking changes documented

### Secondary (MEDIUM confidence)

- [GPU Idle Power Benchmark Guide](https://www.ywian.com/blog/gpu-idle-power-benchmark-fix-it-guide) — Idle power ranges (30-50W high-end)
- [Warmup convergence detection](https://www.emergentmind.com/topics/model-warmup-techniques) — Recent research (2025-2026)
- [Docker shared volumes](https://www.baeldung.com/ops/docker-share-volume-multiple-containers) — Concurrent access patterns
- [Time-series databases for energy monitoring](https://www.mdpi.com/1996-1073/17/21/5478) — Architecture patterns
- [User Acceptance Testing Best Practices](https://research.aimultiple.com/user-acceptance-testing-best-practices/) — UAT methodology

### Tertiary (LOW confidence, needs validation)

- GPU idle power subtraction methodology — Community convergence, no official NVIDIA spec
- Warmup convergence CV threshold <5% — Domain inference, not research-validated
- InfluxDB/TimescaleDB for power timeseries — Appropriate but needs performance testing

---
*Research completed: 2026-01-29*
*Ready for roadmap: yes*
