# Domain Pitfalls — LLM Benchmarking Tools

**Domain:** LLM inference efficiency measurement
**Researched:** 2026-01-29
**Focus:** Energy measurement, Docker orchestration, parameter management, vLLM/TensorRT stability, UAT

## Critical Pitfalls

Mistakes that cause rewrites, systematic measurement errors, or major issues.

---

### Pitfall 1: Idle Power Contamination

**What goes wrong:** Energy measurements include baseline GPU power consumption (idle state), leading to 15-30% systematic overestimation of inference energy. When batch size decreases, experiments take longer and idle consumption starts to weigh on total consumption proportionally more.

**Why it happens:** Energy measurement tools (CodeCarbon, NVML) measure total GPU power draw, not differential energy above idle. The tool captures everything from experiment start to end — including the GPU's baseline power even when it's not actively computing.

**Consequences:**
- Biased energy efficiency comparisons (longer experiments penalised disproportionately)
- Invalid cross-configuration comparisons (batch size 32 vs 128 comparisons are systematically skewed)
- Scientific papers with incorrect efficiency conclusions
- Cannot accurately compare GPU architectures with different idle power profiles

**Prevention:**
1. Measure idle baseline power BEFORE experiment (10-30 second sampling window with no workload)
2. Subtract baseline from total energy: `inference_energy = total_energy - (baseline_power_watts × duration_seconds)`
3. Include warmup to ensure "hot" GPU state before baseline measurement (GPUs consume different power when cold vs warmed up)
4. Record and report both raw and baseline-corrected energy in results schema
5. Flag in results when baseline subtraction was NOT performed (for backwards compatibility)

**Detection:**
- Energy/token stays constant as batch size increases (should decrease due to amortisation)
- Longer experiments show higher total energy but similar per-token energy to shorter runs
- Energy measurements don't scale linearly with computational intensity

**Phase mapping:** v1.19.0 (MEAS-02) — foundational for all subsequent measurements

---

### Pitfall 2: Prefill/Decode Phase Blending

**What goes wrong:** Time to First Token (TTFT) and Inter-Token Latency (ITL) metrics blend two fundamentally different computational phases: prefill (compute-bound) and decode (memory-bound). This masks phase-specific bottlenecks and makes optimisation decisions misleading.

**Why it happens:** Most benchmarking tools measure end-to-end latency without separating prefill (processing input prompt) from decode (generating output tokens). Prefill is compute-heavy (benefits from high batch size), decode is memory-bandwidth-limited (benefits from low batch size). Blended metrics average out these opposing characteristics.

**Consequences:**
- Cannot diagnose whether bottleneck is prefill or decode
- Batch size tuning becomes trial-and-error instead of principled
- Cross-phase interference in vLLM/TensorRT (compute-heavy prefill blocks decode tasks, increasing ITL)
- Disaggregated prefill/decode architectures (emerging best practice) cannot be evaluated

**Prevention:**
1. Measure TTFT separately from ITL (already implemented via streaming callbacks)
2. For deeper analysis: instrument phase boundaries explicitly (end of prefill = first token generation)
3. Report prefill energy vs decode energy separately (requires phase-aware energy sampling)
4. For PyTorch backend: use `torch.profiler` to ground-truth phase boundaries
5. For vLLM/TensorRT: rely on library-provided phase timing if available, otherwise estimate from TTFT

**Detection:**
- TTFT and ITL don't show expected compute/memory-bound scaling behaviour
- Batch size increases improve throughput but ITL degrades unexpectedly
- Users request "why is my decode so slow" without ability to diagnose

**Phase mapping:** v2.1.0 (PREC-01) — precision metrics after core measurement foundations solid

---

### Pitfall 3: NVML Sampling Blind Spots

**What goes wrong:** NVML power measurements are sampled at maximum 66.7 Hz (15ms intervals), but on A100/H100 GPUs only 25% of runtime is actually sampled. This leads to drastic under/overestimation of energy consumed, especially for short inference bursts.

**Why it happens:** NVIDIA GPUs provide instantaneous power readings via NVML, but the sampling is asynchronous to workload execution. For bursty workloads (single-batch inference), power spikes between samples are missed. CodeCarbon samples every 15 seconds by default, compounding the problem.

**Consequences:**
- Energy measurements for short experiments (<1 minute) are unreliable
- High variance across repeat runs of identical experiments
- Cannot accurately measure single-request latency/energy tradeoffs
- Measurement error can reach 73% average, 300% maximum (from research literature)

**Prevention:**
1. Increase sampling frequency for short experiments (configurable interval, default 100ms for <5min experiments)
2. Run multi-cycle experiments with statistical aggregation (already implemented)
3. Report measurement uncertainty: coefficient of variation (CV) across cycles flags high-variance measurements
4. For ground-truth validation: compare against external power meters (Yokogawa WT310, etc.) during UAT
5. Warmup experiments to steady state before measurement window

**Detection:**
- CV > 10% across cycles for identical configurations
- Energy/token varies >15% between runs
- Short experiments show higher variance than long experiments
- Results don't match external power meter readings

**Phase mapping:** v1.19.0 (MEAS-04) — time-series sampling with configurable intervals

**Sources:**
- [Part-time Power Measurements: nvidia-smi's Lack of Attention](https://arxiv.org/html/2312.02741v2)
- [Maximum Sampling Rate of GPU Power Measurement using NVML](https://forums.developer.nvidia.com/t/maximum-sampling-rate-of-gpu-power-measurement-using-nvml/109848)

---

### Pitfall 4: Docker Container State Leakage

**What goes wrong:** Using `docker compose run --rm` for each experiment creates fresh containers every time, discarding model caches, warmup state, and GPU context. This introduces 30-60 second overhead per experiment and invalidates "warm start" measurements. Conversely, long-running containers without proper cleanup accumulate GPU memory leaks, CUDA context bloat, or stale model weights.

**Why it happens:** Docker's execution models have tradeoffs: `docker run --rm` ensures clean state but pays startup cost; long-running containers with `docker exec` are fast but must manage state explicitly. GPU state (CUDA contexts, model weights in VRAM) persists across executions unless explicitly cleared.

**Consequences:**
- Campaign orchestration takes 10x longer than necessary (startup overhead dominates short experiments)
- "Warm start" benchmarks accidentally measure cold starts
- GPU memory leaks accumulate across experiments, causing OOM failures mid-campaign
- Container crashes require manual cleanup and campaign restart
- vLLM/TensorRT with `ipc: host` can conflict across containers if not isolated properly

**Prevention:**
1. Use long-running containers with `docker compose exec` for campaigns (v1.21.0 design)
2. Implement explicit model unload between experiments when `force_cold_start=true`
3. Default to warmup-then-measure (keeps models warm for fair comparisons)
4. Container health checks detect GPU memory leaks (NVML memory monitoring)
5. Graceful teardown clears CUDA contexts before next experiment
6. Shared volumes for results (bind mounts), isolated volumes for model caches (named volumes with PUID/PGID)
7. Per-backend container isolation (vLLM and TensorRT in separate containers due to conflicting PyTorch deps)

**Detection:**
- Experiment duration includes unexpected 30-60s "startup" time
- GPU memory grows monotonically across campaign
- Second experiment in sequence shows different GPU memory baseline than first
- CUDA out-of-memory errors appear mid-campaign but not on fresh container

**Phase mapping:** v1.21.0 (CAMP-01, CAMP-06) — campaign orchestrator redesign

**Sources:**
- [Sharing GPU between Docker containers](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1534)
- [How can two containers share the usage of a GPU safely?](https://forums.developer.nvidia.com/t/how-can-two-containers-share-the-usage-of-a-gpu-safely/258381)

---

### Pitfall 5: Backend API Instability (vLLM/TensorRT Breaking Changes)

**What goes wrong:** vLLM and TensorRT-LLM undergo frequent breaking API changes without deprecation warnings. Parameters are renamed, defaults flip, or entire subsystems change (e.g., TensorRT-LLM switching default backend from C++ to PyTorch). A tool that works with vLLM v0.6.x breaks silently with v0.7.x.

**Why it happens:** Both libraries are pre-1.0 (vLLM aiming for 1.0 "API stability", TensorRT-LLM reached 1.0 in 2026 but introduced breaking changes in the transition). Research-focused libraries prioritise performance and features over backwards compatibility. TensorRT-LLM explicitly states "breaking change" in release notes but doesn't provide migration paths.

**Consequences:**
- Parameter audit (v1.20.0) becomes stale within 3 months
- Users' saved configs break after `pip install --upgrade`
- Docker images pinned to specific versions become outdated quickly
- `extra:` escape hatch for custom kwargs stops working (param names changed)
- CI tests pass but user experiments fail due to version drift

**Prevention:**
1. Pin exact versions in Docker images (e.g., `vllm==0.6.3.post1`, not `vllm>=0.6`)
2. Document version compatibility in config examples and docs
3. Runtime version detection: warn if installed version mismatches tested version
4. Defensive parameter passing: use `**kwargs` filtering to ignore unknown params (don't crash on renamed params)
5. Quarterly parameter audit cycle (not one-time) — add to maintenance roadmap
6. SSOT introspection tests fail loudly when backend API changes
7. Subscribe to vLLM/TensorRT release notes, test RC versions before production updates

**Detection:**
- `TypeError: unexpected keyword argument 'X'` errors in backend inference calls
- Parameters documented in library's docs don't exist in installed version
- Default behaviour changes without config changes (e.g., sampling strategy)
- Runtime tests fail after `pip install --upgrade`

**Phase mapping:** v1.20.0 (PARAM-01 to PARAM-04) — parameter completeness with version pinning

**Sources:**
- [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [vLLM Roadmap Q3 2025](https://github.com/vllm-project/vllm/issues/20336)
- [vLLM vs TensorRT-LLM: Key differences](https://northflank.com/blog/vllm-vs-tensorrt-llm-and-how-to-run-them)

---

### Pitfall 6: Parameter Audit Scope Creep

**What goes wrong:** Attempting to achieve 100% coverage of all possible kwargs across PyTorch, vLLM, and TensorRT leads to infinite scope. Each library has 100+ parameters, many undocumented or experimental. The audit never finishes, and most added parameters are never used by actual users.

**Why it happens:** Well-intentioned completeness goal meets reality: libraries evolve, niche params are for specific hardware (MoE routing, Hopper-specific features), and the long tail of kwargs delivers diminishing returns. The SSOT introspection system makes adding params easy, creating temptation to add everything.

**Consequences:**
- v1.20.0 timeline blows out from 2 weeks to 2 months
- Parameter matrix documentation becomes unreadable
- Runtime tests explode (combinatorial parameter explosion)
- Config validation gets complex, fragile
- Users overwhelmed by options (paradox of choice)

**Prevention:**
1. Define completion criteria: **90%+ of energy/throughput-impactful parameters**, not "all kwargs"
2. Impact heuristic: prioritise params that affect energy, throughput, or latency by >5%
3. `extra:` escape hatch for the long tail (users can pass arbitrary kwargs as dict)
4. Document exclusions: maintain list of deliberately skipped params with rationale
5. User-driven additions: add params when requested, not speculatively
6. Version-specific audits: focus on stable APIs, exclude experimental features

**Detection:**
- Parameter audit backlog keeps growing faster than items are completed
- New parameters added without clear impact justification
- Config examples don't use 80% of available parameters
- Users ask "what should I actually tune?" (too many options)

**Phase mapping:** v1.20.0 (PARAM-01, PARAM-02) — targeted 90%+ coverage

**Sources:**
- Domain expertise (LLenergyMeasure codebase analysis)

---

### Pitfall 7: Thermal Throttling Blind Spots

**What goes wrong:** GPU thermal throttling during experiments silently degrades performance and inflates energy measurements. The tool reports "Model A uses 20% more energy than Model B", but the difference is actually thermal throttling, not model efficiency.

**Why it happens:** Long-running experiments, sequential campaign runs without thermal gaps, or inadequate cooling cause GPU temperatures to exceed throttling thresholds (e.g., 83°C on A100). NVIDIA GPUs automatically reduce clock speeds to prevent damage, but this is invisible to energy measurement tools. Lower clock speeds = same work takes longer = more total energy consumed.

**Consequences:**
- Non-reproducible results (thermal state varies run-to-run)
- Sequential experiments show degrading performance (first run fast, tenth run slow)
- Incorrect efficiency rankings (throttled GPU looks inefficient)
- Campaign results depend on ambient temperature, airflow, GPU slot position

**Prevention:**
1. Monitor NVML performance state (`nvmlDeviceGetPerformanceState`) during experiments
2. Record thermal state in results metadata: clock speeds, temperature, power limits
3. Flag results when throttling detected (`thermal_throttling_detected: true`)
4. Enforce thermal gaps between experiments (already in campaign orchestration)
5. Warmup convergence detection: wait for thermal equilibrium before measurement
6. UAT validation: check that sequential runs show <5% performance variance

**Detection:**
- GPU clock speed drops mid-experiment (NVML `nvmlDeviceGetClockInfo`)
- Temperature exceeds throttling threshold (typically >80°C)
- Performance degrades across sequential runs despite identical config
- Throughput CV > 5% across cycles when it should be <2%

**Phase mapping:** v1.19.0 (MEAS-03) — thermal throttling detection

**Sources:**
- Domain expertise (existing thermal gap implementation in campaigns)

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or reproducibility issues.

---

### Pitfall 8: Environment Metadata Omission

**What goes wrong:** Results don't include CUDA version, driver version, GPU power limits, CPU governor, or container detection. When results can't be reproduced 6 months later, there's no diagnostic information to identify what changed.

**Prevention:**
1. Capture environment snapshot at experiment start (CUDA, driver, GPU config, CPU governor)
2. Include in results schema v3 (v1.19.0 MEAS-01)
3. Hash environment snapshot for quick "configuration drift" detection
4. Compare environment across experiments in multi-cycle campaigns (flag if drift detected)

**Phase mapping:** v1.19.0 (MEAS-01)

---

### Pitfall 9: vLLM ITL Proportional Estimation

**What goes wrong:** vLLM doesn't provide per-token timestamps for non-streaming inference. The tool estimates ITL by dividing total decode time by token count, which masks variance and hides outliers (e.g., first decode token vs subsequent tokens have different latencies).

**Prevention:**
1. Recommend streaming mode for latency benchmarks (per-token timestamps available)
2. Flag in results when ITL is estimated vs measured
3. Report as `itl_mean_ms` with null for variance when estimated
4. Document limitation in vLLM backend docs

**Phase mapping:** v1.19.0 (MEAS-06) — schema flags for estimation vs measurement

---

### Pitfall 10: PUID/PGID Complexity in Docker

**What goes wrong:** Files created by Docker containers are owned by root, causing permission errors when host user tries to read results. The PUID/PGID workaround (run container as host user) requires `.env` file, which users forget to create, leading to opaque entrypoint errors.

**Prevention:**
1. `setup.sh` auto-generates `.env` with PUID/PGID (already implemented)
2. Entrypoint validates PUID/PGID are set, exits with clear error message if missing
3. Named volumes for caches (Docker-managed permissions), bind mounts for results (user-accessible)
4. Document in quickstart: "Run `./setup.sh` first"

**Phase mapping:** Already implemented; v1.22.0 (UAT-04) docs refresh validates clarity

**Sources:**
- [Docker Deployment Guide](docs/deployment.md) — existing implementation

---

### Pitfall 11: Grid Generation Cartesian Explosion

**What goes wrong:** Grid generation creates cartesian product of ALL parameter values across ALL backends, producing thousands of invalid configurations (e.g., PyTorch-specific param × vLLM backend).

**Prevention:**
1. Backend-aware grid generation: only combine params valid for target backend
2. SSOT introspection already knows which params belong to which backend
3. Mutual exclusion constraints (e.g., `load_in_4bit` × `load_in_8bit` = invalid)
4. Estimate grid size before generation, warn if >100 configs

**Phase mapping:** v1.21.0 (CAMP-02) — backend-aware grid generation

**Sources:**
- Domain expertise (SSOT introspection architecture)

---

### Pitfall 12: Configuration Drift Without Detection

**What goes wrong:** User edits a config file, forgets which version they ran 3 months ago, cannot reproduce results. Git tracks file changes, but experiments don't capture config hash or version.

**Prevention:**
1. Hash experiment config (already tracked in results as `config_hash`)
2. Include full config in results JSON (already implemented)
3. Detect if config file changed since experiment ran (compare hash)
4. Campaign manifest links exp_id → config version → result path

**Phase mapping:** Already implemented; v1.21.0 (CAMP-03) campaign manifest reinforces

**Sources:**
- [Configuration Drift Management](https://www.reach.security/blog/what-is-configuration-drift-5-best-practices-for-your-teams-security-posture)

---

## Minor Pitfalls

Mistakes that cause annoyance but are easily fixable.

---

### Pitfall 13: Warmup Overkill vs Underkill

**What goes wrong:** Fixed warmup count (e.g., 3 prompts) is either wasteful (model warmed up after 1 prompt) or insufficient (model still converging after 5 prompts, especially for large models on cold GPUs).

**Prevention:**
1. Convergence detection: continue warmup until throughput CV stabilises (<5%)
2. CycleStatistics already tracks CV — reuse for warmup
3. Max warmup limit to prevent infinite loops (e.g., 10 prompts)

**Phase mapping:** v1.19.0 (MEAS-05) — warmup convergence

---

### Pitfall 14: CodeCarbon Process-Level GPU Limitation

**What goes wrong:** CodeCarbon's `tracking_mode=Process` estimates RAM at process level but not GPU/CPU energy. Users expect per-process GPU energy, get machine-wide readings instead.

**Prevention:**
1. Document limitation in energy backend docs
2. Use `tracking_mode=Machine` (default) for GPU energy
3. For multi-GPU setups: use CUDA_VISIBLE_DEVICES to isolate GPU per process

**Phase mapping:** Documentation fix, no code changes needed

**Sources:**
- [CodeCarbon Methodology](https://mlco2.github.io/codecarbon/methodology.html)

---

### Pitfall 15: MIG Energy Measurement Warning Confusion

**What goes wrong:** Tool flags "MIG instance detected, energy reflects parent GPU" but users don't understand what this means or how to interpret results.

**Prevention:**
1. Clearer warning message: "MIG instance detected. Energy readings include all MIG slices on parent GPU, not just this instance. For accurate per-instance energy, disable MIG or ensure no other workloads on sibling instances."
2. Include `is_mig` and `mig_profile` in metadata
3. Document in deployment guide: MIG energy limitations and workarounds

**Phase mapping:** v1.22.0 (UAT-04) — docs refresh

**Sources:**
- [Docker Deployment Guide](docs/deployment.md) — existing MIG detection

---

## UAT-Specific Pitfalls

Blind spots that only surface during user acceptance testing.

---

### Pitfall 16: The "Works On My Machine" Trap

**What goes wrong:** Developer environment has specific Python version, CUDA version, or library versions that differ from user environment. Tool works perfectly in dev, breaks immediately for users.

**Prevention:**
1. UAT round 1 on fresh clone (v1.19.0 MEAS-07): fresh VM, follow quickstart from scratch
2. Docker images eliminate environment variance (already implemented)
3. Document tested environments: Python 3.10/3.11, CUDA 12.4, driver ≥535
4. CI matrix tests across Python versions

**Phase mapping:** v1.19.0 (MEAS-07), v1.22.0 (UAT-03)

---

### Pitfall 17: Error Message Archaeology

**What goes wrong:** Errors are cryptic stack traces deep in PyTorch internals. Users give up instead of debugging. Example: `RuntimeError: CUDA error: invalid device ordinal` doesn't tell user "you need to set CUDA_VISIBLE_DEVICES for MIG instances".

**Prevention:**
1. Catch common errors at orchestration layer, re-raise with context
2. Custom exception classes with diagnostic hints (`GPUConfigurationError`, `BackendParameterError`)
3. UAT feedback: log all error messages users encounter, improve messages for top 5 errors
4. Validation catches misconfigurations BEFORE experiments run (e.g., invalid backend param)

**Phase mapping:** v1.22.0 (UAT-03) — error message improvements based on UAT feedback

**Sources:**
- [User Acceptance Testing Best Practices](https://research.aimultiple.com/user-acceptance-testing-best-practices/)

---

### Pitfall 18: Incomplete Quickstart Assumptions

**What goes wrong:** Quickstart guide assumes Docker, NVIDIA GPU, HuggingFace token already configured. New users hit errors at step 1, no troubleshooting guidance provided.

**Prevention:**
1. Prerequisites checklist BEFORE quickstart steps
2. Validation script: `make check-prereqs` that tests Docker, nvidia-smi, GPU visibility
3. Troubleshooting section for each common failure mode
4. UAT round 1: observe user following quickstart, note every point of confusion

**Phase mapping:** v1.22.0 (UAT-04) — documentation refresh

**Sources:**
- [User Acceptance Testing Challenges](https://coruzant.com/software/user-acceptance-testing-challenges/)

---

### Pitfall 19: Pass/Fail Criteria Ambiguity

**What goes wrong:** UAT testers don't know what "success" looks like. Does the experiment need to complete? Should results match expected values? How much variance is acceptable?

**Prevention:**
1. Define UAT acceptance criteria per experiment type:
   - Quick test: completes in <5 minutes, results file created
   - Full campaign: all experiments complete, CV <10% across cycles
   - Baseline power: idle measurement within ±5W across runs
2. Automated validation: `lem results validate <exp_id>` checks sanity conditions
3. UAT checklist with clear PASS/FAIL per task

**Phase mapping:** v1.22.0 (UAT-03) — UAT acceptance criteria

**Sources:**
- [UAT Testing Best Practices](https://www.panaya.com/blog/testing/what-is-uat-testing/)

---

### Pitfall 20: Time Pressure Skipping Testing

**What goes wrong:** Business users don't have time for UAT due to day jobs. When operational work piles up, testing gets pushed aside. UAT scheduled for "next week" repeatedly.

**Prevention:**
1. Time-box UAT sessions: 1 hour for round 1 (basic workflow), 2 hours for round 2 (full campaign)
2. Provide pre-configured test scenarios (no setup required, just run)
3. Asynchronous UAT: users can test at their own pace, provide feedback async
4. Prioritise UAT tasks: must-test (basic experiment) vs nice-to-test (edge cases)

**Phase mapping:** v1.22.0 (UAT-01 to UAT-05) — structured UAT plan

**Sources:**
- [Common UAT Challenges](https://coruzant.com/software/user-acceptance-testing-challenges/)

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **v1.19.0 Measurement Foundations** | Baseline power subtraction breaks existing results schema | Additive schema changes only; migrate v2 → v3 with backwards-compatible reader |
| **v1.19.0 Time-series sampling** | High-frequency sampling causes disk I/O bottleneck | Configurable interval with defaults (100ms for <5min, 1s for >5min); in-memory buffer, write at end |
| **v1.20.0 Parameter audit** | Scope creep: attempting 100% coverage | Hard limit: 90%+ of energy-impactful params, document exclusions |
| **v1.20.0 Backend API changes** | New vLLM version breaks param names | Pin exact versions in Docker; version detection warns on mismatch |
| **v1.21.0 Docker exec model** | Container state leakage or memory accumulation | Health checks + graceful cleanup; explicit model unload when `force_cold_start=true` |
| **v1.21.0 Campaign orchestrator** | Complex state machine for multi-backend dispatch | Start simple: sequential dispatch, no parallelism; add complexity only if needed |
| **v1.22.0 UAT round 1** | Confirmation bias (developer testing own code) | External tester or fresh VM setup; observe without helping |
| **v1.22.0 Cleanup pass** | Deleting code that's actually needed | Git branch before cleanup; validate tests still pass; restore if needed |

---

## Research Confidence Levels

| Area | Confidence | Sources | Notes |
|------|-----------|---------|-------|
| Energy measurement pitfalls | **HIGH** | 6 research papers + NVML docs + codebase | Idle power and sampling issues well-documented in literature |
| Docker orchestration gotchas | **MEDIUM** | NVIDIA forums + GitHub issues + docs | Community knowledge, not formal research |
| Backend API stability | **HIGH** | vLLM/TensorRT release notes + migration guides | Breaking changes explicitly documented |
| Prefill/decode separation | **HIGH** | 5 recent papers on disaggregated inference | Active research area with established metrics |
| UAT blind spots | **MEDIUM** | General UAT literature + domain inference | Generic UAT wisdom, not LLM-specific |
| Parameter audit scope creep | **HIGH** | Domain expertise + SSOT codebase analysis | Pattern observed in existing codebase |

---

## Sources

### Energy Measurement
- [Per-query energy consumption of LLMs (Muxup, 2026)](https://muxup.com/2026q1/per-query-energy-consumption-of-llms)
- [Benchmarking Energy Efficiency of Large Language Models Using vLLM (arXiv:2509.08867)](https://arxiv.org/html/2509.08867v1)
- [CodeCarbon Methodology](https://mlco2.github.io/codecarbon/methodology.html)
- [Part-time Power Measurements: nvidia-smi's Lack of Attention (arXiv:2312.02741)](https://arxiv.org/html/2312.02741v2)
- [Accurate and Convenient Energy Measurements for GPUs (IEEE SC'24)](https://ieeexplore.ieee.org/document/10793163/)
- [Maximum Sampling Rate of GPU Power via NVML](https://forums.developer.nvidia.com/t/maximum-sampling-rate-of-gpu-power-measurement-using-nvml/109848)

### Prefill/Decode Phase Separation
- [Prefill-decode disaggregation (BentoML LLM Inference Handbook)](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)
- [Disaggregated Prefilling (vLLM docs)](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [Inside Real-Time LLM Inference: From Prefill to Decode](https://medium.com/@devsp0703/inside-real-time-llm-inference-from-prefill-to-decode-explained-72a1c9b1d85a)

### Docker & GPU Orchestration
- [Sharing GPU between Docker containers (NVIDIA toolkit issue)](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1534)
- [How can two containers share the usage of a GPU safely?](https://forums.developer.nvidia.com/t/how-can-two-containers-share-the-usage-of-a-gpu-safely/258381)
- [Docker Container Orchestration Platforms (2026)](https://www.portainer.io/blog/container-orchestration-platforms)

### Backend API Stability
- [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [vLLM Roadmap Q3 2025](https://github.com/vllm-project/vllm/issues/20336)
- [vLLM vs TensorRT-LLM comparison (Northflank)](https://northflank.com/blog/vllm-vs-tensorrt-llm-and-how-to-run-them)

### UAT & Testing
- [User Acceptance Testing Best Practices (AIMultiple)](https://research.aimultiple.com/user-acceptance-testing-best-practices/)
- [Common UAT Challenges (Coruzant)](https://coruzant.com/software/user-acceptance-testing-challenges/)
- [What is UAT Testing? (Panaya)](https://www.panaya.com/blog/testing/what-is-uat-testing/)
- [LLM Testing in 2026 (Confident AI)](https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies)

### Configuration Management
- [What is Configuration Drift? (2026 Security Explainer)](https://www.reach.security/blog/what-is-configuration-drift-5-best-practices-for-your-teams-security-posture)
- [Configuration Drift Explained (Wiz)](https://www.wiz.io/academy/cloud-security/configuration-drift)
