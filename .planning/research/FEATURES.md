# Feature Audit: LLenergyMeasure v2.0 Decisions vs Peer Tool Evidence

**Domain:** LLM inference efficiency measurement
**Researched:** 2026-02-25
**Mode:** Ecosystem audit of existing `.product/` decisions
**Overall confidence:** MEDIUM-HIGH (8 peer tools surveyed; some via official docs, some via web search)

---

## Executive Summary

The existing `.product/` decisions are largely well-founded. The unified `llem run` design,
composition-based config, sweep grammar, and library-first architecture all align with peer
tool patterns. However, the audit surfaces **three significant gaps** in the existing feature
set that peer tools address and the `.product/` decisions do not:

1. **Phase-aligned energy attribution** (prefill vs decode) -- TokenPowerBench and ML.ENERGY
   both treat this as fundamental; our decisions defer it vaguely to "v2.1"
2. **Power time-series capture** -- Zeus PowerMonitor, TokenPowerBench, and ML.ENERGY v3 all
   capture power traces during inference; our decisions acknowledge this as an "opportunity"
   but do not commit
3. **Pareto analysis / batch-size-energy tradeoff curves** -- ML.ENERGY v3's headline feature
   is the time-energy Pareto frontier; no `.product/` decision addresses post-study analysis

Two decisions are **questionable** when compared to peer evidence. Five are **contradicted** in
minor ways or have gaps. The remaining decisions are **aligned** with or superior to peer
practice.

---

## Part 1: Decision Audit

### Decision: Unified `llem run` (cli-ux.md Sub-decision A)

**Verdict: ALIGNED**

| Peer Tool | Pattern | Notes |
|-----------|---------|-------|
| lm-eval | `lm-eval run` (single command) | Since Dec 2025 refactor: `run`, `ls`, `validate` |
| MLflow | `mlflow run` | Single entry point |
| pytest | `pytest` (single command) | Config file determines scope |
| vLLM bench | `vllm bench serve`, `vllm bench latency`, `vllm bench throughput`, `vllm bench sweep serve` | Multiple subcommands -- different scope from llem |
| optimum-benchmark | `optimum-benchmark --config-dir X --config-name Y` | Single command, Hydra determines scope |

The unified `llem run` where YAML determines scope matches the dominant pattern (lm-eval,
MLflow, pytest, optimum-benchmark). vLLM's bench suite is the exception -- it uses separate
subcommands for different benchmark types -- but vLLM is a backend-specific tool, not a
cross-backend framework.

**Confidence:** HIGH (verified via official docs)

---

### Decision: 2 commands + 1 flag (cli-ux.md)

**Verdict: ALIGNED**

lm-eval uses 3 subcommands (`run`, `ls`, `validate`). `ls` lists available tasks --
analogous to nothing in llem's domain (we do not have a task registry). `validate` is a
config checker -- subsumed by llem's `--dry-run` flag on `run`. Two commands (`run`, `config`)
plus `--version` is the minimum viable surface for a research tool.

No peer tool has a `check`, `init`, or `results` command. This was already well-documented in
the existing `11-peer-cli-patterns.md` research and remains correct.

**Confidence:** HIGH

---

### Decision: Zero-config `llem run --model X` (cli-ux.md Sub-decision C)

**Verdict: ALIGNED, with one gap**

| Peer Tool | Zero-config story |
|-----------|-------------------|
| lm-eval | `lm-eval run --model hf --tasks hellaswag` -- 2 required flags |
| optimum-benchmark | Requires YAML config or full Hydra overrides -- no zero-config |
| vLLM bench | Requires `--model` at minimum, many optional flags |

llem's zero-config with just `--model X` is better than most peers. The gap: **no peer tool
uses interactive prompts for backend selection**. The `llem run --model X` design says
"interactive prompt selects backend if multiple are installed (matches `gh pr create` pattern)".
This is novel for ML tooling. Recommendation: keep it, but ensure `--backend` flag can bypass
the prompt for scripted usage (already implied by the design).

**Confidence:** MEDIUM (interactive prompt pattern is untested in ML CLI space)

---

### Decision: Composition config with optional backend sections (config-architecture.md C1)

**Verdict: ALIGNED**

| Peer Tool | Config pattern |
|-----------|---------------|
| optimum-benchmark | Inheritance -- separate `PyTorchConfig`, `VLLMConfig`, etc. |
| lm-eval | Flat `model_args` string -- no structured config |
| TokenPowerBench | Declarative YAML with engine/model/hardware sections |
| vLLM bench | CLI flags only, no config objects |

optimum-benchmark uses inheritance (the rejected option). TokenPowerBench uses a flat
declarative config. The composition approach is defensible -- the rejection of inheritance
(type pollution through every layer) is well-reasoned and the `.product/` rationale is strong.

**Confidence:** HIGH

---

### Decision: Dotted sweep notation `pytorch.batch_size: [1, 8, 32]` (config-architecture.md C2)

**Verdict: ALIGNED, no direct peer precedent but closest analogue validates approach**

| Peer Tool | Sweep syntax |
|-----------|-------------|
| Hydra | `backend.device=cpu,cuda` -- dotted notation with `-m` flag |
| W&B Sweeps | Flat `parameters:` dict, no backend scoping |
| vLLM bench sweep | JSON file with parameter combinations, Cartesian product |
| Optuna | Programmatic `trial.suggest_*()` |

The dotted notation is closest to Hydra's override syntax (`backend.device=cpu,cuda`).
vLLM's approach (JSON file with explicit parameter dicts) is the closest alternative for
a benchmark-specific sweep. The `.product/` decision's per-backend Cartesian product with
independent grids is novel but logically sound.

**Confidence:** MEDIUM (novel syntax, no exact peer match, but Hydra precedent validates the
dotted-key concept)

---

### Decision: Execution profiles as CLI shorthand only (cli-ux.md Sub-decision D)

**Verdict: ALIGNED**

No peer tool uses named execution profiles for statistical rigour presets. This was already
well-researched in `16-execution-profiles-patterns.md`. The decision to separate machine-local
thermal settings from study rigour settings is correct and unique to llem's energy measurement
domain.

vLLM bench sweep defaults to 3 runs per configuration -- similar to llem's CLI effective
default of n_cycles=3. This validates the "3 cycles is reasonable default" decision.

**Confidence:** HIGH

---

### Decision: `ExperimentConfig` = pure, `StudyConfig` = thin resolved container (experiment-study-architecture.md Option C)

**Verdict: ALIGNED**

| Peer Tool | Architecture |
|-----------|-------------|
| optimum-benchmark | `BenchmarkConfig` wraps `ScenarioConfig` + `BackendConfig` + `LauncherConfig` |
| vLLM bench sweep | Separate `serve_params` + `bench_params`, Cartesian product at runtime |
| TokenPowerBench | Flat declarative config, no study/experiment distinction |

The Option C architecture (sweep resolution at YAML parse time, StudyConfig as resolved list)
is cleaner than any peer implementation. optimum-benchmark's Hydra-based composition is the
closest analogue. The decision is well-supported.

**Confidence:** HIGH

---

### Decision: `llem.run()` unified library API (experiment-study-architecture.md Q3)

**Verdict: QUESTIONABLE -- union return type is unusual**

| Peer Tool | Library API |
|-----------|------------|
| lm-eval | `simple_evaluate()` -- returns `dict` always |
| optimum-benchmark | `Benchmark.launch(config)` -- returns `BenchmarkReport` always |
| Zeus | `ZeusMonitor.end_window()` -- returns `Measurement` always |

No peer tool returns a union type (`ExperimentResult | StudyResult`). They all return a single
type. The llem design's union return creates downstream issues:

1. Library callers must check the type or know their input type
2. Type narrowing is required for correct IDE autocomplete
3. The convenience of unwrapping `StudyResult.experiments[0]` for single experiments adds a
   semantic discontinuity

**Challenge:** Consider always returning `StudyResult` and providing a convenience accessor
like `result.experiment` (singular) that returns the first experiment for single-run cases.
This eliminates the union type while preserving the ergonomic convenience.

**Counter-argument:** The union return was explicitly decided to avoid wrapping single
experiments in unnecessary study containers. The decision acknowledges this tradeoff. It is
defensible but worth revisiting -- no peer tool uses this pattern.

**Confidence:** MEDIUM (architectural preference, not a hard error)

---

### Decision: Exit codes 0/1/2/130 (error-handling.md K1)

**Verdict: ALIGNED**

Matches lm-eval, pytest, and standard Unix convention. No issues.

**Confidence:** HIGH

---

### Decision: Rich + tqdm progress display (live-observability.md)

**Verdict: ALIGNED, with one observation**

| Peer Tool | Progress display |
|-----------|-----------------|
| vLLM bench | Rich terminal UI with GPU monitoring |
| AIPerf | Dashboard TUI, simple (progress bars), or none |
| lm-eval | Minimal progress (tqdm) |
| GuideLLM | Rich terminal UI with performance metrics |

The decision to use Rich at study level and tqdm at experiment level is pragmatic. However,
**AIPerf and vLLM bench suite now have real-time TUI dashboards** showing GPU metrics during
runs. The `.product/` decision defers live power display to v2.1, which is fine, but the TUI
trend in peer tools suggests this will become expected faster than anticipated.

**Confidence:** HIGH

---

### Decision: v2.x CLI-first, v3.0 lm-eval, v4.0 web (versioning-roadmap.md)

**Verdict: QUESTIONABLE -- the v2.0-v2.4 micro-versioning is too granular**

The task brief says the old v2.0-v2.4 micro-versions have been collapsed into a single v2.0.
This is the right call. The old roadmap's v2.1 (measurement depth), v2.2 (Docker), v2.3
(parameter completeness), v2.4 (shareability) granularity assumes a linear single-developer
path. A single v2.0 that delivers the core research tool, with Docker as a follow-on when
needed, is more realistic.

However, the v3.0 = lm-eval integration decision deserves scrutiny. **lm-eval has restructured
its packaging** (Dec 2025): the base package no longer includes transformers/torch, and backends
are installed separately (`pip install lm_eval[hf]`, `pip install lm_eval[vllm]`). This makes
lm-eval lighter-weight to integrate as an optional dependency. The integration may be simpler
than originally estimated.

**Confidence:** MEDIUM (version granularity is a project management call, not a technical one)

---

### Decision: Output as flat JSON per experiment (output-storage.md)

**Verdict: ALIGNED, with gap**

| Peer Tool | Output format |
|-----------|--------------|
| lm-eval | JSON per evaluation run |
| optimum-benchmark | `benchmark_report.json` + `benchmark_report.txt` per run |
| vLLM bench sweep | Results directory with CSV and JSON per parameter combination |
| TokenPowerBench | Per-run results with power traces |

Flat JSON per experiment is standard. The gap: **no decision addresses power time-series
storage**. If/when power traces are captured (Zeus PowerMonitor), they produce high-frequency
data (1-10 Hz) that does not belong in the main JSON result. A separate Parquet or CSV file
per experiment is needed. This is not addressed in `output-storage.md`.

**Confidence:** HIGH (JSON is fine; time-series storage is a gap)

---

### Decision: `extra = "forbid"` on configs (config-architecture.md)

**Verdict: ALIGNED**

Hydra does not have this (it allows arbitrary overrides). Pydantic's `extra = "forbid"` is
stricter -- correct for a measurement tool where a typo in `bachsize: 8` could produce
scientifically wrong results. No peer tool has this level of config strictness, but the
rationale (scientific integrity) is specific to llem's domain.

**Confidence:** HIGH

---

## Part 2: Features Peers Have That `.product/` Decisions Do Not Address

### Gap 1: Phase-Aligned Energy Attribution (Prefill vs Decode)

**Priority: HIGH -- this is a credibility gap for an energy measurement tool**

| Peer Tool | Phase attribution |
|-----------|-----------------|
| TokenPowerBench | Core feature: "phase-aligned metrics pipeline that attributes energy to prefill and decode stages of every request" |
| ML.ENERGY v3 | Measures energy during "steady state" of batch-saturated generation; distinguishes phases |
| AIEnergyScore | Measures "preprocess, prefill, and decode" separately |
| llem `.product/` | No decision addresses this. `versioning-roadmap.md` mentions "prefill/decode phase split" at v2.3. |

**Assessment:** All three energy-focused peer tools measure prefill and decode energy
separately. This is not a v2.3 feature -- it is a fundamental aspect of LLM energy
measurement. A 70B model's prefill phase consumes dramatically different power than its decode
phase (prefill is compute-bound, decode is memory-bound). Without phase attribution, llem's
energy numbers are a single aggregate that hides the most interesting information.

**Recommendation:** Move prefill/decode phase-split energy attribution to v2.0 scope. The
implementation is not complex: it requires timestamping the transition from prefill to decode
(which llem already tracks via TTFT) and aligning power measurements to those timestamps.

**Confidence:** HIGH (three peer tools, two papers)

---

### Gap 2: Power Time-Series Capture

**Priority: HIGH -- enables phase attribution and is table stakes for energy tools**

| Peer Tool | Power time-series |
|-----------|-------------------|
| Zeus | `PowerMonitor` with `get_power_timeline()` -- 1-10 Hz sampling |
| TokenPowerBench | NVML/DCGM sampling at 1-10 Hz, temporal alignment with inference phases |
| ML.ENERGY v3 | Power draw trends across configurations |
| llem `.product/` | `live-observability.md` defers "live power display" to v2.1. No decision on power data capture. |

**Assessment:** The `.product/` decisions conflate "displaying live power" (a UI feature) with
"capturing power time-series data" (a measurement feature). These are independent. Capturing
the data is essential for phase attribution and post-hoc analysis. Displaying it live is a
nice-to-have.

**Recommendation:** Add power time-series capture to v2.0 scope (via Zeus `PowerMonitor` if
Zeus is the energy backend, or via direct NVML polling otherwise). Store traces as Parquet
alongside the experiment JSON. Defer live power display to later.

**Confidence:** HIGH

---

### Gap 3: Pareto Analysis / Time-Energy Tradeoff Curves

**Priority: MEDIUM -- differentiator for studies, not single experiments**

| Peer Tool | Pareto analysis |
|-----------|----------------|
| ML.ENERGY v3 | "Time-energy Pareto frontier" -- core feature. Recommends minimum-energy config for a given latency constraint. |
| vLLM bench sweep | `sweep plot_pareto` command for Pareto charts |
| AIPerf | "Pareto analysis and visualisation" |
| Zeus | Batch size optimiser finds energy-optimal batch size |
| llem `.product/` | No decision addresses post-study analysis or visualisation. |

**Assessment:** For a tool that measures "how implementation choice affects efficiency," the
natural output of a study is a tradeoff curve: "here are the configurations on the Pareto
frontier of latency vs energy." ML.ENERGY v3's headline is exactly this. llem's study output
is a set of JSON files with no analysis layer on top.

**Recommendation:** Add a simple Pareto extraction to `StudyResult` -- not a visualisation
tool, but a data structure that identifies the Pareto-optimal experiments from a study. Users
can then plot with matplotlib/seaborn. This belongs in v2.0 as a library feature, not a CLI
command.

**Confidence:** MEDIUM (ML.ENERGY and vLLM bench do this; it is becoming expected for sweep
tools)

---

### Gap 4: Environment Metadata Capture

**Priority: HIGH -- already identified in old FEATURES.md but no `.product/` decision exists**

| Peer Tool | Metadata captured |
|-----------|-------------------|
| TokenPowerBench | Hardware topology, GPU model, driver version, CUDA version |
| optimum-benchmark | System information in benchmark report |
| vLLM bench | GPU model, VRAM, driver info in results |
| AIEnergyScore | Hardware standardised (H100 only), but records full system info |
| llem `.product/` | No decision in `decisions/`. `versioning-roadmap.md` mentions "env metadata" at v2.1. |

**Assessment:** The old FEATURES.md flagged this as "P0 critical for credibility." It remains
correct. An energy measurement without recording the GPU model, driver version, CUDA version,
and system state is not reproducible. This should be in v2.0.

The `designs/result-schema.md` should include an `EnvironmentSnapshot` with:
- GPU model, VRAM, compute capability, driver version
- CUDA version, backend library versions
- Python version, llem version
- CPU model, total RAM
- Baseline GPU temperature at start

**Recommendation:** This is a v2.0 table-stakes feature. Add to scope.

**Confidence:** HIGH

---

### Gap 5: Confidence Intervals / Bootstrap CI

**Priority: MEDIUM -- differentiator for multi-cycle studies**

| Peer Tool | Statistical reporting |
|-----------|---------------------|
| AIEnergyScore | 10 runs per model, reports average |
| ML.ENERGY v3 | Reports per-token energy with variance analysis |
| vLLM bench sweep | 3 runs per configuration by default, reports mean |
| lm-eval | Reports mean/stderr for accuracy metrics |
| llem `.product/` | `versioning-roadmap.md` mentions "bootstrap CI" at v2.1. Multi-cycle exists. |

**Assessment:** No peer tool currently reports confidence intervals for performance/energy
metrics. lm-eval reports stderr for accuracy, but that is a different domain. The opportunity
is real but not urgent. Deferring to post-v2.0 is defensible.

**Recommendation:** Keep as post-v2.0. Report mean, median, std dev, min, max for multi-cycle
results in v2.0. Add bootstrap CI when there is user demand.

**Confidence:** MEDIUM

---

### Gap 6: `--dry-run` for Grid Preview

**Priority: MEDIUM -- no peer tool has this, but it is mentioned in `.product/` without a decision**

| Peer Tool | Dry run / grid preview |
|-----------|----------------------|
| Hydra | `--cfg job` shows resolved config; no grid preview |
| vLLM bench sweep | No preview; runs immediately |
| W&B Sweeps | No preview |
| llem `.product/` | `cli-ux.md` mentions `--dry-run` for validation. No design for study grid preview. |

**Assessment:** The `.product/` decisions mention `--dry-run` covers "active config validation"
but do not specify what it shows for studies. For a study with `sweep:` grammar generating 50+
experiments, users need to preview the grid before committing hours of GPU time.

**Recommendation:** `llem run study.yaml --dry-run` should:
1. Resolve the sweep grammar
2. Show the full experiment grid (count, configurations)
3. Estimate total runtime and VRAM requirements
4. Validate all configs (pre-flight for each)
5. Exit without running

This is a v2.0 feature. Add to scope.

**Confidence:** MEDIUM (no peer does this well, but it is logically necessary for expensive
experiment grids)

---

### Gap 7: VRAM Pre-Flight Estimation

**Priority: MEDIUM -- partially addressed in research but no decision exists**

| Peer Tool | VRAM estimation |
|-----------|----------------|
| vLLM | Internal profile_run during engine init (not externally callable) |
| HF Accelerate | `accelerate estimate-memory` CLI (weight memory only) |
| llem `.product/` | `10-sweep-validation-patterns.md` research covers this in detail. No decision. |

**Assessment:** The research file `10-sweep-validation-patterns.md` has a thorough analysis of
VRAM estimation patterns and even proposes an `estimate_vram()` function. But no `.product/`
decision commits to building this. For expensive sweeps, pre-flight VRAM checking prevents
wasting GPU time on OOM configurations.

**Recommendation:** Add `estimate_vram()` as a pre-flight check in v2.0. Use the formula from
the research file. Not a separate command -- part of `--dry-run` and optional pre-flight.

**Confidence:** MEDIUM

---

### Gap 8: HuggingFace Hub Results Sharing

**Priority: LOW for v2.0 -- but worth noting**

| Peer Tool | Results sharing |
|-----------|---------------|
| optimum-benchmark | Push results directly to HuggingFace Hub |
| lm-eval | `--hf_hub_log_args` flag to log results to Hub |
| llem `.product/` | `versioning-roadmap.md` defers `llem results push` to v2.4 |

**Assessment:** Two major peer tools support HF Hub integration. Deferring to v2.4 is
defensible for a v2.0 release, but the implementation is lightweight (HF Hub API is simple).

**Recommendation:** Keep deferred. But note that HF Hub integration is simpler than building a
custom central DB (the v2.4 plan). Consider HF Hub as the first sharing mechanism rather than
a custom DB.

**Confidence:** MEDIUM

---

## Part 3: Feature Landscape Summary

### Table Stakes (must have for credibility)

| Feature | Status in `.product/` | Peer Evidence | Verdict |
|---------|----------------------|---------------|---------|
| TTFT, ITL, throughput, latency | Decided, exists in v1 | Universal | OK |
| Multi-backend support | Decided (3 backends) | optimum-benchmark has 15+, but 3 is defensible | OK |
| Batch size / precision / quantisation control | Decided | Universal | OK |
| Config-driven experiments (YAML) | Decided | optimum-benchmark (Hydra), TokenPowerBench (declarative) | OK |
| Warmup runs | Decided | Universal | OK |
| JSON result output | Decided | Universal | OK |
| Environment metadata capture | **NOT decided** | TokenPowerBench, optimum-benchmark, vLLM bench | **GAP** |
| Prefill/decode phase-split energy | Deferred to v2.3 | TokenPowerBench, ML.ENERGY, AIEnergyScore | **GAP** |
| Power time-series capture | Deferred to v2.1 | Zeus, TokenPowerBench | **GAP** |
| Pre-flight validation | Decided (implicit in run) | No peer does this well | AHEAD |
| Instructive error messages | Decided | No peer does this well | AHEAD |

### Differentiators (set llem apart)

| Feature | Status in `.product/` | Peer Evidence | Verdict |
|---------|----------------------|---------------|---------|
| Energy per token (Joules) | Decided, exists in v1 | Only 4 tools do this | STRONG |
| Cross-backend energy comparison | Decided | No peer does this | UNIQUE |
| Sweep grammar with backend scoping | Decided (dotted notation) | Novel; Hydra closest | UNIQUE |
| Thermal gap management (configurable) | Decided | No peer does this | UNIQUE |
| Baseline-adjusted energy | Mentioned in research | TokenPowerBench does this | EXPECTED |
| Config hash for reproducibility | Decided | Novel | STRONG |
| `--dry-run` with grid preview | Mentioned but not designed | No peer does this well | OPPORTUNITY |
| Pareto frontier extraction | **NOT decided** | ML.ENERGY v3, vLLM bench sweep | **GAP** |
| FLOPs estimation | Decided, exists in v1 | Rare; academic value | STRONG |

### Anti-Features (correctly excluded)

| Anti-Feature | `.product/` Status | Peer Evidence | Verdict |
|--------------|-------------------|---------------|---------|
| Training benchmarks | Not planned | Correct -- different domain (MLPerf Training) | CORRECT |
| LLM-as-judge quality scoring | Not planned | Correct -- scope creep (lm-eval does this) | CORRECT |
| Closed-loop optimisation | Not planned | Zeus has batch size optimiser, but not benchmarking | CORRECT |
| API endpoint benchmarking | Not planned | Different scope (AIPerf, LLMPerf) | CORRECT |
| Perfect reproducibility claims | Not planned | Correct -- impossible with BF16 | CORRECT |

---

## Part 4: Peer Tool Feature Matrix

| Feature | llem (decided) | optimum-bench | lm-eval | vLLM bench | TokenPowerBench | ML.ENERGY | AIPerf | Zeus | AIEnergyScore |
|---------|---------------|---------------|---------|------------|-----------------|-----------|--------|------|---------------|
| Energy measurement | Yes (CodeCarbon) | Yes (CodeCarbon) | No | No | Yes (NVML/RAPL) | Yes (Zeus) | DCGM telemetry | Yes (NVML/RAPL) | Yes (CodeCarbon) |
| Prefill/decode energy | **No (v2.3)** | No | No | No | **Yes** | **Yes** | No | No | **Yes** |
| Power time-series | **No (v2.1)** | No | No | No | **Yes** | **Yes** | Yes | **Yes** | No |
| Multi-backend | 3 | 15+ | 4+ (HF, vLLM, GGUF, API) | 1 (vLLM) | 4 | 1 (vLLM) | API | N/A | 1 |
| Sweep/grid | Yes (dotted) | Yes (Hydra) | No | Yes (JSON) | Partial | Yes | No | No | No |
| Pareto analysis | **No** | No | No | **Yes** | No | **Yes** | **Yes** | **Yes** (batch opt) | No |
| Library API | `llem.run()` | `Benchmark.launch()` | `simple_evaluate()` | Scripts only | CLI only | Scripts only | CLI only | `ZeusMonitor` | Scripts only |
| Env metadata | **No** | Yes | Partial | Yes | Yes | Yes | Yes | N/A | Yes |
| CI/statistics | Multi-cycle | No | Mean/stderr | 3 runs/mean | No | Variance | Percentiles | No | 10 runs/mean |
| Process isolation | Yes (subprocess) | Yes (process/torchrun) | No | Separate server | No | N/A | N/A | N/A | N/A |
| Docker images | Planned (v2.2) | Yes (4 images) | No | No | No | No | No | No | No |
| Config validation | `extra="forbid"` | Hydra | No | No | No | No | No | N/A | No |

---

## Part 5: Feature Dependencies for v2.0

```
Environment Metadata (EnvironmentSnapshot)
    |-- no dependencies, implement first
    |
    v
Power Time-Series Capture (Zeus PowerMonitor or direct NVML polling)
    |-- depends on: Zeus integration (already planned as energy backend)
    |
    v
Phase-Aligned Energy Attribution (prefill vs decode)
    |-- depends on: TTFT timestamp (already have) + power time-series
    |-- depends on: aligning power samples to inference phases
    |
    v
Pareto Frontier Extraction (from StudyResult)
    |-- depends on: study execution (already designed)
    |-- depends on: per-experiment energy + latency in results
    |-- no dependency on power time-series (uses aggregate metrics)

--dry-run Grid Preview
    |-- depends on: sweep resolution (already designed in Option C)
    |-- depends on: VRAM estimation (research exists, no decision)
    |-- independent of energy features
```

**Critical path for v2.0:** Environment metadata --> power time-series capture --> phase
attribution. These three form a dependency chain. Pareto extraction and dry-run preview are
independent and can be implemented in parallel.

---

## Part 6: MVP Recommendation for v2.0

Based on the audit, the `.product/` decisions should be updated to include:

### Must-add to v2.0 scope (peer evidence demands it)

1. **Environment metadata capture** -- EnvironmentSnapshot in every ExperimentResult. GPU
   model, VRAM, driver, CUDA, Python, llem version, backend versions, baseline temperature.
   Every peer tool does this.

2. **Prefill/decode phase-split energy** -- Pull forward from v2.3. TokenPowerBench,
   ML.ENERGY, and AIEnergyScore all measure this. Without it, llem's energy numbers are less
   useful than peers'.

3. **Power time-series capture** -- Pull forward from v2.1. Required for phase attribution.
   Store as Parquet alongside experiment JSON.

4. **`--dry-run` grid preview for studies** -- Resolve sweep, show grid, estimate runtime,
   validate all configs. No peer does this well; it is a differentiator for expensive GPU
   experiments.

### Should-add to v2.0 scope (strong peer evidence, moderate effort)

5. **Pareto frontier extraction in StudyResult** -- Not a CLI command or visualisation tool,
   but a `pareto_optimal: list[ExperimentResult]` field or method on StudyResult. ML.ENERGY v3
   and vLLM bench sweep both do this.

6. **VRAM pre-flight estimation** -- The research exists (`10-sweep-validation-patterns.md`).
   Build the `estimate_vram()` function. Use in `--dry-run` and optionally in pre-flight.

### Keep deferred (peer evidence supports deferral)

7. **Confidence intervals / bootstrap CI** -- No peer tool does this for energy/perf metrics.
   Mean + median + std dev is sufficient for v2.0.

8. **HF Hub results sharing** -- Two peers support it, but it is a convenience, not a
   credibility feature. v2.4+ is fine.

9. **Live power display** -- UI feature, not measurement feature. v2.1+ is fine once power
   capture (the measurement) is in v2.0.

10. **Additional backends (SGLang, llama.cpp)** -- Desirable but not v2.0. Three backends is
    a defensible starting point.

---

## Part 7: Recommendations for `.product/` Decision Updates

### 1. Create new decision: `environment-metadata.md`
Content: EnvironmentSnapshot schema, what to capture, where to store it, how it affects
reproducibility claims. Reference TokenPowerBench and optimum-benchmark patterns.

### 2. Update `versioning-roadmap.md`
Pull "prefill/decode phase split" from v2.3 to v2.0.
Pull "power time-series" and "env metadata" from v2.1 to v2.0.
The rationale: these are table-stakes for an energy measurement tool in 2026. Peers already
have them.

### 3. Update `experiment-study-architecture.md`
Add Pareto frontier extraction as a method on `StudyResult`. Document the interface.

### 4. Create new decision: `dry-run-design.md`
Content: what `--dry-run` shows for single experiments vs studies. Grid preview, VRAM
estimation, runtime estimation.

### 5. Update `output-storage.md`
Add decision for power time-series storage format (Parquet per experiment, alongside JSON).

### 6. Consider revisiting union return type in `llem.run()`
The union `ExperimentResult | StudyResult` return has no peer precedent. Consider always
returning `StudyResult` with a convenience accessor. Low priority -- the current design works,
it is just unusual.

---

## Sources

### Official Documentation (HIGH confidence)
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM Parameter Sweeps](https://docs.vllm.ai/en/latest/benchmarking/sweeps/)
- [Zeus Measuring Energy](https://ml.energy/zeus/measure/)
- [lm-eval Python API](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)
- [AI Energy Score Methodology](https://huggingface.github.io/AIEnergyScore/)
- [optimum-benchmark README](https://github.com/huggingface/optimum-benchmark/blob/main/README.md)

### Research Papers (HIGH confidence)
- [TokenPowerBench (Dec 2025)](https://arxiv.org/html/2512.03024v1) -- phase-aligned energy
- [ML.ENERGY Benchmark (May 2025)](https://arxiv.org/html/2505.06371v1) -- Pareto frontiers
- [ML.ENERGY Leaderboard v3 blog](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/)

### GitHub Repositories (MEDIUM-HIGH confidence)
- [AIPerf (NVIDIA)](https://github.com/ai-dynamo/aiperf)
- [Zeus (ml-energy)](https://github.com/ml-energy/zeus)
- [GuideLLM (vllm-project)](https://github.com/vllm-project/guidellm)
- [lm-evaluation-harness releases](https://github.com/EleutherAI/lm-evaluation-harness/releases)

### Web Search (MEDIUM confidence)
- [NVIDIA LLM Benchmarking Fundamentals](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [AI Energy Score v2 blog](https://huggingface.co/blog/sasha/ai-energy-score-v2)
- [Red Hat GuideLLM article](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference)
