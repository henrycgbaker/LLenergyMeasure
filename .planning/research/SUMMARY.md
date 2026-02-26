# Project Research Summary

**Project:** LLenergyMeasure v2.0
**Domain:** LLM inference efficiency measurement and benchmarking
**Researched:** 2026-02-25
**Confidence:** MEDIUM-HIGH

## Executive Summary

LLenergyMeasure's `.product/` decision set is broadly well-designed. Across six research files auditing stack, features, architecture, pitfalls, decision consistency, and the API return-type question, the majority of decisions hold up against peer evidence from 8+ tools (lm-eval, optimum-benchmark, Zeus, vLLM, MLflow, TokenPowerBench, ML.ENERGY, AIEnergyScore). The core architecture -- library-first, composition-based config, subprocess isolation, Option C experiment-study separation, unified `llem run` CLI -- is sound and in several areas ahead of peer practice. However, the research surfaces **three blocking issues**, **several stale dependency pins**, **a fundamental cross-document inconsistency**, and **three missing table-stakes features** that must be resolved before the harmonisation phase produces a consistent document set for implementation.

The three most consequential findings are: (1) the unified `run()` library API with union return type `ExperimentResult | StudyResult` contradicts all peer practice (0/10 tools use this pattern) and official Python typing guidance -- the original split API (`run_experiment()` / `run_study()`) was the correct design; (2) FLOPs as a "primary metric" is misleading for this tool's stated purpose because FLOPs are deterministic for a given model+input and do not vary between deployment configurations, backends, or batch sizes -- the actual primary metrics are `energy_per_output_token` and `tokens_per_second`; (3) three features that every energy-focused peer tool treats as table-stakes are currently deferred past v2.0: environment metadata capture, prefill/decode phase-split energy, and power time-series capture. Additionally, the `zeus-ml` PyPI package has been renamed to `zeus`, making the current `pyproject.toml` install stale code.

Key risks are concentrated in measurement methodology: the 30-second thermal floor is under-calibrated (MLPerf requires 60 seconds under load), NVML accuracy is +/-5 watts (not +/-5 percent as documented), and baseline power subtraction is under-specified. These are addressable through well-defined calibration work. The cross-document inconsistency (15+ files still reference `llem study` and `run_experiment()`/`run_study()` after the 2026-02-25 unification decision) is a documentation debt that must be resolved during harmonisation to prevent implementers from building the wrong API.

---

## 1. Critical Issues Requiring Immediate Resolution

These block the harmonisation phase. They must be resolved before writing implementation specs.

### 1.1 Library API: Union Return Type Must Be Reverted to Split Functions

**Source:** `UNIFIED-API-RESEARCH.md`, `ARCHITECTURE.md` (Decision H), `DECISION-AUDIT.md` (section 2.5, CI-1)

The `experiment-study-architecture.md` decision (2026-02-25) mandated `llem.run(config) -> ExperimentResult | StudyResult`. This directly contradicts:
- **0/10 peer tools** use a union return type based on input count (pytest, Optuna, Ray Tune, lm-eval, MLflow, optimum-benchmark, W&B, scikit-learn, Hydra, Zeus)
- **Official Python typing guidance** (typing.python.org): "Avoid union return types, since they require isinstance() checks"
- **Guido van Rossum** (mypy issue #1693): explicitly advises against union return types
- **The project's own `designs/library-api.md`**, which argued against union returns and designed `run_experiment()` + `run_study()` for exactly this reason

The `@overload` escape hatch does not work for the common case: `llem.run("config.yaml")` takes `str | Path`, so the type checker cannot resolve whether the return is `ExperimentResult` or `StudyResult`.

**Recommendation:** Revert to the `library-api.md` design: two public functions with unambiguous return types, backed by an internal `_run(StudyConfig) -> StudyResult` runner. This matches the Option C internal architecture perfectly while giving callers type-safe return types.

```
Public:  run_experiment(config) -> ExperimentResult
         run_study(config) -> StudyResult
Internal: _run(StudyConfig) -> StudyResult  (Option C architecture unchanged)
```

The CLI remains unified `llem run` regardless -- this is a library API question only.

### 1.2 Cross-Document Inconsistency: 15+ Files Reference Superseded API

**Source:** `DECISION-AUDIT.md` (sections CI-1 through CI-7)

The 2026-02-25 unification decision changed the CLI from 3 commands to 2 and the library API from split to unified, but the change was NOT propagated. The following still reference `llem study`, `run_experiment()`, and/or `run_study()`:

- `.product/CLAUDE.md` (the instructions file for future Claude sessions -- highest-risk staleness)
- `designs/library-api.md`, `backward-compatibility.md`, `experiment-isolation.md`
- `study-execution-model.md`, `live-observability.md`, `study-resume.md`
- `output-storage.md`, `documentation-strategy.md`, `progressive-disclosure.md`
- `versioning-roadmap.md`, `installation.md`, `architecture.md` (sub-decision J)

**Resolution:** During harmonisation, all files must be updated to reflect whichever API shape is chosen (the research recommends reverting to the split API, which means updating `experiment-study-architecture.md` Q3 instead of updating the other 15 files).

### 1.3 zeus-ml Package Renamed to zeus

**Source:** `STACK.md` (section 1)

The PyPI package `zeus-ml` stopped at v0.11.0. The project moved to `zeus` (currently v0.13.1). Installing `zeus-ml` installs stale code. The `pyproject.toml` extras must be updated from `zeus-ml` to `zeus` immediately.

### 1.4 FLOPs Framing as "Primary Metric" Must Change

**Source:** `PITFALLS.md` (CP-3), `DECISION-AUDIT.md` (section 2.14)

FLOPs = `2 * N_params * tokens`. This is deterministic for a given model and input. It does not change between PyTorch/vLLM/TensorRT-LLM, between batch sizes, between precision settings, or between any deployment parameter this tool measures. Positioning FLOPs as a "primary cross-run comparison metric" is scientifically misleading for a tool whose purpose is measuring implementation choice effects.

**Recommendation:** Demote FLOPs to reference metadata. Promote `energy_per_output_token` (J/token) and `tokens_per_second` as primary metrics. Keep FLOPs in the result schema for cross-model normalisation. Defer MFU (which IS diagnostic) to v2.1 as planned.

---

## 2. Decisions Confirmed by Research

These hold up under peer evidence and should proceed as-is.

| Decision | Source | Evidence |
|----------|--------|----------|
| Library-first architecture | `ARCHITECTURE.md` (A) | lm-eval, optimum-benchmark, Zeus, CodeCarbon all library-first |
| Module structure (`src/llenergymeasure/`) | `ARCHITECTURE.md` (B) | All peers use single package |
| Unified `llem run` CLI (YAML determines scope) | `FEATURES.md`, `DECISION-AUDIT.md` (2.7) | lm-eval, pytest, MLflow all use single unified command |
| 2 commands + 1 flag | `FEATURES.md` | Minimum viable surface; no peer has `init`, `check`, or `results` commands |
| Composition config (not inheritance) | `ARCHITECTURE.md` (C1) | Inheritance creates union type pollution; composition is cleaner |
| Dotted sweep notation | `ARCHITECTURE.md` (C2), `FEATURES.md` | Novel but closest to Hydra's dotted override syntax; well-designed |
| Execution profiles as CLI shorthand only | `DECISION-AUDIT.md` (2.10) | No peer uses named profiles for statistical rigour presets |
| Option C: ExperimentConfig pure, StudyConfig resolved | `ARCHITECTURE.md` | Cleanest architecture; sweep at parse time is correct |
| Subprocess isolation via `multiprocessing.Process` + `spawn` | `ARCHITECTURE.md` (I) | Matches optimum-benchmark exactly; CUDA correctness requirement |
| Energy vs CO2 separation | `STACK.md` (section 1), `ARCHITECTURE.md` (G) | Zeus = energy only, CodeCarbon = CO2 only; peers confirm |
| Pydantic v2 (reject Hydra) | `STACK.md` (section 3) | Hydra conflicts with parse-time resolution architecture |
| Typer for v2.0 CLI | `STACK.md` (section 4) | Adequate for 2-command surface; cyclopts evaluation at v3.0 |
| `extra = "forbid"` on configs | `FEATURES.md` | Correct for scientific integrity; no peer is this strict |
| Error handling: `LLEMError` hierarchy + exit codes 0/1/2/130 | `ARCHITECTURE.md`, `DECISION-AUDIT.md` (2.18) | Matches httpx/SQLAlchemy pattern; standard |
| `__init__.py` exports as sole stable API | `ARCHITECTURE.md` | Matches lm-eval, httpx, Pydantic |
| Zero backend deps at base; pip extras | `DECISION-AUDIT.md` (2.12) | Exact pattern used by lm-eval, optimum-benchmark, Transformers |
| Fixed-count warmup (reject CV convergence) | `DECISION-AUDIT.md` (2.13) | No peer implements CV convergence; adds unbounded runtime |
| Study ships at v2.0 | `ARCHITECTURE.md` (J) | Sweeps are the differentiator; deferring would ship an inferior single-experiment tool |
| Dataset: AIEnergyScore default, BYOJSONL, synthetic | `DECISION-AUDIT.md` (2.22) | Externally validated standard; enables cross-tool comparison |
| Result schema migration via `schema_version` | `DECISION-AUDIT.md` (2.23) | Better than lm-eval (which has no versioning) |
| HPC/SLURM deferred to v3.x | `DECISION-AUDIT.md` (2.28) | lm-eval also has no SLURM integration; correct for v2.0 audience |
| Rejecting Hydra | `ARCHITECTURE.md` | Hydra takes over entry point; incompatible with library-first; team moving away |
| SGLang accelerated to v2.2 candidate | `STACK.md` (section 2) | PyTorch ecosystem member; RadixAttention creates genuinely different energy profiles |
| llama.cpp deferred to v3.x | `STACK.md` (section 2) | CPU vs GPU energy comparability concern requires design work |

---

## 3. Decisions Challenged by Research

These need updating during harmonisation, with evidence citations.

### 3.1 Three-Layer Config Model -- Simplify to Two Sources + Auto-Capture

**Source:** `ARCHITECTURE.md` (C), `DECISION-AUDIT.md` (2.3)

No peer tool uses a named three-layer config model. The valid separation is:
1. **User config** (`~/.config/llenergymeasure/config.yaml`) -- machine-local preferences
2. **Experiment config** (YAML) -- what to measure, shareable
3. **Environment snapshot** -- auto-captured, stored in results (not a config layer)

"Layer 3" is output metadata, not configuration. Calling it a config layer confuses implementers. Drop the "three-layer" naming; keep the separation.

### 3.2 NVML Accuracy Claims -- Wrong Units

**Source:** `PITFALLS.md` (CP-1)

The `.product/` docs claim Zeus/NVML is "~5% accurate" and CodeCarbon is "~15% accurate". NVML accuracy is actually +/-5 **watts** (not percent). At 300W (A100 under load) that is 1.7%. At 40W (idle) that is 12.5%. Replace fixed-percentage accuracy table with conditional formula: `accuracy_pct = 5W / mean_power_W * 100`.

### 3.3 30-Second Thermal Floor -- Under-Calibrated

**Source:** `PITFALLS.md` (CP-2)

The 30-second figure has no cited source. MLPerf Power requires a minimum 60-second measurement window. Thermal stabilisation under load takes 45-90 seconds on A100/H100. The current 5 warmup runs at 2 tokens each provide ~2-3 seconds of actual GPU compute -- insufficient for thermal conditioning.

**Recommendation:** Increase warmup to 10 full-length runs (matching optimum-benchmark), OR add a separate 60-second thermal conditioning phase running the actual workload before measurement.

### 3.4 Warmup Reduced-Output Strategy -- Insufficient for Energy Benchmarks

**Source:** `PITFALLS.md` (MP-3)

2-token warmup runs do not warm the KV cache at operational size, do not trigger vLLM's continuous batching scheduler, and do not load the GPU thermally. For energy benchmarks, full-length warmup runs are necessary. Make configurable: `warmup_mode: "fast" | "full"` with `full` as default.

### 3.5 Baseline Power Subtraction -- Over-Simplified

**Source:** `PITFALLS.md` (CP-4)

The linear correction `energy_adjusted = total - (idle_power * duration)` assumes constant idle power, but idle power increases as the GPU heats up. More importantly, no peer tool publishes baseline-corrected energy. Raw energy is sufficient for relative comparisons (A vs B on same hardware). Keep `baseline_power_w` as optional metadata; do NOT make `energy_adjusted_j` a primary metric.

### 3.6 Bootstrap CI -- Use BCa, Not Percentile

**Source:** `PITFALLS.md` (CP-5), `STACK.md` (section 8)

Percentile bootstrap has poor coverage for skewed distributions (energy data is typically right-skewed). Use BCa method via `scipy.stats.bootstrap(method='BCa')`. Increase default resamples from 1,000 to 2,000. For energy CIs, require multi-cycle studies and bootstrap over per-cycle totals (not per-request, since Zeus measures total energy per window).

### 3.7 Access Control (.env Pattern) -- Over-Engineered

**Source:** `DECISION-AUDIT.md` (2.26)

ML researchers already use `huggingface-cli login` and `HF_TOKEN` env vars. The `.env` file pattern is unfamiliar to this audience. Rely on `huggingface_hub`'s existing auth chain instead.

### 3.8 Output Contract Divergence

**Source:** `DECISION-AUDIT.md` (2.16)

Unified `llem run` produces either a flat JSON file OR a subdirectory depending on YAML content. Users cannot predict output location without parsing their YAML. Consider always producing a directory (even for single experiments), matching Hydra's approach.

---

## 4. Feature Gaps Identified

Peer evidence shows these features are expected by the research community but missing or deferred in `.product/`.

### Table Stakes (must add to v2.0)

| Gap | Peer Evidence | Current Status | Source |
|-----|---------------|----------------|--------|
| **Environment metadata capture** | TokenPowerBench, optimum-benchmark, vLLM bench, AIEnergyScore -- all capture GPU model, VRAM, driver, CUDA version | No `.product/` decision exists; deferred to v2.1 | `FEATURES.md` (Gap 4) |
| **Prefill/decode phase-split energy** | TokenPowerBench, ML.ENERGY, AIEnergyScore -- core feature in all three | Deferred to v2.3 | `FEATURES.md` (Gap 1) |
| **Power time-series capture** | Zeus PowerMonitor, TokenPowerBench, ML.ENERGY v3 | Deferred to v2.1 | `FEATURES.md` (Gap 2) |
| **`--dry-run` grid preview for studies** | No peer does this well, but logically necessary for expensive sweeps | Mentioned but not designed | `FEATURES.md` (Gap 6) |

### Should-Add (strong evidence, moderate effort)

| Gap | Peer Evidence | Source |
|-----|---------------|--------|
| **Pareto frontier extraction from StudyResult** | ML.ENERGY v3 headline feature; vLLM bench sweep `plot_pareto` | `FEATURES.md` (Gap 3) |
| **VRAM pre-flight estimation** | vLLM internal profile_run; HF Accelerate `estimate-memory` | `FEATURES.md` (Gap 7) |
| **Minimum measurement duration** (10s or loop until 10s) | MLPerf Power 60s minimum; NVML 100ms update period with 25% sampling | `PITFALLS.md` (MP-5) |
| **GPU persistence mode check** | All serious benchmarking assumes persistence mode | `PITFALLS.md` (UA-1) |
| **CPU-GPU synchronisation requirement** | Zeus `sync_execution_with`; ML.ENERGY blog explicitly requires it | `PITFALLS.md` (mP-1) |

### Keep Deferred (peer evidence supports deferral)

| Feature | Rationale | Source |
|---------|-----------|--------|
| Confidence intervals / bootstrap CI | No peer does this for energy/perf; mean + std dev sufficient for v2.0 | `FEATURES.md` (Gap 5) |
| HF Hub results sharing | Convenience, not credibility; v2.4+ fine | `FEATURES.md` (Gap 8) |
| Live power display (TUI) | UI feature; capture the data in v2.0, display later | `FEATURES.md` |
| SGLang, llama.cpp backends | Desirable but three backends is defensible for v2.0 | `STACK.md` |
| Visualisation library | No peer bundles one; JSON/CSV output is standard | `STACK.md` (section 6) |

---

## 5. Stack Updates Required

**Source:** `STACK.md` (full audit)

| Item | Current Pin | Required Pin | Severity | Reason |
|------|------------|-------------|----------|--------|
| `zeus-ml` (PyPI) | `zeus-ml` | `zeus>=0.13.1` | **CRITICAL** | Package renamed; old name installs stale v0.11.0 |
| CodeCarbon | `>=2.8.0` | `>=3.2.2` | HIGH | Major version jump; private API may have changed |
| Transformers | `>=4.49.0` | `>=5.0.0` | HIGH | v5 is a major version; v4 is behind |
| vLLM | `>=0.6.0` | `>=0.15.0` | HIGH | 9 minor versions ahead; test Transformers compat |
| TensorRT-LLM | `>=0.12.0` | `>=1.0.0` | HIGH | Approaching 1.0 stable; breaking from 0.12 |
| nvidia-ml-py | `>=12.0.0` | `>=13.590.48` | MEDIUM | Updated per STACK.md audit |
| scipy | not explicit | `>=1.12` | MEDIUM | Required for `scipy.stats.bootstrap` BCa |
| pytest-mock | not present | `>=3.12` | MEDIUM | GPU/backend mocking for unit tests |
| pytest-xdist | not present | `>=3.5` | MEDIUM | Parallel test execution in CI |
| Poetry | current manager | Consider uv migration | MEDIUM | uv now exceeds Poetry in downloads; 10-100x faster |

---

## 6. Recommended Priority Ordering for Harmonisation

Based on the dependency structure across all research, harmonisation should proceed in this order.

### Priority 0: Resolve API Shape (blocks everything)

Decide between:
- (a) Unified `run()` with union return (current `experiment-study-architecture.md`)
- (b) Split `run_experiment()` / `run_study()` (original `library-api.md`, supported by all research)

**Research recommendation:** Option (b). Then update only `experiment-study-architecture.md` Q3 and `architecture.md` H instead of updating 15+ other files.

### Priority 1: Propagate API Decision Through Document Set

Update all stale references per `DECISION-AUDIT.md` CI-1 through CI-7. Most critically: `.product/CLAUDE.md` (the instructions file read at session start by future Claude instances).

### Priority 2: Add Missing Table-Stakes Decisions

Write new decision files for:
- `environment-metadata.md` -- EnvironmentSnapshot schema
- `dry-run-design.md` -- what `--dry-run` shows for experiments vs studies
- Update `versioning-roadmap.md` -- pull phase-split energy and power time-series into v2.0

### Priority 3: Fix Challenged Decisions

Update in-place (with supersession annotations per `.product/CLAUDE.md` rules):
- `flops-estimation.md` -- demote from primary to reference metadata
- `warmup-strategy.md` -- increase thermal floor, add full-length warmup option
- `architecture.md` C/D/E -- simplify three-layer naming to two sources + auto-capture
- `designs/energy-backends.md` -- replace fixed-percentage accuracy with conditional formula
- `access-control.md` -- simplify to `huggingface_hub` auth chain

### Priority 4: Update Stack Pins

Update `pyproject.toml` and test compatibility:
- `zeus-ml` -> `zeus` (critical)
- Version pins for CodeCarbon, Transformers, vLLM, TRT-LLM
- Add `scipy`, `pytest-mock`, `pytest-xdist`

### Priority 5: Feature Gap Design Work

Write designs for:
- EnvironmentSnapshot schema (fields, capture mechanism)
- Power time-series capture + storage format (Parquet alongside JSON)
- Prefill/decode phase attribution (timestamp alignment with TTFT)
- Pareto frontier extraction on StudyResult
- `--dry-run` grid preview behaviour

---

## 7. Implications for Roadmap

### Suggested Phase Structure

Based on dependency chains identified across the research:

```
Environment Metadata (no dependencies)
    |
    v
Power Time-Series Capture (depends on Zeus integration)
    |
    v
Phase-Aligned Energy Attribution (depends on TTFT + power time-series)

Pareto Frontier Extraction (independent, parallel)
--dry-run Grid Preview (independent, parallel)
```

### Phase 1: Document Harmonisation + Stack Update

**Rationale:** Cannot implement with contradictory decision documents and stale dependency pins.
**Delivers:** Consistent `.product/` decisions; updated `pyproject.toml`; resolved API shape.
**Addresses:** All cross-document inconsistencies (CI-1 through CI-7); all critical stack pins.
**Avoids:** Implementers building the wrong API; installing stale Zeus package.

### Phase 2: Core Measurement Foundations

**Rationale:** Measurement accuracy is foundational. Phase-split energy, power time-series, and environment metadata are table stakes that peers already have.
**Delivers:** EnvironmentSnapshot in every ExperimentResult; power time-series capture via Zeus PowerMonitor; prefill/decode phase attribution; corrected NVML accuracy documentation; thermal conditioning; CPU-GPU sync requirement; minimum measurement duration.
**Addresses:** FEATURES.md gaps 1, 2, 4; PITFALLS.md CP-1, CP-2, CP-3, MP-3, MP-5, mP-1.
**Uses:** `zeus>=0.13.1`, `nvidia-ml-py>=13.590.48`, `scipy>=1.12`.

### Phase 3: Library + CLI Implementation

**Rationale:** With measurement methodology settled and stack updated, implement the library API and CLI.
**Delivers:** `run_experiment()`, `run_study()` public API; `llem run`, `llem config` CLI; config validation with `extra="forbid"`; subprocess isolation; sweep resolution at YAML parse time.
**Implements:** Option C architecture; composition config; dotted sweep notation.
**Avoids:** PITFALLS.md Docker overhead (not yet -- Docker is v2.2).

### Phase 4: Study Execution + Results

**Rationale:** Studies depend on the library layer. Pareto extraction and `--dry-run` preview are study-level features.
**Delivers:** StudyConfig resolution; multi-experiment execution; `--dry-run` grid preview; VRAM pre-flight estimation; Pareto frontier extraction on StudyResult; result output (JSON per experiment, study directory structure).
**Addresses:** FEATURES.md gaps 3, 6, 7.

### Phase 5: Docker Multi-Backend (v2.2)

**Rationale:** Docker is opt-in for multi-backend studies. Three backends can run locally as single-backend studies in v2.0.
**Delivers:** Docker container per backend; `python-on-whales` orchestration; TRT engine caching.
**Avoids:** PITFALLS.md MP-1 (Docker energy overhead -- document, don't correct).

### Phase 6: Statistical Enhancements + Polish

**Rationale:** Bootstrap CI, baseline power correction, and statistical tools are enhancements over the core.
**Delivers:** BCa bootstrap CI via `scipy.stats.bootstrap`; baseline power measurement (P0 state); outlier detection; MFU calculation.
**Addresses:** PITFALLS.md CP-4, CP-5.

### Research Flags

**Needs deeper research during planning:**
- **Phase 2** (measurement foundations): Thermal conditioning protocol needs empirical calibration per GPU model. Power time-series storage format (Parquet schema) needs design. NVML energy counter accuracy on consumer GPUs (RTX series) is under-documented.
- **Phase 4** (study execution): `--dry-run` runtime estimation is novel (no peer does it well). VRAM estimation formula needs validation.
- **Phase 5** (Docker): TRT engine cache invalidation on version change needs design. NVML single-session owner enforcement mechanism needs specification.

**Standard patterns (skip research):**
- **Phase 1** (harmonisation): Straightforward document updates.
- **Phase 3** (library + CLI): Well-documented patterns; Option C architecture is clear; Pydantic + Typer are well-understood.
- **Phase 6** (statistics): `scipy.stats.bootstrap` is well-documented; BCa is a standard method.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Version numbers verified against PyPI; package rename confirmed; peer version matrices checked |
| Features (audit) | MEDIUM-HIGH | 8 peer tools surveyed; some via official docs, some via web search; feature matrix is comprehensive |
| Architecture | HIGH | Source code of 10 tools inspected; typing guidance from official Python docs and Guido directly |
| Pitfalls (measurement) | MEDIUM-HIGH | NVML +/-5W from API docs; thermal ramp from MLPerf paper; FLOPs argument is mathematical; energy counter accuracy under-documented by NVIDIA |
| Decision audit | HIGH | Every assessment cites specific peer tool implementations; cross-referenced against `.product/research/` files 01-16 |
| API return type | HIGH | 10 tools verified; official typing guidance; Guido van Rossum quote; `@overload` limitation demonstrated |

**Overall confidence:** MEDIUM-HIGH

### Gaps Remaining

| Gap | How to Handle |
|-----|---------------|
| NVML energy counter accuracy on consumer GPUs | Empirical testing during Phase 2; RTX 6000 Ada discrepancy reported but unresolved |
| CodeCarbon 3.x Pydantic v2 compatibility | Must verify before shipping `[codecarbon]` extra; may need isolation boundary |
| vLLM + Transformers v5 compatibility matrix | Must test specific version combinations; known pain point |
| Thermal conditioning duration per GPU model | Empirical calibration needed; 60s is MLPerf minimum but may need per-model tuning |
| TRT-LLM 1.0 stable API surface | 1.3.0rc4 is pre-release; API may change before stable |
| Power time-series storage format | Parquet per experiment is the recommendation; schema needs design |
| Docker energy overhead quantification | ~1-3% estimated from reasoning; no direct energy measurement study exists |
| NVLink interconnect power gap in multi-GPU | 3-10% understatement estimated; no per-measured correction; document limitation |

---

## Sources

Aggregated from all six research files. Full citation lists in each file.

### Peer-Reviewed Papers (HIGH confidence)
- [Part-time Power Measurements (Burtscher et al., arXiv:2312.02741)](https://arxiv.org/html/2312.02741v2) -- NVML accuracy
- [ML.ENERGY Benchmark (NeurIPS D&B 2025)](https://arxiv.org/html/2505.06371v1) -- steady-state methodology, Pareto frontiers
- [MLPerf Power Benchmark (IEEE HPCA 2025)](https://arxiv.org/html/2410.12032v2) -- 60s minimum, full-system measurement
- [TokenPowerBench (Dec 2025)](https://arxiv.org/html/2512.03024v1) -- phase-aligned energy attribution
- [Mind the Memory Gap (arXiv:2503.08311)](https://arxiv.org/html/2503.08311v2) -- FLOPs vs memory bandwidth

### Official Documentation (HIGH confidence)
- [NVIDIA NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) -- +/-5W accuracy spec
- [Python typing best practices](https://typing.python.org/en/latest/reference/best_practices.html) -- avoid union returns
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)
- [Zeus v0.13.1](https://github.com/ml-energy/zeus) -- energy measurement; package renamed from zeus-ml
- [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) -- BCa method

### Peer Tool Source Code (HIGH confidence)
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) -- `simple_evaluate()` return type; `__init__.py` exports
- [optimum-benchmark](https://github.com/huggingface/optimum-benchmark) -- `Benchmark.launch()` return type; process isolation
- [vLLM benchmark CLI + sweeps](https://docs.vllm.ai/en/latest/benchmarking/) -- Pareto analysis; parameter sweeps
- [Ray Tune ResultGrid](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.fit.html) -- always returns collection type
- [AIEnergyScore](https://huggingface.github.io/AIEnergyScore/) -- phase-split energy; environment metadata

### Community + Industry (MEDIUM confidence)
- [pytorch/torchtitan Hydra issue](https://github.com/pytorch/torchtitan/issues/1415) -- teams moving away from Hydra
- [mypy issue #1693](https://github.com/python/mypy/issues/1693) -- Guido on union return types
- [Databricks LLM inference best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) -- memory bandwidth, not FLOPs
- [uv vs Poetry 2026](https://cuttlesoft.com/blog/2026/01/27/python-dependency-management-in-2026/)

---
*Research completed: 2026-02-25*
*Ready for harmonisation: yes -- resolve API shape first, then propagate*
