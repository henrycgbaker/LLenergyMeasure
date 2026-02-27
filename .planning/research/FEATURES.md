# Feature Research: Study/Sweep Execution (M2)

**Domain:** Multi-experiment study execution in LLM benchmarking tools
**Researched:** 2026-02-27
**Confidence:** HIGH — primary sources are `.product/` decisions and designs (fully
researched prior sessions), supplemented by targeted web search for any gaps.

> **Note on this file's scope:** This is a milestone-scoped research file for M2
> (study/sweep execution). It supersedes the prior FEATURES.md which audited
> M1 features against peer tools. M1 features (ExperimentConfig, ExperimentResult,
> llem run single, energy backends) are **already decided and largely implemented**.
> This file covers only the incremental features needed to add multi-experiment
> study execution to the working single-experiment foundation.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that researchers expect from any multi-experiment sweep tool. Missing these
makes llem feel incomplete compared to optimum-benchmark, vLLM bench sweep, and lm-eval.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Grid sweep grammar | Any sweep tool needs a way to declare the parameter space. Researchers expect YAML-based grid declaration, not Python code. | MEDIUM | Already designed: dotted notation `pytorch.batch_size: [1, 8]`, backend-scoped grids. Cartesian product at YAML parse time before Pydantic. |
| Explicit experiment list | Non-Cartesian combinations (e.g. specific model+backend pairs) require an explicit list. W&B, vLLM bench, and MLflow all support this. | LOW | Already designed: `experiments:` list alongside or instead of `sweep:`. Additive to sweep grid. |
| Subprocess isolation per experiment | CUDA allocator state does not fully clear within a process. Every peer tool (optimum-benchmark, AIEnergyScore) uses fresh process per experiment. Users expect clean measurements. | MEDIUM | Fully designed: `multiprocessing.get_context("spawn")`, `Pipe` IPC, `daemon=False`, `p.join(timeout=...)`. Pseudocode in `designs/experiment-isolation.md`. |
| Progress display during study | A study running 36 experiments (12 configs × 3 cycles) takes hours. Silence = users think it crashed. All peer tools show progress. | MEDIUM | Designed: tqdm progress bar at study level, per-experiment status lines (`✓ / ▶ / · / ✗`), inline key metrics on completion. Multi-process Queue → consumer thread on parent. |
| Thermal gap countdown | Unique to energy measurement tools. During the `config_gap_seconds` pause, the user needs a countdown — not silence. Without it, a 5-minute gap between cycles looks like a hang. | LOW | Designed in observability.md: `· [4/12]  pytorch / batch=16 / bf16   →  waiting thermal gap  (55s remaining)`. |
| Skip-and-continue on experiment failure | A study with 36 experiments must not abort on experiment 3 crashing. Peer tools (optimum-benchmark, vLLM bench, Hydra multirun) all skip failed configs and continue. | MEDIUM | Designed: `skipped` (Pydantic ValidationError, never touches GPU), `failed` (subprocess crash), `ran` — three distinct outcomes. Error logged to manifest, study continues. |
| Study manifest / checkpoint | Long studies (hours) need a checkpoint in case of interruption. Users expect to inspect study progress without waiting for completion. | MEDIUM | Designed: `StudyManifest` written at study start, updated after each experiment. Persists to `study_manifest.json` alongside results. Distinct from `StudyResult` (final return). |
| n_cycles repetition | Energy measurements vary run to run. Repeating each configuration N times is expected for any scientifically serious tool. vLLM bench defaults to 3. AIEnergyScore runs 10. | LOW | Fully decided: `n_cycles` in study.yaml `execution:` block (Pydantic default=1, CLI effective default=3). |
| Dry-run grid preview | Before spending hours of GPU time, users need to see what experiments will run. Especially important when sweep grammar produces 50+ configs. | MEDIUM | Designed in designs/study-yaml.md: `llem run study.yaml --dry-run` shows resolved grid, invalid config skips, VRAM estimates, experiment count. |

---

### Differentiators (Competitive Advantage)

Features that set llem apart from peer tools specifically in multi-experiment execution.
These are grounded in decisions already made; implementation complexity is the question.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Backend-scoped sweep dimensions | `pytorch.batch_size: [1, 8]` and `vllm.max_num_seqs: [64, 256]` produce **independent** Cartesian grids per backend. No peer tool has backend-scoped sweep dimensions. | MEDIUM | Novel; Hydra dotted notation is the closest precedent but produces cross-backend Cartesian product. llem's backend-independent grids are essential for multi-backend studies where parameters are not comparable. |
| Execution/study separation with portable hashing | `study_design_hash` excludes the `execution:` block. Same study run at `n_cycles=3` vs `n_cycles=5` shares the same hash. "Topping up" a study for publication is possible without losing identity. No peer tool does this. | LOW | Designed in decisions/study-execution-model.md Decision B. Hash = SHA-256[:16] of `model_dump(exclude={"execution"})`. |
| cycle_order: interleaved (thermal fairness) | Sequential execution (config A three times, then B three times) introduces thermal autocorrelation — config A gets cold-start advantages, config B gets warm-GPU advantages. `interleaved` (A→B→C, A→B→C) distributes thermal effects fairly. No peer tool addresses thermal ordering of measurement cycles. | LOW | Decided: `sequential | interleaved | shuffled`. Interleaved is CLI effective default. vLLM bench sweep runs all bench_params per serve_comb (sequential) — no cycle ordering. |
| Manifest-first checkpoint (always-on, not opt-in) | `StudyManifest` is written on every study run from experiment 1, regardless of flags. No peer tool (optimum-benchmark, vLLM bench, lm-eval) writes a checkpoint manifest by default. Hydra multirun fails silently when a run crashes. | LOW | Enables post-hoc inspection, reproducibility documentation, and future `--resume`. The cost is negligible: one small JSON write per experiment state change. |
| Structured per-experiment failure IPC | When a subprocess crashes, llem sends a structured `{type, message, traceback}` dict via Pipe before exiting. The study runner records this in the manifest. Peer tools (Hydra multirun) only log file-level tracebacks; no structured failure data flows to the orchestrator. | MEDIUM | Designed in designs/experiment-isolation.md. Three-terminal patterns: timeout → SIGKILL + synthetic event; non-zero exit → read pipe if data available; success → read result from pipe. |
| Multi-level pre-flight validation before study starts | Before running any experiment, llem validates all `N` generated configs with Pydantic and shows all invalid combinations at once. Users see `"3 of 15 configs skipped (invalid — shown before any experiments run)"`. Hydra, W&B, and vLLM bench run-and-fail invalid configs. | MEDIUM | Designed in designs/study-yaml.md. Three validation levels: L1 Pydantic (always, zero cost), L2 VRAM estimation (opt-in via `--dry-run`), L3 runtime graceful failure (always-on). |

---

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem reasonable but are explicitly rejected based on peer research and product decisions.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Named execution profiles in user config (`execution_profiles:`) | Users want to define `publication` or `quick` profiles once and reference them in all study files. Nextflow `profiles {}` and dbt `targets` are well-known precedents. | Portability footgun: `profile: publication` in study.yaml creates hidden dependency on target machine's user config. Two researchers get different results from the same study.yaml. No peer tool uses named profiles for statistical rigour presets. | Direct fields in study.yaml (`n_cycles: 5`). `--profile quick` and `--profile publication` as CLI shorthands expanding to flag combinations at CLI level only. |
| `metric:` / `direction: minimize` optimisation | W&B Sweeps, Ray Tune, and Optuna all support this. Users running HPO workflows expect it. | llem measures everything and optimises nothing. Adding a metric target collapses the measurement to a search problem — a fundamentally different product. | Report all metrics in `ExperimentResult`. Users run analysis tools (pandas, matplotlib, Optuna) on the results themselves. |
| `max_concurrent_trials:` / parallel experiment execution | Users with multiple GPUs or distributed setups want parallel sweeps. Ray Tune and Hydra joblib support this. | GPU ownership (NVML, Zeus energy windows) prevents within-machine parallelism — two simultaneous experiments on the same GPU contaminate each other's energy measurements. Multi-GPU-aware parallelism requires GPU routing which is a separate M3+ concern. | Sequential by design. Multi-machine parallelism via SLURM array jobs (launch separate `llem run` per GPU). |
| `timeout_seconds:` / time budget | Ray Tune `time_budget_s:` stops sweeps after N seconds. Users running CI jobs want time limits. | Silently truncates measurements. A researcher who doesn't know which experiments were cut may publish incomplete results. | Use `n_cycles=1` and `--no-gaps` for fast CI runs. YAML `execution:` block makes the protocol explicit and auditable. |
| `early_terminate:` / pruning | W&B Hyperband, Optuna TrialPruned, Ray Tune schedulers. Users doing optimisation want this. | Optimisation-framework concept, not measurement concept. Pruning introduces selection bias that invalidates comparative energy measurements. | Run all experiments. Filter Pareto-optimal results in post-processing. |
| `max_experiments:` / run cap | "Run at most 20 of my 100 experiments." W&B supports this. | Undefined semantics for Cartesian grids: which 20 of 100? Random? First 20? Any answer introduces bias. | Generate a smaller sweep. Or use `experiments:` explicit list to manually curate the subset. |
| `callbacks:` / hook system | Ray Tune, Optuna have hook systems. Framework integrators want them. | Not a framework; adds API surface without clear user value. Scope creep at v2.0. | Library API: `run_study()` returns `StudyResult`. Post-process results directly. |

---

## Feature Dependencies

```
[Single Experiment (M1 — already done)]
    └──required by──> [StudyRunner subprocess dispatch]
                          └──required by──> [Study-level progress display]
                          └──required by──> [StudyManifest writes]
                          └──required by──> [StudyResult assembly]

[StudyConfig (M2)]
    └──required by──> [Sweep resolution (YAML parse time)]
                          └──required by──> [Multi-level pre-flight validation]
                          └──required by──> [--dry-run grid preview]

[StudyConfig resolution]
    └──required by──> [Manifest entry generation (all N × cycles entries at study start)]

[Subprocess isolation (STU-01 to STU-04)]
    └──required by──> [Structured failure IPC (Pipe)]
    └──required by──> [Progress Queue → consumer thread]
    └──required by──> [SIGKILL on timeout]

[StudyManifest (STU-08 to STU-09)]
    └──enables (later)──> [--resume (STU-10 to STU-11, M4)]

[StudyResult (RES-13 to RES-15)]
    └──required by──> [run_study() return type]
    └──required by──> [study_summary.json output]

[Multi-backend study (CM-10)]
    └──blocked by──> [Docker runner (M3)]
```

### Dependency Notes

- **Subprocess isolation requires M1 ExperimentOrchestrator:** The child process spawned by `StudyRunner` runs `ExperimentOrchestrator` unchanged. M1 must ship a working `ExperimentOrchestrator` before `StudyRunner` can wrap it.
- **Sweep resolution must precede pre-flight:** All N experiment configs are generated at YAML parse time, before any subprocess starts. Pre-flight validation runs against the resolved list. This is the "show all invalid configs upfront" feature.
- **Manifest generation requires knowing all experiments upfront:** `StudyManifest.experiments` is populated at study start with all `N × cycles` entries in `pending` state. This requires the full resolved list before experiment 1 starts.
- **Multi-backend study conflicts with M2 scope:** `CM-10` says multi-backend without Docker → hard error at pre-flight. Docker execution is M3. M2 supports single-backend studies (multiple configs of the same backend). A study.yaml with `backend: [pytorch, vllm]` in M2 raises `PreFlightError`.
- **`--resume` depends on `StudyManifest`:** The manifest is written in M2 (always-on). The `--resume` logic that reads it is deferred to M4. This is intentional: write the checkpoint now, implement resume later.

---

## MVP Definition (M2 Scope)

### Launch With (M2)

The minimum set to deliver study/sweep execution as a usable feature.

- [x] `StudyConfig` Pydantic model with `sweep:`, `experiments:`, `execution:` blocks — CFG-11 through CFG-16
- [x] Sweep resolution (Cartesian product) at YAML parse time — CFG-12, CFG-13, CFG-14
- [x] `StudyRunner` with `multiprocessing.get_context("spawn")` subprocess per experiment — STU-01 through STU-04
- [x] IPC via `multiprocessing.Pipe` (file fallback >1MB) — STU-02
- [x] Thermal gap enforcement between experiments — STU-06 (`config_gap_seconds`, `cycle_gap_seconds`)
- [x] `cycle_order: sequential | interleaved | shuffled` — STU-07
- [x] `StudyManifest` with `ManifestWriter` (`mark_running`, `mark_completed`, `mark_failed`) — STU-08, STU-09
- [x] `StudyResult` schema with `study_design_hash`, `measurement_protocol`, `result_files` — RES-13 through RES-15
- [x] `run_study()` library function — LA-02, LA-05
- [x] Study-mode CLI flags: `--cycles`, `--no-gaps`, `--order` — CLI-05
- [x] Thermal gap countdown display — CLI-11
- [x] Multi-level pre-flight validation (L1 Pydantic skip, L3 runtime graceful failure) — implicit in STU-01 + study loop design
- [x] Study progress display: tqdm bar + per-experiment status lines — extends CLI-08 to study context
- [x] Hard error for multi-backend study without Docker — CM-10

### Add After Validation (Later M2 Iteration)

Features that can ship after the core study loop works.

- [ ] `--dry-run` L2 (VRAM estimation for sweep grid preview) — extends CLI-07 to study context. Depends on VRAM estimation utility from `research/10-sweep-validation-patterns.md`.
- [ ] Study-level summary display (final `Study complete: 11/12 ran, 1 failed` line) — extends CLI-08.

### Future Consideration (M4+)

Features deferred per decisions made in `.product/`.

- [ ] `--resume` from interrupted study (STU-10, STU-11) — M4. Manifest is ready; resume logic is later.
- [ ] `cold_start: true` (model unload between experiments) — M4, as it changes measurement semantics beyond basic study execution.
- [ ] Bootstrap confidence intervals on study-level aggregation — deferred post-v2.0 (DEF-01).
- [ ] Cross-file `base:` reference in study.yaml (study.yaml referencing standalone experiment.yaml) — deferred to v2.2.

---

## Feature Prioritisation Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Sweep grammar (CFG-11 to CFG-14) | HIGH | MEDIUM | P1 |
| Subprocess isolation (STU-01 to STU-04) | HIGH | MEDIUM | P1 |
| n_cycles + cycle_order (STU-07) | HIGH | LOW | P1 |
| Thermal gap + countdown display (STU-06, CLI-11) | HIGH | LOW | P1 |
| StudyManifest checkpoint (STU-08, STU-09) | HIGH | LOW | P1 |
| StudyResult schema (RES-13 to RES-15) | HIGH | LOW | P1 |
| Skip-and-continue failure handling | HIGH | MEDIUM | P1 |
| Study progress display (tqdm + per-experiment lines) | HIGH | LOW | P1 |
| Multi-backend hard error (CM-10) | HIGH | LOW | P1 |
| run_study() library function (LA-02, LA-05) | HIGH | LOW | P1 |
| Study CLI flags --cycles, --no-gaps, --order (CLI-05) | MEDIUM | LOW | P2 |
| --dry-run VRAM estimation for sweep grid | MEDIUM | MEDIUM | P2 |
| Study summary display (final line) | MEDIUM | LOW | P2 |
| --resume from interrupted study | HIGH | HIGH | P3 (M4) |
| cold_start: true | MEDIUM | HIGH | P3 (M4) |
| Bootstrap CI on study aggregation | MEDIUM | HIGH | P3 (post-v2.0) |

**Priority key:**
- P1: Must have for M2 to be usable
- P2: Should have, add in same milestone when core is working
- P3: Future consideration, explicitly deferred

---

## Peer Tool Comparison

The following table focuses on multi-experiment study execution specifically,
not single-experiment features.

| Feature | optimum-benchmark | vLLM bench sweep | lm-eval | llem M2 |
|---------|------------------|-----------------|---------|---------|
| Sweep grammar | Hydra `--multirun x=1,2,3` | JSON param files, Cartesian product | No sweep | `sweep:` block, dotted backend-scoped notation |
| Subprocess isolation | `multiprocessing.Process` + `Pipe` per experiment | Long-running server per serve_comb | No isolation (in-process) | `multiprocessing.get_context("spawn")` per experiment |
| IPC mechanism | `multiprocessing.Pipe` (file fallback >1MB) | HTTP REST API | N/A | `multiprocessing.Pipe` (file fallback >1MB) |
| n_runs / cycles | Via Hydra grid (add `seed` param) | `--num-runs` (default 3) | No | `n_cycles` in `execution:` block |
| Cycle ordering | No concept — sequential per Hydra | No concept | No | `sequential | interleaved | shuffled` |
| Thermal gaps | Not addressed | Not addressed | Not addressed | `config_gap_seconds` + `cycle_gap_seconds` (machine-local user config) |
| Progress display | Hydra run logs per config | Rich terminal, per-config status | tqdm | tqdm + per-experiment status lines + thermal countdown |
| Failure handling | Run fails, logged, sweep continues | Run fails, logged, sweep continues | Task fails, others continue | Skip (Pydantic) + fail-and-continue (subprocess crash) + structured Pipe IPC |
| Manifest / checkpoint | No | No | No | `StudyManifest` written at study start, updated per-experiment |
| Result aggregation | Per-run `benchmark_report.json` in subdirs | Per-run JSON in results dir | Single JSON with all task results | `StudyResult` with `result_files` paths + `study_summary.json` |
| Study identity hash | No | No | No | `study_design_hash` (excludes execution block) |
| Pre-flight validation | No | No | No | Multi-level: Pydantic L1 upfront + VRAM L2 dry-run + runtime L3 |

**Confidence:** HIGH for optimum-benchmark (source code verified in `research/13-execution-isolation-patterns.md`); MEDIUM for vLLM bench sweep (official docs verified); MEDIUM for lm-eval (official docs, Dec 2025 refactor noted).

---

## Sources

### Authoritative (HIGH confidence — from prior sessions, source-verified)

- `decisions/study-execution-model.md` — study design hash, execution block placement, CLI precedence
- `decisions/experiment-isolation.md` — subprocess isolation rationale, Pipe IPC, SIGKILL, timeout, pseudocode
- `designs/study-yaml.md` — full StudyConfig schema, sweep grammar, cycle_order semantics, invalid combination handling
- `designs/study-resume.md` — StudyManifest vs StudyResult disambiguation, ManifestWriter API
- `designs/observability.md` — progress display design, thermal gap countdown, verbosity levels
- `research/13-execution-isolation-patterns.md` — optimum-benchmark Process launcher code (verbatim), vLLM bench sweep architecture, AIEnergyScore Docker pattern
- `research/10-sweep-validation-patterns.md` — Hydra, W&B, Optuna, ConfigSpace sweep constraint patterns; VRAM estimation formulae
- `research/16-execution-profiles-patterns.md` — named preset decision (rejected); n_cycles direct field rationale
- `REQUIREMENTS.md` — STU-01 to STU-11, CFG-11 to CFG-17, RES-13 to RES-15, CLI-05, CM-10

### Web Search Supplemental (MEDIUM confidence)

- [optimum-benchmark README](https://github.com/huggingface/optimum-benchmark/blob/main/README.md) — confirms serial-by-default multirun, Hydra launcher plugins for parallelism
- [vLLM Benchmarking](https://docs.vllm.ai/en/latest/benchmarking/) — confirms param_sweep module, result aggregation utilities
- [lm-eval harness releases](https://github.com/EleutherAI/lm-evaluation-harness/releases) — Dec 2025 lighter install; no study-level sweep capability added
- [CodeCarbon validation study](https://arxiv.org/html/2509.22092v1) — energy measurement accuracy benchmarking context

---
*Feature research for: M2 study/sweep execution — LLenergyMeasure v2.0*
*Researched: 2026-02-27*
