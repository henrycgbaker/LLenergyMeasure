# Decision Audit: LLenergyMeasure Product Decisions

**Date**: 2026-02-25
**Branch**: planning/dry-refactor
**Scope**: All `.product/decisions/` and `.product/designs/` files, cross-referenced against peer tool behaviour
**Method**: Evidence-first — each assessment cites specific peer tool implementations and research findings

---

## 1. Executive Summary

| Rating | Count | Files |
|--------|-------|-------|
| ALIGNED | 12 | architecture (A/B/F/I/J), installation, warmup strategy, dataset-handling, result-schema-migration, error-handling, backward-compatibility, testing-strategy |
| QUESTIONABLE | 8 | config-architecture (C2 sweep grammar), experiment-isolation, flops-estimation, multi-gpu, output-storage, docker-execution, reproducibility, carbon-intensity |
| CONTRADICTED | 2 | library API return type (union run()), FLOPs formula primary metric |
| OVER-ENGINEERED | 3 | three-layer config model (C/D/E sub-decisions), access-control (.env), study-design-hash exclusion |
| UNDER-SPECIFIED | 4 | warmup (thermal floor calibration), reproducibility (no measurement variance quantification), lora adapter support (TRT constraint surfacing), CI policy for measurement accuracy |

**Critical finding**: There is a **fundamental cross-document contradiction** between the newest decision (experiment-study-architecture.md, 2026-02-25 — unified `llem run` + `llem.run()`) and a large set of still-active documents (experiment-isolation.md, study-execution-model.md, live-observability.md, study-resume.md, output-storage.md, documentation-strategy.md, progressive-disclosure.md, backward-compatibility.md, versioning-roadmap.md, designs/library-api.md) which still say `llem study`, `run_experiment()`, and `run_study()`. The DRY refactor branch has not propagated the unified architecture decision through the document set.

**Second critical finding**: The library-api.md design file still specifies `run_experiment` / `run_study` as separate functions with the explicit argument that "A single `run()` has an ambiguous return type" — but the experiment-study-architecture.md decision (accepted 2026-02-25) mandates `llem.run()` with union return type. These two documents directly contradict each other.

---

## 2. Per-Decision Assessment

---

### 2.1 Architecture — Sub-decision A: Library-First Pattern

**File**: `.product/decisions/architecture.md` (sub-decision A)
**Claim**: "Library-first at v2.0. CLI is a thin wrapper."

**Peer evidence**:
- lm-eval: `simple_evaluate()` library function; `lm-eval` CLI wraps it. HIGH confidence.
- Zeus: `ZeusMonitor` library object; no dedicated CLI. HIGH confidence.
- CodeCarbon: `EmissionsTracker` library; `codecarbon` CLI wraps it. HIGH confidence.
- Optimum-Benchmark: `Benchmark.run()` library; CLI wraps it. HIGH confidence.

**Rating**: ALIGNED

The library-first pattern is standard across all peer tools in the measurement/benchmarking space. This decision is correct and well-grounded.

---

### 2.2 Architecture — Sub-decision B: Module Structure

**File**: `.product/decisions/architecture.md` (sub-decision B)
**Claim**: "`src/llenergymeasure/` layout. CLI and study modules inside the package."

**Peer evidence**:
- lm-eval: `src/lm_eval/` layout, CLI inside package. HIGH confidence.
- Optimum-Benchmark: `src/optimum/benchmark/` layout. HIGH confidence.
- Zeus: flat `zeus/` layout (no `src/`). MEDIUM confidence.

**Rating**: ALIGNED

`src/` layout is the modern Python standard per PEP 517/518. CLI inside the package is the correct call for a tool with no separate release cycle.

---

### 2.3 Architecture — Sub-decision C/D/E: Three-Layer Config Model

**File**: `.product/decisions/architecture.md` (sub-decisions C, D, E)
**Claim**: Three distinct config layers: (1) runners/machine, (2) experiment/study, (3) infrastructure context.

**Peer evidence**:
- Nextflow: Two-layer model. `nextflow.config` for execution; `main.nf` for pipeline logic. No explicit "infrastructure context" layer.
- Snakemake: `--profile` for execution environment; `Snakefile` for workflow. Two layers.
- DVC: `~/.dvc/config` (global), `dvc.yaml` (pipeline). Two layers.
- MLflow: `mlflow.yml` for run config; environment captured automatically. Two layers.
- lm-eval: Single config file (`config.yaml`). One layer.
- W&B: Two surfaces — `wandb.init()` config and sweep config. Two layers.

**Challenger's argument**: Every peer tool the project cites as a reference uses a two-layer model (what + where/how). The three-layer model is a bespoke design with no direct peer precedent. The "infrastructure context as scientific record" framing is intellectually sound, but it describes a data-capture concern (what to store in results), not a config model concern — and conflating these two distinct problems is the root of the complexity.

**Rating**: QUESTIONABLE

**What peer practice suggests**: Separate the config model (two layers: experiment definition + machine-local execution settings) from the result capture model (what infrastructure context to record in output). The three-layer model would become: (1) user config for machine settings including carbon/PUE defaults, (2) experiment/study YAML for measurement definition. Layer 3 is simply "what gets stored in results" — not a third config layer. This is how Nextflow, Snakemake, and MLflow all handle it.

The sub-decisions C, D, and E have `**TODO come back to**` markers in the architecture.md decision table — flagging that even the decision authors recognised these were unresolved.

---

### 2.4 Architecture — Sub-decision G: Energy vs CO2 Separation

**File**: `.product/decisions/architecture.md` (sub-decision G); `.product/decisions/carbon-intensity.md`
**Claim**: "CO2 decoupled from energy. Base = NVML polling. `[zeus]` = accurate energy. `[codecarbon]` = CO2."

**Peer evidence**:
- Zeus: measures energy (joules/watts) only; no CO2. HIGH confidence.
- CodeCarbon: measures CO2 estimates; uses its own energy measurement internally. HIGH confidence.
- Optimum-Benchmark: delegates energy to Zeus or CodeCarbon via optional dep. MEDIUM confidence.
- AIEnergyScore: reports CO2 without distinguishing the energy measurement underneath. MEDIUM confidence.

**Challenger's argument**: CodeCarbon does its own energy measurement internally and then converts to CO2 — it is not purely a CO2 layer on top of an energy layer. Enabling both Zeus and CodeCarbon simultaneously creates potential double-counting (two energy measurement sessions against the same GPU) which the NVML single-session owner constraint addresses but only partially: CodeCarbon's internal energy measurement may still conflict with Zeus.

**Rating**: ALIGNED (with implementation caveat)

The conceptual separation is correct. The NVML single-session owner constraint is correctly identified. The implementation caveat (CodeCarbon has its own energy measurement that must yield to Zeus) is acknowledged in the decision.

---

### 2.5 Architecture — Sub-decision H: Library API Surface

**File**: `.product/decisions/architecture.md` (sub-decision H); `.product/decisions/experiment-study-architecture.md`
**Claim** (architecture.md): "`run`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult` in `__init__.py`"
**Claim** (experiment-study-architecture.md, 2026-02-25): "`llem.run(config) -> ExperimentResult | StudyResult`. Single function."
**Claim** (designs/library-api.md, still v2.0 pre-unification): `run_experiment()` + `run_study()` as separate functions, with explicit statement: "A single `run()` has an ambiguous return type."

**Peer evidence**:
- lm-eval: `simple_evaluate()` returns a dict; all evaluation modes use this one function; NO separate `evaluate_task()` / `evaluate_group()`. HIGH confidence.
- pytest: `pytest.main()` handles everything — no `run_test()` / `run_suite()`. HIGH confidence.
- MLflow: `mlflow.run()` accepts any entry point; no separate single/multi variant. HIGH confidence.

**Rating**: CONTRADICTED — Three documents make mutually incompatible claims about the library API:

1. `experiment-study-architecture.md` (most recent, 2026-02-25): `llem.run()` unified, union return type
2. `architecture.md` sub-decision H (last updated 2026-02-25): lists `run` as the export — consistent with unification
3. `designs/library-api.md` (last updated 2026-02-19, not updated since unification decision): `run_experiment()` + `run_study()`, with an explicit argument against union return type

**The union return type argument in library-api.md** ("A single `run()` has an ambiguous return type") deserves scrutiny: lm-eval's `simple_evaluate()` returns `dict[str, dict]` — no split by mode. pytest's `pytest.main()` returns an `ExitCode` enum — no split. The argument that union return types make type checking harder is true but manageable with `isinstance()` or Python overloads (`@overload`). The experiment-study-architecture.md decision already resolves this by noting that `run()` unwraps to `ExperimentResult` for single-experiment inputs. This is sound.

**Action required**: `designs/library-api.md` must be updated to reflect the unified `run()` API. The `run_experiment()` / `run_study()` API specified there is superseded.

---

### 2.6 Architecture — Sub-decision I: Subprocess Isolation

**File**: `.product/decisions/experiment-isolation.md`
**Claim**: "`multiprocessing.Process` per experiment with `spawn` start method."

**Peer evidence**:
- Optimum-Benchmark: `multiprocessing.Process` per benchmark with `spawn` context — exact same mechanism. HIGH confidence (confirmed via research/13-execution-isolation-patterns.md).
- AIEnergyScore: Ephemeral `docker run` per experiment. HIGH confidence.
- vLLM benchmark: Long-running server process — but this is a throughput tool, not a measurement tool with state isolation requirements. Different objective.

**Challenger's argument**: The decision correctly identifies that single-experiment `llem run` runs in-process (sub-decision I: "subprocess isolation requirement applies to `llem study` / `StudyRunner` only"). But the document still uses `llem study` terminology after the 2026-02-25 unification to `llem run`. The substance is correct; the terminology is stale.

**Rating**: ALIGNED (terminology stale post-2026-02-25 unification)

The `multiprocessing.Process` + `spawn` + `Pipe` pattern matches optimum-benchmark exactly. The SIGKILL-on-timeout rationale (CUDA doesn't handle SIGTERM reliably) is well-documented. The pipe buffer edge case (64KB OS pipe limit) is correctly handled with the 1MB file-based fallback.

---

### 2.7 CLI: Unified `llem run` Command

**File**: `.product/decisions/cli-ux.md`; `.product/decisions/experiment-study-architecture.md`
**Claim**: "2 commands: `llem run` + `llem config` + `--version` flag. YAML determines scope."

**Peer evidence**:
- lm-eval: `lm-eval run` handles everything — no separate `lm-eval study`. HIGH confidence.
- pytest: `pytest` handles everything — no `pytest suite`. HIGH confidence.
- MLflow: `mlflow run` handles everything. HIGH confidence.
- cargo: `cargo run` handles everything. HIGH confidence.
- dbt: `dbt run` handles everything. HIGH confidence.

**Challenger's argument**: The tools that use a single unified command (lm-eval, pytest, MLflow) do so because their "collection" concept adds no structural complexity beyond the single unit. llem's StudyConfig adds genuine complexity (sweep DSL, execution protocol, n_cycles). The older `study-execution-model.md` Decision D made exactly this argument for keeping `llem study` as a separate command, citing W&B sweeps, vLLM bench sweep, and Kubeflow Katib as tools that maintain separate commands when the collection has structural complexity.

The decision log shows this was explicitly relitigated in experiment-study-architecture.md (2026-02-25), which concluded that Option C (sweep resolution at YAML parse time, before Pydantic) means StudyConfig is a thin container without structural complexity at the Python type level. This is the correct resolution. The sweep DSL is a YAML parsing concern, not a type-level concern, and the unified runner code path follows naturally.

**Rating**: ALIGNED

The unification is architecturally clean given Option C. The YAML file is a genuine complexity signal (a study.yaml with `sweep:` is visually distinct from an experiment.yaml). The decision is well-reasoned and matches peer practice.

---

### 2.8 Config Architecture: Composition vs Inheritance

**File**: `.product/decisions/config-architecture.md` (sub-decision C1)
**Claim**: "Single `ExperimentConfig` with optional backend sections (`pytorch:`, `vllm:`, `tensorrt:`)."

**Peer evidence**:
- Optimum-Benchmark: Uses inheritance (`InferenceConfig` subclasses per backend). LOW-MEDIUM confidence. This is the counter-example the decision explicitly rejected.
- Hydra: File-based composition via config groups. HIGH confidence.
- lm-eval: Flat string `model_args` dict, no nested sections. HIGH confidence.

**Challenger's argument**: The decision correctly identifies that inheritance creates a union type that pollutes every layer. However, the composition approach also has costs: an `ExperimentConfig` for a PyTorch experiment carries `vllm: None` and `tensorrt: None` as dead weight. For a tool where the backend determines which config section is relevant, this creates a validation problem: a user who sets `pytorch.batch_size = 8` but `backend: vllm` will see the `pytorch` section silently ignored. The decision acknowledges this but the validation rule ("scoped key references backend not in list → ValidationError") only catches the sweep grammar case, not the static YAML case.

**Rating**: ALIGNED

The composition choice is defensible and the tradeoffs are well-documented. Inheritance is objectively worse for a tool with a unified run function and multi-backend studies.

---

### 2.9 Config Architecture: Sweep Grammar (Dotted Notation)

**File**: `.product/decisions/config-architecture.md` (sub-decision C2)
**Claim**: "Dotted notation: `pytorch.batch_size: [1, 8, 32]`. Per-backend Cartesian product."

**Peer evidence**:
- Hydra: Dotted notation for overrides (`model.learning_rate=0.01`). HIGH confidence. But Hydra overrides reference config nodes, not sweep dimensions with per-group Cartesian products.
- W&B sweeps: Flat dict, no dotted notation. No per-group Cartesian products. HIGH confidence.
- Optuna: No YAML grammar; Python API. HIGH confidence (not relevant).
- Ray Tune: `param_space` dict in Python. HIGH confidence (not relevant).
- MLflow: No sweep grammar built in. HIGH confidence.

**Challenger's argument**: The dotted notation is a custom grammar invented for llem with no direct peer precedent. The closest peer (Hydra) uses dotted notation for something structurally different (config node overrides, not sweep dimensions). A researcher trying to understand `pytorch.batch_size: [1, 8]` in a study YAML has no existing mental model to draw on. The "split on first dot only" rule for nested params (`pytorch.attn.implementation`) is an undocumented edge case that will surprise users.

The alternative rejected (Option B — backend lock: `sweep:` only for universal params) is arguably more explicit and less surprising, even if it requires more YAML.

**Rating**: QUESTIONABLE

**What peer practice suggests**: The swept parameter spaces in peer tools (W&B, Ray Tune, Hydra) uniformly use explicit per-backend config blocks or Python dicts with clear scoping. The dotted notation is novel and requires careful documentation. It is workable but should be treated as a UX risk requiring user testing.

**Specific concern**: The "split on first dot only" behaviour for `pytorch.attn.implementation` (passing `attn.implementation` as a nested kwarg to PyTorchConfig) is specified but not validated in the edge-case table. What happens when PyTorchConfig has no `attn.implementation` field? With `extra="forbid"`, this should be a `ValidationError` — but the spec says it's "passed as nested kwarg" which implies it bypasses the Pydantic model. This edge case needs tightening.

---

### 2.10 Execution Profiles: `--profile` as CLI Flag Only

**File**: `.product/decisions/cli-ux.md` (sub-decision D); `.product/decisions/study-execution-model.md` (Decision A)
**Claim**: "`--profile quick|publication` is a CLI flag shorthand only. Not a config layer."

**Peer evidence**:
- pytest: `--timeout`, `--dist`, `-n` flags for execution tuning; no named profile system. HIGH confidence.
- cargo test: `--release` and flags; no named profile for execution rigour. HIGH confidence.
- lm-eval: `--batch-size`, `--num-fewshot`; no profile system. HIGH confidence.
- Nextflow: Profiles are for execution *environment* (local vs SLURM), not rigour level. HIGH confidence.
- Snakemake: Same as Nextflow. HIGH confidence.

**Challenger's argument**: The research correctly found that no peer tool uses named profiles for statistical repetition settings (n_cycles equivalent). The decision to make `--profile` a CLI shorthand only is well-grounded. The specific concern is the **split placement of `config_gap_seconds` / `cycle_gap_seconds` in user config vs `n_cycles` / `cycle_order` in study YAML**. This "portability" argument is coherent but creates an asymmetry: two execution settings are portable (in study YAML) and two are machine-local (in user config). This is not how any peer tool structures execution settings.

**Rating**: ALIGNED

The rejection of named profiles for rigour presets is correct and evidence-based. The field placement split (n_cycles in study YAML, gaps in user config) is an unusual but defensible design that serves the portability goal.

---

### 2.11 Experiment / Study Architecture: Option C

**File**: `.product/decisions/experiment-study-architecture.md`
**Claim**: "`ExperimentConfig` = pure data type. `StudyConfig` = thin resolved container. Sweep at parse time."

**Peer evidence**:
- Optuna: `Trial` (pure data) + `Study` (container managing objectives). HIGH confidence. The analogy is apt.
- pytest: Individual test item (pure, parameterised via `@pytest.mark.parametrize`) + test suite (collection). HIGH confidence.
- Ray Tune: `TrialConfig` + `Experiment` container. MEDIUM confidence.

**Rating**: ALIGNED

Option C cleanly resolves the DRY violation of Option A and the type-level confusion of Option B. The key insight — sweep grammar is a YAML parsing concern, not a Pydantic model concern — is architecturally sound and matches how Hydra handles config resolution (before model construction).

**One concern**: The single-experiment `llem run` path runs in-process (no subprocess isolation), but the unified `llem run` for study.yaml runs with subprocess isolation. The output contract (flat JSON vs subdirectory) is determined by `len(study.experiments)`. This means two invocations of `llem run` may produce structurally different output depending on whether the YAML has a `sweep:` block. Users may find this surprising. Peer tools that unify (pytest, lm-eval) don't have this diverging output contract because they don't have the single vs multi distinction.

---

### 2.12 Installation / Extras Model

**File**: `.product/decisions/installation.md`
**Claim**: "Zero backend deps at base. Extras: `[pytorch]`, `[vllm]`, `[tensorrt]`, `[zeus]`, `[codecarbon]`."

**Peer evidence**:
- lm-eval: `pip install lm_eval[vllm]`, `pip install lm_eval[anthropic]` — exact same pattern. HIGH confidence.
- Zeus: `pip install zeus-ml[full]` and `pip install zeus-ml[trace]`. HIGH confidence.
- Optimum-Benchmark: `pip install optimum-benchmark[onnxruntime]`, `pip install optimum-benchmark[pytorch]`. HIGH confidence.
- Transformers: `pip install transformers[torch]`, `pip install transformers[flax]`. HIGH confidence.

**Rating**: ALIGNED

Pip extras per backend is the dominant pattern across every relevant peer. The rejection of a single `[all]` extra due to vLLM+TRT incompatibility is correctly grounded. Progressive disclosure via helpful errors (no `llem init`) matches lm-eval, Zeus, and MLflow.

---

### 2.13 Warmup Strategy

**File**: `.product/decisions/warmup-strategy.md`
**Claim**: "Fixed count (n=5 runs, 2 tokens max). 30s thermal floor for energy."

**Peer evidence**:
- Optimum-Benchmark: `warmup_runs=20` fixed count, reduced output. HIGH confidence (from research/14).
- Zeus: `warmup_iters=10` fixed count. MEDIUM confidence.
- AIEnergyScore: `n_warmup=10` fixed count. MEDIUM confidence.
- DeepSpeed Profiler: Fixed warmup. HIGH confidence.
- MLPerf: Fixed warmup runs. HIGH confidence.
- vLLM: `--num-scheduler-steps` warm-up in server mode. MEDIUM confidence.

**Rating**: ALIGNED

Fixed count warmup matches every peer tool. The rejection of CV convergence is well-grounded: no peer implements it, and it introduces unbounded runtime. The 30-second thermal floor for energy benchmarks is an additive mechanism that peer tools don't implement — this is a genuine improvement over peers.

**Under-specification concern**: The 30-second thermal floor value is stated but not calibrated. What is the source of this figure? GPU power state ramp-up times vary significantly between GPU models (A100 vs RTX 4090 vs H100). A 30-second floor may be too short for A100 (known to take 60+ seconds to reach stable frequencies) and unnecessarily long for consumer GPUs. The decision should cite a source or acknowledge this as an empirical estimate requiring per-GPU calibration.

---

### 2.14 FLOPs Estimation

**File**: `.product/decisions/flops-estimation.md`
**Claim**: "Analytical estimation using PaLM/Chinchilla formula (`2×N×T`). Phase split. No runtime profiler."

**Peer evidence**:
- lm-eval: Does not track FLOPs. Confirmed via research/14. HIGH confidence.
- Optimum-Benchmark: Does not measure FLOPs. HIGH confidence.
- vLLM: No FLOPs output (PR #12341 draft, not merged). HIGH confidence.
- DeepSpeed FLOPs Profiler: Has FLOPs but cannot handle autoregressive KV-cache generation. HIGH confidence.
- PaLM paper: Uses `2×N×T` formula. HIGH confidence.
- Chinchilla paper: Same formula. HIGH confidence.
- LLM-Inference-Bench: Uses `2×N×T`. MEDIUM confidence.
- TokenPowerBench: Uses FLOPs per token. MEDIUM confidence.

**Rating**: CONTRADICTED (on the primary metric framing)

The formula is correct and industry-standard. The rejection of runtime profilers is correct and well-evidenced. However, the decision names `flops_per_output_token` as the "primary cross-run comparison metric." This is not how the LLM efficiency community uses FLOPs:

1. FLOPs are a property of the *model and workload* (fixed for a given model + input/output length), not of the *implementation*. `flops_per_output_token` is **identical** for PyTorch and vLLM on the same model with the same inputs. If the tool's purpose is to measure *implementation* choice effects, a metric that cannot differ between implementations is a poor primary metric.

2. The dominant primary metric in publishable LLM efficiency research is `energy_per_output_token` or `tokens_per_second`. FLOPs are used for model normalisation (e.g., FLOPs/token vs energy/token to compare across model sizes), not as an efficiency metric for implementation comparison.

3. The decision's own context states "The tool does not use FLOPs as a latency predictor" and "FLOPs correlate poorly with GPU latency" — if FLOPs cannot predict latency and cannot vary between implementations, they are a normalisation tool, not a measurement output.

**What peer practice suggests**: Report FLOPs as a normalisation denominator (enabling comparison like "energy / FLOPs across hardware") rather than as a standalone efficiency metric. The primary metrics for implementation comparison are `energy_total_j`, `tokens_per_second`, `ttft_ms`, `itl_ms`, and `energy_per_output_token`. FLOPs belong in the methodology section (to characterise workload) not as primary output metrics.

---

### 2.15 Multi-GPU Support

**File**: `.product/decisions/multi-gpu.md`
**Claim**: "Passive — detect, record, aggregate. No new TP/PP sweep params in v2.0."

**Peer evidence**:
- Zeus: Measures per-GPU energy via `Measurement.gpu_energy` dict. HIGH confidence.
- Optimum-Benchmark: Records GPU count and names in output. MEDIUM confidence.
- TokenPowerBench (arxiv 2512.03024): Benchmarks energy across TP/PP configurations. MEDIUM confidence.
- MLPerf: Records hardware configuration. HIGH confidence.

**Rating**: QUESTIONABLE

The "passive" stance is reasonable for v2.0, but the decision conflates two distinct concerns:

1. **Recording multi-GPU configuration** (correct to do passively — record what GPUs are used)
2. **Measuring multi-GPU energy correctly** (this is not passive — it requires active decisions about aggregation)

The decision states "aggregate energy across all GPUs" as if it is obvious, but it raises an under-specified question: for PyTorch with `device_map="auto"` (pipeline parallelism), the model spans GPUs but only the primary GPU runs the full forward pass logically. Zeus measures actual power draw per device, which is correct. But NVML polling in the base package may not measure all GPUs consistently depending on how `pynvml` is initialised. This should be explicit in the implementation design.

**Under-specification**: `energy_per_output_token` is named as the primary efficiency metric for multi-GPU experiments. For a 2-GPU run vs a 1-GPU run of the same model, `energy_per_output_token` will differ because power draw scales with GPU count. This is the correct comparison metric. But the decision doesn't state whether `energy_per_output_token` uses `energy_total_j` (sum across all GPUs) or a per-GPU figure. The result schema shows `MultiGPUMetrics.energy_per_output_token_j` which suggests it uses the aggregate — this should be explicit.

---

### 2.16 Output Storage: Results Directory Structure

**File**: `.product/decisions/output-storage.md`
**Claim**: "Single experiment → flat JSON file. Study → `{name}_{timestamp}/` subdir. Counter suffix on collision."

**Peer evidence**:
- lm-eval: Single flat results JSON. HIGH confidence.
- MLflow: Per-run subdirectory under experiment directory. HIGH confidence.
- Hydra: Timestamped output dirs (`outputs/2026-02-18/14-30-00/`). HIGH confidence.
- W&B: Results on remote server; local files are per-run directories. MEDIUM confidence.

**Rating**: QUESTIONABLE

The decision is well-reasoned for its stated goals (human-readable, `ls results/` navigable). Two concerns:

1. **The output contract divergence is a UX risk** (noted in the experiment-study-architecture.md decision). With unified `llem run`, the same command can produce either a flat JSON file or a subdirectory depending on YAML content. This diverging output contract is harder to explain than it appears. The design says "output format determined by `len(experiments)`" — but users can't easily predict output location without knowing whether their YAML produces 1 or N experiments. Hydra solves this by always producing a timestamped subdirectory.

2. **Parquet export decision is buried in designs/result-schema.md** but not in the output-storage.md decision. The decision for when Parquet is written alongside JSON (always for studies, on request for single experiments) should be in output-storage.md.

---

### 2.17 Docker Execution Architecture

**File**: `.product/decisions/docker-execution.md`
**Claim**: "Ephemeral `docker run` per experiment. Config via env var. Results via shared volume."

**Peer evidence**:
- AIEnergyScore: Ephemeral `docker run` per experiment. HIGH confidence.
- MLPerf: One container per backend run. HIGH confidence.
- No peer tool uses persistent containers for benchmarking (confirmed in research).

**Rating**: QUESTIONABLE

The decision is mostly correct. The specific concern is the TRT engine compilation strategy.

**TRT compilation concern**: The decision caches TRT engines at `~/.llenergymeasure/trt-engines/` with a content-addressed key. The cache key excludes `builder_opt_level` because "same engine is functionally equivalent at different opt levels." This may be incorrect: TRT engines compiled at different optimisation levels can have different performance profiles (higher opt level = potentially faster but slower to compile). If a researcher varies `builder_opt_level` in a sweep, they need different cache keys. The current key design would silently serve a cached engine at the wrong optimisation level.

**Under-specification**: The decision doesn't cover what happens when the cache is stale (e.g., TRT-LLM version update changes engine format). Cache invalidation on version change is a correctness requirement for a research tool.

---

### 2.18 Error Handling

**File**: `.product/decisions/error-handling.md`
**Claim**: "`LLEMError` hierarchy. Exit codes 0/1/2/130. `ValidationError` passes through."

**Peer evidence**:
- httpx: `HTTPError` → `TransportError` → `ConnectError` hierarchy. HIGH confidence.
- pytest: Exit codes 0/1/2/3/4/5. HIGH confidence.
- lm-eval: No custom hierarchy; plain exceptions. HIGH confidence.
- Click/Typer: Exit code 2 for usage errors (automatic). HIGH confidence.

**Rating**: ALIGNED

The exception hierarchy is well-designed and follows the httpx/SQLAlchemy pattern. Allowing `ValidationError` to pass through without wrapping is the correct call — Pydantic's `.errors()` provides richer information than anything we could add by wrapping. The documented asymmetry (broad-catch `except LLEMError` won't catch `ValidationError`) is acceptable and documented.

One note: pytest uses exit codes 3 (internal error), 4 (CLI usage), 5 (no tests collected). For a study tool, exit code 5 ("no experiments generated") might be worth distinguishing from exit code 1 (runtime error). The decision correctly rejects this — keeping it at 0/1/2/130 is the simpler choice that matches lm-eval.

---

### 2.19 Backward Compatibility / API Stability

**File**: `.product/decisions/backward-compatibility.md`
**Claim**: "`__init__.py` exports only as stable API. One minor version deprecation window."

**Note**: This document still lists `run_experiment` and `run_study` as the stable v2.0.0 exports (lines 45-52), which contradicts the 2026-02-25 decision to unify as `run()`. This is a stale reference that must be updated.

**Peer evidence**:
- httpx: `__init__.py` exports as stable boundary. HIGH confidence.
- SQLAlchemy: Same pattern. HIGH confidence.
- Pydantic: Same pattern. HIGH confidence.
- lm-eval: Same pattern. HIGH confidence.

**Rating**: ALIGNED (content correct; stale API names need updating)

The one-minor-version deprecation window is aggressive but appropriate for a research tool in early v2.x. The rationale is sound.

---

### 2.20 Testing Strategy

**File**: `.product/decisions/testing-strategy.md`
**Claim**: "pytest. Two tiers: unit (no GPU) + integration (`@pytest.mark.gpu`). Protocol injection for mocks."

**Peer evidence**:
- lm-eval: `@pytest.mark.slow` and `@pytest.mark.no_cuda` markers. HIGH confidence.
- Zeus: GPU-gated tests. MEDIUM confidence.
- CodeCarbon: `@pytest.mark.parametrize` for multiple platforms. MEDIUM confidence.
- Optimum-Benchmark: Separate `tests/` directories per backend. MEDIUM confidence.

**Rating**: ALIGNED

Two-tier testing with GPU markers is the standard pattern. Protocol injection for mocks is the correct approach given the existing `InferenceBackend` Protocol design. The deferred Hypothesis decision is appropriate — sweep grammar testing with property-based testing would be valuable but adds complexity before the grammar stabilises.

**Under-specification concern**: "Measurement accuracy testing is out of scope" is stated but the concern is larger than acknowledged. A research tool that produces measurements used in publications needs *some* form of accuracy validation. The suggestion that "users should calibrate manually" is not sufficient for publishable research claims. The decision should at minimum point to a documented calibration procedure (e.g., run against a known power draw device) that users can follow.

---

### 2.21 Reproducibility

**File**: `.product/decisions/reproducibility.md`
**Claim**: "`environment_snapshot` in every result. `random_seed: 42`. Built-in datasets pinned."

**Peer evidence**:
- lm-eval: Records `lm-eval` version + `git_hash` in results. HIGH confidence.
- Optimum-Benchmark: Records full `environment_info` including all package versions. HIGH confidence.
- MLflow: Auto-logs Python version, pip packages, git hash. HIGH confidence.
- MLPerf: Full hardware and software specification required. HIGH confidence.

**Rating**: QUESTIONABLE

The reproducibility decision is honest about what can and cannot be controlled. The `±5-15% variance` statement in `reproducibility_notes` is good scientific communication. However:

1. **`pip freeze` approach**: The decision mentions "pip list --format=freeze" for environment capture. `pip freeze` is fragile: it only captures pip-installed packages, not conda-installed packages or system libraries (which include CUDA, cuDNN). Optimum-Benchmark captures `torch.version.cuda` and `torch.__version__` directly (more reliable than pip freeze). The full environment snapshot should include GPU driver version, CUDA version, cuDNN version, and NCCL version — all of which affect measurement reproducibility.

2. **Proposed `reproducibility_notes` static string**: Hardcoding `"Energy measurements have ±5-15% variance from thermal and scheduler effects"` as a static string is imprecise. The actual variance depends on GPU model, workload duration, and whether multiple cycles were run. This should be a structured field, not a static disclaimer.

3. **`random_seed: 42` for sampling**: This controls model output sampling but doesn't control GPU non-determinism. The decision acknowledges this but doesn't mention `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` — the standard PyTorch determinism controls. These are not perfect but reduce a significant source of variance.

---

### 2.22 Dataset Handling

**File**: `.product/decisions/dataset-handling.md`
**Claim**: "Named built-ins (AIEnergyScore default) + BYOJSONL + synthetic. Pinned inside package."

**Peer evidence**:
- lm-eval: Downloads datasets from HuggingFace Hub at first use (caches locally). HIGH confidence.
- AIEnergyScore: Ships `text_generation` dataset (1,000 prompts). HIGH confidence.
- MLPerf: Specific datasets required per task. HIGH confidence.
- Optimum-Benchmark: No built-in datasets — user supplies. HIGH confidence.

**Rating**: ALIGNED

The decision to ship datasets inside the package for air-gapped HPC is correct and well-reasoned. The AIEnergyScore dataset choice is strong: it's an externally validated standard from HuggingFace/CMU that enables direct comparison with published AIEnergyScore benchmarks. The synthetic mode is a genuine research tool — holding prompt length constant while varying batch size is a standard ablation pattern.

**Under-specification**: NEEDS_ADDRESSING.md item 16 notes that the `aienergyscore.jsonl` file has not been created yet. The "200-prompt subset" for default experiments and "full 1,000 via `n: 1000`" design implies a two-tier built-in dataset, but only one file is discussed. This implementation gap should be resolved before Phase 5.

---

### 2.23 Result Schema Migration

**File**: `.product/decisions/result-schema-migration.md`
**Claim**: "`schema_version` field. Missing fields load as `None`. No migration functions in v2.x."

**Peer evidence**:
- MLflow: `schema_version` in run metadata. HIGH confidence.
- lm-eval: No schema version (problematic for cross-version aggregation). HIGH confidence.
- Parquet: Schema evolution via `pyarrow.Schema.from_pandas()`. HIGH confidence.

**Rating**: ALIGNED

The "re-run to get new fields" stance is scientifically correct — fabricating measured values from old data would be dishonest. The warning-on-aggregation approach provides appropriate user guidance. Adding `schema_version` to every result file is a better practice than lm-eval's lack of versioning.

---

### 2.24 Carbon Intensity / CO2 Estimation

**File**: `.product/decisions/carbon-intensity.md`
**Claim**: "Delegate lookup to CodeCarbon. Base package: user-specified `grid_carbon_intensity_gco2_kwh` only."

**Peer evidence**:
- CodeCarbon: Maintains regional carbon intensity data. HIGH confidence.
- Zeus: No CO2 estimation. HIGH confidence.
- AIEnergyScore: Uses global average CO2 conversion factor (not region-aware). MEDIUM confidence.

**Rating**: QUESTIONABLE

The decision is well-reasoned but has one significant gap: CodeCarbon's IP-based location detection will not work in air-gapped HPC environments (explicitly acknowledged). This means HPC users who don't know their grid intensity get `co2_estimate: None`. For a tool positioned for research use at universities (which frequently run on HPC), this is a common case.

**What peer practice suggests**: AIEnergyScore uses a simple global average (not region-aware) for CO2 conversion, which at least produces *some* figure. The decision explicitly rejects this ("tool never invents a CO2 number") which is the correct scientific stance. But the consequence is that HPC users who forget to set `LLEM_CARBON_INTENSITY` get no CO2 figure at all — and the "no figure" case is easy to miss when aggregating results. A prominent warning in the result (not just `co2_estimate: None`) would be valuable.

---

### 2.25 LoRA Adapter Support

**File**: `.product/decisions/lora-adapter-support.md`
**Claim**: "Optional `lora:` block in v2.0. Hub ID + local path. `merge_weights=false` default."

**Peer evidence**:
- lm-eval: `peft=<hub-id>` in model args. HIGH confidence.
- vLLM: First-class `LoRARequest` API. HIGH confidence.
- Optimum-Benchmark: PEFT extra; delegates to backend. MEDIUM confidence.
- TRT-LLM: Requires pre-merged weights. HIGH confidence.

**Rating**: ALIGNED

The reversal from "defer to v2.1" to "include in v2.0" is correct given peer tool evidence. LoRA is standard, not an extension. The `merge_weights=false` default (measuring adapter overhead) is the right scientific default — it measures what actually happens in production LoRA deployments.

**One under-specification**: The TRT-LLM constraint ("raises validation error if `lora:` specified") is correct, but the error message design is not specified. Given that TRT-LLM users who need LoRA must pre-merge weights externally, the error message needs to tell them exactly what to do — and ideally where the pre-merged weights should go. This is a UX gap.

---

### 2.26 Access Control / Authentication

**File**: `.product/decisions/access-control.md`
**Claim**: "`.env` file for auth. Hard error if auth fields in experiment/study YAML."

**Peer evidence**:
- lm-eval: Shell environment variables only (`HF_TOKEN`). HIGH confidence.
- Zeus: No auth handling. HIGH confidence.
- MLflow: Shell environment or config file (`~/.mlflow/.credentials`). MEDIUM confidence.
- Docker Compose: `.env` file pattern. HIGH confidence.
- Hugging Face Hub: `~/.cache/huggingface/token` file. HIGH confidence.

**Rating**: OVER-ENGINEERED

The `.env` file approach is more complex than necessary and introduces a new pattern inconsistent with how the target users (ML researchers) already handle `HF_TOKEN`. Researchers using HuggingFace models already use `huggingface-cli login` (which writes to `~/.cache/huggingface/token`) and/or shell environment variables (`export HF_TOKEN=...`). The `.env` file pattern is familiar to web developers but less familiar to ML researchers.

**What peer practice suggests**: lm-eval simply reads `HF_TOKEN` from the environment. Hugging Face Hub's own Python library (`huggingface_hub`) already handles auth token discovery (`~/.cache/huggingface/token` → env var → manual login). llem should simply rely on the `huggingface_hub` auth chain rather than introducing a `.env` file pattern that adds a new credential location to manage.

The `.env.example` template and `.gitignore` entry are useful but the added complexity of detecting and hard-erroring on auth fields in YAML is over-engineered for the v2.0 scope.

---

### 2.27 Observability (Live Progress)

**File**: `.product/decisions/live-observability.md`
**Claim**: "Rich Progress + Live for study-level. tqdm retained at experiment level. Three verbosity levels."

**Peer evidence**:
- lm-eval: Rich-based progress display. HIGH confidence.
- Zeus: No rich progress; simple logging. HIGH confidence.
- Optimum-Benchmark: tqdm-based. MEDIUM confidence.

**Rating**: ALIGNED (with stale terminology)

The Rich + tqdm mixing is a pragmatic v2.0 decision (retain working code). The three verbosity levels match lm-eval and pytest conventions. The stdout/stderr split for pipeline composability is correct Unix practice.

**Stale terminology**: The live-observability.md document still says `llem study` throughout (line 168: "`llem study` (multi-experiment study)"). Post 2026-02-25 unification, all study invocations use `llem run study.yaml`. This document needs updating.

---

### 2.28 HPC/SLURM Deferral

**File**: `.product/decisions/hpc-slurm.md`
**Claim**: "Defer to v3.x. Layer 1 runner system accommodates Singularity without breaking changes."

**Peer evidence**:
- lm-eval: No SLURM integration in core. Users run `lm-eval` inside SLURM jobs manually. HIGH confidence.
- Nextflow: Native SLURM support via executor. HIGH confidence (not a benchmarking tool).
- Snakemake: Native SLURM support. HIGH confidence (not a benchmarking tool).

**Rating**: ALIGNED

Deferring HPC to v3.x is correct for the target v2.0 audience (local GPU measurement). The runner abstraction in Layer 1 (Singularity runner key reserved) is the right structural decision to make HPC addition non-breaking later.

---

## 3. Cross-Document Inconsistencies

These are contradictions where Document A says X and Document B implies Y, without explicit supersession annotation.

---

### CI-1: CRITICAL — Library API (`run()` vs `run_experiment()` / `run_study()`)

**Documents**:
- `experiment-study-architecture.md` (2026-02-25): "`llem.run(config) -> ExperimentResult | StudyResult`. Single function."
- `designs/library-api.md` (2026-02-19, not updated): `run_experiment()` + `run_study()` as separate functions, with active argument against union return type.
- `backward-compatibility.md` (2026-02-19, not updated): Lists `run_experiment, run_study` as stable v2.0.0 exports.
- `architecture.md` sub-decision H (updated 2026-02-25): Lists `run` as the export — consistent with unification.
- `designs/CLAUDE.md` index: `library-api.md` status listed as "Confirmed (session 3, naming updated session 4)" — predates the 2026-02-25 unification.

**The contradiction**: The most recent decision (experiment-study-architecture.md) mandates a unified `run()` with union return type. The design file it should have updated (`designs/library-api.md`) was not updated. Implementers reading `designs/library-api.md` will implement the wrong API.

**Resolution needed**: `designs/library-api.md` must be updated to reflect `run()` with union return type and overloads. `backward-compatibility.md` stable exports list must be updated.

---

### CI-2: CRITICAL — `llem study` command in active documents

**Documents using `llem study` as if still valid (post-2026-02-25 unification to `llem run`)**:
- `experiment-isolation.md` lines 101, 273: "requirement applies to `llem study` / `StudyRunner` only"
- `experiment-isolation.md` line 320: "llem study batch-size-sweep.yaml"
- `study-execution-model.md` lines 102-103: `llem study file.yaml --profile quick`
- `study-execution-model.md` multiple: "llem study" as CLI command
- `study-resume.md`: `llem study resume` command throughout
- `output-storage.md` lines 16, 108: "llem run and llem study both write result files"
- `live-observability.md` line 168: "`llem study` (multi-experiment study)" section header
- `documentation-strategy.md` line 77: `studies.md — llem study: YAML format`
- `versioning-roadmap.md` line 77: "CLI 15→3 commands" and "3 CLI commands: `llem run`, `llem study`, `llem config`"
- `progressive-disclosure.md` line 43: "llem run study.yaml OR llem study study.yaml"
- `architecture.md` sub-decision J: "`llem study` is the primary differentiator"
- `installation.md` Step 6: "llem study batch-size-effects.yaml"
- `backward-compatibility.md` line 155: references `run_study()` n_cycles behaviour
- `designs/CLAUDE.md` index: `cli-commands.md` described as "3 commands + 1 flag"
- `.product/CLAUDE.md` (meta rules): "CLI: 3 commands — `llem run`, `llem study`, `llem config`" and "Library: `run_experiment()`, `run_study()`"

The `.product/CLAUDE.md` quick reference section at the bottom **still states the pre-unification API as the current truth** — this is the instructions file for future Claude sessions working on this project. This is a high-risk staleness.

**Resolution needed**: All references to `llem study` as a CLI command must be updated to `llem run [study.yaml]`. All references to `run_experiment()` / `run_study()` as the library API must be updated to `run()`.

---

### CI-3: study-execution-model.md Decision D vs experiment-study-architecture.md

**Documents**:
- `study-execution-model.md` Decision D (2026-02-20): "D1 — reject unification. `experiment` and `study` remain distinct, first-class concepts. Two CLI commands, two config types."
- `experiment-study-architecture.md` (2026-02-25): Supersedes Decision D, accepts Option C (unified `llem run`, unified `run()`).

**Status**: Decision D is marked as "Superseded" in `experiment-study-architecture.md`. But the supersession annotation has NOT been added to the `study-execution-model.md` file itself — someone reading `study-execution-model.md` would see Decision D as the active decision.

**Resolution needed**: Add a `> **Superseded (2026-02-25):** by experiment-study-architecture.md Option C.` annotation to Decision D in study-execution-model.md.

---

### CI-4: n_cycles CLI effective default — Inconsistent across documents

**Documents**:
- `study-execution-model.md` measurement protocol table: `cycle_order` Pydantic default = `"sequential"`, CLI effective default = `"interleaved"`
- `cli-ux.md` execution profiles table: CLI effective default for `n_cycles = 3`, cycle_order not specified separately for the no-profile-flag case.
- `study-execution-model.md` line 280: `cycle_order: "interleaved"` is the CLI effective default
- `experiment-study-architecture.md` Q2: "`llem run --model X` (no YAML) → CLI effective default n_cycles=3"

**The inconsistency**: Is the CLI effective default for `cycle_order` "sequential" (Pydantic default) or "interleaved" (stated in the SSOT field table)? The SSOT table in `study-execution-model.md` says interleaved is the CLI effective default, but no document explains *where* this default is applied (it's not in the Pydantic model, so it must be applied in the CLI layer). This is an unimplemented design element.

**Resolution needed**: Clarify where CLI effective defaults for `cycle_order` are applied (CLI argument defaults vs Pydantic model defaults) and ensure consistency across cli-ux.md and study-execution-model.md.

---

### CI-5: study-design-hash excludes execution — but study.yaml is the file

**Documents**:
- `study-execution-model.md` Decision B: "`study_design_hash` excludes execution block — same study at different rigour = same hash."
- `designs/result-schema.md`: `StudyResult.study_design_hash` = "SHA-256[:16] of sweep+experiments only (execution block excluded)"
- `study-execution-model.md` SSOT field table: `execution: block` — "Not in `study_design_hash`"

**Under-specification**: If a user changes `n_cycles` from 3 to 5 ("topping up"), the same hash applies. But the result file set at n_cycles=3 and n_cycles=5 are stored together in the same `{study_name}_{timestamp}/` directory (they would each have different timestamps and thus different subdirectories). How does a user "top up" a study — do they run `llem run study.yaml` again (which creates a new `{name}_{timestamp_2}/` directory) and manually aggregate? The hash design enables top-up identity, but the workflow for actually doing it is unspecified. This should be in designs/study-resume.md but the resume decision defers the implementation.

---

### CI-6: Dataset default — `alpaca` reference in library-api.md

**Document**: `designs/library-api.md` line 95: `dataset="alpaca"` as example default.
**Correct decision**: `dataset: "aienergyscore"` is the default (confirmed in dataset-handling.md and NEEDS_ADDRESSING.md item 26 which states this was fixed in cli-commands.md and installation.md).

The `library-api.md` design file still shows `alpaca` as the example dataset in code samples (lines 95, 102). This is a stale reference that contradicts the confirmed AIEnergyScore default.

---

### CI-7: NVML single-session owner — Implementation path unclear

**Documents**:
- `architecture.md` sub-decision G: "NVML single-session owner: Only one NVML session active at a time. When Zeus is installed and active, base NVML poller must yield."
- `installation.md`: Same constraint noted.
- `designs/energy-backends.md`: Listed in DESIGNS CLAUDE.md index as containing the "Energy backend plugin system; backend registry; accuracy table; NVML conflict" — but this file was not read during audit (it is a designs file, not decisions).

**Concern**: The NVML single-session owner constraint is described in two decisions files but the *mechanism* for enforcement is only promised in a designs file. The decisions should reference the enforcement mechanism. As stated, the constraint could be violated silently if CodeCarbon is installed and initialised before Zeus. The enforcement must be at import time or measurement start time, not a documentation-only constraint.

---

## 4. Recommendations for Harmonisation Phase

Ordered by priority.

---

### P0 — Critical (blocks implementation correctness)

**P0.1 Update `designs/library-api.md` to reflect `run()` unified API**
- Replace `run_experiment()` + `run_study()` with `run(config) -> ExperimentResult | StudyResult`
- Remove the argument against union return types (superseded by experiment-study-architecture.md)
- Add `@overload` signatures for the three input forms
- Status: Superseded content actively present in the SSOT design file

**P0.2 Update `.product/CLAUDE.md` quick reference**
- Remove: `CLI: 3 commands — llem run, llem study, llem config`
- Add: `CLI: 2 commands + 1 flag — llem run, llem config, --version`
- Remove: `Library: run_experiment(), run_study()`
- Add: `Library: run(config) -> ExperimentResult | StudyResult`
- This file is read at session start by Claude instances — staleness here propagates into all future work.

**P0.3 Add supersession annotation to `study-execution-model.md` Decision D**
- Add `> **Superseded (2026-02-25):** by experiment-study-architecture.md Option C — unified llem run command.` at the start of the Decision D section.

**P0.4 Update `backward-compatibility.md` stable exports list**
- Replace `run_experiment, run_study` with `run`
- This file defines the v2.0.0 stability contract — incorrect names here create false commitments.

---

### P1 — High (terminology drift; will confuse implementers)

**P1.1 Update `experiment-isolation.md`**
- Replace `llem study` with `llem run study.yaml` throughout
- Replace `run_experiment()` / `run_study()` with `run()` in the API diagram

**P1.2 Update `study-execution-model.md` non-superseded sections**
- Replace `llem study file.yaml --profile quick` with `llem run file.yaml --profile quick`
- Clarify where CLI effective default for `cycle_order: "interleaved"` is applied (CI-4)

**P1.3 Update `live-observability.md`**
- Replace "`llem study` (multi-experiment study)" section header with "`llem run study.yaml`"

**P1.4 Update `versioning-roadmap.md` v2.0 scope**
- Replace "3 CLI commands: `llem run`, `llem study`, `llem config`" with "2 commands + 1 flag"

**P1.5 Update `output-storage.md`**
- Replace "llem run and llem study both write result files" with unified `llem run` language

**P1.6 Update `installation.md` progressive disclosure flow**
- Step 6 shows `llem study batch-size-effects.yaml` — update to `llem run batch-size-effects.yaml`

**P1.7 Fix `designs/library-api.md` dataset default**
- Replace `dataset="alpaca"` with `dataset="aienergyscore"` in all code examples (CI-6)

---

### P2 — Medium (under-specification; risks research validity)

**P2.1 Calibrate warmup thermal floor (warmup-strategy.md)**
- Replace "30 seconds (chosen)" with "30 seconds for consumer/data-centre GPUs; calibration procedure documented in reproducibility.md"
- Add note that A100/H100 may require longer thermal stabilisation (documented in manufacturer power management guides)

**P2.2 Reframe FLOPs as normalisation metric, not primary metric (flops-estimation.md)**
- FLOPs are identical for any implementation of the same model on same inputs — they cannot differentiate PyTorch vs vLLM
- Primary metrics for implementation comparison should be: `energy_total_j`, `tokens_per_second`, `ttft_ms`, `itl_ms`, `energy_per_output_token_j`
- FLOPs belong in the methodology section and as a denominator for hardware comparison, not as a standalone "efficiency" metric
- The `flops_per_output_token` metric *is* useful for cross-hardware comparison (how efficiently does GPU X execute these FLOPs?) — keep it, but reframe its purpose

**P2.3 Specify TRT-LLM engine cache invalidation policy (docker-execution.md)**
- When TRT-LLM version is updated, cached engines may be invalid (format changes between TRT-LLM versions)
- Add: "Cache is invalidated by including `tensorrt_llm.__version__` in the cache key"

**P2.4 Tighten sweep grammar edge case: `pytorch.attn.implementation` (config-architecture.md)**
- Clarify what happens when a nested dotted key (`pytorch.attn.implementation`) resolves to a field that doesn't exist in `PyTorchConfig`
- With `extra="forbid"`, this should be a `ValidationError` at sweep resolution time, not silently passed
- Add this case to the edge-case table

**P2.5 Add TRT LoRA error message specification (lora-adapter-support.md)**
- Specify the exact error message when `lora:` is used with TRT-LLM backend
- Should include: (1) what the constraint is, (2) how to pre-merge weights, (3) where to point merged weights in config

**P2.6 Replace static `reproducibility_notes` string with structured field (reproducibility.md)**
- The `±5-15%` range is model-dependent and should not be hardcoded as a static string
- Consider a `measurement_variance_notes: dict` with per-metric documented uncertainty

**P2.7 Multi-GPU energy aggregation for PyTorch device_map="auto" (multi-gpu.md)**
- Explicitly state whether NVML is polled per-device or aggregate for `device_map="auto"` (pipeline parallelism)
- Confirm that `pynvml.nvmlDeviceGetHandleByIndex(i)` is called for each GPU index (not just GPU 0)

---

### P3 — Low (design improvements; not blocking)

**P3.1 Reconsider `.env` approach for auth (access-control.md)**
- Target users are ML researchers who already use `huggingface-cli login` and shell env vars
- `.env` file adds a new credential location that HF users don't expect
- Simplest correct approach: rely on `huggingface_hub.HfApi()` auth chain (which already handles token file + env var)
- Only action needed: pre-flight check for `HF_TOKEN` env var or `~/.cache/huggingface/token` file

**P3.2 Align three-layer config framing with peer two-layer models (architecture.md)**
- Reframe "Layer 3" (infrastructure context) as "result metadata" not as a third config layer
- The config model has two surfaces: user config (Layer 1) and experiment/study YAML (Layer 2)
- Infrastructure context is captured into results — it is not a "config" that users read or write
- This reframing removes complexity without changing any actual behaviour

**P3.3 Specify top-up study workflow (study-execution-model.md, study-resume.md)**
- The hash design enables identifying "same study at different rigour" but the workflow for aggregating results across two study runs is unspecified
- Add a section to study-resume.md or output-storage.md explaining the expected user workflow

**P3.4 Add `torch.backends.cudnn.deterministic` to reproducibility controls (reproducibility.md)**
- This is a standard PyTorch determinism control that reduces (but doesn't eliminate) GPU non-determinism
- Should be documented as an optional control that llem can enable/disable

**P3.5 Address AIEnergyScore dataset file gap (dataset-handling.md / NEEDS_ADDRESSING.md item 16)**
- Specify pinned commit hash for the AIEnergyScore dataset before Phase 5 implementation
- The "200-prompt subset vs full 1000 via n:1000" design implies the file exists — it doesn't yet

---

## 5. Summary Table

| Decision Area | File | Rating | Priority Action |
|---------------|------|--------|-----------------|
| Library-first architecture | architecture.md (A) | ALIGNED | None |
| Module structure | architecture.md (B) | ALIGNED | None |
| Three-layer config model | architecture.md (C/D/E) | QUESTIONABLE | P3.2 reframe |
| Energy vs CO2 separation | architecture.md (G) | ALIGNED | None |
| Library API surface | architecture.md (H) + library-api.md | CONTRADICTED | P0.1 critical |
| Subprocess isolation | experiment-isolation.md | ALIGNED (stale terms) | P1.1 |
| Unified `llem run` | cli-ux.md | ALIGNED | None |
| Composition vs inheritance | config-architecture.md (C1) | ALIGNED | None |
| Sweep grammar (dotted) | config-architecture.md (C2) | QUESTIONABLE | P2.4 tighten |
| Execution profiles | cli-ux.md (D) | ALIGNED | None |
| Option C architecture | experiment-study-architecture.md | ALIGNED | P0.3, CI-3 |
| Installation/extras | installation.md | ALIGNED | P1.6 |
| Warmup strategy | warmup-strategy.md | ALIGNED | P2.1 calibrate |
| FLOPs estimation | flops-estimation.md | CONTRADICTED | P2.2 reframe |
| Multi-GPU | multi-gpu.md | QUESTIONABLE | P2.7 |
| Docker execution | docker-execution.md | QUESTIONABLE | P2.3 |
| Error handling | error-handling.md | ALIGNED | None |
| Output storage | output-storage.md | QUESTIONABLE | P1.5 |
| Backward compatibility | backward-compatibility.md | ALIGNED (stale API) | P0.4 |
| Testing strategy | testing-strategy.md | ALIGNED | None |
| Reproducibility | reproducibility.md | QUESTIONABLE | P2.6 |
| Dataset handling | dataset-handling.md | ALIGNED | P3.5 |
| Schema migration | result-schema-migration.md | ALIGNED | None |
| Carbon intensity | carbon-intensity.md | QUESTIONABLE | note only |
| LoRA adapter support | lora-adapter-support.md | ALIGNED | P2.5 |
| Multi-GPU | multi-gpu.md | QUESTIONABLE | P2.7 |
| Access control | access-control.md | OVER-ENGINEERED | P3.1 |
| Observability | live-observability.md | ALIGNED (stale terms) | P1.3 |
| HPC/SLURM deferral | hpc-slurm.md | ALIGNED | None |
| Product vision | product-vision.md | ALIGNED | None |
| Versioning roadmap | versioning-roadmap.md | ALIGNED (stale count) | P1.4 |

---

## 6. Confidence Notes

- All peer tool behaviours cited are HIGH confidence unless marked otherwise (verified against research files 01-16 in `.product/research/` and cross-referenced against CLI/library documentation)
- FLOPs reframing (§2.14) is a domain-specific challenge based on LLM efficiency benchmarking practice — MEDIUM confidence on the specific "implementation-invariant" claim; HIGH confidence that `energy_per_output_token` is the dominant primary metric in published work
- The thermal floor calibration concern (§P2.1) is based on GPU manufacturer power management documentation (known to vary by GPU model) — HIGH confidence that 30s is insufficient for some GPU models
