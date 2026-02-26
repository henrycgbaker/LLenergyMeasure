# Architecture Audit: LLenergyMeasure v2.0 Decisions vs Peer Evidence

**Audited:** 2026-02-25
**Scope:** Decisions A through J in `.product/decisions/architecture.md`, plus config-architecture.md, experiment-study-architecture.md, experiment-isolation.md, error-handling.md, backward-compatibility.md
**Method:** Fresh research against peer tools (optimum-benchmark, lm-eval-harness, Zeus, MLflow, vLLM, Hydra/OmegaConf), cross-referenced with existing `.product/research/` evidence

---

## Audit Summary

| Decision | Verdict | Confidence | One-Liner |
|----------|---------|------------|-----------|
| **A. Library-first** | ALIGNED | HIGH | Industry standard; lm-eval, optimum-benchmark, Zeus all do this |
| **B. Module structure** | ALIGNED | HIGH | Single package with internal modules is universal |
| **C. Three-layer config** | QUESTIONABLE | MEDIUM | Over-engineered vs peers; no peer tool uses three explicit layers |
| **D. Field placement** | OVER-ENGINEERED | MEDIUM | The "three questions" framework adds complexity without peer precedent |
| **E. Runner resolution** | ALIGNED | MEDIUM | Keeping runner out of experiment YAML is correct; precedence chain is fine |
| **F. Infrastructure metadata** | ALIGNED | HIGH | Auto-detect + store is exactly what Zeus and CodeCarbon do |
| **G. Energy vs CO2** | ALIGNED | HIGH | Independent concerns; matches Zeus/CodeCarbon split exactly |
| **H. Library API surface** | QUESTIONABLE | HIGH | Union return type from `run()` contradicts peer practice and typing guidance |
| **I. Subprocess isolation** | ALIGNED | HIGH | `multiprocessing.Process` per experiment matches optimum-benchmark exactly |
| **J. Study ships at v2.0** | ALIGNED | MEDIUM | Reasonable scope; sweeps are the differentiator |

**Config architecture (C1/C2):** ALIGNED for composition; ALIGNED for dotted sweep notation
**Experiment-study architecture (Option C):** QUESTIONABLE on unified `run()` return type
**Error handling:** ALIGNED; exception hierarchy is standard
**Backward compatibility:** ALIGNED; `__init__.py` exports is industry standard

---

## Decision-by-Decision Analysis

### A. Architecture Pattern: Library-First at v2.0

**Verdict: ALIGNED**
**Confidence: HIGH**

Every credible peer tool in this space is library-first:

| Tool | Library API | CLI relationship |
|------|------------|-----------------|
| lm-eval-harness | `simple_evaluate()`, `evaluate()` | CLI wraps these via `__main__.py` |
| optimum-benchmark | `Benchmark.launch(BenchmarkConfig)` | CLI is Hydra `@hydra.main` wrapping `Benchmark.launch()` |
| Zeus | `ZeusMonitor()`, `begin_window()`/`end_window()` | CLI (`zeus monitor`) wraps the library |
| MLflow | `mlflow.log_metric()`, `MlflowClient()` | CLI wraps the client |
| CodeCarbon | `EmissionsTracker()`, `tracker.stop()` | No separate CLI (pure library) |

lm-eval exports exactly three items from `__init__.py`: `evaluate`, `simple_evaluate`, `__version__`. This is the narrowest public surface in the ecosystem and matches the llem decision precisely.

optimum-benchmark's `Benchmark.launch(config)` is a static method that spawns a `multiprocessing.Process` internally. The API is: construct config objects, call one function, get a report back. This is the exact same pattern as llem's `run()`.

**No challenge to this decision.** Library-first is unambiguously correct.

**Sources:** [lm-eval-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness), [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark), [Zeus GitHub](https://github.com/ml-energy/zeus)

---

### B. Module Structure: Single Package

**Verdict: ALIGNED**
**Confidence: HIGH**

All peer tools use a single package:

- `lm_eval/` with `evaluator.py`, `models/`, `tasks/`, `api/`
- `optimum_benchmark/` with `benchmark/`, `backends/`, `launchers/`, `scenarios/`, `trackers/`
- `zeus/` with `monitor/`, `optimizer/`, `device/`, `utils/`

No peer tool splits CLI, study/sweep, or backend modules into separate installable packages. The `src/llenergymeasure/` layout with `cli/`, `study/`, `core/`, `config/` inside is standard.

optimum-benchmark's structure is the closest analogue:

```
optimum_benchmark/
    backends/        # inference backends (PyTorch, vLLM, TRT-LLM, etc.)
    benchmark/       # BenchmarkConfig, Benchmark.launch()
    launchers/       # process, inline, torchrun
    scenarios/       # inference, training
    trackers/        # energy, memory, latency
    cli.py           # Hydra CLI entry point
```

This maps well to llem's planned structure. The key architectural parallel: backends, launchers (isolation), scenarios (what to measure), and trackers (how to measure) are all separate modules within one package.

**No challenge to this decision.**

---

### C. Three-Layer Config Model

**Verdict: QUESTIONABLE**
**Confidence: MEDIUM**

This is the most over-engineered decision relative to peer practice. No peer tool uses an explicit "three-layer" config model with named layers.

**What peers actually do:**

| Tool | Config layers | Notes |
|------|-------------|-------|
| **optimum-benchmark** | 1 layer: `BenchmarkConfig` (launcher + scenario + backend composed) | Environment auto-captured separately |
| **lm-eval** | 1 layer: `simple_evaluate()` kwargs or YAML task config | No user config file; model_args is a flat string |
| **Zeus** | 0 layers: constructor args to `ZeusMonitor(gpu_indices=[0,1])` | No config file at all |
| **vLLM bench sweep** | 2 JSON files: serve_params.json + bench_params.json | No user config; environment is implicit |
| **Hydra** | N layers: config groups composed at runtime | Most complex; also the most complained-about |
| **MLflow** | 1 config + tracking server | Environment/artifact store config is server-side |

**The core issue:** Layer 1 (user config), Layer 2 (experiment YAML), and Layer 3 (infrastructure context) are presented as a fundamental architectural concept requiring a "three questions" decision framework. In practice:

- **Layer 1** is just a user preferences file (`~/.config/...`). Every tool that has user preferences has this. It does not need a special "layer" name.
- **Layer 2** is the experiment definition. This IS the config. Calling it "Layer 2" adds jargon without adding clarity.
- **Layer 3** is auto-captured runtime metadata stored with results. This is not config at all -- it is output. Zeus stores GPU info with measurements. CodeCarbon stores hardware info with emissions. Neither calls this a "config layer".

**The valid separation underneath the jargon is sound:**
1. Machine-local preferences (results_dir, runner mappings) -- user config file
2. Experiment definition (model, backend, params) -- experiment YAML, shareable
3. Runtime environment snapshot (GPU model, CUDA version) -- auto-captured, stored with results

This is a two-config-source + auto-capture pattern, not a three-layer config model. The distinction matters because "three layers" implies three things the user must understand and configure. In reality, Layer 3 is invisible to the user (auto-captured) and Layer 1 has sensible defaults that most users never touch.

**Recommendation:** Keep the separation (it is correct). Drop the "three-layer" naming and framework. Call them what they are: user config, experiment config, and environment snapshot. The "three questions" in Decision D add cognitive overhead for implementors without preventing any real mistakes.

**Risk if unchanged:** Implementors treat Layer 3 as something that needs config-level validation, precedence resolution, and user-facing documentation. It does not. It is auto-captured metadata.

---

### D. Field Placement (Three Questions Framework)

**Verdict: OVER-ENGINEERED**
**Confidence: MEDIUM**

The "three questions" framework ("Does this vary between machines?", "Does this define what I'm measuring?", "Does this describe the physical environment?") is a decision heuristic that is more complex than needed. In practice, the field placement table in architecture.md already resolves every field -- the questions are a retroactive justification, not a discovery tool.

**What peers do instead:** Fields belong in the experiment config if the user specifies them. Everything else is either auto-detected or has a sensible default. There is no formal placement framework.

optimum-benchmark example: `BenchmarkConfig` takes `launcher`, `scenario`, and `backend` configs. The `environment` field is auto-populated from `get_system_info()` at construction time. No placement decision framework needed.

**The field placement table itself is useful** (it documents where every field lives). The "three questions" heuristic adds no value on top of the table.

**Recommendation:** Keep the field placement table. Remove the "three questions" framework. It is solving a problem that does not recur -- once the table is written, it is the SSOT.

---

### E. Runner Resolution

**Verdict: ALIGNED**
**Confidence: MEDIUM**

Keeping `runner:` out of experiment YAML is correct. The precedence chain (env var > user config > CLI flag > default) is standard.

**Peer evidence:**
- optimum-benchmark: launcher type is specified in the Hydra config but is conceptually separate from the experiment definition. The backend config (what to measure) is independent of the launcher config (how to run).
- vLLM bench sweep: `--serve-cmd` is a CLI argument, not in the sweep params JSON.
- lm-eval: No runner concept -- runs in-process always.

The only concern: the precedence chain has four levels (`LLEM_RUNNER_<BACKEND>` env var > user config > `--runner` CLI flag > default). Four levels is one more than most tools. Env var override of user config is standard. CLI flag overriding user config is standard. Both overriding each other introduces a question: which wins, env var or CLI flag?

The current precedence puts env var above CLI flag. This is unconventional -- most tools put CLI flags above env vars (CLI is more explicit). However, for runner selection in CI/Docker environments, env var taking precedence is defensible because the environment determines what runners are available.

**Minor concern only.** The decision is sound.

---

### F. Infrastructure Metadata as Scientific Record

**Verdict: ALIGNED**
**Confidence: HIGH**

Auto-detect + store with results is exactly what peers do:

- **Zeus**: `ZeusMonitor` auto-detects GPU count, model, NVML capabilities. Energy results include GPU indices and device info.
- **CodeCarbon**: Auto-detects CPU, GPU, cloud provider, region. Stores in `emissions.csv` alongside energy data.
- **optimum-benchmark**: `environment` field on `BenchmarkConfig` auto-populates from `get_system_info()` + `get_hf_libs_info()` at instantiation.
- **lm-eval**: Captures `model_info`, `versions` dict, git hash, and full config in result output.

Calling this "Layer 3" is the jargon issue (see Decision C). The practice itself is industry-standard.

**No challenge to this decision.**

---

### G. Energy vs CO2 as Independent Concerns

**Verdict: ALIGNED**
**Confidence: HIGH**

This maps perfectly to the Zeus/CodeCarbon split:

- **Zeus**: Measures energy (Joules, Watts). No CO2 estimation.
- **CodeCarbon**: Estimates CO2 emissions. Uses its own energy measurement internally (NVML polling or RAPL).
- **ML.ENERGY leaderboard**: Reports energy in Joules. CO2 is a separate calculation using regional carbon intensity.

The decision to make base package = NVML polling, `[zeus]` = accurate energy, `[codecarbon]` = CO2 matches the ecosystem perfectly. The NVML single-session owner constraint (only one of base NVML poller or Zeus can be active) is a genuine technical constraint documented in Zeus.

**No challenge to this decision.**

---

### H. Library API Surface: Unified `run()` with Union Return Type

**Verdict: QUESTIONABLE**
**Confidence: HIGH**

This is the second most concerning decision. The unified `run()` function returns `ExperimentResult | StudyResult` based on input type. This creates a union return type that callers must narrow.

**The typing problem is well-documented.** From python/mypy#1693: "Having a union return type is often a problem... any attempt to use operations specific to one return type will fail a type check." Python's `@overload` can help if the input type determines the output type, but in llem's case:

```python
# These both take str | Path -- overload cannot distinguish them
result = llem.run("experiment.yaml")   # -> ExperimentResult
result = llem.run("study.yaml")        # -> StudyResult
```

The overload only works for `ExperimentConfig` vs `StudyConfig` input. For path-based input (the common case for YAML users), the return type is `ExperimentResult | StudyResult` and every call site needs `isinstance()` or `assert`.

**What peers do:**

| Tool | API shape | Return type |
|------|----------|-------------|
| lm-eval | `simple_evaluate(model, tasks)` | Always `EvalResults` (one type) |
| optimum-benchmark | `Benchmark.launch(config)` | Always `BenchmarkReport` (one type) |
| Zeus | `monitor.end_window(name)` | Always `Measurement` (one type) |
| MLflow | `mlflow.start_run()` context | Returns `Run` (one type) |

**Every peer tool returns exactly one type from its entry point.** None uses a union return type.

**The existing designs/library-api.md actually anticipated this.** It originally had `run_experiment()` returning `ExperimentResult` and `run_study()` returning `StudyResult` -- two functions, each with an unambiguous return type. The justification for unification was "Why not `run("config.yaml")`? A single `run()` has an ambiguous return type" -- the doc correctly identified the problem, then the experiment-study-architecture.md decision overrode it.

**Alternatives:**

1. **Revert to two functions** (`run_experiment()` + `run_study()`): Matches peer practice. Unambiguous types. The cost (two functions to learn) is trivial.

2. **Always return `StudyResult`**: Since internally all paths go through `_run(StudyConfig)`, the natural return type is `StudyResult`. Single experiments return a `StudyResult` with `len(experiments) == 1`. Callers always get the same type. This is analogous to how lm-eval always returns `EvalResults` even for a single task.

3. **Keep unified `run()` but accept the typing tax**: Use `@overload` for the `ExperimentConfig`/`StudyConfig` input case and accept that path-based input returns a union.

**Recommendation:** Option 1 (two functions) is the cleanest. It was the original design for good reason. The argument that unified `run()` is "simpler" is undermined by the union return type making every call site more complex.

However: Option 2 (always `StudyResult`) is also viable and has the advantage of truly unifying the internal model. A single experiment becomes `StudyResult(experiments=[result])`. Callers who want the single result do `result.experiments[0]`. This is ugly but honest about the internal architecture.

**The decision to unify `llem run` at the CLI level is separate and correct.** CLI commands do not have return types. The CLI can present single-experiment results differently from study results without API typing concerns. The library API and CLI do not need to mirror each other -- the existing docs already note this ("CLI names differ from library names").

---

### I. Subprocess Isolation via `multiprocessing.Process`

**Verdict: ALIGNED**
**Confidence: HIGH**

This decision matches optimum-benchmark's core architecture exactly. The research in `13-execution-isolation-patterns.md` already documented this comprehensively.

Key confirmations from fresh research:

1. **`spawn` start method is mandatory for CUDA.** PyTorch documentation (2025) confirms: "spawn is preferred because it avoids copying the CUDA context into child processes." Fork is unsafe with CUDA -- this is not a design choice, it is a correctness requirement.

2. **`multiprocessing.Pipe` for IPC is correct.** optimum-benchmark uses exactly this pattern, including the 1MB file-based fallback threshold.

3. **`daemon=False` is correct.** Allows clean CUDA teardown. optimum-benchmark also uses `daemon=False`.

4. **Timeout via `p.join(timeout=...)` + SIGKILL is correct.** optimum-benchmark uses a polling loop with `p.is_alive()` and `parent_connection.poll()` rather than `p.join(timeout=...)`, but both achieve the same goal. The SIGKILL on timeout is the only reliable mechanism for hung CUDA calls.

**One minor note:** optimum-benchmark uses sync checkpoints (two `sync_with_child()` calls) before starting the benchmark. This ensures the child process is ready before the parent begins waiting for results. The llem design does not include sync checkpoints. This is a robustness improvement worth considering but not a fundamental architectural difference.

**No challenge to this decision.**

---

### J. Study Module Scope: Ships at v2.0

**Verdict: ALIGNED**
**Confidence: MEDIUM**

Parameter sweeps are the core value proposition of this tool ("how much does implementation choice affect efficiency?"). Deferring sweeps to v2.2 would ship a single-experiment tool that is just a less mature version of existing tools.

**Peer context:**
- optimum-benchmark includes sweep support from the start (via Hydra `--multirun`).
- vLLM bench sweep is a first-class feature, not an afterthought.
- lm-eval runs multiple tasks in a single invocation by default.

Docker multi-backend deferred to v2.2 is reasonable. Local single-backend studies at v2.0 provide immediate research value.

**No challenge to this decision.**

---

## Cross-Cutting Decisions

### Config Architecture: Composition (C1) + Dotted Sweep Notation (C2)

**Verdict: ALIGNED**
**Confidence: MEDIUM**

**C1 (Composition):** The decision to use a single `ExperimentConfig` with optional backend sections is well-justified. The rejected inheritance approach would propagate a union type through every layer. optimum-benchmark uses inheritance (separate backend config classes) but does not have the same problem because its `BenchmarkConfig` takes `backend: Any` -- the type system does not constrain the backend type. llem's composition approach is more type-safe than optimum-benchmark's `Any` while avoiding the union proliferation of full inheritance.

**C2 (Dotted sweep notation):** The `pytorch.batch_size: [1, 8, 32]` syntax is custom but well-designed. The split-on-first-dot parsing is clean. The per-backend Cartesian product is the correct semantic.

**Peer comparison on sweep grammar:**

| Tool | Sweep syntax | Notes |
|------|-------------|-------|
| vLLM bench sweep | Separate JSON files for serve_params and bench_params | Cartesian product of serve x bench |
| optimum-benchmark | Hydra `--multirun` with comma-separated overrides | `backend.device=cpu,cuda` |
| W&B Sweeps | YAML with `parameters:` block, flat keys | `batch_size: {values: [1, 8, 32]}` |
| Optuna | Programmatic `trial.suggest_*()` API | No declarative grammar |

llem's dotted notation is more compact than vLLM's separate files and more structured than Hydra's override syntax. It handles the multi-backend case (where backend-specific params are not interchangeable) better than any peer's approach.

**Concern:** The `extra = "forbid"` with an explicit `extra: dict` escape hatch is clever but could confuse users who encounter the Pydantic error message for typos vs the `extra:` field. This is a documentation challenge, not an architectural one.

---

### Experiment-Study Architecture (Option C)

**Verdict: ALIGNED on internal architecture, QUESTIONABLE on library API**

The Option C architecture is sound:
- `ExperimentConfig` = pure data type, zero study knowledge. Correct.
- `StudyConfig` = resolved container (`list[ExperimentConfig]` + `ExecutionConfig`). Correct.
- Sweep resolution at YAML parse time, before Pydantic. Correct.
- Single runner `_run(StudyConfig)`. Correct.

This matches the "config file carries the complexity signal" pattern identified in pytest, lm-eval, and `mlflow run`. The internal unification is good architecture.

The concern is the library API surface (see Decision H above).

---

### Error Handling (K1/K2/K3)

**Verdict: ALIGNED**
**Confidence: HIGH**

**K1 (Exit codes 0/1/2/130):** Correct. No ML benchmarking tool uses granular exit codes. lm-eval uses 0/1.

**K2 (Custom exception hierarchy):** Correct. `LLEMError` root with typed subclasses is standard for library-first tools. httpx, SQLAlchemy, Pydantic all do this.

**K3 (Pydantic ValidationError passthrough):** Correct. Wrapping Pydantic's rich error in a custom exception would lose information. The asymmetry (`ValidationError` is not `LLEMError`) is a known, documented trade-off.

**One note:** The `ExperimentError` class has `config: dict` and `cause: Exception` as class attributes in the design, not `__init__` parameters. These should be instance attributes set in `__init__`. This is a design doc issue, not an architectural issue.

---

### Backward Compatibility

**Verdict: ALIGNED**
**Confidence: HIGH**

`__init__.py` exports as the sole stable API is industry standard. One minor version deprecation window is aggressive but appropriate for a pre-1.0-adoption tool.

lm-eval `__all__ = ["evaluate", "simple_evaluate", "__version__"]` is the exact precedent.

**Note:** The backward-compatibility.md still lists `run_experiment` and `run_study` as the stable exports, while experiment-study-architecture.md changed this to unified `run()`. These docs are internally inconsistent. Whichever API shape is chosen, the docs must be synchronised.

---

## Over-Engineering Assessment

### What is appropriately complex:
- **Subprocess isolation** -- correctness requirement, not optional complexity
- **Backend config composition** -- genuine multi-backend complexity
- **Dotted sweep notation** -- solves a real problem compactly
- **Exception hierarchy** -- standard library-first pattern
- **Infrastructure auto-capture** -- scientific rigour requirement

### What is more complex than needed:
- **Three-layer config naming** -- two sources (user config + experiment YAML) plus auto-capture. Not three "layers".
- **Field placement three-questions framework** -- the table is the SSOT; the questions add nothing
- **Union return type from `run()`** -- solves a problem that does not exist at the CLI level and creates a problem at the library level

### What is missing:
- **Sync checkpoints between parent and child process** -- optimum-benchmark has this, llem does not. Worth adding for robustness.
- **Device isolation monitoring** -- optimum-benchmark's `device_isolation: true` detects foreign GPU processes. Useful for measurement integrity.
- **Progress callback API for library users** -- the design includes Rich progress for CLI but no callback mechanism for library users who want progress events programmatically.

---

## The Hydra Question: Was Rejecting Hydra Correct?

**Verdict: YES, the rejection was correct.**

optimum-benchmark uses Hydra and pays a significant cost:
- `BenchmarkConfig` fields are typed `Any` (not the specific config types) because Hydra's structured configs use dataclasses, not Pydantic
- Config composition via file-based groups is powerful but adds operational complexity
- The `--multirun` sweep is less expressive than llem's dotted notation for multi-backend sweeps
- Hydra takes over the application entry point (`@hydra.main`) -- not compatible with a library-first design where the CLI is a thin wrapper

**Key evidence:** pytorch/torchtitan opened an issue requesting Hydra/OmegaConf adoption (2025) and the discussion revealed that many teams are moving away from Hydra toward simpler approaches. The Facebook Research stopes project uses Hydra but documents significant configuration complexity.

llem's approach (Pydantic models + YAML + custom sweep resolution) is simpler, more type-safe, and more compatible with library-first architecture. Hydra would force the config system to use dataclasses instead of Pydantic, losing validation, `extra = "forbid"`, and the None-as-sentinel pattern.

**Sources:** [pytorch/torchtitan Hydra issue](https://github.com/pytorch/torchtitan/issues/1415), [facebookresearch/hydra composition bug](https://github.com/facebookresearch/hydra/issues/1592)

---

## Specific Challenges and Recommendations

### Challenge 1: Simplify Config Naming (Decisions C and D)

**Current:** Three-layer model (Layer 1, Layer 2, Layer 3) with a three-questions placement framework.

**Proposed:** Two config sources + auto-capture:
- **User config** (`~/.config/llenergymeasure/config.yaml`) -- machine-local preferences, runner mappings
- **Experiment config** (experiment.yaml / study.yaml) -- what to measure, shareable
- **Environment snapshot** -- auto-captured at runtime, stored with results (not a config layer)

The separation is identical. The naming is simpler and matches what every peer calls these things. "Layer 1/2/3" is project-specific jargon that will confuse new contributors.

### Challenge 2: Resolve the Library API Return Type (Decision H)

**Current:** `run()` returns `ExperimentResult | StudyResult`.

**Options ranked by recommendation:**

1. **Two functions** (`run_experiment()` + `run_study()`): Cleanest types. Original design. Matches lm-eval's `evaluate()` + `simple_evaluate()` pattern (two entry points, no ambiguity).

2. **Always return `StudyResult`**: Consistent with internal architecture. `StudyResult` with `len(experiments) == 1` for single runs. Less ergonomic but type-safe.

3. **Accept the union**: Workable with `@overload` for typed inputs; path-based input returns union. Tax at every call site.

The CLI remains unified `llem run` regardless. This is a library API question only.

### Challenge 3: Add Sync Checkpoints to Subprocess Isolation (Decision I)

**Current:** Parent spawns child, waits on `p.join(timeout=...)`.

**Missing:** No verification that child process initialised successfully before starting the wait. optimum-benchmark uses two sync checkpoints:
1. After child spawns (verify it is alive)
2. After device isolation setup (verify GPU is accessible)

Without sync checkpoints, a child that fails during import/setup will cause the parent to wait for the full timeout before detecting failure. Adding sync checkpoints reduces failure-to-detection time from `timeout` seconds to near-instant.

### Challenge 4: Document the Inconsistency Between API Docs

**backward-compatibility.md** lists `run_experiment` and `run_study` as stable exports.
**experiment-study-architecture.md** specifies unified `run()`.
**designs/library-api.md** still documents `run_experiment()` and `run_study()` with full overload signatures.

These must be synchronised after resolving Challenge 2.

---

## Peer Architecture Comparison Matrix

| Aspect | optimum-benchmark | lm-eval | Zeus | vLLM bench | **llem (planned)** |
|--------|-------------------|---------|------|-----------|-------------------|
| **Entry point** | `Benchmark.launch(config)` | `simple_evaluate(model, tasks)` | `ZeusMonitor()` API | CLI only | `llem.run(config)` |
| **Config system** | Hydra + dataclasses | Flat kwargs + YAML tasks | Constructor args | JSON param files | Pydantic + YAML |
| **Return type** | `BenchmarkReport` | `EvalResults` | `Measurement` | N/A (files) | `ExperimentResult \| StudyResult` |
| **Process isolation** | `multiprocessing.Process` | None (in-process) | None (measurement lib) | `subprocess.Popen` | `multiprocessing.Process` |
| **Sweep support** | Hydra `--multirun` | Task groups | N/A | JSON Cartesian product | Dotted notation sweep |
| **Backend selection** | Config group | `--model` string | N/A | `--serve-cmd` | `backend:` field |
| **User config** | Hydra defaults | None | None | None | `~/.config/.../config.yaml` |
| **Environment capture** | `get_system_info()` auto | Versions dict | GPU indices auto | N/A | Auto-detect (Layer 3) |
| **Error handling** | Exceptions (unstructured) | Plain exceptions | Plain exceptions | Exit codes | `LLEMError` hierarchy |
| **API stability** | No explicit policy | No explicit policy | No explicit policy | No explicit policy | `__init__.py` exports + SemVer |

**Key takeaway:** llem's architecture is more thoroughly designed than any peer. The risk is not under-engineering -- it is over-engineering. The decisions are overwhelmingly sound; the naming and abstraction layers are where complexity creeps in without proportional value.

---

## Sources

- [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark) -- BenchmarkConfig, launcher system, process isolation
- [optimum-benchmark README](https://github.com/huggingface/optimum-benchmark/blob/main/README.md) -- Python API usage examples
- [lm-eval-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness) -- `__init__.py` exports, `simple_evaluate()` signature
- [lm-eval architecture blog](https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html) -- Architecture overview
- [Zeus project](https://ml.energy/zeus/) -- Monitor API, measurement architecture
- [Zeus GitHub](https://github.com/ml-energy/zeus) -- Multi-platform energy measurement
- [vLLM parameter sweeps](https://docs.vllm.ai/en/latest/benchmarking/sweeps/) -- Sweep config format
- [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) -- Benchmark architecture
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) -- Run/experiment hierarchy
- [PyTorch multiprocessing docs](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) -- Spawn method for CUDA
- [python/mypy#1693](https://github.com/python/mypy/issues/1693) -- Union return types as anti-pattern
- [python/typing overload spec](https://typing.python.org/en/latest/spec/overload.html) -- @overload limitations
- [pytorch/torchtitan Hydra issue](https://github.com/pytorch/torchtitan/issues/1415) -- Hydra adoption challenges
- `.product/research/13-execution-isolation-patterns.md` -- Existing peer isolation research (HIGH confidence)
- `.product/research/15-config-architecture-patterns.md` -- Existing peer config research
