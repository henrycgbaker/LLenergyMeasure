# Study YAML Design

> **Naming note (session 4, updated 2026-02-25)**: The CLI command is unified `llem run`
> (YAML determines scope). The file is `study.yaml`. Previously called `campaign`. All
> examples use `study.yaml` below. The library function is `run_study()`.
> Config architecture DECIDED — see [../decisions/config-architecture.md](../decisions/config-architecture.md).

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/installation.md](../decisions/installation.md),
                     [../decisions/cli-ux.md](../decisions/cli-ux.md),
                     [../decisions/config-architecture.md](../decisions/config-architecture.md),
                     [../decisions/study-execution-model.md](../decisions/study-execution-model.md),
                     [../decisions/experiment-study-architecture.md](../decisions/experiment-study-architecture.md)
**Status**: Confirmed (session 3+4+6+7). CLI unified to `llem run` 2026-02-25.

## Confirmed: Sweep Grammar

Two modes, combinable in one file:

### Mode A — Grid Sweep (Cartesian product, with optional backend-scoped dimensions)

**Universal params only (single-backend study):**
```yaml
# study.yaml — single-backend grid sweep
model: meta-llama/Llama-3.1-8B
backend: pytorch
dataset: alpaca
n: 100

sweep:
  batch_size: [1, 8, 16, 32]       # 4 values  ← shared param for pytorch
  precision: [fp16, bf16, int8]     # 3 values
# → 12 experiments: every batch_size × precision combination
```

**Universal + backend-scoped params (multi-backend study — dotted notation):**
```yaml
# study.yaml — multi-backend sweep with backend-scoped dimensions
model: meta-llama/Llama-3.1-8B
backend: [pytorch, vllm]
dataset: aienergyscore
n: 100

sweep:
  precision: [bf16, fp16]              # universal — applies to both backends
  pytorch.batch_size: [1, 8, 32]       # scoped — pytorch's grid only (+1 dimension)
  vllm.max_num_seqs: [64, 256]         # scoped — vllm's grid only (+1 dimension)

# → pytorch grid: precision × batch_size  = 2 × 3 = 6 experiments
# → vllm grid:    precision × max_num_seqs = 2 × 2 = 4 experiments
# → total: 10 experiments (independent grids per backend)
```

**Rule:** Each backend gets the Cartesian product of `(universal keys) × (its own scoped keys)`.
Backends have **independent grids** — a scoped key for backend A adds nothing to backend B's grid.

**Dotted notation syntax:** `<backend>.<param>: [values]`. Split on first dot only, so
`pytorch.attn.implementation` → backend=`pytorch`, param=`attn.implementation`.

**Validation:**
- Scoped key references a backend not in `backend:` list → `ValidationError` at parse time
- No `backend:` field with scoped keys → backend inferred from the scoped keys (error if ambiguous)

All parameters not in `sweep:` are fixed across all experiments.

**Peer reference:** W&B Sweeps `method: grid` with `parameters:` block;
Hydra `--multirun batch_size=1,8,16,32 precision=fp16,bf16,int8`.

See [../decisions/config-architecture.md — Sweep Grammar Decision](../decisions/config-architecture.md) for the full resolved semantics, grid algorithm, and edge case table.

### Mode B — Explicit Experiment List

```yaml
# study.yaml — explicit list example
experiments:
  - model: meta-llama/Llama-3.1-8B
    backend: pytorch
    batch_size: 32
    precision: fp16
    dataset: alpaca
    n: 100
  - model: meta-llama/Llama-3.1-8B
    backend: vllm               # different backend → triggers Docker
    batch_size: 32
    precision: fp16
    dataset: alpaca
    n: 100
```

Use when: specific pairs required (not all combinations), or when mixing backends.

**Peer reference:** MLflow experiment tracking (explicit run configuration);
Nextflow process inputs (specific parameter sets per process).

### Mode C — Combined (Grid + Additional Explicit)

```yaml
# Base config for sweep
model: meta-llama/Llama-3.1-8B
backend: pytorch
dataset: alpaca
n: 100

# Grid sweep
sweep:
  batch_size: [1, 8, 16, 32]
  precision: [fp16, bf16]
# → 8 experiments from grid

# Explicit additions (supplementary to sweep)
experiments:
  - backend: vllm
    batch_size: 32
    precision: fp16
# → 1 additional experiment (vllm, outside pytorch sweep)
# Total: 9 experiments
```

## Runner Field

> **Superseded 2026-02-19:** The `runner:` field was removed from `study.yaml` entirely.
> `study.yaml` is silent on execution environment. Runner selection is a user config / env var / CLI
> flag concern — study files must be portable (same file runs on laptop, HPC, CI).
> See [../decisions/architecture.md — Runner Resolution](../decisions/architecture.md) and
> [../designs/user-config.md — Runner Selection Logic](user-config.md).
>
> **Runner resolution precedence (2026-02-19):**
> 1. `LLEM_RUNNER_<BACKEND>` env var — highest
> 2. `~/.config/llenergymeasure/config.yaml` `runners.<backend>`
> 3. CLI flag: `llem run --runner docker` — overrides all backends
> 4. Default: `local`

**Rejected design (preserved for reference):**

```yaml
# REJECTED — runner: field no longer valid in study.yaml

# Auto-detect (default — omit runner field)
# Tool inspects backends used → local if single-backend, docker if multi-backend

# Explicit override
runner: local     # force local — ONLY valid for single-backend studies
runner: docker    # force Docker (e.g. formal/reproducible mode with single backend)
runner: auto      # explicit auto-detect (same as omitting runner)
```

**Why rejected:** No peer tool (Nextflow, Snakemake, DVC, lm-eval, Hydra) embeds execution
environment selection in the portable pipeline/study file. The study defines *what* to measure;
*where* to run it is a runtime/machine concern. Runner profiles in study.yaml also pointed at
a user config section (`runner_profiles:`) that was later removed — the use case is covered
by having different `~/.config/llenergymeasure/config.yaml` files per machine.

## Confirmed: Base Config Reference

`base:` is **optional** in study.yaml. References an existing `experiment.yaml` as the
starting point. All fields in `sweep:` / `experiments:` override it. Purely a DRY
convenience — no requirement to use it.

**Resolution order (later overrides earlier):**
```
1. Tool built-in defaults   (n=100, batch_size=1, precision=bf16, dataset=alpaca)
2. base: experiment.yaml    (optional — user-maintained starting point)
3. Inline study fields      (override base)
4. sweep: block             (turns specific fields into Cartesian product)
5. experiments: list        (explicit additions, additive to sweep)
```

**With `base:` (DRY, reuse validated experiment config):**
```yaml
# study.yaml
base: experiment.yaml       # inherit model, backend, dataset, n from here

sweep:
  batch_size: [1, 8, 16, 32]
  precision: [fp16, bf16]
# → 8 experiments. model/dataset/n/backend come from experiment.yaml
```

**Without `base:` (self-contained — simple, portable):**
```yaml
# study.yaml
model: meta-llama/Llama-3.1-8B
backend: pytorch
dataset: alpaca
n: 100

sweep:
  batch_size: [1, 8, 16, 32]
  precision: [fp16, bf16]
# → 8 experiments. identical result.
```

`sweep:` always lives in study.yaml — never in experiment.yaml. experiment.yaml
defines a single experiment; only studies have sweeps.

**Peer reference:** Hydra multi-file config composition; W&B Sweeps is always
self-contained. We support both patterns — simple studies use self-contained,
complex multi-study workflows use `base:`.

## Confirmed: Invalid Combination Handling

Grid sweeps over LLM parameters produce many invalid combinations (e.g.,
`tensorrt + fp32`, `awq + pytorch`, `speculative_decoding + tensorrt`). These are
**knowable in advance** from backend documentation — not runtime surprises.

**Peer research finding:** No major sweep framework (W&B, Hydra, Ray Tune, Optuna)
provides native categorical constraint validation. The universal industry pattern is a
hand-rolled `validate_experiment_config()` function that runs before GPU allocation.
ConfigSpace has declarative constraint syntax but is niche academic tooling not worth
the dependency.

### Multi-level validation (confirmed)

**Industry pattern (from peer research):** No production ML framework uses a separate
constraint registry file. vLLM, TRT-LLM, lm-eval, and PyTorch Lightning all embed
constraints inside Pydantic `@model_validator` methods. The SSOT is small module-level
dicts that feed INTO the Pydantic model — not a parallel validation system.

#### Level 1 — Pydantic model validation (always runs, zero cost)

**SSOT: single positive dict per field per backend.** Enumerate what IS valid. Anything
absent is implicitly invalid. No separate "forbidden" list — that would encode the same
truth twice. See [experiment-config.md](experiment-config.md) for the full ExperimentConfig
schema including SSOT dicts and validators.

**Study loop — three semantically distinct outcomes:**

```python
skipped, failed, ran = [], [], []

for raw_config in generated_experiment_configs:
    # Pydantic parse — ValidationError = "skipped" (never touches GPU)
    try:
        config = ExperimentConfig(**raw_config)
    except ValidationError as e:
        skipped.append({"config": raw_config, "reason": e.errors()})
        continue

    # Execution — any exception here = "failed"
    try:
        result = run_experiment(config)
        ran.append(result)
    except Exception as e:
        failed.append({"config": config.model_dump(), "error": str(e)})
        continue  # log and continue
```

Pre-flight output (all invalid configs shown upfront before any experiment starts):
```
Study: 15 configs generated from grid
  Skipping 3 invalid combinations (shown before any experiments run):
    ✗ tensorrt + precision=fp32   tensorrt does not support precision='fp32'. Valid: ['fp16', 'int8', 'fp8', 'int4']
    ✗ tensorrt + precision=bf16   tensorrt does not support precision='bf16'
    ✗ pytorch  + quantization=awq pytorch does not support quantization. Valid: []
  Running 12 valid experiments.
```

**Execution always continues** — log failure, run next experiment. No stop-on-failure flag.

**Keeping the SSOT current:**
No automated doc-scraping pipeline — too brittle. No automated constraint discovery from
runtime errors — no framework does this. The feedback loop is human: review structured
failure logs, identify patterns, update the SSOT dicts manually. Runtime failure logs contain
the full config and exception for exactly this purpose.

#### Level 2 — VRAM estimation (opt-in via `llem run --dry-run`)

Estimates VRAM per experiment before any GPU time spent. Uses `AutoConfig.from_pretrained()`
— reads from local HF cache if model already downloaded (no network); gracefully skips
estimate and warns if model not yet cached.

```
weight_gb   = (num_params × dtype_bytes) / 1e9
kv_cache_gb = (2 × num_layers × hidden_size × seq_len × batch_size × dtype_bytes) / 1e9
overhead_gb = weight_gb × 0.15
total_gb    = weight_gb + kv_cache_gb + overhead_gb
```

`dtype_bytes`: fp32=4, fp16=2, bf16=2, int8=1, int4=0.5

Dry-run output:
```
llem run sweep.yaml --dry-run

  12 experiments to run (3 skipped — invalid config)

  VRAM estimates (A100 80GB available, 72GB effective limit):
  ✓ llama-3.1-8b  pytorch  bf16  batch=1   →  ~17 GB
  ✓ llama-3.1-8b  pytorch  bf16  batch=32  →  ~31 GB
  ✗ llama-3.1-8b  pytorch  fp32  batch=32  →  ~72 GB   LIKELY OOM

  11 will run, 1 likely OOM.
```

#### Level 3 — Graceful runtime failure (automatic, always on)

Per-experiment OOM or unexpected backend error → catch, log full config + exception +
traceback, continue study.

Runtime failure logs contain enough to diagnose patterns and update SSOT dicts manually:
```json
{
  "status": "failed",
  "config": {"backend": "tensorrt", "precision": "fp16", "quantization": "awq"},
  "exception_type": "RuntimeError",
  "error_message": "TensorRT-LLM: AWQ quantization requires int4 precision"
}
```

Study summary always distinguishes all three outcomes:
```
Study complete: 11/12 ran, 1 failed (RuntimeError: CUDA out of memory)
  Skipped: 3 (invalid config — see study_summary.json)
Results: results/batch-size-effects_2026-02-18T14-30/
```

## Full Schema (Annotated)

> **Updated (2026-02-25):** `runner:` field removed from study.yaml. Runner selection is a
> user config / env var / CLI flag concern. See Runner Field section above.

```yaml
# study.yaml — full schema with all optional fields
# Command: llem run study.yaml
# NO runner: field — runner selection is user config / env var / CLI only

# --- Base configuration (applies to all experiments unless overridden in sweep/experiments) ---
model: meta-llama/Llama-3.1-8B   # HuggingFace model ID (required)
backend: pytorch                    # default backend if not specified per-experiment
dataset: alpaca                     # default dataset
n: 100                              # default sample count

# --- Grid sweep (optional) ---
sweep:
  batch_size: [1, 8, 16, 32]       # Cartesian product with precision
  precision: [fp16, bf16, int8]
  # Any experiment config field can appear here

# --- Explicit experiments (optional, additive to sweep) ---
experiments:
  - backend: vllm
    batch_size: 32
    precision: fp16
  - backend: tensorrt
    batch_size: 16
    precision: int8
```

## Peer Format Comparison

| Feature | LLenergyMeasure | W&B Sweeps | Hydra multirun | Optuna |
|---------|-----------------|------------|----------------|--------|
| Grid sweep | `sweep:` block | `method: grid` + `parameters:` | `--multirun x=1,2,3` | `suggest_*` API |
| Explicit list | `experiments:` | N/A | Override files | Trial definition |
| Base config | `base:` (optional, confirmed) | Script defaults | Config composition | Study defaults |
| Constraint validation | Multi-level (confirmed) | None | None | Pruning API |
| Runner selection | User config / CLI (not in YAML) | Not applicable | Launcher config | Not applicable |

W&B Sweeps reference: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
Hydra multirun reference: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

---

## Confirmed: Study Identity

```yaml
name: batch-size-sweep-2026-02   # optional — used in result directory name + study_summary.json
```

If omitted, auto-generated from timestamp: `study_2026-02-18T14-30`.

**Peer reference:** W&B Sweeps `name:` field; Ray Tune `RunConfig(name=...)`.

---

## Confirmed: Execution Block (nested — session 4 reversion)

All execution/orchestration fields live under a nested `execution:` block. This separates
**WHAT to measure** (top-level: model, dataset, sweep, experiments) from **HOW to run the
collection** (nested: n_cycles, cycle_order, thermal gaps).

```yaml
# HOW to run the collection of experiments
execution:
  n_cycles: 3                  # repeat full experiment set N times (Pydantic default: 1; CLI default: 3)
  cycle_order: interleaved     # sequential | interleaved | shuffled (Pydantic default: sequential)
  config_gap_seconds: 60       # optional: override user config machine default (thermal gap between experiments)
  cycle_gap_seconds: 300       # optional: override user config machine default (thermal reset between cycles)
```

**This block is excluded from `study_design_hash`** and stored as `measurement_protocol` in
`StudyResult`. See [../decisions/study-execution-model.md — Decision B](../decisions/study-execution-model.md).
Rationale: the hash answers *"did you test the same parameter space?"*, not *"how rigorously?"*.
Same sweep + more cycles = same hash → enables "topping up" a study without a new identity.

**Design rationale (session 4):** These are "how to orchestrate the collection", not "what to
measure". Lifting them to top level (as in session 3) cluttered the top-level namespace and
mixed two different semantic layers. The `execution:` block is analogous to Nextflow's
`process` resource directives (separated from computation definition) and Docker Compose's
`deploy:` block. Nesting preserves the top-level for experiment identity fields.

**`n_cycles`:** Runs the complete experiment set N times. Essential for statistical robustness
in energy measurement — GPU power draw varies between runs. `n_cycles=3` with 4 sweep
experiments → 12 total experiment runs; statistics computed across the 3 repetitions per
config.

**`cycle_order`:**
- `interleaved`: A→B→C, A→B→C, A→B→C — **default**. Best for thermal fairness.
- `shuffled`: A→C→B, B→A→C, C→B→A — randomises order within each cycle.
- `grouped`: A,A,A → B,B,B → C,C,C — fastest, worst for thermal fairness.

**`config_gap_seconds`:** Pause between experiments for GPU thermal recovery. Default 60s.

**`cycle_gap_seconds`:** Pause between full cycles for complete thermal reset to idle.
Default 300s (5 min).

**`profile:`** — **Removed (2026-02-20).** The `profile:` key is no longer valid in the
`execution:` block. Named execution profiles in user config were rejected — they conflated
machine-local thermal settings with study portability. Use direct fields instead.
`--profile quick` and `--profile publication` are CLI flag shorthands, not profile references.
See [../decisions/cli-ux.md — Execution Profiles](../decisions/cli-ux.md) and
[../decisions/study-execution-model.md — Decision A](../decisions/study-execution-model.md).

**No peer equivalent:** These fields have no parallel in W&B, Ray Tune, Optuna, or Hydra
because none of those tools measure GPU energy. Thermal management is unique to our domain
and scientifically justified.

**`warmup:` placement:** `warmup:` lives in `ExperimentConfig` (single-experiment level), NOT
in `execution:`. Warmup is "how to measure THIS experiment" (how many prompts to discard before
measurement). Execution is "how to run THE COLLECTION" (cycles, ordering, thermal gaps). These
are different levels of abstraction and correctly live in different config objects.

**Rescued from existing codebase:** `cycles`, `structure`, `config_gap_seconds`,
`cycle_gap_seconds` existed in `CampaignExecutionConfig`. Renamed and moved to `execution:`
block; `warmup_prompts`/`warmup_timeout_seconds` from `CampaignExecutionConfig` correctly
live in `ExperimentConfig.warmup`.

## Confirmed: Cold Start (re-added session 4)

```yaml
cold_start: false   # unload model between experiments to measure model-loading energy (default: false)
```

Cold start measurement (model loading energy + TTFT) is scientifically valid for a measurement
tool — PyTorch, vLLM, and TRT all have different loading costs that are implementation choices.

`cold_start: true` means: unload model from GPU memory between experiments, reload from disk,
measure the full loading + first-inference cycle. This is **process-level unload/reload** within
a running container (not container restart).

**Interaction with Docker**: Cold start semantics differ between local (bare metal) and Docker:
- **Local**: process-level model unload (`del model; torch.cuda.empty_cache()`)
- **Docker**: unload within running container, OR container restart for clean state
- Full Docker cold start semantics are TBD — see [../decisions/docker-execution.md](../decisions/docker-execution.md)

**Not re-added**: `restart_container: bool` (was `CampaignColdStartConfig.restart_container`).
That is a Docker-internal concern (v2.0 Docker milestone), not user-facing config.

---

## Peer Features Explicitly Rejected

| Peer Feature | Source | Reason for Rejection |
|---|---|---|
| `metric:` + `direction: min/max` | W&B, Ray Tune, Optuna | Optimisation tools need this; we measure everything and optimise nothing |
| `timeout_seconds:` / `time_budget_s:` | Ray Tune | Silently truncates measurements — wrong for scientific energy studies |
| `max_experiments:` / `run_cap:` | W&B | Undefined semantics for Cartesian grids; which N of 100 experiments run? |
| `max_concurrent_trials:` / `n_jobs:` | Ray Tune, Optuna | GPU ownership prevents within-machine parallelism; moot |
| `early_terminate:` / pruning | W&B, Optuna | Optimisation framework concept; orthogonal to measurement |
| `callbacks:` / hook system | Ray Tune, Optuna | Not a framework; adds API surface without clear user value |
| Nested parameter hierarchies | W&B only | Flat `sweep:` dict covers all our use cases; niche and complex |
| `health_check:` | Existing codebase | Wrong abstraction level; Docker health is v2.0 Docker internal; OOM → log + continue is sufficient |
| `restart_container:` | Existing codebase | v2.0 Docker-internal concern, not user-facing config (note: `cold_start: bool` RE-ADDED as a simple top-level field — see Confirmed: Cold Start above) |
| `daemon:` / `schedule:` | Existing codebase | Scheduling is OS/cron concern; no peer tool has this in study YAML |

---

## Updated Full Schema (updated 2026-02-25)

```yaml
# study.yaml — complete schema (2026-02-25 — WHAT / HOW separation)
# Command: llem run study.yaml
# NO runner: field — runner selection is user config / env var / CLI only (removed 2026-02-19)

# ─── WHAT: Study identity ───────────────────────────────────────────────────
name: batch-size-sweep-2026-02           # optional; auto-generated from timestamp if omitted

# ─── WHAT: Experiment defaults ──────────────────────────────────────────────
model: meta-llama/Llama-3.1-8B          # HuggingFace model ID (required if not in base:)
backend: pytorch                         # default backend; can be overridden per-experiment
dataset: aienergyscore                   # default dataset
n: 100                                   # default sample count

# ─── WHAT: Optional base config (DRY inheritance) ───────────────────────────
base: experiment.yaml                    # model/dataset/n/backend from here; sweep/experiments override

# ─── WHAT: Cold start measurement (default: false) ──────────────────────────
cold_start: false                        # unload model between experiments to measure loading energy

# ─── WHAT: Grid sweep (universal + optional backend-scoped dims) ─────────────
sweep:
  precision: [fp16, bf16]               # universal — all backends
  pytorch.batch_size: [1, 8, 32]        # scoped — adds dimension to pytorch's grid only
  vllm.max_num_seqs: [64, 256]          # scoped — adds dimension to vllm's grid only
  # → independent Cartesian products per backend (decided 2026-02-19)

# ─── WHAT: Explicit experiments (additive to sweep) ─────────────────────────
experiments:
  - backend: vllm
    vllm:
      max_num_seqs: 256                  # vLLM-specific — not comparable to pytorch batch_size
  - backend: tensorrt
    tensorrt:
      max_batch_size: 16
      quantization: int8_sq

# ─── HOW: Execution orchestration (nested to separate from WHAT) ─────────────
# This entire block is EXCLUDED from study_design_hash — stored as measurement_protocol in results
# NO profile: key — use direct fields. --profile quick|publication are CLI flag shorthands only.
execution:
  n_cycles: 3                            # repeat full experiment set N times (Pydantic default: 1)
  cycle_order: interleaved               # sequential | interleaved | shuffled (Pydantic default: sequential)
  config_gap_seconds: 60                 # optional: override user config machine default
  cycle_gap_seconds: 300                 # optional: override user config machine default
  # total runs: (sweep experiments + explicit experiments) × n_cycles
```
