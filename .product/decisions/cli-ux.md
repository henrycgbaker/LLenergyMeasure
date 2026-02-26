# CLI & UX Decisions

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-26
**Research:** N/A (peer review of 9 CLIs inline)

## Decision

**v2.0 CLI surface — 2 commands + 1 flag:**
```
llem run    [config.yaml] [--model X] [--backend X]  # Run experiment(s) — YAML determines scope
llem config [--verbose]                               # Show environment + config guidance
llem --version                                        # Version flag (not subcommand)
```

| Sub-decision | Resolution |
|---|---|
| **A. Command set** | Unified `llem run` — YAML file carries complexity signal. No separate `llem study`. |
| **B. CLI/Library names** | Intentional divergence: `llem run` ↔ `run_experiment()` / `run_study()`. Unambiguous return types. |
| **C. Zero-config** | `llem run --model X --param_of_interest_a --param_of_interest_b` works with sensible defaults, allowing user to quickly run a check with their param(s) of interest. Pre-flight validates before every run. |
| **D. Execution profiles** | ~~`--profile quick\|publication`~~ **Superseded (2026-02-25):** `--profile` flag dropped entirely. 0/5 peers use named rigour profiles. Use `--cycles`, `--order`, `--no-gaps` flags directly; study YAML `execution:` block for portable settings. |
| **E. --dry-run + sweep validation** | Three-layer validation: `validate_experiment_config()` (always), `estimate_vram()` (--dry-run + optional pre-flight), runtime failure handling (catch + continue). Grid preview for studies. |

---

## Context

v1.x CLI used `llem experiment` with various subcommands. The v2.0 redesign needed to rationalise the command surface to match researcher workflows and ML tool conventions. Research into 9 peer CLIs (lm-eval, mlflow, dbt, cargo, poetry, wandb, optimum-benchmark, AIEnergyScore, docker) shaped the decision space.

Forces:
1. CLI must serve quick one-off measurement and structured multi-experiment studies
2. All ML ecosystem peer tools use `run` for execution — diverging creates unfamiliar UX
3. Environment inspection and setup guidance belong in one re-runnable command, not two separate single-lifecycle commands
4. Execution profile presets must not conflate machine thermal settings (machine-local) with study rigour (portable)
5. When the internal architecture unifies experiments and studies (Option C in [experiment-study-architecture.md](experiment-study-architecture.md)), a second command name is redundant — the YAML file carries the complexity signal

---

## Sub-decision A — Command Set

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **2 commands: `run`, `config` + unified internal architecture (chosen 2026-02-25)** | Minimal surface; YAML file carries the complexity signal; matches pytest, lm-eval, mlflow; one runner code path internally | Users lose the explicit "I'm in study mode" CLI signal — mitigated by YAML file being that signal |
| 3 commands: `run`, `study`, `config` | Two commands communicate scope clearly at the CLI level | Redundant when internal architecture is unified (Option C); second command name carries no information the YAML file doesn't already carry |
| `llem experiment` (keep old name) | No rename needed | Zero ML ecosystem recognition for `experiment` as a CLI verb; `run` is dominant across 9 peers |
| `llem sweep` instead of `llem study` | Familiar in ML hyperparameter tuning context | W&B `sweep` carries strong hyperparameter optimization connotation; llem does measurement not optimization |
| `llem status` + `llem init` as separate commands | Explicit separation of display and setup | `status` and `init` serve the same information at different lifecycle moments; `config` is re-runnable and serves both |
| `llem check` (active validation) | Explicit pre-flight invocation | `--dry-run` flag on run covers this; adding a 3rd command has no peer precedent |
| `llem results list` / `show` | Explicit result browsing | Human-readable filenames + `ls results/` is sufficient; no peer tool found this necessary |

**All cut commands and why:**

| Command | Why Cut |
|---------|---------|
| `llem experiment` | Renamed to `llem run` — dominant CLI idiom (docker run, cargo run, kubectl run, dbt run, mlflow run). |
| `llem study` (separate command) | Folded into `llem run` (2026-02-25). With Option C architecture (see [experiment-study-architecture.md](experiment-study-architecture.md)), the internal runner is unified — a single run is a degenerate study. The YAML file carries the complexity signal, not the command name. Matches pytest, lm-eval, mlflow run. |
| `llem status` / `llem info` | Folded into `llem config`. Same information, better name that implies it shows and manages user configuration. |
| `llem init` (separate command) | Folded into `llem config`. Standalone init implies one-time-only; `llem config` is re-runnable. `--init` flag planned for post-v2.0 when interactive wizard is designed. |
| `llem check` (active validation) | `--dry-run` flag on `llem run` covers active config validation without adding a command. Pre-flight in `run` is the enforcement point. |
| `llem sweep` | Rejected — W&B association with hyperparameter optimization is too strong in ML community. llem does measurement not optimization. |
| `llem results list` / `show` | Human-readable filenames + `ls results/` is sufficient. |
| `llem config validate` | Subsumed by implicit pre-flight in run. |
| `llem campaign` | Renamed to `llem study` (session 4), then folded into `llem run` (2026-02-25). Campaign has zero ML ecosystem recognition. |
| `llem batch` | Subsumed by `llem run` (studies). |
| `llem schedule` | No industry precedent. Use cron. |
| `llem aggregate` | Automatic post-experiment, not a user command. |
| `llem datasets` / `llem presets` / `llem gpus` | Covered by `llem config` + docs. |
| `llem version` | Replaced by `--version` flag. |

**Rejected: 3 commands (`run` + `study` + `config`) (2026-02-25)**

> **Superseded (2026-02-25):** The original decision (2026-02-19) used 3 commands: `llem run` for single experiments and `llem study` for multi-experiment studies. This was revised to a unified `llem run` after accepting Option C in [experiment-study-architecture.md](experiment-study-architecture.md), which unifies the internal architecture. A second command name is redundant when the internal runner treats a single experiment as a degenerate study. The YAML file carries the complexity signal — matches pytest, lm-eval, mlflow run.

**Rejected: Separate `experiment` / `study` concepts in CLI (2026-02-25)**

> **Superseded (2026-02-25):** Evaluated and initially rejected unification (2026-02-20) on the grounds that `StudyConfig` has genuine extra complexity. Option C resolves this — sweep grammar is resolved at YAML parse time, `StudyConfig` is a thin resolved container, and a single internal runner handles both. The domain terminology (experiment = atomic measurement, study = structured investigation) is preserved in the *type* names (`ExperimentConfig`, `StudyConfig`) but not in the *CLI* — users interact with `llem run` only.

### Decision

We will use 2 commands: `llem run`, `llem config` + `--version` flag.

- `run`: dominant ML ecosystem CLI idiom for execution (lm-eval, mlflow, dbt, cargo, poetry). Handles both single experiments and multi-experiment studies — the YAML file (or lack thereof) determines scope.
- `config`: passive display + setup guidance, re-runnable; absorbs `status` and `init`

The internal domain terminology is preserved: `ExperimentConfig` (one measurement point) and `StudyConfig` (resolved container of experiments). But the CLI has a single entry point. See [experiment-study-architecture.md](experiment-study-architecture.md) for the full architecture.

### v2.0 Command Set

```
llem run    [config.yaml] [--model X] [--backend X]  # Run experiment(s) — YAML determines scope
llem config [--verbose]                               # Show environment + config guidance
llem --version                                        # Version flag (not subcommand)
```

2 commands + 1 flag. Complete v2.0 CLI surface.

**How `llem run` determines scope:**
```bash
llem run --model meta-llama/Llama-3.1-8B                  # quick: CLI flags, defaults
llem run --model X --backend pytorch --precision fp16      # more specific, still flags
llem run experiment.yaml                                   # full YAML, single experiment
llem run study.yaml                                        # YAML with sweep/experiments block
```

The YAML file is the complexity signal. No second command needed.

**Future additions:**
- `llem run --resume study-dir/` — resume interrupted study (post-v2.0)
- `llem config --init` — interactive wizard that writes `~/.config/llenergymeasure/config.yaml` (post-v2.0)
- `llem results push <file>` — upload to central DB (post-v2.0, blocked on trust model)

Full command signatures and options: [designs/cli-commands.md](../designs/cli-commands.md)
Output examples (llem config, pre-flight): [designs/observability.md](../designs/observability.md)

### Consequences

Positive: Minimal surface; matches peer ML tool conventions; `llem run --model X` is the zero-config onboarding path; `llem run my-study.yaml` is the research power user path. One command to learn.

Negative / Trade-offs: Users lose the explicit "I'm in study mode" CLI signal — mitigated by the YAML file being that signal. Output format (flat JSON vs study subdirectory) is determined by `len(experiments)`, not the command name.

Neutral: `--init` flag deferred to post-v2.0.

---

## Sub-decision B — CLI/Library Name Divergence

> **Updated 2026-02-25:** The library API uses intentional divergence: `llem run` CLI maps to
> two library functions `run_experiment()` / `run_study()` with unambiguous return types.
> This follows the lm-eval pattern (`lm-eval run` CLI → `simple_evaluate()` library function).

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Intentional divergence: `llem run` ↔ `run_experiment()` / `run_study()` (chosen 2026-02-25)** | Library names are self-documenting; unambiguous return types; industry-standard pattern (lm-eval: `lm-eval run` → `simple_evaluate()`); 0/10 peer tools use union return | Two library functions for one CLI command — mitigated by the CLI being a thin wrapper that dispatches based on YAML content |
| Unified `llem run` ↔ `llem.run()` with union return | No divergence; clean; one function | Union return type (`ExperimentResult \| StudyResult`) contradicts all peer practice (0/10 tools); requires isinstance() checks; `@overload` cannot resolve for `str \| Path` input |
| Same name everywhere: `llem run` ↔ `run()` | No divergence | Too generic for a library function; doesn't convey what it runs |

> **Superseded (2026-02-25):** The brief period (2026-02-25 session) where a unified
> `llem.run()` with union return type was considered has been reversed. Research
> (UNIFIED-API-RESEARCH.md) found 0/10 peer tools use union return types, and the
> `@overload` escape hatch cannot resolve for `str | Path` input. The split API was
> the original and correct design. See [experiment-study-architecture.md](experiment-study-architecture.md).

### Decision

CLI and library names are intentionally divergent: `llem run` dispatches to either
`run_experiment()` or `run_study()` based on YAML content. This follows the standard
pattern where CLI wrappers translate user intent into specific library calls.

```python
import llenergymeasure as llem

# Single experiment — returns ExperimentResult (always)
result = llem.run_experiment(model="meta-llama/Llama-3.1-8B", backend="pytorch")
result = llem.run_experiment(ExperimentConfig(model="X", backend="pytorch"))
result = llem.run_experiment("experiment.yaml")

# Multi-experiment study — returns StudyResult (always)
result = llem.run_study("study.yaml")
result = llem.run_study(StudyConfig(experiments=[...]))
```

Internally, both paths construct a `StudyConfig` and call `_run(StudyConfig) -> StudyResult`.
`run_experiment()` unwraps the single-element study result to return `ExperimentResult`.
The Option C architecture is unchanged — only the public API surface differs.

Reference: lm-eval uses `lm-eval run` CLI → `simple_evaluate()` library. pytest uses
`pytest` CLI → `pytest.main()` library. Neither mirrors its CLI name in its library API.

### Consequences

Positive: Unambiguous return types; no isinstance() checks needed; type checkers can infer
return types statically; matches lm-eval and pytest patterns.

Negative / Trade-offs: Two functions to learn instead of one — mitigated by clear naming
and the CLI remaining a single `llem run` command.

Neutral: N/A

---

## Sub-decision C — Zero-Config Defaults and Fail-Fast Pre-Flight

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Zero-config defaults + fail-fast pre-flight with exact fix commands (chosen)** | Onboarding requires no config file; errors are actionable | Pre-flight adds latency before GPU work |
| Require config file before first run | Explicit; no magic defaults | Barrier to entry; contradicts progressive disclosure pattern |
| Silent fallback on missing config (no fail-fast) | No friction | Silently wrong configurations; hard to debug; violates scientific claim integrity |

### Decision

`llem run --model X` runs with sensible defaults: AIEnergyScore dataset, 100 prompts,
batch_size=1, bf16. Interactive prompt selects backend if multiple are installed (matches
`gh pr create` pattern).

Pre-flight validation runs automatically before every experiment or study. Every missing
item gets the exact command to resolve it (e.g. `pip install llenergymeasure[pytorch]`,
`export HF_TOKEN=...`). No `--skip-preflight` escape. Implicit inside `llem run`.

### Consequences

Positive: Onboarding is zero-config; errors are immediately actionable; no silent wrong
configurations.

Negative / Trade-offs: Pre-flight latency before every run; no escape hatch (deliberate —
the escape would enable silently wrong experiments).

Neutral: No separate `llem check` command — `--dry-run` covers interactive validation.

---

## Sub-decision D — Execution Profiles

> **Updated 2026-02-20:** The named `execution_profiles:` design was rejected.
> See [study-execution-model.md — Decision A](study-execution-model.md) for full rationale.

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`--profile` as CLI flag shorthand only (chosen)** | No config layer; study files remain portable; clear separation of concerns | Only two built-in shorthands; less flexible for custom rigour |
| Named `execution_profiles:` in user config | Flexible named presets per machine | Conflates machine thermal (gap seconds — machine-local) with study rigour (n_cycles — portable); study files depend on profile names that may not exist on a colleague's machine |
| `profile:` key in study.yaml `execution:` block | Self-contained per study | Ties the study file to machine-specific settings; makes files non-portable |

**Rejected: Named `execution_profiles:` in user config (2026-02-20)**

**Rejected (2026-02-20):** The following design was active as of 2026-02-19 and explicitly
rejected. See [../research/16-execution-profiles-patterns.md](../research/16-execution-profiles-patterns.md)
for the peer tool survey underpinning this rejection.

**Why rejected:**
1. Conflates two orthogonal concerns: `config_gap_seconds`/`cycle_gap_seconds` (machine thermal
   characteristics — correct in user config) with `n_cycles`/`cycle_order` (study rigour —
   correct in study.yaml). A named preset bundles both together, making the study file depend
   on a profile name that may not exist on a colleague's machine.
2. No peer tool uses named profiles for statistical rigour presets. Nextflow/Snakemake profile
   systems are for *execution environment switching* (local vs SLURM), not rigour levels.
3. `LLEM_PROFILE` env var was confusing — it selects a named preset, but named presets mix
   machine and study concerns. Eliminated together with the profile system.

```yaml
# REJECTED DESIGN — preserved for reference

# study file — reference a named profile
execution:
  profile: publication   # ← removed: profile: key no longer valid in execution block
  n_cycles: 3            # inline fields override profile

# ~/.config/llenergymeasure/config.yaml
execution_profiles:      # ← removed: execution_profiles: block no longer in user config
  quick:
    n_cycles: 1
    config_gap_seconds: 0
    cycle_gap_seconds: 0
  standard:
    n_cycles: 3
    cycle_order: interleaved
    config_gap_seconds: 60
    cycle_gap_seconds: 300
  publication:
    n_cycles: 5
    cycle_order: shuffled
    config_gap_seconds: 120
    cycle_gap_seconds: 600
```

**Previous precedence chain (no longer valid):**
```
1. built-in Pydantic defaults (n_cycles=1, ...)
2. standard profile  ← CLI applied this as baseline
3. LLEM_PROFILE env var + execution_profiles.<name> from user config
4. study file execution: block (inline fields override profile)
5. CLI flag  ← highest priority
```

### Decision

> **Superseded (2026-02-25):** The `--profile quick|publication` CLI flag shorthand has been
> dropped entirely. Research confirmed 0/5 peer tools use named rigour profiles — all use
> individual flags/config fields. The `--profile` concept was a premature abstraction over
> two flags (`--cycles` and `--order`). Users set execution parameters directly via CLI flags
> (which override YAML) or study YAML `execution:` block (which is portable and versioned).
> Execution environment profiles (local/Docker/SLURM) deferred to post-v2.0.

Execution settings are specified directly:

```yaml
# study.yaml — portable, versioned
execution:
  n_cycles: 5                  # how many times to repeat the full experiment set
  cycle_order: shuffled         # sequential | interleaved | shuffled
  config_gap_seconds: 120       # optional: override user config machine default
  cycle_gap_seconds: 600        # optional: override user config machine default
```

```yaml
# ~/.config/llenergymeasure/config.yaml — machine-local thermal defaults only
execution:
  config_gap_seconds: 60        # thermal gap between experiments
  cycle_gap_seconds: 300        # thermal gap between full cycles
```

`n_cycles` and `cycle_order` are NOT in user config — they define study rigour (portable,
study-level concern). Gap seconds ARE in user config — they describe the machine's thermal
characteristics (machine-local concern). See [study-execution-model.md — Decision A](study-execution-model.md).

**Precedence** (later overrides earlier):
```
1. Pydantic defaults:  n_cycles=1, cycle_order="sequential"
2. User config:        execution.config_gap_seconds / execution.cycle_gap_seconds
3. Study file:         execution: { n_cycles, cycle_order, [gap overrides] }
4. CLI flag:           --cycles N, --no-gaps, --order X
```

**`n_cycles` effective default:**

| Context | Effective `n_cycles` | Why |
|---|---|---|
| `llem run study.yaml` (no execution: block) | **3** — CLI effective default | Statistically sound minimum for measurement |
| `run_experiment()` / `run_study()` library API | **1** — Pydantic field default | Library callers have explicit control; no CLI default applied |
| `execution: {n_cycles: 5}` inline in study file | **5** — explicit value | Explicit study file value always wins over CLI default |
| `llem run study.yaml --cycles 1` | **1** — CLI flag | Highest priority override |
| `llem run study.yaml --cycles 1 --no-gaps` | **1** | Explicit CLI override |

The CLI applies an effective default of 3 cycles — a study file with no `execution:` block runs **3 cycles** (not the Pydantic default of 1). Single-cycle studies produce statistically unreliable energy measurements; requiring users to opt OUT of rigour is better than requiring them to opt IN.

The `run_experiment()` / `run_study()` library API does NOT apply the 3-cycle default — library callers have explicit programmatic control and the Pydantic default of `1` is correct for their context.

### Consequences

Positive: study.yaml is portable across machines; all execution parameters are explicit and discoverable via `--help`.

Negative / Trade-offs: No shorthand for common parameter combinations — users must learn the individual flags. Mitigated by example YAML files shipping with the package.

Neutral: Gap seconds defaults live in user config and can be overridden per-study in study.yaml.

---

## Sub-decision E — `--dry-run` and Sweep Validation

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Three-layer validation: `validate_experiment_config()` (always) + `estimate_vram()` (--dry-run / optional pre-flight) + runtime failure handling (chosen)** | Catches known-invalid combos before GPU allocation; VRAM estimation prevents OOM waste on large sweeps; runtime layer handles the unknowable; no external dependency | Requires maintaining a SSOT dict of valid backend-precision combos; VRAM formula is approximate (±15-20%) |
| Declarative constraint syntax (ConfigSpace `ForbiddenClause` style) | Clean DSL for expressing forbidden combos; sampling guarantees validity | Adds niche academic dependency (ConfigSpace); no major sweep framework uses declarative constraints; overkill for ~50 lines of validation logic |
| Run-and-collect failures (W&B pattern) | Zero pre-flight work; sampler learns from failures in Bayesian search | Wastes GPU time on known-invalid combos; unacceptable for expensive grid sweeps where every combination runs once; no sampler to learn from in Cartesian grids |
| Separate `llem check` command | Explicit validation invocation | Adds a third command with no peer precedent; `--dry-run` on `run` covers this without expanding the command surface |

### Decision

We will add a `--dry-run` flag to `llem run` backed by three independent validation layers. No peer tool (Hydra, W&B, vLLM, lm-eval) provides effective `--dry-run` grid preview — this is a differentiator for expensive GPU experiments. Research in [../research/10-sweep-validation-patterns.md](../research/10-sweep-validation-patterns.md) confirmed that no major sweep framework offers declarative categorical constraint syntax worth depending on; the industry standard is a hand-rolled validator function.

**Layer 1 — `validate_experiment_config()` (always, before every experiment)**

A ~50-line function that checks known-invalid parameter combinations against a single positive SSOT dict per backend. Runs before any GPU allocation — both during `--dry-run` and during normal execution. Invalid configs raise `PreFlightError` (see [error-handling.md](error-handling.md)) with an instructive message naming the valid alternatives. In a study loop, invalid configs are logged as "skipped" and the study continues.

This is the universal industry pattern: Optuna uses `raise TrialPruned()`, W&B users add early-exit guards, Hydra users add `hydra-filter-sweeper` expressions. All amount to the same thing — a validity check before GPU init. We build ours into the config layer rather than the sweep layer, so it protects single experiments and library callers equally.

**Layer 2 — `estimate_vram()` (during `--dry-run` and optionally as pre-flight warning)**

Formula-based VRAM estimation using model metadata from `AutoConfig.from_pretrained()` (reads from local HuggingFace cache; gracefully skips with a warning if the model is not yet cached):

```
weight_gb   = num_params * bytes_per_dtype / 1e9
kv_cache_gb = 2 * n_layers * hidden_size * seq_len * batch_size * bytes_per_dtype / 1e9
total_gb    = weight_gb + kv_cache_gb + (weight_gb * 0.15)
```

Configs exceeding 90% of available GPU VRAM are flagged as "LIKELY OOM". The 90% threshold follows vLLM's standard `gpu_memory_utilization` default. HuggingFace Accelerate's `estimate-memory` CLI covers weight memory only and has no programmatic API suitable for grid filtering; vLLM's internal `profile_run` requires loading the full model. A purpose-built estimator is the only practical option for pre-flight grid checking. Full formulae and source analysis in [../research/10-sweep-validation-patterns.md §5](../research/10-sweep-validation-patterns.md).

During `--dry-run`, VRAM estimates are always shown. During normal execution, VRAM estimation is an optional pre-flight warning (not a hard gate) — the estimate is approximate and should not prevent runs that might succeed.

**Layer 3 — Runtime failure handling (normal execution only)**

Per-experiment OOM or unexpected backend errors are caught, logged with full config + exception + traceback, and the study continues to the next experiment. The study summary distinguishes three outcomes: **ran** (success), **skipped** (Layer 1 — known-invalid config), **failed** (Layer 3 — runtime error). This is the same catch-log-continue pattern used by Hydra multirun and W&B Sweeps.

**`--dry-run` behaviour by scope:**

| Invocation | What `--dry-run` does |
|---|---|
| `llem run experiment.yaml --dry-run` | Validates config (Layer 1), estimates VRAM (Layer 2), prints summary, exits 0 |
| `llem run study.yaml --dry-run` | Resolves sweep grammar, validates all configs (Layer 1), estimates VRAM per experiment (Layer 2), prints grid preview (see below), exits 0 |
| `llem run --model X --dry-run` | Validates CLI-constructed config (Layer 1), estimates VRAM (Layer 2), prints summary, exits 0 |

**Grid preview output for studies:**

```
$ llem run study.yaml --dry-run

Study: batch-size-sweep-2026-02
Model: meta-llama/Llama-3.1-8B

Grid: 15 configs generated, 3 skipped (invalid), 12 to run
  Skipped:
    - tensorrt + precision=fp32   tensorrt does not support precision 'fp32'. Valid: fp16, int8, fp8, int4
    - tensorrt + precision=bf16   tensorrt does not support precision 'bf16'
    - pytorch  + precision=int4   pytorch does not support precision 'int4'

VRAM estimates (A100 80GB detected, 72GB usable at 90%):
  #   Backend   Precision  Batch  Est. VRAM   Status
  1   pytorch   fp16       1      ~16.8 GB    OK
  2   pytorch   fp16       8      ~18.2 GB    OK
  3   pytorch   fp16       32     ~24.1 GB    OK
  4   pytorch   bf16       1      ~16.8 GB    OK
  5   pytorch   bf16       8      ~18.2 GB    OK
  6   pytorch   bf16       32     ~24.1 GB    OK
  7   vllm      fp16       1      ~16.8 GB    OK
  8   vllm      fp16       32     ~24.1 GB    OK
  9   vllm      bf16       1      ~16.8 GB    OK
  10  vllm      bf16       32     ~24.1 GB    OK
  11  pytorch   fp32       1      ~33.5 GB    OK
  12  pytorch   fp32       32     ~72.4 GB    LIKELY OOM

Execution: 3 cycles x 12 experiments = 36 runs (interleaved)
Est. duration: ~2h 15m (assuming ~3.5 min/experiment + thermal gaps)

11 OK, 1 likely OOM. Run without --dry-run to start.
```

The grid preview resolves the full sweep grammar, applies Layer 1 validation, runs Layer 2 VRAM estimation, and presents a summary. This addresses a gap identified in peer tools: Hydra's `--cfg job` shows a single resolved config (no grid), W&B has no preview, vLLM bench sweep runs immediately with no preview. See [../research/10-sweep-validation-patterns.md §6](../research/10-sweep-validation-patterns.md) and FEATURES.md §Gap 6–7 for the full peer comparison.

### Consequences

Positive: Researchers can preview large sweep grids and catch configuration errors before committing hours of GPU time. Known-invalid combinations never touch the GPU. VRAM estimation prevents the most common source of wasted time (OOM on large batch sizes). No external dependency — the validator and estimator are purpose-built and total ~100 lines.

Negative / Trade-offs: The SSOT dict of valid backend-precision combinations must be maintained manually as backends evolve. VRAM estimation is approximate (does not account for backend-specific memory overhead, attention implementation differences, or quantisation-specific memory patterns) — it will produce false positives and false negatives at the margins. The "LIKELY OOM" flag is advisory, not a hard gate, for this reason.

Neutral: `--dry-run` exits 0 regardless of how many configs are invalid or flagged OOM — it is an informational tool, not a gate. The three layers are orthogonal and can be tested independently. Runtime duration estimation (shown in `--dry-run` output) is a rough heuristic and clearly labelled as such.

---

## Related

- [experiment-study-architecture.md](experiment-study-architecture.md) — Option C architecture, unified `llem run`, library API
- [study-execution-model.md](study-execution-model.md) — Decisions A/B/C: measurement protocol placement, hashing (Decision D superseded by experiment-study-architecture.md)
- [installation.md](installation.md) — progressive disclosure install flow
- [../designs/cli-commands.md](../designs/cli-commands.md) — full command signatures and options
- [../designs/observability.md](../designs/observability.md) — output examples (llem config, pre-flight)
- [../research/16-execution-profiles-patterns.md](../research/16-execution-profiles-patterns.md) — peer tool survey for execution profiles
- [../research/10-sweep-validation-patterns.md](../research/10-sweep-validation-patterns.md) — sweep validation patterns, VRAM estimation formulae, peer constraint analysis
- [error-handling.md](error-handling.md) — `PreFlightError` exception for Layer 1 validation failures
- [../designs/study-yaml.md](../designs/study-yaml.md) — three-level validation design and VRAM estimation formula
