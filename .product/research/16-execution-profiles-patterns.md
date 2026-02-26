# Research: Execution Profiles — Named Presets in Peer Tools

**Date**: 2026-02-20
**Cited by**: [decisions/cli-ux.md](../decisions/cli-ux.md) (Execution Profiles section)
**Question**: What do peer tools do for named execution presets, and is this pattern
               necessary complexity for llem v2.0?
**Confidence**: MEDIUM-HIGH (peer tool patterns confirmed via docs/search; primary source
                inspection limited by WebFetch unavailability)

---

## 1. Current llem Design (from cli-ux.md)

The confirmed design (2026-02-19) has three layers:

**Built-in profiles** (hardcoded in CLI, not file-dependent):

| Profile | n_cycles | cycle_order | config_gap_seconds | cycle_gap_seconds | Use case |
|---------|----------|-------------|-------------------|-------------------|----------|
| `quick` | 1 | sequential | 0 | 0 | CI / fast iteration |
| `standard` | 3 | interleaved | 60 | 300 | Default for all `llem study` runs |
| `publication` | 5 | shuffled | 120 | 600 | Publishable results |

**User-local config** (`~/.config/llenergymeasure/config.yaml`) holds additional named
profiles under `execution_profiles:` and can override built-in definitions.

**Study file** (`study.yaml`) references a profile by name and/or overrides individual
fields in an `execution:` block.

**Precedence** (later overrides earlier):
```
Pydantic field defaults
  → standard profile (CLI always applies as baseline)
    → LLEM_PROFILE env var + user config profile
      → study file execution: block
        → CLI flag (--cycles N)
```

The `standard` profile is the CLI's effective default — a study with no `execution:` block
runs 3 cycles. The library API (`run_study()`) does not apply any profile; Pydantic default
of `n_cycles=1` applies.

---

## 2. Peer Tool Survey

### 2.1 Nextflow — `profiles {}` block in `nextflow.config`

**Source**: https://nextflow.io/docs/stable/config.html (HIGH confidence — official docs)

Nextflow profiles are the canonical reference for this pattern. A `nextflow.config` file
defines named profiles in a `profiles {}` scope:

```groovy
profiles {
    standard {
        process.executor = 'local'
    }
    cluster {
        process.executor = 'sge'
        process.queue     = 'long'
        process.memory    = '10GB'
    }
    cloud {
        process.executor  = 'cirrus'
        process.container = 'cbcrg/imagex'
        docker.enabled    = true
    }
}
```

Key design decisions:
- **Profile controls HOW to run, not WHAT to run.** The pipeline definition (the `.nf` file)
  is completely silent on execution environment. Profiles are pure execution concern.
- **`standard` is the default.** If no `-profile` flag is given, Nextflow applies `standard`.
  This is an explicit, named convention — not "no profile".
- **Profiles are machine-local by convention.** The `nextflow.config` file is typically
  gitignored or in a separate config repo (nf-core/configs). The pipeline `.nf` file is
  portable; the config is machine-specific.
- **Profiles stack with commas**: `nextflow run script.nf -profile standard,docker` merges
  two profiles, later overrides earlier.
- **No user-level config for profiles**: profiles live in `nextflow.config` in the working
  directory or a referenced config file, not in `~/.config/`. This is a difference from
  the current llem design.

**What Nextflow does NOT have**: Profiles do not have a hierarchy of built-in names vs
user-defined names. Every profile is equal — `standard` is merely a convention.

**Relevance to llem**: The Nextflow model validates the core separation: pipeline definition
is portable; execution environment is local config. However, Nextflow profiles control
executor type, memory limits, container engines — infrastructure concerns. llem's profiles
control statistical rigour (n_cycles, gaps) — a different kind of concern.

---

### 2.2 Snakemake — `--profile` directory with `config.yaml`

**Source**: https://snakemake.readthedocs.io/en/stable/executing/cli.html (HIGH confidence)

Snakemake profiles are directories containing a `config.yaml` that sets CLI flag defaults:

```yaml
# ~/.config/snakemake/my_profile/config.yaml
executor: slurm
default-resources:
  mem_mb: 4096
  runtime: 60
jobs: 100
latency-wait: 60
use-conda: true
restart-times: 3
```

Activated with: `snakemake --profile my_profile`

Key design decisions:
- **Profiles are pure CLI default overrides.** Every key maps to a CLI flag. No special
  semantics — `cores: 4` is identical to `snakemake --cores 4`.
- **Two kinds of profiles**: (1) global, in `~/.config/snakemake/` or `/etc/xdg/snakemake/`;
  (2) workflow-specific, in `profile/default/` next to the Snakefile. This is analogous to
  llem's distinction between user config (Layer 1) and study-file-local settings.
- **No built-in profiles.** Snakemake has no `standard` or `quick` preset. The user creates
  all profiles. This means zero magic, but also more configuration burden.
- **`$SNAKEMAKE_PROFILE` env var** sets the default profile — same pattern as proposed
  `LLEM_PROFILE` env var.

**Relevance to llem**: Snakemake's design is maximally simple — profiles are just YAML
representations of CLI flags. No indirection through named presets within the profile file.
This is a counterargument to having built-in profiles in llem: Snakemake works fine without
them by making each flag directly configurable.

---

### 2.3 dbt — `profiles.yml` targets (named environments)

**Source**: https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml (HIGH confidence)

dbt's `profiles.yml` (in `~/.dbt/`) defines named **targets** within a profile:

```yaml
# ~/.dbt/profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      schema: dbt_{{ env_var('DBT_USER') }}
    prod:
      type: postgres
      host: prod-db.company.com
      schema: analytics
```

Activated with: `dbt run --target prod`

Key design decisions:
- **`target` is the named preset.** Each target is a complete, named environment config.
- **Default target is explicit**: `target: dev` is the default, set at profile level.
- **File lives in home dir, never versioned.** `~/.dbt/profiles.yml` is gitignored by
  convention — same Layer 1 principle as llem user config.
- **No built-in targets.** dbt has no built-in `dev` or `prod` — they're conventions the
  user names themselves.
- **Direct field override**: `dbt run --vars '{"schema": "staging"}'` provides inline
  override without switching targets — analogous to llem's inline `execution:` block
  overriding a profile.

**Relevance to llem**: dbt's pattern is closest to the proposed llem design in spirit:
named environments (targets) in user-local config, with a default, a flag to select
alternatives, and inline override. The key difference: dbt targets are environments
(connection strings, schemas); llem profiles are statistical rigour presets. The abstraction
fits better for infrastructure settings than for statistical parameters.

---

### 2.4 Hydra — Config Groups

**Source**: https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/ (HIGH confidence)

Hydra config groups allow multiple named configurations per concern:

```
conf/
  db/
    mysql.yaml
    postgresql.yaml
  server/
    development.yaml
    production.yaml
```

Selected at CLI: `python app.py db=postgresql server=production`

Key design decisions:
- **Config groups are composable.** You select one option per group; Hydra merges them.
- **Override at CLI**: `python app.py db=postgresql db.port=5432` — inline key override
  without needing a new named config.
- **No built-in names.** All groups and options are user-defined. Zero magic defaults.
- **Optimum-Benchmark uses Hydra** for this exact purpose: `--config-name cuda_pytorch_bert`
  selects a named benchmark configuration. But OB's configs are full experiment definitions,
  not just execution presets.

**Relevance to llem**: Hydra's config group pattern is powerful but heavyweight. It requires
Hydra as a dependency and fundamentally restructures the config system. Not appropriate for
llem's simpler Pydantic-based approach. However, the composable named-group concept is
the conceptual ancestor of the `profile:` reference in llem's study file.

---

### 2.5 Optimum-Benchmark (HuggingFace)

**Source**: https://github.com/huggingface/optimum-benchmark (MEDIUM confidence — confirmed
from research/15-config-architecture-patterns.md + search results)

OB uses Hydra for config composition. Its `scenario` config (the execution parameters) is
a separate config group from the `backend` config:

```yaml
# examples/cuda_pytorch_bert.yaml
defaults:
  - backend: pytorch
  - launcher: process
  - scenario: inference

backend:
  model: bert-base-uncased
  device: cuda

scenario:
  warmup_runs: 10
  iterations: 10  # this is the n_runs equivalent
  latency: true
  memory: true
```

Key design decisions:
- **`scenario` and `backend` are orthogonal config groups.** The scenario (inference vs
  training) is separate from the backend (PyTorch vs ONNX). Analogous to llem's separation
  of experiment definition from execution settings.
- **`iterations`/`warmup_runs` are direct scalar fields, not named presets.** OB does not
  have a `profile: publication` concept — you just set `iterations: 10` directly. No
  indirection.
- **No "standard" / "quick" presets.** OB treats all repetition counts as user-specified
  without built-in names. The tradeoff: zero magic, but users must know what number is
  appropriate.

**Assessment**: OB's direct-field approach is simpler than named profiles, and it works
for OB because OB is a library/framework used by benchmark engineers who know what
`iterations: 10` means. llem targets a broader audience (researchers who might not know
whether 3 or 5 cycles is appropriate for publishable results), which is the key justification
for named presets.

---

### 2.6 Hyperfine — Direct CLI Flags, No Profiles

**Source**: https://github.com/sharkdp/hyperfine (HIGH confidence)

Hyperfine is a command-line benchmarking tool. Its repetition/statistical settings are all
direct CLI flags:

```bash
hyperfine --runs 20 --warmup 3 --min-runs 10 'cmd1' 'cmd2'
```

No profile concept at all. Every execution parameter is a direct flag with a documented
default (`--runs` defaults to 10, `--warmup` defaults to 0). The defaults are chosen to be
correct for hyperfine's use case (wall-clock time benchmarking).

**Relevance to llem**: Hyperfine's UX is maximally simple — and it works because hyperfine
users are technical (they're benchmarking programs) and the tool has a single well-known
use case. The defaults are universally appropriate. llem's case is more complex: the
"right" number of cycles is context-dependent (CI vs research paper), which is the
justification for profiles vs plain defaults.

---

### 2.7 pytest-benchmark — Direct Flags + .ini Configuration

**Source**: https://pytest-benchmark.readthedocs.io/en/latest/ (HIGH confidence)

pytest-benchmark controls repetition through direct flags and `.ini` config:

```ini
# pytest.ini
[pytest]
benchmark_min_rounds = 5
benchmark_warmup = true
benchmark_warmup_iterations = 1
benchmark_disable_gc = false
```

CLI override: `pytest --benchmark-min-rounds=20 --benchmark-warmup-iterations=5`

No named presets. The `.ini` sets defaults; CLI overrides. The tool's philosophy is that
round counts should be determined automatically (it runs until statistical stability) rather
than preset.

**Relevance to llem**: pytest-benchmark's "run until stable" approach is not applicable to
llem because llem measures energy (which is not statistically self-terminating in the same
way as latency). Fixed n_cycles is correct for energy measurement. But the direct-field-in-
config approach (without named presets) is a valid alternative design.

---

### 2.8 MLflow — No Execution Presets

**Source**: https://mlflow.org/docs/latest/index.html (HIGH confidence)

MLflow does not have execution profiles. `mlflow run` takes direct flags for parallelism
and environment settings. No concept of "repeat this experiment N times with thermal gaps."
MLflow tracks runs, not measurement protocols.

**Relevance to llem**: MLflow's lack of profiles is expected — it's a tracking tool, not
a measurement protocol tool. Confirms that the profiles concept is specific to tools that
run experiments repeatedly for statistical reliability.

---

### 2.9 W&B Sweeps — Search Method, Not Repetition Control

**Source**: https://docs.wandb.ai/models/sweeps/sweep-config-keys (HIGH confidence)

W&B sweeps control hyperparameter search strategy (grid, random, Bayesian) and define the
parameter space. There is no concept of "run each config N times with a thermal gap." W&B
sweep repetition comes from adding a `seed` parameter to the sweep grid (to run the same
config with different seeds), not from a dedicated repeat mechanism.

**Relevance to llem**: Confirms that "profiles for repetition" is not a W&B concept. Also
confirms the earlier decision to reject "sweep" as llem's terminology — W&B sweeps are
about search, not measurement.

---

### 2.10 AIEnergyScore — Fixed Protocol, No User Configuration

**Source**: https://github.com/huggingface/AIEnergyScore (MEDIUM confidence)

AIEnergyScore runs a fixed, hard-coded protocol: specific dataset (1,000 samples per task),
specific hardware (H100), no user-configurable repetitions or profiles. The benchmark is
designed for comparability, not flexibility. Users provide the model; AIEnergyScore controls
everything else.

**Relevance to llem**: This is the opposite extreme from llem. AIEnergyScore achieves
reproducibility by removing user choice. llem's design philosophy is different: it must
accommodate different machine capabilities and different statistical rigour requirements.
This justifies having *some* configurability around n_cycles and gaps — but the question
is whether named profiles are the right mechanism.

---

## 3. Analysis: Is the Profile Abstraction Necessary?

### What problem do profiles actually solve?

The profiles concept solves two things:

**A. Communicating statistical intent**

"Run with `profile: publication`" communicates more than "run with `n_cycles: 5,
cycle_order: shuffled, config_gap_seconds: 120, cycle_gap_seconds: 600`". The named
preset encodes domain knowledge: this set of parameters produces publication-quality
results. A researcher who doesn't know what n_cycles is appropriate for a paper can
trust the `publication` preset.

**B. Reusability across study files**

If a team runs ten different study files for the same paper, they can all say
`profile: publication` and know they're using consistent settings. Changing the standard
from "5 cycles" to "7 cycles" would require editing ten files — unless there's a named
preset in user config that all of them reference.

### What complexity do profiles add?

1. **Indirection**: `profile: publication` requires knowing what `publication` expands to.
   Direct fields are self-documenting; profile names are not.

2. **Two-layer resolution**: The CLI must look up the profile name in user config, then
   apply inline overrides on top. This is non-trivial logic.

3. **User config dependency**: A study file that says `profile: custom-hpc` is not
   portable — it requires that the target machine has `custom-hpc` defined in its user
   config. This is a reproducibility footgun.

4. **Precedence complexity**: The 5-level precedence chain (Pydantic defaults → standard
   profile → env var profile → study file → CLI flag) is hard to explain and harder to debug.

### The critical question: who is the user?

llem's users (researchers) fall into two groups:

- **Infrastructure-aware**: Know what n_cycles, thermal gaps mean. Can write `n_cycles: 5`
  directly. Don't need presets.
- **Infrastructure-naive**: Don't know what's appropriate for a paper. Need guidance.
  But — do they need *named presets*, or do they need *good defaults and documentation*?

The second group needs the correct defaults more than they need preset names. A study file
that says `n_cycles: 5` is equally clear as one that says `profile: publication`, if the
documentation says "use n_cycles: 5 for publication results."

### Comparison: What does OB do for this?

Optimum-Benchmark — llem's closest direct peer — uses **direct fields** (`warmup_runs: 10,
iterations: 10`) with no named preset concept. Their users are experienced benchmark
engineers, so there's no need to abstract "what settings make a good benchmark."

llem's counter-argument is that it targets a broader audience. But the real question is:
does `profile: publication` actually solve the knowledge gap, or does it just move the
documentation requirement from "what does iterations=10 mean?" to "what does publication
mean?"

### Key finding: The split between "execution-scheduling" and "statistical protocol"

No peer tool separates "scheduling" concerns (gaps between experiments, cycle ordering)
from "statistical" concerns (n_cycles). They're either both user-specified directly, or
both hidden behind an abstraction. The current llem design bundles them into profiles,
which is coherent — but it's worth noting that the scheduling parameters
(`config_gap_seconds`, `cycle_gap_seconds`) are more machine-dependent (a hot GPU needs
longer gaps) than the statistical parameters (`n_cycles`, `cycle_order`).

This suggests a cleaner split:
- `n_cycles` and `cycle_order` belong in the study file (statistical protocol for this study)
- `config_gap_seconds` and `cycle_gap_seconds` belong in user config (machine capability)

---

## 4. Recommendation

### The simplest design that solves the actual problem

**Key insight from peer review**: No peer tool (Nextflow, Snakemake, OB, hyperfine,
pytest-benchmark) has a `profile:` indirection that maps to a bundle of execution settings.
The closest analogs (Nextflow profiles, dbt targets) are environment-switchers (local vs
cluster vs cloud), not statistical-rigour-presets. The profile concept in llem conflates
two concerns that peers keep separate:

1. **Machine-local scheduling** (gap seconds, cycle order preference): varies by machine
   capability. Belongs in user config.
2. **Statistical protocol** (n_cycles): defines the study's rigour. Belongs in the study
   file as a direct field.

### Recommended: Simplify to direct fields + split concerns

**Study YAML** (versioned, portable):
```yaml
# batch-size-effects.yaml
model: meta-llama/Llama-3.1-8B
backend: [pytorch, vllm]
sweep:
  batch_size: [1, 4, 8, 16]

execution:
  n_cycles: 3          # how many times to repeat the full experiment set
  cycle_order: interleaved   # sequential | interleaved | shuffled
```

**User config** (`~/.config/llenergymeasure/config.yaml`, machine-local):
```yaml
execution:
  config_gap_seconds: 60    # thermal gap between configs (machine-dependent)
  cycle_gap_seconds: 300    # thermal gap between cycles (machine-dependent)
```

**CLI defaults** (applied when study file has no `execution:` block):
```
n_cycles: 3  (not 1 — same as now, for statistical reliability)
cycle_order: interleaved
```

**Precedence** (simplified from 5 levels to 3):
```
1. User config execution defaults  (machine-local gaps)
2. Study file execution: block     (portable statistical protocol)
3. CLI flag: --cycles N            (one-off override)
```

**Built-in `--profile` flag for convenience** (not a separate config layer):
```bash
llem study batch-size-effects.yaml --profile quick
llem study batch-size-effects.yaml --profile publication
```

The `--profile` flag expands to a set of CLI-flag-level overrides:
- `quick` → `--cycles 1 --no-gaps`
- `publication` → `--cycles 5 --order shuffled`

This is a flag that inlines known-good presets — it does NOT create a YAML config layer.
User config does not store profile definitions. The study file does not reference profiles
by name.

### What this gains

1. **Study files become self-documenting**: `n_cycles: 3` is more informative than
   `profile: standard`. Reviewers of a paper can see directly that 3 cycles were used
   without consulting profile definitions.

2. **Portability restored**: A study file with `n_cycles: 5, cycle_order: shuffled` is
   fully reproducible on any machine, regardless of what that machine's user config says.
   The current design (`profile: publication`) is not portable if `publication` is
   customised in user config.

3. **Simpler precedence**: 3 levels instead of 5. Eliminates `LLEM_PROFILE` env var
   (which was solving a problem created by the profile indirection).

4. **Separates machine concerns from study concerns**: Gap seconds (machine-dependent) live
   in user config; cycle count (study-dependent) lives in the study file.

5. **Still has named preset shorthand**: `--profile quick` / `--profile publication` as CLI
   flags cover the "I don't know what to set" use case without creating a config layer.

### Precedence model (clean version)

```
Concern                   | Where it lives              | Who sets it
--------------------------|-----------------------------|-----------------------
config_gap_seconds        | User config (default: 60s)  | Machine operator
cycle_gap_seconds         | User config (default: 300s) | Machine operator
n_cycles                  | Study file (default: 3)     | Researcher
cycle_order               | Study file (default: interleaved) | Researcher
--cycles / --profile      | CLI flag                    | One-off override
```

The CLI `--profile quick` flag sets `n_cycles=1` and zeroes gap seconds — the latter
is a temporary override, not changing user config.

---

## 5. What to Defer

The following profile-system features should be deferred to v2.2+:

| Feature | Reason to Defer | v2.0 Alternative |
|---------|----------------|------------------|
| Custom user-defined profiles (in `~/.config/`) | Adds config parsing complexity; no peer tool has this for statistical presets | Study files with explicit fields |
| `LLEM_PROFILE` env var | Only needed if profile names are a config layer; drops out with simplified design | `LLEM_CYCLES=5` env var for CI (simpler) |
| `--init` flag that writes profile config | Requires profile config layer to exist | Docs showing recommended n_cycles values |
| Profile override fields (`execution: {profile: X, n_cycles: 3}`) | Complex merge semantics; `n_cycles` in study file already covers this | Direct field in study file |
| Named profiles in user config (`execution_profiles: {}`) | Premature abstraction | Direct `execution:` defaults in user config |

**For CI use case** (the primary motivation for `quick` profile):

```bash
# Current (proposed):
llem study file.yaml --profile quick

# Simplified equivalent:
llem study file.yaml --cycles 1 --no-gaps

# Or via env var:
LLEM_CYCLES=1 LLEM_GAPS=0 llem study file.yaml
```

The CI case does not require named profiles — it just requires a flag that suppresses gaps
and sets n_cycles=1. `--profile quick` is a convenience alias for those flags, not a
separate config system.

---

## 6. What Stays Unchanged

The following decisions from the current design are validated by peer research and should
**not** change:

1. **CLI always applies n_cycles=3 as default** (not 1). This is correct: single-cycle
   energy measurements are not reliable. Requiring users to opt out of rigour is right.
   Peer: Nextflow's `standard` profile concept validates applying an opinionated default.

2. **Library API (run_study()) does NOT apply any default profile.** Library callers have
   explicit control. Pydantic default n_cycles=1 is correct for programmatic use.
   Peer: no peer library (lm-eval's `simple_evaluate()`, OB's `BenchmarkConfig`) applies
   execution presets automatically.

3. **Gap seconds are machine-local config, not study-file concerns.** Correct separation.
   Peer: Snakemake's `latency-wait` (equivalent) lives in profile config, not workflow
   definition.

4. **study.yaml is silent on runner/execution environment.** Correct and strongly
   validated by Nextflow and Snakemake patterns.

---

## 7. Summary Assessment

| Question | Finding | Confidence |
|----------|---------|------------|
| Do peer tools use named execution presets? | Rarely, and only for infra-switching (local/cluster/cloud), not statistical rigour | HIGH |
| Does OB (closest peer) use named profiles for repetition? | No — direct fields only | HIGH |
| Are n_cycles and gap_seconds the same concern? | No — n_cycles is study rigour; gap_seconds is machine capability | HIGH |
| Is `profile:` in study YAML harmful to portability? | Yes — creates hidden dependency on user config | HIGH |
| Is the `--profile quick` CLI shorthand valuable? | Yes — covers CI use case cleanly without a config layer | MEDIUM |
| Are user-defined custom profiles in user config needed in v2.0? | No — premature abstraction | HIGH |
| Should `LLEM_PROFILE` env var exist? | Not needed if profile is a CLI flag, not a config layer | MEDIUM |

**Bottom line**: The profile concept conflates two separable concerns and adds a config
layer that no peer tool has for this purpose. The simplest design that solves the problem
is: direct fields in study YAML (`n_cycles`, `cycle_order`) + machine-local defaults for
gap seconds in user config + `--profile quick/publication` as CLI convenience aliases that
expand to specific flag combinations. This eliminates the 5-level precedence chain,
restores study-file portability, and removes the user-config profile-definition complexity
entirely.

---

## Sources

- Nextflow configuration docs: https://nextflow.io/docs/stable/config.html
- Snakemake CLI reference: https://snakemake.readthedocs.io/en/stable/executing/cli.html
- Snakemake profile blog: http://bluegenes.github.io/Using-Snakemake_Profiles/
- dbt profiles.yml: https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml
- Hydra config groups: https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/
- Optimum-Benchmark: https://github.com/huggingface/optimum-benchmark
- hyperfine: https://github.com/sharkdp/hyperfine
- pytest-benchmark: https://pytest-benchmark.readthedocs.io/en/latest/
- W&B sweep config: https://docs.wandb.ai/models/sweeps/sweep-config-keys
- AIEnergyScore: https://github.com/huggingface/AIEnergyScore
- Prior llem research: [15-config-architecture-patterns.md](15-config-architecture-patterns.md)
