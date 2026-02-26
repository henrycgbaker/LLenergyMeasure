# Study & Execution Model

**Status:** Accepted
**Date decided:** 2026-02-20
**Last updated:** 2026-02-25
**Research:** [../research/16-execution-profiles-patterns.md](../research/16-execution-profiles-patterns.md)

**Prerequisite for:** [../designs/study-yaml.md](../designs/study-yaml.md),
                      [../designs/user-config.md](../designs/user-config.md),
                      [cli-ux.md](cli-ux.md) § Execution Profiles,
                      [architecture.md](architecture.md) § Field Placement Table

## Decision

| Sub-decision | Resolution |
|---|---|
| **A. Measurement protocol placement** | Split by portability: `n_cycles`/`cycle_order` in study YAML (portable); gap seconds in user config (machine-local). |
| **B. Hashing semantics** | `study_design_hash` excludes execution block — same study at different rigour = same hash. |
| **C. experiment.yaml ↔ study.yaml** | Single run is syntactic sugar for `StudyConfig(experiments=[config])`. Same runner. |
| **D. Experiment vs study unification** | **Superseded** by [experiment-study-architecture.md](experiment-study-architecture.md) — Option C accepted. |

---

## Context

Three interrelated questions were surfaced during DRY refactor review (2026-02-20). The core
tension: the config model has **four distinct concerns**, but the prior design did not
explicitly name measurement protocol, leaving it placed inconsistently:

```
User Config — Machine/Infrastructure (HOW to run, WHERE)
  → runners, output dirs, energy backend, carbon intensity, PUE
  → user config: ~/.config/llenergymeasure/config.yaml

Measurement Protocol (HOW RIGOROUSLY) ← explicitly named
  → n_cycles, cycle_order (in study.yaml, excluded from hash)
  → config_gap_seconds, cycle_gap_seconds (in user config, machine-local)

Experiment/Study YAML — Experimental Design (WHAT to measure)
  → model, backend, precision, n, dataset, decoder, warmup, backend params
  → experiment.yaml (single measurement point) / study.yaml (parameter space)
  → study_design_hash covers this only — not measurement protocol

Environment Snapshot — Infrastructure Context (WHAT you're measuring ON)
  → auto-detected: GPU model, VRAM, CUDA version, driver, CPU
  → stored with results, never in config files
```

---

## Decision A — Measurement Protocol Placement

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **A1 Modified: split by portability (chosen)** | `n_cycles`/`cycle_order` in study.yaml (portable, reproducible); gap seconds in user config (machine-local) | Requires discipline to place fields correctly |
| A2: All measurement protocol in study.yaml | Self-contained study files | Gap seconds are machine characteristics, not study design — a hot shared cluster and a cold workstation need different gaps for the same study |
| A3: All measurement protocol in user config | Machine-appropriate defaults everywhere | `n_cycles` is study design (sample size), not machine preference; non-portable |
| Three-file model (experiment.yaml + study.yaml + execution.yaml) | Clean separation | Every peer tool uses 2 config surfaces; third surface creates new precedence conflict with user config; separation achievable with two labelled sections in study.yaml |
| Named profiles in user config (`execution_profiles:`) | Flexible named presets | Conflates machine thermal settings (gap seconds) with study rigour (n_cycles/cycle_order) in one named preset; study files become machine-dependent |

**Rejected: Three-file model** (experiment.yaml + study.yaml + execution.yaml)

Considered and rejected 2026-02-20. Reasons:
1. Every peer tool uses 2 config surfaces (what + how). None use 3 for this purpose.
2. User config already IS the machine execution settings file — adding `execution.yaml` creates
   a fourth surface and a new precedence conflict between user config and execution.yaml.
3. The separation of concerns the three-file model provides can be achieved with two clearly
   labelled sections within study.yaml (experimental design + execution block).

**Rejected: Named profiles in user config** (`execution_profiles:` block)

The profile concept conflates two distinct concerns (infrastructure-switching vs rigour presets)
into one named-preset mechanism. No peer tool uses named presets for statistical repetition
settings. Users with unusual n_cycles requirements set them directly in the study file or via
`--cycles` CLI flag. The `execution_profiles:` block in user config is removed.

### Decision

Modified A1 — split by portability:

| Field | Home | Rationale |
|-------|------|-----------|
| `n_cycles` | study.yaml `execution:` block | Portable — equivalent to sample size; part of the study design record a colleague needs to reproduce |
| `cycle_order` | study.yaml `execution:` block | Portable — affects statistical validity (interleaved vs sequential reduces thermal autocorrelation) |
| `config_gap_seconds` | user config `execution.*` (default) | Machine-local — hot shared cluster needs 120s; cold workstation needs 0s; not a study design choice |
| `cycle_gap_seconds` | user config `execution.*` (default) | Machine-local — same reasoning |

Gap seconds can be **overridden in study.yaml** `execution:` block for specific cases, but they
are NOT part of the `study_design_hash`. The user config provides the machine-appropriate default.

**Precedence for execution fields (3 levels):**
```
1. User config:  execution.config_gap_seconds / execution.cycle_gap_seconds  ← machine defaults
2. Study file:   execution: { n_cycles, cycle_order, [gap overrides] }        ← explicit always wins
3. CLI flag:     --cycles N, --no-gaps                                         ← highest priority
```

**CLI convenience flags** (not a config layer):
```bash
llem run file.yaml --cycles 1 --no-gaps          # quick single-cycle study
llem run file.yaml --cycles 5 --order shuffled    # publication-grade rigour
```

> **Superseded (2026-02-25):** `--profile quick|publication` shorthand removed. 0/5 peers
> use named rigour profiles. Direct flags are clearer and match peer patterns.

**Date**: 2026-02-20

### Consequences

Positive: Study files are portable across machines; colleagues can reproduce studies with the
same experimental design regardless of their machine's thermal characteristics.

Negative / Trade-offs: `config_gap_seconds` must be set appropriately in user config per machine
(no automatic detection of thermal recovery time).

Neutral: Gap overrides can be embedded in study.yaml for specific cases (e.g. publication-grade
study on a cluster with known thermal characteristics).

---

## Decision B — Hashing Semantics

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **B2a: Execution block excluded from `study_design_hash` (chosen)** | Same study run at different rigour levels shares the same hash; supports "topping up" with more cycles | Hash alone doesn't show how rigorously the study was run |
| B2b: Execution block included in hash | Hash captures everything about the study run | Same experimental design run at n_cycles=3 vs n_cycles=5 creates two different identities; "topping up" impossible without losing the link |
| Hash the raw YAML file (`study_yaml_hash`) | Simple to compute | Fragile — whitespace, comments, and field ordering change the hash even when experimental design is identical |

**Rejected (2026-02-20): `study_yaml_hash`**

Hashing the full YAML file is fragile (whitespace, comments, field ordering all change the hash
even when the experimental design is identical). `study_design_hash` hashes the canonical
`model_dump(exclude={"execution"})` of the experimental design portion only — two runs of the
same study at different rigour levels share the same hash.

### Decision

B2a — execution block excluded from `study_design_hash`. Stored as metadata.

```python
def compute_study_design_hash(config: StudyConfig) -> str:
    """Hash of experimental design only. Excludes execution block.

    Two studies with identical sweep/experiment list but different n_cycles
    get the same hash — they are the same study run at different rigour levels.
    Supports "topping up" a study with more cycles without creating a new identity.
    """
    config_dict = config.model_dump(exclude={"execution"})
    canonical = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

**Rationale**: The hash answers *"did you test the same thing?"*, not *"how rigorously?"*.

- n_cycles is a *quality attribute* of a study run, not an *identity attribute* of a study.
  Changing the dataset or model changes what you measured. Changing n_cycles changes how
  confidently you measured it. These are different things.
- "Topping up" a study (3 cycles → 5 cycles for submission) should produce the same hash.
  Under B2b these would be different studies and results couldn't be linked.
- Mirrors the pattern already established for `measurement_config_hash`: infrastructure
  context is stored with results but excluded from the experiment identity hash.

**StudyResult stores measurement protocol as metadata (not hashed):**
```json
{
  "study_design_hash": "a3f2...",
  "measurement_protocol": {
    "n_cycles": 5,
    "cycle_order": "shuffled",
    "config_gap_seconds_used": 120,
    "cycle_gap_seconds_used": 600
  }
}
```

**Date**: 2026-02-20

### Consequences

Positive: "Topping up" a study (adding more cycles for a publication) produces the same hash;
study identity is stable across rigour levels.

Negative / Trade-offs: Hash alone doesn't capture how rigorously the study was run — check
`measurement_protocol` in the StudyResult.

Neutral: Mirrors the `measurement_config_hash` pattern: execution context stored as metadata,
excluded from the identity hash.

---

## Decision C — experiment.yaml ↔ study.yaml Relationship

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **C1: Unified runner infrastructure (chosen)** | Single internal runner; no code duplication | Requires careful framing — "degenerate study" framing is misleading |
| C2: Fully separate runner implementations | No accidental coupling | Code duplication; two execution paths to maintain |

> **Updated (2026-02-25):** Decision C originally framed as "two distinct CLI commands" (`llem run`
> + `llem study`). CLI was subsequently unified to `llem run` (YAML determines scope). The
> internal architecture (single runner) is unchanged. Decision D (experiment vs study
> unification) is superseded by [experiment-study-architecture.md](experiment-study-architecture.md).

### Decision

C1 (unified runner). All execution — single experiment or study — shares the same internal
runner infrastructure (`_run(StudyConfig) -> StudyResult`). The CLI uses unified `llem run`;
the library exposes `run_experiment()` and `run_study()` for type-safe return types. See
[experiment-study-architecture.md](experiment-study-architecture.md) for full architecture.

```
ExperimentConfig  ←  the atom: one measurement point (WHAT to measure, fully specified)
StudyConfig       ←  the investigation: parameter space + measurement protocol
                      (WHERE in parameter space, HOW MANY CYCLES, WHAT ORDER)

llem run          ←  unified CLI entry point (YAML determines scope)
run_experiment()  ←  library function for single experiment → ExperimentResult
run_study()       ←  library function for studies → StudyResult

Internally: all paths construct StudyConfig and call _run(StudyConfig).
A single experiment is StudyConfig(experiments=[config]) — same runner, same pipeline.
```

**Single-experiment behaviour** (via `llem run experiment.yaml` or `run_experiment()`):
- Executes one experiment
- Applies CLI default of n_cycles=3 (unless `--cycles N` flag overrides)
- Output: single ExperimentResult JSON per cycle, named with timestamp
- Does NOT have an `execution:` block in experiment.yaml — measurement protocol is CLI-driven

**Cross-file references** (study.yaml referencing standalone experiment.yaml files) — **deferred
to v2.2**. For v2.0, study.yaml defines all experiments inline (sweep grammar or explicit list).

**Date**: 2026-02-20

### Consequences

Positive: Single runner implementation to maintain; single-experiment runs via `llem run` benefit
from the same subprocess isolation and result-writing infrastructure as study runs.

Negative / Trade-offs: The "shared internals" framing can mislead — must be explicit in docs
that this is an implementation detail, not conceptual equivalence.

Neutral: Cross-file references deferred to v2.2.

---

## SSOT: Field Inventory

This is the canonical field inventory. Supersedes scattered references in architecture.md,
cli-ux.md, study-yaml.md, and user-config.md.

### User Config — Machine/Infrastructure

`~/.config/llenergymeasure/config.yaml` — machine-local, never versioned, never shared.

| Field | User config key | Default | Env var override |
|-------|----------------|---------|-----------------|
| Backend runner | `runners.<backend>` | `local` | `LLEM_RUNNER_<BACKEND>` |
| Results directory | `output.results_dir` | `./results` | — |
| Model cache directory | `output.model_cache_dir` | HF default | `HF_HOME` |
| Energy backend | `measurement.energy_backend` | `nvml` | — |
| Carbon intensity | `measurement.carbon_intensity_gco2_kwh` | `None` (lookup by location) | `LLEM_CARBON_INTENSITY` |
| Datacenter PUE | `measurement.datacenter_pue` | `1.0` | `LLEM_DATACENTER_PUE` |
| Datacenter location | `measurement.datacenter_location` | `None` | `LLEM_DATACENTER_LOCATION` |
| Config gap (machine default) | `execution.config_gap_seconds` | `60` | — |
| Cycle gap (machine default) | `execution.cycle_gap_seconds` | `300` | — |
| UI verbosity | `ui.verbosity` | `normal` | — |
| UI interactive prompt | `ui.prompt` | `true` | — |

**Removed from user config**: `execution_profiles:` block (dropped — profiles replaced by direct
fields and `--profile` CLI flag shorthands).

### Measurement Protocol (in study.yaml, excluded from hash)

In study.yaml `execution:` block. Stored as `measurement_protocol` in StudyResult. NOT hashed.

| Field | Pydantic default | CLI effective default | Notes |
|-------|-----------------|----------------------|-------|
| `n_cycles` | `1` | `3` | Explicit study file value always wins over CLI default |
| `cycle_order` | `"sequential"` | `"interleaved"` | |
| `config_gap_seconds` | user config value | user config value | Can be overridden in study.yaml |
| `cycle_gap_seconds` | user config value | user config value | Can be overridden in study.yaml |

### Experiment/Study YAML — Experimental Design

Versioned, shareable. `study_design_hash` computed over this only.

**ExperimentConfig** (single measurement atom — used with `llem run` and inside StudyConfig):

| Field | Type | Default | In `measurement_config_hash`? |
|-------|------|---------|------------------------------|
| `model` | `str` | required | ✓ |
| `backend` | `Literal["pytorch","vllm","tensorrt"]` | required | ✓ |
| `precision` | `Literal[...]` | `"bf16"` | ✓ |
| `dataset` | `str \| SyntheticDatasetConfig` | `"aienergyscore"` | ✓ |
| `n` | `int` | `100` | ✓ |
| `random_seed` | `int` | `42` | ✓ |
| `decoder` | `DecoderConfig` | defaults | ✓ |
| `warmup` | `WarmupConfig` | defaults | ✓ |
| `baseline` | `BaselineConfig` | defaults | ✓ |
| `pytorch` | `PyTorchConfig \| None` | `None` | ✓ |
| `vllm` | `VLLMConfig \| None` | `None` | ✓ |
| `tensorrt` | `TensorRTConfig \| None` | `None` | ✓ |
| `lora` | `LoRAConfig \| None` | `None` | ✓ |
| `extra` | `dict \| None` | `None` | ✓ |

**StudyConfig** (parameter space — used with `llem run` when YAML contains `sweep:` or `experiments:`):

| Field | In `study_design_hash`? | Notes |
|-------|------------------------|-------|
| `sweep:` (dotted notation grid) | ✓ | e.g. `pytorch.batch_size: [1, 8, 32]` |
| `experiments:` (explicit list) | ✓ | List of ExperimentConfig overrides |
| `execution:` block | ✗ | Measurement protocol — stored as metadata, excluded from hash |

### Environment Snapshot — Infrastructure Context (auto-captured, stored with results)

Never in config files. Written to result JSON at measurement time.

| Field | Source | Notes |
|-------|--------|-------|
| `gpu_names` | `nvidia-smi` | `list[str]` |
| `gpu_vram_gb` | `nvidia-smi` | `list[float]` |
| `gpu_count` | `nvidia-smi` | `int` |
| `cuda_version` | multi-source detection | `str \| None` |
| `driver_version` | `nvidia-smi` | `str \| None` |
| `cpu_model` | `/proc/cpuinfo` | `str` |
| `carbon_intensity_gco2_kwh` | from user config / env var | `float \| None` |
| `datacenter_pue` | from user config / env var | `float` |
| `datacenter_location` | from user config / env var | `str \| None` |

---

## Precedence Chains

### Infrastructure context (environment snapshot)

```
1. User config default:  measurement.carbon_intensity_gco2_kwh / .datacenter_pue / .datacenter_location
2. Env var override:     LLEM_CARBON_INTENSITY / LLEM_DATACENTER_PUE / LLEM_DATACENTER_LOCATION
3. Stored in environment snapshot:  written to result JSON at measurement time (immutable after)
```

### Measurement protocol

```
1. User config default:  execution.config_gap_seconds / execution.cycle_gap_seconds
2. Study file:           execution: { n_cycles, cycle_order, [gap overrides] }
3. CLI flag:             --cycles N / --no-gaps / --order [sequential|interleaved|shuffled]
```

CLI `--profile quick` and `--profile publication` are shorthand for CLI flag combinations at
step 3 — not a separate config layer.

### Experiment config fields

```
1. Pydantic model default
2. experiment.yaml / inline experiment definition in study.yaml
3. CLI flag:  --model, --backend, --batch-size, --precision, etc.
```

---

## Files Updated by These Decisions

| File | Change |
|------|--------|
| `decisions/architecture.md` | Field placement table rows 101, 107, 110, 111 updated |
| `decisions/cli-ux.md` | Execution Profiles section: remove named profiles in user config; document `--profile` as CLI flag shorthand |
| `designs/study-yaml.md` | `execution:` block schema: remove `profile:` key; document hash exclusion |
| `designs/user-config.md` | Remove `execution_profiles:` block; add `execution.config_gap_seconds` / `execution.cycle_gap_seconds` |
| `designs/result-schema.md` | Add `study_design_hash` + `measurement_protocol` to StudyResult |

---

## Decision D — Experiment vs. Study Unification

> **Superseded (2026-02-25):** By [experiment-study-architecture.md](experiment-study-architecture.md)
> which accepted Option C. The CLI was unified to `llem run` (YAML determines scope). The
> conceptual distinction between experiment (atom) and study (investigation) is preserved in
> config types (`ExperimentConfig` vs `StudyConfig`) and library API (`run_experiment()` vs
> `run_study()`), but the CLI no longer has separate commands. The reasoning below about why
> unification was wrong specifically argued against CLI unification — which Option C resolves
> by making YAML the complexity signal instead of the command name.

**Date**: 2026-02-20
**Research:** peer tool survey in this section; Optuna, vLLM bench, lm-eval, W&B, Ray Tune,
MLflow, MLPerf, Kubeflow Katib, Hugging Face Lighteval

### Considered Options

Three structural options were evaluated:

| Option | Description |
|--------|-------------|
| **D1 (originally chosen, superseded)** | Keep `experiment` (atom) + `study` (investigation) as distinct concepts, two CLI commands, two config types |
| **D2** | Everything is a study; drop `ExperimentConfig` and `llem run` as separate concepts |
| **D3** | Everything is an experiment; rename `StudyConfig` → `ExperimentConfig`, drop `llem study` |

The trigger: Decision C's original phrasing — "`llem run` IS a degenerate study" — implied
conceptual equivalence. Decision D evaluated this and rejected it — but was subsequently
superseded by Option C in experiment-study-architecture.md which achieves CLI unification
while preserving the type-level and library API distinction.

### Decision

> **Superseded (2026-02-25):** See experiment-study-architecture.md. The resolution:
> unified `llem run` CLI (YAML determines scope), but distinct config types
> (`ExperimentConfig` / `StudyConfig`) and library functions (`run_experiment()` / `run_study()`).

D1 was originally chosen — reject unification. `experiment` and `study` remain distinct at the
type level, but share a unified CLI entry point.

---

### Why Unification Was Originally Rejected (preserved for record)

> **Note (2026-02-25):** The arguments below were written for D1 (two CLI commands). The
> resolution in experiment-study-architecture.md preserves the valid points (distinct config
> types, different output contracts, different hashing semantics) while unifying the CLI.
> The "two CLI commands serve two user mental models" argument (point 5) was specifically
> addressed by Option C's insight that the YAML file itself is the complexity signal.

#### 1. The configs are structurally different, not cosmetically

`ExperimentConfig` defines *what to measure* (~15 fields).

`StudyConfig` adds three structurally distinct components:
- A **sweep DSL** — `sweep: { pytorch.batch_size: [1, 8, 32] }` is a grid generator, not a list
  of configs. It is a domain-specific mini-language that doesn't belong in an experiment atom.
- An **execution protocol** — `n_cycles`, `cycle_order`. This is a study-level concept. A single
  experiment has no execution protocol; it runs once (with its own internal warmup/baseline).
- An **explicit experiment list** — `experiments: [...]` for non-Cartesian combinations.

Collapsing to one type means either: (a) every single-run YAML carries optional sweep/execution
fields as dead weight, or (b) the type is conditionally structured by presence of certain
fields — which is just the two-type split without names. Neither is an improvement.

#### 2. The output contracts are different

| Command | Output |
|---------|--------|
| `llem run` | `results/llama-3.1-8b_pytorch_2026-02-18T14-30.json` — flat JSON file |
| `llem study` | `results/batch-effects_2026-02-18T14-30/` — subdirectory with `study_summary.json` + N JSONs |

A study-of-1 should not produce a subdirectory when a user ran `llem run --model X`. A study that
sometimes produces a flat file is inconsistent. The output contract divergence is real.

#### 3. The hashing semantics are different

`measurement_config_hash` (per-experiment): identity of a single measurement point.
`study_design_hash` (per-study, excludes execution block): identity of a parameter space.
These serve different purposes and cannot be collapsed to a single hash concept.

#### 4. Peer evidence: tools that unify do so because their collection has no extra complexity

The tools with a single unified command or type — lm-eval (`lm-eval run`), MLflow (`mlflow run`)
— do so because their "collection" is purely organisational. lm-eval task groups have no sweep
grammar, no execution protocol, no measurement cycles. MLflow experiments are namespace metadata.

The tools whose "collection" adds genuine structural complexity (W&B sweeps, vLLM `bench sweep`,
Kubeflow Katib) all maintain separate concepts and commands. llem's `StudyConfig` is structurally
closer to these than to lm-eval groups.

#### 5. Two CLI commands serve two user mental models

`llem run --model X` → "measure this configuration now" (zero-config, onboarding path)
`llem run my-study.yaml` → "conduct this structured investigation" (power user, research path)

This is the `docker run` / `docker compose run` split: universally understood, intentional
ergonomic differentiation. Collapsing would make the quick one-off feel heavier than it is, and
the structured study feel more casual than it should.

#### 6. Domain terminology is correct

In research methodology: *experiment* = one controlled test; *study* = structured investigation
involving multiple experiments. Researchers write "we conducted a study of N experiments across
batch sizes." Calling a 50-configuration sweep an "experiment" is technically incorrect in the
domain. The naming reflects domain expertise, not arbitrary labelling.

---

### What Would Have Been Lost Under D2/D3 (preserved for record)

| Loss | Impact |
|------|--------|
| Zero-config entry point `llem run --model X` | Critical — primary onboarding path for new users |
| Type safety: `run_experiment() -> ExperimentResult` | High — unambiguous return type |
| Distinct flat-file vs directory output contracts | High — different filesystem affordances |
| Sweep DSL scoped to StudyConfig only | Medium — sweep grammar doesn't bleed into single-experiment context |
| Precise research vocabulary in docs and papers | Medium — "study of N experiments" is standard in research communication |

> **Note (2026-02-25):** Option C (experiment-study-architecture.md) preserves all of the above
> while unifying the CLI. The type-level distinction, return type safety, and output contract
> differences are all maintained.

---

### Conceptual Model (updated 2026-02-25)

```
ExperimentConfig  ←  the atom: one measurement point
StudyConfig       ←  the investigation: N atoms + measurement protocol + sweep grammar

llem run          ←  unified CLI entry point (YAML determines scope)
run_experiment()  ←  library entry point for single experiment → ExperimentResult
run_study()       ←  library entry point for study → StudyResult

The conceptual distinction is preserved at the type level and library API level.
The CLI is unified — the YAML file is the complexity signal, not the command name.
```

### Consequences

Positive: Domain terminology preserved at type level; type safety in library API
(`run_experiment() -> ExperimentResult`, `run_study() -> StudyResult`); distinct output contracts.

Negative / Trade-offs: Two concepts to document and explain; slightly more surface area for
new users to understand.

Neutral: CLI unified to `llem run` (YAML determines scope). The conceptual distinction lives
in the library API and config types, not the CLI command name.

---

## Related

- [architecture.md](architecture.md): Two sources + auto-capture model (field placement table updated)
- [experiment-study-architecture.md](experiment-study-architecture.md): Option C — supersedes Decision D
- [../research/16-execution-profiles-patterns.md](../research/16-execution-profiles-patterns.md): Peer tool survey underpinning Decision A
- [../designs/study-yaml.md](../designs/study-yaml.md): StudyConfig schema (updated)
- [../designs/config-model.md](../designs/config-model.md): SSOT for field placement and data flow
- [../designs/user-config.md](../designs/user-config.md): User config schema (updated)
- [../NEEDS_ADDRESSING.md](../NEEDS_ADDRESSING.md): Item 30
