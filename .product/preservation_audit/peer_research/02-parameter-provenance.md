# Peer Research: Parameter Provenance Tracking

> Generated 2026-02-26. Peer evidence for preservation audit item P-07.

Our v1.x tool has ~230 LOC implementing a `ParameterSource` enum
(`PYDANTIC_DEFAULT`, `PRESET`, `CONFIG_FILE`, `CLI`) with a 4-layer resolution chain.
Each parameter's source is stored per-field in `ExperimentResult` so researchers can
answer "where did this value come from?" This document assesses whether that pattern
is standard practice, over-engineering, or somewhere in between.

---

## Evidence Per Tool

### 1. Hydra / OmegaConf (facebookresearch)

**Does it track where each config value came from?** Partially — at the file level, not per-field.

Hydra saves three files in its `.hydra/` output directory per run:

| File | Contents |
|------|----------|
| `config.yaml` | The fully resolved/merged configuration |
| `overrides.yaml` | Only the CLI overrides applied to that run |
| `hydra.yaml` | Hydra's own internal configuration |

This means a researcher can diff `config.yaml` against `overrides.yaml` to infer which
values came from CLI, but the separation is **file-level**, not per-field. There is no
metadata on each config node saying "this came from defaults.yaml, this from the CLI."

OmegaConf (the underlying config library) provides `is_missing()` and
`is_interpolation()` node queries, and supports merge semantics with documented
precedence (defaults list → config groups → CLI overrides). But it does **not** track
which merge pass introduced each value. No per-field `source` attribute exists.

**Granularity**: File-level (overrides.yaml vs config.yaml). Not per-field.
**In results?**: The `.hydra/` directory sits alongside outputs. Not embedded in results.
**Precedence documented?**: Yes — Defaults List documentation explains composition order.

**Sources**:
- [Hydra Output/Working Directory](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/)
- [Hydra Configure Hydra Overview](https://hydra.cc/docs/configure_hydra/intro/)
- [OmegaConf Usage Docs](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html)

---

### 2. lm-eval (EleutherAI)

**Does it track where each config value came from?** No.

lm-eval saves a `results.json` containing the resolved task configuration, model args,
metric values, and version numbers per task. The `--show_config` flag prints the full
`TaskConfig` for reproducibility. However, no field in the output distinguishes
"this value was a default" from "this value was user-specified."

The reproducibility strategy is: save the full resolved config + commit hash + task
YAML version number. A researcher can reconstruct what happened by comparing the saved
config against the task YAML defaults, but the tool does not do this comparison for them.

**Granularity**: None. Resolved config only.
**In results?**: Full resolved config saved; no provenance metadata.
**Precedence documented?**: CLI args override config file. Not formally documented as a chain.

**Sources**:
- [lm-eval interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- [lm-eval task_guide.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

---

### 3. optimum-benchmark (Hugging Face)

**Does it track where each config value came from?** No.

optimum-benchmark uses Hydra for configuration and saves a `benchmark.json` containing
both the benchmark report and the resolved configuration. Since it sits on top of Hydra,
it inherits Hydra's `.hydra/overrides.yaml` artifact, but adds no provenance layer of
its own. The `BenchmarkConfig` dataclasses have defaults but no mechanism to record
which values were user-overridden.

**Granularity**: None beyond what Hydra provides (file-level).
**In results?**: Resolved config in `benchmark.json`. No per-field source.
**Precedence documented?**: Inherited from Hydra conventions.

**Sources**:
- [optimum-benchmark README](https://github.com/huggingface/optimum-benchmark/blob/main/README.md)
- [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark)

---

### 4. MLflow

**Does it track where each config value came from?** No per-field provenance.

MLflow's `log_param()` / `log_params()` API stores parameter key-value pairs as flat
strings. All parameters are stored uniformly — there is no field or tag distinguishing
"user explicitly set this" from "framework default." The `autolog()` feature logs
hyperparameters including defaults provided by the ML library, but tags the **run** (not
individual params) with `mlflow.autologging: sklearn` (or similar). This tells you the
run used autologging, not which specific params were defaults.

MLflow does capture contextual metadata (git hash, source file, software versions) but
this is run-level provenance, not parameter-level.

**Granularity**: None. All params stored uniformly.
**In results?**: Params stored in tracking server; no source metadata per param.
**Precedence documented?**: N/A — MLflow is a logging system, not a config resolution system.

**Sources**:
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Autolog](https://mlflow.org/docs/latest/ml/tracking/autolog/)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html)

---

### 5. Weights & Biases (W&B)

**Does it track where each config value came from?** Partially — internal locking, not
exposed per-field.

W&B has an internal parameter locking mechanism: when a sweep controls a parameter,
updates from user code are blocked with a warning: `"Config item 'learning_rate' was
locked by 'sweep' (ignored update)"`. This implies W&B internally tracks **who owns
each config key** (sweep vs user code), but this is:

- Used for conflict prevention, not for provenance reporting
- Not exposed in the run config or API as a per-field "source" attribute
- Not saved in results for post-hoc analysis

The documented precedence is: `wandb.init(config=...)` overrides `config-defaults.yaml`;
programmatic settings override environment variables. But this is operational precedence,
not recorded provenance.

**Granularity**: Internal per-key locking (sweep vs user). Not exposed in results.
**In results?**: Only the resolved config is saved to the run.
**Precedence documented?**: Yes — init args > env vars > config-defaults.yaml.

**Sources**:
- [W&B Configure Experiments](https://docs.wandb.ai/models/track/config)
- [W&B Env Var Override Docs](https://docs.wandb.ai/support/environment_variables_overwrite_parameters/)
- [W&B Sweep Lock Issue #4168](https://github.com/wandb/wandb/issues/4168)

---

### 6. Pydantic v2 (and pydantic-settings)

**Does it track where each config value came from?** Partially — set vs unset only.

Pydantic v2 provides `model_fields_set`: a `set[str]` of field names that were
**explicitly provided** during instantiation. Fields that used their default value are
absent from this set. This is a binary distinction (set / not-set), not a multi-source
provenance chain.

```python
class Config(BaseModel):
    batch_size: int = 1
    model: str

c = Config(model="gpt2")
c.model_fields_set  # {'model'}
# batch_size is absent — used default
```

`pydantic-settings` adds multi-source resolution (env vars, .env files, init kwargs,
secrets directory) with a configurable priority order via `settings_customise_sources()`.
However, **after resolution, there is no per-field record of which source won**. The
priority chain is documented as a resolution algorithm, not as recorded metadata.

**Granularity**: Binary per-field (set vs default) via `model_fields_set`. No multi-source tracking.
**In results?**: `model_fields_set` is an instance attribute, not serialised by default.
**Precedence documented?**: Yes — `pydantic-settings` documents source priority order.

**Sources**:
- [Pydantic Models — model_fields_set](https://docs.pydantic.dev/latest/concepts/models/)
- [Pydantic Settings Management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

---

### 7. Click (pallets) — Python CLI framework

**Does it track where each config value came from?** Yes — per-parameter, 5 sources.

Click (which underlies Typer, our CLI framework) provides `ParameterSource`, an enum
with per-parameter source tracking since v8.0:

| Enum Member | Meaning |
|-------------|---------|
| `COMMANDLINE` | Value provided via CLI args |
| `ENVIRONMENT` | Value from environment variable |
| `DEFAULT_MAP` | Value from `Context.default_map` |
| `DEFAULT` | Built-in default for the parameter |
| `PROMPT` | Value from interactive prompt |

Usage: `ctx.get_parameter_source("port")` returns the source for that specific parameter.

This is the **closest peer precedent** to our `ParameterSource` enum. Click tracks
provenance per-parameter at the CLI layer. However:

- It is used internally for CLI behaviour (e.g., distinguishing explicit `--port 8080`
  from the default), not serialised into output files.
- It does not extend beyond the CLI layer — config file values are not distinguished
  from defaults unless routed through `default_map`.

**Granularity**: Per-parameter, 5 discrete sources.
**In results?**: No — internal to Click's context, not serialised.
**Precedence documented?**: Implicitly via enum ordering and docs.

**Sources**:
- [Click Commands and Groups — get_parameter_source](https://click.palletsprojects.com/en/stable/commands-and-groups/)
- [Click API — ParameterSource](https://click.palletsprojects.com/en/stable/api/)

---

### 8. Terraform / Pulumi (Infrastructure-as-Code)

**Does it track where each config value came from?** Terraform: documented precedence,
no runtime tracking. Pulumi: no.

**Terraform** has a well-documented 6-level variable precedence chain:

1. `-var` and `-var-file` flags (highest, last wins)
2. `*.auto.tfvars` / `*.auto.tfvars.json` (lexical order)
3. `terraform.tfvars.json`
4. `terraform.tfvars`
5. `TF_VAR_<name>` environment variables
6. `default` in `variable` block (lowest)

However, Terraform does **not** expose "where did this value come from" in plan output
or state. The precedence is a resolution algorithm, not recorded provenance. Users must
manually reason about which source won. There is no `terraform show --provenance` or
equivalent.

**Pulumi** stores config in `Pulumi.<stack>.yaml` with project-level defaults in
`Pulumi.yaml`. Provider config can come from stack config or explicit `Provider()`
arguments, with provider args taking precedence. No per-field provenance tracking exists.

**Granularity**: None at runtime. Precedence is a documented algorithm.
**In results?**: Not in state or plan output.
**Precedence documented?**: Yes — Terraform's 6-level chain is extensively documented.

**Sources**:
- [Terraform Variables](https://developer.hashicorp.com/terraform/language/values/variables)
- [Terraform Variable Precedence Guide](https://learning-ocean.com/tutorials/terraform/terraform-variable-precedence/)
- [Pulumi Configuration](https://www.pulumi.com/docs/iac/concepts/config/)

---

### 9. 12-Factor App / python-dotenv

**Does it track where each config value came from?** No — pattern only.

The 12-Factor App methodology establishes the principle: environment variables > config
files > defaults. `python-dotenv` implements this with `load_dotenv(override=False)`:
env vars win unless `override=True`.

This is a **precedence convention**, not a tracking mechanism. After resolution, the
application has a value but no record of which source provided it.

**Granularity**: None.
**In results?**: N/A.
**Precedence documented?**: Yes — 12-Factor factor III. python-dotenv documents `override` semantics.

**Sources**:
- [python-dotenv PyPI](https://pypi.org/project/python-dotenv/)
- [python-dotenv GitHub](https://github.com/theskumar/python-dotenv)

---

## Summary Table

| Tool | Per-field provenance? | Stored in results? | Granularity | Precedence documented? |
|------|----------------------|-------------------|-------------|----------------------|
| **Hydra/OmegaConf** | No (file-level only) | `.hydra/overrides.yaml` alongside output | File-level | Yes |
| **lm-eval** | No | Full resolved config only | None | Informal |
| **optimum-benchmark** | No (inherits Hydra) | Resolved config in JSON | None | Via Hydra |
| **MLflow** | No | Flat key-value params | None | N/A |
| **W&B** | Internal (sweep lock) | Resolved config only | Per-key internal | Yes |
| **Pydantic v2** | Binary (set/unset) | `model_fields_set` (not serialised) | Per-field binary | Yes |
| **Click** | **Yes** (5 sources) | No (internal to context) | **Per-parameter** | Implicit |
| **Terraform** | No | Not in state/plan | None | **Yes (6-level)** |
| **Pulumi** | No | No | None | Partial |
| **12-Factor/dotenv** | No | N/A | None | Convention only |
| **LLenergyMeasure v1.x** | **Yes** (4 sources) | **Yes** (in ExperimentResult) | **Per-field** | **Yes (4-layer)** |

---

## Analysis

### What the ecosystem actually does

The overwhelming pattern is: **document the precedence algorithm, save the resolved
config, do not track per-field provenance.** This is true for Hydra, lm-eval,
optimum-benchmark, MLflow, W&B, Terraform, and Pulumi.

The two partial exceptions are:

1. **Click's `ParameterSource`** — genuine per-parameter source tracking with a 5-member
   enum. This is used internally for CLI behaviour but never serialised into output files.
   It validates the *concept* as sound engineering (the Click maintainers deemed it worth
   implementing), but not the practice of *persisting it in results*.

2. **Hydra's `overrides.yaml`** — saves CLI overrides as a separate artifact, enabling
   post-hoc diff against the resolved config. This is the closest to "provenance in
   outputs" but is coarse-grained (file-level, not per-field).

3. **Pydantic's `model_fields_set`** — tracks which fields were explicitly provided vs
   defaulted. Binary only (no multi-source chain), and not serialised.

### What makes our case different

Our tool measures the effect of implementation parameters on inference efficiency. A
researcher running a study needs to answer: "was `batch_size=8` something I deliberately
set, or did a preset sneak it in?" This is not a debugging convenience — it is a
**scientific audit requirement** when the parameter values *are the independent variables*.

In experiment tracking tools (MLflow, W&B), the parameters are logged *by the user* and
are not the subject of study. In config management tools (Hydra, Terraform), the values
drive infrastructure/training and the resolution history is transient. In our case, the
config parameters *are what we are measuring*, making their provenance part of the
experimental record.

### Risk of dropping it

- No peer tool does full per-field provenance in results, so there is no ecosystem
  expectation for it.
- However, Click validates the enum pattern (4 sources) as sound, and Hydra validates
  the "save overrides separately" principle.
- The 230 LOC is modest. The `ResolvedConfig` wrapper, the 4-layer resolution, and the
  `to_summary_dict()` serialisation are well-scoped.
- The display layer (`display_non_default_summary()` in `summaries.py`) uses provenance
  to show researchers a compact "what changed from defaults" view — a genuine UX
  differentiator.

---

## Recommendation

**Incorporate — keep at v2.0.**

Our per-field parameter provenance tracking is an outlier relative to peers, but it is a
*justified* outlier for a tool where config parameters are the independent variables under
study. The pattern is validated by Click's `ParameterSource` (same concept, different
scope) and by Hydra's `overrides.yaml` (same principle of preserving override history).

Specific actions:

1. **Keep** `ParameterSource` enum, `ParameterProvenance` model, and `ResolvedConfig`
   wrapper. The ~230 LOC is proportionate to the value.
2. **Simplify the v2.0 layer count** if presets are dropped or merged into the YAML
   config mechanism. If v2.0 has no presets layer, the enum shrinks to 3 sources
   (`DEFAULT`, `CONFIG_FILE`, `CLI`) — which maps directly to Click's model.
3. **Continue embedding provenance in `ExperimentResult`** via `to_summary_dict()`. This
   is the differentiator: peers save resolved config; we save resolved config + source
   annotations.
4. **Leverage Pydantic's `model_fields_set`** during resolution as a cheaper alternative
   to the current `compare_dicts()` approach. Pydantic already knows which fields were
   explicitly set — use it rather than diffing flattened dicts.
5. **Document the precedence chain** in the config design doc (matching Terraform's
   practice of explicit precedence documentation).
