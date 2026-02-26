# CLI Commands — v2.0 Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/cli-ux.md](../decisions/cli-ux.md)
**Target**: 2 commands + 1 flag (down from 15 in codebase, down from 9 in prior draft)

> **Superseded (2026-02-25):** The previous design (2026-02-19) specified 3 commands
> (`llem run`, `llem study`, `llem config`). The unified `llem run` command now handles
> both single experiments and multi-experiment studies — the YAML file determines scope.
> See [cli-ux.md Sub-decision A](../decisions/cli-ux.md) for the decision rationale.

## Full Command Signatures

### `llem run`

```
llem run [CONFIG] [OPTIONS]
```

Run inference experiment(s). YAML file determines scope: a single experiment config runs
one experiment; a study YAML with `sweep:` or `experiments:` blocks runs a multi-experiment
study. Auto-detects runner (local vs Docker based on backend mix).

**Arguments:**
```
CONFIG    Path to experiment.yaml or study.yaml (optional — if omitted, uses --model + defaults)
```

**Options:**
```
--model     -m   TEXT    HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)
--backend   -b   TEXT    Inference backend: pytorch|vllm|tensorrt
--dataset   -d   TEXT    Dataset alias or HuggingFace path (default: aienergyscore)
-n               INT     Number of prompts (default: 100)
--batch-size     INT     Batch size (default: 1)
--precision      TEXT    fp32|fp16|bf16|int8 (default: bf16)
--output    -o   PATH    Output directory (default: results/)
--dry-run        FLAG    Validate config and show experiment plan; do not run
--cycles         INT     Override n_cycles (study mode)
--no-gaps        FLAG    Disable thermal gap timers (study mode)
--order          TEXT    Cycle order: sequential|interleaved|shuffled (study mode)
--resume         PATH    Resume interrupted study from directory
```

**Zero-config invocation (single experiment):**
```bash
llem run --model meta-llama/Llama-3.1-8B
# → defaults to pytorch backend (sensible default even when multiple installed)
# → uses defaults: aienergyscore dataset, n=100, batch_size=1, bf16
# → output: results/llama-3.1-8b_pytorch_2026-02-18T14-30/result.json
```

**Config-driven invocation (single experiment):**
```bash
llem run experiment.yaml
llem run experiment.yaml --batch-size 32  # CLI overrides YAML
```

**Study invocation (multi-experiment):**
```bash
llem run study.yaml                        # YAML with sweep/experiments block
llem run study.yaml --dry-run              # preview grid, validate, estimate VRAM
llem run study.yaml --cycles 5 --order shuffled   # rigorous: 5 cycles, shuffled order
llem run study.yaml --cycles 1 --no-gaps   # quick single-cycle study
```

> **Superseded (2026-02-25):** `--profile quick|publication` flag removed. 0/5 peer tools
> use named rigour profiles — all use individual flags. Statistical rigour settings
> (`n_cycles`, `cycle_order`, `gap_seconds`) live in study YAML `execution:` block;
> CLI flags override. Effective values recorded in results.

**Auto-runner logic (studies):**
```
study.yaml contains multiple backends (pytorch + vllm)?
  → Docker required (backends are process-incompatible)
  → Fail with helpful error if Docker unavailable
study.yaml contains single backend?
  → Local by default
  → Docker if `runner: docker` in YAML or user config
```

**Library equivalent:**
```python
import llenergymeasure as llem

# Single experiment — returns ExperimentResult
result = llem.run_experiment(model="meta-llama/Llama-3.1-8B", backend="pytorch")
result = llem.run_experiment(ExperimentConfig.from_yaml("exp.yaml"))

# Study — returns StudyResult
result = llem.run_study("study.yaml")
result = llem.run_study(StudyConfig(experiments=[...]))
```

**Peer reference:** lm-eval pattern — `lm-eval --model hf --tasks hellaswag` works with
minimal args; `lm-eval --config eval_config.yaml` for full control. Hydra multirun pattern
for parameter sweeps. W&B sweeps pattern for grid/random sweep definition in single YAML.

---

### `llem config`

```
llem config [--verbose]
```

Passive environment snapshot + user config display. Shows GPU, installed backends, Docker
availability, active user config, and next-step guidance. Non-blocking (informational only,
no file writes in v2.0). Does not replace pre-flight.

**Options:**
```
--verbose    FLAG    Show full environment details (all GPU properties, all config values)
```

`--init` flag planned for a later v2.0 milestone (interactive config wizard).

**Peer reference:** `flutter doctor`, `dbt debug`, `gh config list`. Not found in lm-eval,
MLflow, Optimum-Benchmark (they omit it entirely). Justified here by multi-backend complexity.

---

### `--version` flag

```
llem --version
```

Shows version + installed backend versions. Implemented via Typer `version_option()`.

```
llenergymeasure 2.0.0
  pytorch:   transformers 4.47.0 / torch 2.4.0
  zeus:      0.13.1
```

**Peer reference:** `gh --version`, `cargo --version`, `pip --version` — all flags, not
subcommands.

---

## Rename

`lem` → `llem` at v2.0. Clean break — no alias, no shim.

```toml
# pyproject.toml
[project.scripts]
llem = "llenergymeasure.cli:app"
```

## Example YAML Files (Ship with Package)

```
src/llenergymeasure/
  examples/
    experiment.yaml.example    # template single experiment config
    study.yaml.example         # template study with sweep
```

Users copy these as starting points. No `llem init` command.

## Milestone Phasing (All v2.0)

> **Updated (2026-02-25):** All features below are v2.0 scope, delivered across milestones.
> No separate v2.2 — Docker multi-backend and study resume are later v2.0 milestones.

| Command | Milestone | Purpose |
|---------|-----------|---------|
| `llem run --resume study-dir/` | Later v2.0 milestone | Resume interrupted study |
| `llem config --init` | Later v2.0 milestone | Interactive wizard to write user config |
| `llem results push <file>` | Post-v2.0 (v3.0+) | Upload result to central DB |
