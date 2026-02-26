# Output Storage: Results Directory Structure

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

All experiments produce a timestamped subdirectory: `results/{name}_{timestamp}/`. Single experiment and study use the same structure — always a subdirectory. Collision policy: append monotonic suffix (`_2`, `_3`), never overwrite. Human-readable filenames — `ls results/` replaces a dedicated `llem results` command. Sidecar Parquet files for time-series data (power, energy, throughput) alongside JSON result.

---

## Context

`llem run` writes result files to disk (single experiment or study, depending on YAML content).
Three questions required decisions:

- **J1**: Should single-experiment output be a flat file or a per-experiment subdirectory?
- **J2**: Should study output use a flat directory or a timestamped subdirectory?
- **J3**: What should happen when an output path already exists (collision policy)?

The guiding principle from `decisions/cli-ux.md` is the `ls results/` idiom: results must be
navigable without a separate command or database. Human-readable filenames replace `llem results list`.

---

## J1 — Single-Experiment Output: Flat File

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| Flat file in results dir (original, superseded) | `ls results/` shows each experiment directly; matches lm-eval's single-result-JSON pattern | Inconsistent with study output; can't predict output location; breaks when sidecar files added |
| **Always subdirectory (chosen, revised 2026-02-25)** | Consistent structure for single and study; predictable output location; supports sidecar Parquet time-series files; matches Hydra pattern | One level of nesting even for single experiments |

> **Superseded (2026-02-25):** Flat file approach replaced with always-subdirectory. Reasons:
> (1) time-series Parquet sidecar files are now v2.0 scope — single experiments produce multiple
> artefacts, so subdirectories are needed regardless; (2) output contract divergence (flat vs
> subdir) meant users couldn't predict output location without knowing YAML content; (3) Hydra
> always produces timestamped subdirectories and users find this predictable.

### Decision

Every experiment run produces a timestamped subdirectory:

```
results/
  llama-3.1-8b_pytorch_2026-02-18T14-30/
    result.json                    # ExperimentResult
    timeseries.parquet             # power, energy, throughput, temperature
  llama-3.1-8b_vllm_2026-02-18T15-45/
    result.json
    timeseries.parquet
```

Rationale: Time-series data (power, cumulative energy, throughput, temperature at 1–10 Hz)
requires sidecar files — a single JSON file is no longer sufficient. Consistent subdirectory
structure for all runs (single and study) eliminates the output contract divergence.

### Consequences

Positive: Predictable output location; supports multi-artefact results; `ls results/` still works.
Negative / Trade-offs: One level of nesting even for simple single experiments.
Neutral: Study output (J2) continues to use subdirectories — now consistent with single experiments.

---

## J2 — Study Output: Timestamped Subdirectory

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`{study_name}_{timestamp}/` subdir (chosen)** | Groups all related experiment JSONs + summary + manifest; `ls results/` shows one entry per study run; matches MLflow run-per-directory pattern | One extra level of nesting |
| Flat directory — all study experiment files alongside single-experiment files | No nesting | `ls results/` becomes unnavigable when a study produces 12+ files; no way to distinguish study files from single-run files |
| `results/studies/{study_name}_{timestamp}/` | Separates studies from single runs | Extra nesting (`studies/`) for no benefit |
| `results/{study_name}/` (no timestamp) | Simple | Ambiguous if same study runs twice; second run overwrites first |

**Rejected (2026-02-19):** `results/studies/batch-size-effects_2026-02-18T14-30/` — extra
nesting (`studies/`) for no benefit.

**Rejected (2026-02-19):** `results/batch-size-effects/` (no timestamp) — ambiguous if same
study runs twice; overwrites previous.

### Decision

We will create a subdirectory under `results/` named `{study_name}_{timestamp}/`.

```
results/
  batch-size-effects_2026-02-18T14-30/
    study_manifest.json                          ← in-progress checkpoint (written during run)
    study_summary.json                           ← StudyResult (written at end)
    llama-3.1-8b_pytorch_bf16_batch1_2026-...json
    llama-3.1-8b_pytorch_bf16_batch8_2026-...json
    llama-3.1-8b_vllm_bf16_seqs64_2026-...json
    ...
  precision-comparison_2026-02-19T09-00/
    ...
```

**Study name:** From `name:` field in study YAML. If omitted, auto-generated: `study_{timestamp}`.
Descriptive filenames are encouraged — `batch-size-effects.yaml` → `batch-size-effects_...` subdir.

Rationale: Study output is a group of related files. Subdirectory keeps them together and makes
`ls results/` navigable (one entry per study run, not N experiment files interleaved). MLflow
pattern: each "run" gets its own subdirectory under the experiment directory.

### Consequences

Positive: `ls results/` shows one logical entry per study; study artefacts grouped.
Negative / Trade-offs: Two different output structures (flat for single experiment, subdir for
study) — users must know which YAML they ran to know where to look.
Neutral: `study_manifest.json` written progressively during run enables resumability checking.

---

## J3 — Overwrite Policy: Counter Suffix on Collision

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Append `_1`, `_2`, etc. on collision (chosen)** | Non-destructive; silent; reproducible scripting works (same YAML twice for comparison); matches Hydra's counter pattern | Adds a suffix that was not in the user's original intent |
| Hard error on collision | Explicit | Breaks reproducible scripting (running same YAML twice); unusable in CI |
| Silent overwrite | Simple | Loses data from the previous run with no warning |

**Rejected (2026-02-19):** Hard error on collision — stops a run because `results/` already
has a file from a previous run; breaks reproducible scripting.

**Rejected (2026-02-19):** Silent overwrite — loses data from the previous run with no warning.

### Decision

If the target path already exists, append `_1`, `_2`, etc. to avoid collision. Never silently
overwrite.

```
# First run:   results/batch-size-effects_2026-02-18T14-30/  ← created
# Second run (same second, edge case):
#              results/batch-size-effects_2026-02-18T14-30_1/ ← suffixed
```

Single experiments: ISO timestamp is second-granular — collisions are rare but possible
(same model + backend + second in a script). Suffix `_1`, `_2` handles it.
Studies: Same policy applies to the subdir name.

Rationale: Hydra uses timestamped output dirs (`outputs/2026-02-18/14-30-00/`) with
same-second collision handled by appending a counter. This is the dominant pattern in ML tooling.

### Consequences

Positive: Non-destructive; reproducible scripting works; consistent with industry pattern.
Negative / Trade-offs: Files with `_1` suffix are slightly less predictable to parse
programmatically. Mitigation: `StudyResult.output_dir` always contains the actual path used.
Neutral: In practice, second-granular timestamps make collisions rare outside scripts.

---

## Filename Format

### Single experiment

```
{model_slug}_{backend}_{timestamp}.json
```

Examples:
```
llama-3.1-8b_pytorch_2026-02-18T14-30.json
mixtral-8x7b_vllm_2026-02-18T15-45.json
llama-3.1-70b_tensorrt_2026-02-19T09-12.json
```

`model_slug`: HuggingFace ID with `/` → `-` and lowercase. `meta-llama/Llama-3.1-8B` → `llama-3.1-8b`.
`timestamp`: ISO 8601, minute-granular (`2026-02-18T14-30`), colons replaced with hyphens for
filesystem compatibility.

### Study experiment (within study subdir)

```
{model_slug}_{backend}_{sweep_params}_{timestamp}.json
```

`sweep_params`: key-value pairs of swept dimensions, abbreviated. `precision=bf16,batch=8`.
Length limit: 80 chars total filename. Truncate sweep_params if needed.

### Study summary files

```
{study_name}_{timestamp}/study_summary.json    ← StudyResult
{study_name}_{timestamp}/study_manifest.json   ← StudyManifest (checkpoint, written during run)
```

---

## User Config Integration

`output.results_dir` in `~/.config/llenergymeasure/config.yaml` sets the base directory.
`./results/` is the default if not configured.

CLI `--output PATH` overrides the user config for that invocation.
Library API `output_dir` parameter overrides explicitly.

```yaml
# ~/.config/llenergymeasure/config.yaml
output:
  results_dir: /scratch/my_username/llem_results   # HPC scratch
```

---

## Future Considerations

> **Research note (2026-02-25):** `.planning/research/FEATURES.md` (Gap 2) identifies power
> time-series capture as a table-stakes feature for energy measurement tools. If/when power
> traces are captured (e.g. via Zeus `PowerMonitor`), they produce high-frequency data (1–10 Hz)
> that does not belong in the main JSON result. A separate Parquet or CSV file per experiment
> would be needed, which would require revisiting the flat-file decision for single experiments
> (J1) — a per-experiment subdirectory may become necessary to group the JSON result with its
> power trace file.

---

## Related

- [cli-ux.md](cli-ux.md): Human-readable filenames + `ls results/` idiom
- [local-result-navigation.md](local-result-navigation.md): How users discover and inspect results
- [../designs/user-config.md](../designs/user-config.md): `output.results_dir` user config field
- [../designs/library-api.md](../designs/library-api.md): Library API output path parameter
- [../designs/result-schema.md](../designs/result-schema.md): `StudyResult` / `ExperimentResult` schema
