# N-X06: Batch Processing Command

**Module**: `src/llenergymeasure/cli/batch.py`
**Risk Level**: MEDIUM
**Decision**: Pending — this command IS the functionality of `llem study`, but implements it as a flat batch runner rather than a sweep-aware study runner. The decision is whether to (a) merge its functionality into `llem study`, (b) cut it entirely (functionality subsumed), or (c) keep it as a power-user shortcut.
**Planning Gap**: `decisions/cli-ux.md` explicitly lists `llem batch` as cut ("Subsumed by `llem campaign`"). However, `batch.py` remains in the codebase and the parallel execution capability it provides (`--parallel N`) has no equivalent in the current `llem campaign` implementation.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/cli/batch.py`
**Key classes/functions**:
- `batch_run_cmd()` (line 18) — Typer command function

**Parameters**:
- `config_pattern: str` — glob pattern (e.g., `configs/*.yaml`)
- `dataset: str | None` — override dataset for all configs
- `sample_size: int | None` — override prompt count for all configs
- `parallel: int | None` — run N configs concurrently via `ProcessPoolExecutor`
- `dry_run: bool` — list configs without running

**Behaviour**:
1. Expands glob pattern via `glob.glob(config_pattern)` (line 52)
2. Validates ALL configs first via `load_config(Path(path))` (line 63) — upfront validation before any runs
3. Lists validation results: green tick for valid, red cross for invalid
4. If `dry_run`: prints count and exits (line 78–80)
5. Sequential mode: loops through `valid_configs`, calls `run_config()` which spawns `python -m llenergymeasure.cli experiment <path>` as a subprocess
6. Parallel mode: uses `ProcessPoolExecutor(max_workers=parallel)` (line 107) with `as_completed()` for non-blocking result collection
7. Summary: `N succeeded, M failed` (line 130)

The `run_config()` inner function (line 83) constructs the command as `[sys.executable, "-m", "llenergymeasure.cli", "experiment", str(config_path)]` — it delegates to the existing `experiment` command.

Total: 134 lines.

## Why It Matters

This command provides two capabilities not yet confirmed in the `llem study` design:

1. **Glob-based config discovery**: users with `configs/llama-7b.yaml`, `configs/llama-13b.yaml`, `configs/mistral-7b.yaml` can run `llem batch 'configs/*.yaml'` without writing a study YAML. This is a lightweight alternative to study sweeps for ad-hoc batch execution.

2. **Parallel execution**: `--parallel N` runs N experiments concurrently. This is explicitly not in the `llem study` design (study runs experiments sequentially by design, for clean GPU state and thermal isolation). The parallel flag here sacrifices thermal isolation for wall-clock speed — a valid trade-off for CPU-only or multi-node scenarios.

The upfront validation-before-run pattern is valuable: it fails fast on config errors before any GPU time is spent.

## Planning Gap Details

`decisions/cli-ux.md` (line: "`llem batch` — Subsumed by `llem campaign`"): this cut was confirmed before `campaign` was renamed to `study`. The planning note assumes `llem study` handles the batch use case — but the v2.0 study design uses a `sweep:` grammar in a YAML file, not a glob pattern over existing config files.

The two models serve different workflows:
- `llem study study.yaml` — researcher writes a parameterised study definition with sweep grammar
- `llem batch 'configs/*.yaml'` — researcher runs a set of independently-maintained config files without a study YAML

These are not equivalent. The study sweep generates configs programmatically; the batch command runs pre-existing configs.

The `--parallel N` flag has no equivalent in `llem study` (by design — sequential is required for energy isolation). If `batch` is cut, parallel execution of multiple independent experiments is lost.

## Recommendation for Phase 5

**Decision required — three options**:

**Option A — Cut (confirmed by planning)**: Remove `batch.py`. Users who want to run multiple configs write a `study.yaml` with `experiments:` list (not `sweep:`). The glob shortcut is lost; parallel is lost. Accept the planning decision.

**Option B — Keep as undocumented power-user shortcut**: Retain `llem batch` but do not advertise it in docs or `llem --help`. It remains for users who know about it. Risk: adds CLI surface area inconsistent with the 3-command design principle.

**Option C — Merge glob support into `llem study`**: Allow `llem study 'configs/*.yaml'` where a glob resolves to a list of config files instead of requiring a `study.yaml`. This extends `llem study` without adding a new command. The parallel execution question remains open.

The upfront validation pattern (lines 59–76) and the `ProcessPoolExecutor` pattern (lines 104–115) are good implementations regardless of the final decision — if either feature is kept, re-use this code.
