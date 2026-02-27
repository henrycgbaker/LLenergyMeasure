---
phase: 07-cli
plan: 02
subsystem: cli
tags: [cli, run-command, tqdm, error-handling, typer]
dependency_graph:
  requires: [07-01]
  provides: [llem-run-command]
  affects: [cli/__init__.py]
tech_stack:
  added: []
  patterns: [typer-command-registration, deferred-import-for-torch, tqdm-spinner, exit-codes]
key_files:
  created:
    - src/llenergymeasure/cli/run.py
    - tests/unit/test_cli_run.py
  modified:
    - src/llenergymeasure/cli/__init__.py
decisions:
  - "Module-level imports for all non-torch symbols — enables patch() in tests; torch loads on-call not on import"
  - "CliRunner(mix_stderr=True) in tests — typer's CliRunner doesn't reliably separate stderr without it"
  - "SIGINT handled as both signal handler and KeyboardInterrupt catch — belt-and-suspenders for SystemExit(130)"
metrics:
  duration: "5 min"
  completed: "2026-02-27"
  tasks_completed: 2
  files_created: 2
  files_modified: 1
---

# Phase 7 Plan 2: llem run Command Summary

`llem run` command wiring `load_experiment_config()` + `run_experiment()` to CLI flags, full error exit codes, tqdm spinner, and dry-run path.

## What Was Built

### `src/llenergymeasure/cli/run.py` (167 LOC)

Implements the `llem run` Typer command with:

- **10 CLI flags**: `[CONFIG]`, `--model`, `--backend`, `--dataset`, `-n`, `--batch-size`, `--precision`, `--output`, `--dry-run`, `--quiet`, `--verbose`
- **SIGINT handler**: installs `signal.signal(SIGINT, ...)` raising `SystemExit(130)`; `KeyboardInterrupt` also caught as belt-and-suspenders
- **Error dispatch** via try/except in `run()`:
  - `ConfigError` → exit 2
  - `PreFlightError | ExperimentError | BackendError` → exit 1
  - `ValidationError` → exit 2 (formatted with `format_validation_error`)
  - `KeyboardInterrupt` → `SystemExit(130)`
- **`_run_impl()`** helper with:
  - CLI overrides dict built from non-None flags only; `batch_size` maps to `pytorch.batch_size` (dotted key for `_unflatten()`)
  - Early `ConfigError` if no config path and no `--model`
  - **Dry-run branch**: calls `estimate_vram()` + `print_dry_run()`, returns (exit 0)
  - **Run branch**: `print_experiment_header()` → tqdm spinner → `run_experiment()` → `print_result_summary()`; optional `result.save()` if `output_dir` set
- **tqdm spinner**: `total=None` (indeterminate), `disable=quiet or not sys.stderr.isatty()` satisfies `--quiet` and non-TTY suppression

### `src/llenergymeasure/cli/__init__.py` (updated)

Replaced skeleton comment with registration of the `run` command:

```python
from llenergymeasure.cli.run import run as _run_cmd
app.command(name="run", help="Run an LLM efficiency experiment")(_run_cmd)
```

### `tests/unit/test_cli_run.py` (190 LOC, 11 tests)

Full test coverage of error paths and flag behaviour:

| Test | What it verifies |
|------|-----------------|
| `test_run_help` | `--help` exits 0, shows --model/--backend/--dry-run |
| `test_run_version` | `--version` exits 0, shows "llem v" |
| `test_run_no_args_exits_2` | No config/no model → ConfigError → exit 2 |
| `test_run_config_error_exits_2` | Mocked ConfigError → exit 2, "ConfigError" in output |
| `test_run_validation_error_exits_2` | Misspelled backend "pytorh" → ValidationError → exit 2 |
| `test_run_preflight_error_exits_1` | Mocked PreFlightError → exit 1 |
| `test_run_experiment_error_exits_1` | Mocked ExperimentError → exit 1 |
| `test_run_dry_run_exits_0` | `--dry-run` → exit 0, `print_dry_run` called |
| `test_run_dry_run_calls_estimate_vram` | `estimate_vram(config)` and `get_gpu_vram_gb()` called |
| `test_run_quiet_flag_accepted` | tqdm called with `disable=True` when `--quiet` |
| `test_run_success_prints_summary` | `print_result_summary(result)` called on success |

## Deviations from Plan

### Auto-fixed: Module-level imports for testability (Rule 2)

**Found during:** Task 2 implementation
**Issue:** Plan specified deferred imports inside `_run_impl`. `patch("llenergymeasure.cli.run.load_experiment_config")` fails with `AttributeError` when the symbol isn't a module-level attribute.
**Fix:** Moved all non-torch imports (`load_experiment_config`, `run_experiment`, display functions, `estimate_vram`) to module level. `run_experiment` importing `from llenergymeasure` doesn't load torch at import time — torch loads only when `_run()` calls `get_backend()`. The original deferred import concern was overly conservative.
**Files modified:** `src/llenergymeasure/cli/run.py`

### Auto-fixed: CliRunner mix_stderr=True (Rule 1)

**Found during:** Task 2 test development
**Issue:** `typer.testing.CliRunner(mix_stderr=False)` doesn't reliably capture `print(..., file=sys.stderr)` in `.stderr` — the captured output is empty.
**Fix:** Used `CliRunner(mix_stderr=True)` and checked `.output` (combined stdout+stderr). This is the documented typer/click testing pattern for commands that write to both streams.
**Files modified:** `tests/unit/test_cli_run.py`

## Self-Check: PASSED

- `src/llenergymeasure/cli/run.py` — FOUND
- `tests/unit/test_cli_run.py` — FOUND
- `src/llenergymeasure/cli/__init__.py` — modified (verified)
- Commit `e3073cf` (feat: implement llem run command) — FOUND
- Commit `9a4cb75` (test: add 11 unit tests) — FOUND
- 11/11 tests passing
