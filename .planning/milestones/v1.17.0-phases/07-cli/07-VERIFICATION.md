---
phase: 07-cli
verified: 2026-02-27T12:00:00Z
status: passed
score: 13/14 must-haves verified
re_verification: false
notes: |
  CLI-03 "auto-detects single vs study YAML" is partially satisfied: single-experiment YAML
  is fully supported. Study YAML detection is deferred because run_study() intentionally
  raises NotImplementedError in M1 (Phase 3 verified design). This is an in-scope M1
  limitation, not a gap introduced by Phase 7.
human_verification:
  - test: "Run llem run --model gpt2 --backend pytorch with an actual GPU environment"
    expected: "Indeterminate tqdm spinner displays during measurement, result summary prints to stdout on completion"
    why_human: "GPU not available on host; tqdm TTY detection and spinner display cannot be verified programmatically"
  - test: "Pipe llem run output to a file: llem run --model gpt2 --backend pytorch > out.txt"
    expected: "tqdm spinner suppressed (non-TTY), only result summary appears in out.txt"
    why_human: "Non-TTY tqdm suppression requires a real terminal session; CliRunner is always non-TTY"
  - test: "Press Ctrl-C during llem run measurement"
    expected: "Process exits with code 130, stderr shows 'Interrupted.'"
    why_human: "SIGINT handling requires an interactive terminal session"
---

# Phase 7: CLI Verification Report

**Phase Goal:** Researchers interact with the tool entirely through `llem run` and `llem config` — plain text output, no Rich dependency, correct exit codes on all error paths, and a `--dry-run` that validates without running.
**Verified:** 2026-02-27
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `llem run` command exists and handles all 10 documented flags | VERIFIED | `run.py` 195 LOC, all flags present: `--model`, `--backend`, `--dataset`, `-n`, `--batch-size`, `--precision`, `--output`, `--dry-run`, `--quiet`, `--verbose` |
| 2 | `llem config` command exists and shows GPU, backends, energy, user config path | VERIFIED | `config_cmd.py` 184 LOC; GPU section via `_probe_gpu()`, backends via `importlib.util.find_spec`, energy priority Zeus>NVML>CodeCarbon, user config path via `get_user_config_path()` |
| 3 | `llem --version` exits 0 and prints version string | VERIFIED | `__init__.py` `version_callback` prints `f"llem v{__version__}"` and raises `typer.Exit()`; `test_run_version` passes |
| 4 | No Rich dependency in any new CLI file | VERIFIED | `grep -r "import rich"` on all 5 new CLI files returns nothing; `test_config_no_rich_import` passes via AST parse |
| 5 | All result output to stdout; progress/header to stderr | VERIFIED | `print_result_summary` uses bare `print()` (stdout); `print_experiment_header` uses `print(..., file=sys.stderr)`; `tqdm(file=sys.stderr)` |
| 6 | `--dry-run` validates config (Pydantic) and estimates VRAM, exits 0 | VERIFIED | `_run_impl` calls `load_experiment_config()` then `estimate_vram()` + `print_dry_run()`; `test_run_dry_run_exits_0` passes |
| 7 | ConfigError exits code 2; PreFlightError/ExperimentError exits code 1; SIGINT exits 130 | VERIFIED | `run()` dispatches: `ConfigError` → `typer.Exit(2)`, `(PreFlightError\|ExperimentError\|BackendError)` → `typer.Exit(1)`, `KeyboardInterrupt` → `SystemExit(130)`; all 4 tests pass |
| 8 | Pydantic ValidationError formatted with friendly header, exits 2 | VERIFIED | `format_validation_error()` produces "Config validation failed (N error(s)):" header with did-you-mean; `ValidationError` → `typer.Exit(2)`; `test_run_validation_error_exits_2` passes |
| 9 | `--quiet` suppresses tqdm spinner, keeps final summary | VERIFIED | `tqdm(disable=quiet or not sys.stderr.isatty())`; `test_run_quiet_flag_accepted` verifies `disable=True` in tqdm call kwargs |
| 10 | Result summary prints Energy/Performance/Timing sections to stdout | VERIFIED | `print_result_summary()` 55 LOC in `_display.py` with three labelled sections; all numeric values via `_sig3()` |
| 11 | FLOPs displayed with method/confidence from `process_results[0].compute_metrics` | VERIFIED | `print_result_summary` reads `result.process_results[0].compute_metrics.flops_method` and `.flops_confidence`; falls back to bare scalar if empty |
| 12 | VRAM estimation returns weights, KV cache, overhead, total in GB | VERIFIED | `estimate_vram()` returns `{"weights_gb", "kv_cache_gb", "overhead_gb", "total_gb"}` or `None`; `test_vram_dtype_bytes` passes |
| 13 | All 37 unit tests pass without GPU or network access | VERIFIED | `python3.10 -m pytest tests/unit/test_cli_display.py tests/unit/test_cli_run.py tests/unit/test_cli_config.py` → 37 passed in 1.85s |
| 14 | CLI-03 study YAML auto-detection (partial — single-experiment only) | PARTIAL | `llem run` fully handles single-experiment YAML and `--model` flag; study YAML detection deferred because `run_study()` intentionally raises `NotImplementedError` in M1 (Phase 3 verified design decision). The `llem run [CONFIG]` command syntax is fully implemented. |

**Score:** 13/14 truths fully verified; 1 partial (in-scope M1 limitation)

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|-------------|--------|---------|
| `src/llenergymeasure/cli/_display.py` | 80 | 296 | VERIFIED | All 7 functions present: `_sig3`, `_format_duration`, `print_result_summary`, `print_dry_run`, `format_error`, `format_validation_error`, `print_experiment_header` |
| `src/llenergymeasure/cli/_vram.py` | 30 | 133 | VERIFIED | `estimate_vram()`, `get_gpu_vram_gb()`, `DTYPE_BYTES` with 5 precision types |
| `tests/unit/test_cli_display.py` | 40 | 177 | VERIFIED | 19 tests covering `_sig3`, `_format_duration`, `DTYPE_BYTES`, `format_validation_error`, `format_error` |
| `src/llenergymeasure/cli/run.py` | 100 | 195 | VERIFIED | Full `llem run` command with `_run_impl`, SIGINT handler, error dispatch, dry-run and run branches |
| `src/llenergymeasure/cli/__init__.py` | — | 47 | VERIFIED | `from llenergymeasure.cli.run import run as _run_cmd` present; both `run` and `config` registered |
| `tests/unit/test_cli_run.py` | 50 | 251 | VERIFIED | 11 tests covering help, version, error paths, dry-run, quiet flag, success path |
| `src/llenergymeasure/cli/config_cmd.py` | 60 | 184 | VERIFIED | `config_command`, `_probe_gpu`, `_probe_backend_version`, GPU/backends/energy/user-config/Python sections |
| `tests/unit/test_cli_config.py` | 30 | 119 | VERIFIED | 7 tests: help, basic output, no-GPU, verbose, user config path, exits-0, no-rich-import |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_display.py` | `ExperimentResult` | `print_result_summary(result)` reads result fields | VERIFIED | Function present at line 75; reads `result.experiment_id`, `total_energy_j`, `avg_tokens_per_second`, `duration_sec`, `process_results[0].compute_metrics`, `latency_stats`, `measurement_warnings` |
| `_display.py` | `pydantic.ValidationError` | `format_validation_error(e)` wraps Pydantic errors | VERIFIED | Function present at line 227; calls `e.errors()`, iterates, uses `difflib.get_close_matches` for did-you-mean |
| `_vram.py` | `ExperimentConfig` | `estimate_vram(config)` uses `config.model`, `config.precision` | VERIFIED | Function present at line 24; reads `config.model`, `config.precision`, `config.max_input_tokens` |
| `run.py` | `llenergymeasure._api.run_experiment` | `from llenergymeasure import run_experiment` | VERIFIED | Import at line 23; called at line 187 inside tqdm context manager |
| `run.py` | `llenergymeasure.config.loader.load_experiment_config` | `from llenergymeasure.config.loader import load_experiment_config` | VERIFIED | Import at line 22; called at line 166 |
| `run.py` | `llenergymeasure.cli._display` | `from llenergymeasure.cli._display import ...` | VERIFIED | Import at lines 14–20; `print_result_summary`, `print_dry_run`, `format_error`, `format_validation_error`, `print_experiment_header` all imported and called |
| `run.py` | `tqdm.auto` | `from tqdm.auto import tqdm` | VERIFIED | Import at line 12; used at line 181 with `total=None`, `file=sys.stderr`, `disable=quiet or not sys.stderr.isatty()` |
| `__init__.py` | `run.py` | `app.command(name="run", ...)(_run_cmd)` | VERIFIED | Line 36–38; `from llenergymeasure.cli.run import run as _run_cmd`; registered as "run" command |
| `config_cmd.py` | `pynvml` | `_probe_gpu()` calls `pynvml.nvmlInit/nvmlDeviceGetCount` | VERIFIED | `nvmlInit()` at line 29 inside `_probe_gpu`; wrapped in `try/except Exception` for non-blocking |
| `config_cmd.py` | `importlib.util.find_spec` | `_probe_backend()` checks package availability | VERIFIED | `importlib.util.find_spec(package)` at line 108; checks `transformers`, `vllm`, `tensorrt_llm` |
| `config_cmd.py` | `llenergymeasure.config.user_config` | `load_user_config` deferred import inside function body | VERIFIED | Imported at line 141 (`get_user_config_path`) and line 148 (`UserConfig`, `load_user_config`); tests patch at source module |
| `__init__.py` | `config_cmd.py` | `app.command(name="config", ...)(_config_cmd)` | VERIFIED | Lines 40–42; `from llenergymeasure.cli.config_cmd import config_command as _config_cmd`; registered as "config" command |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CLI-01 | 07-03 | 2 commands + 1 flag: `llem run`, `llem config`, `llem --version` | SATISFIED | All three present in `__init__.py`; `llem run --help` and `llem config --help` both exit 0 |
| CLI-02 | 07-02 | Entry point renamed `lem` → `llem`. No backward-compat shim. | SATISFIED | `pyproject.toml` line 47: `llem = "llenergymeasure.cli:app"`; old `lem` entry not present |
| CLI-03 | 07-02 | `llem run [CONFIG] [OPTIONS]` — auto-detects single vs study YAML | PARTIAL | Single-experiment YAML fully handled via `load_experiment_config()`; study YAML detection deferred — `run_study()` intentionally `NotImplementedError` in M1. M1 scope only requires single-experiment. |
| CLI-04 | 07-02 | `llem run` flags: `--model`, `--backend`, `--dataset`, `-n`, `--batch-size`, `--precision`, `--output`, `--dry-run`, `--quiet`, `--verbose` | SATISFIED | All 10 flags present in `run()` function signature with correct types and Typer annotations |
| CLI-06 | 07-03 | `llem config [--verbose]` — environment display + setup guidance | SATISFIED | `config_command(verbose=False)` in `config_cmd.py`; GPU, backends, energy, user config, Python all displayed |
| CLI-07 | 07-02 | `--dry-run`: L1 (Pydantic validation, always) + L2 (VRAM estimate) | SATISFIED | `load_experiment_config()` provides L1 Pydantic validation; `estimate_vram()` + `print_dry_run()` provides L2 VRAM estimate |
| CLI-08 | 07-01 + 07-02 | Plain text output (~200 LOC). tqdm for progress. No Rich dependency. | SATISFIED | Zero Rich imports in all 5 new CLI files; `tqdm` spinner at line 181; `_display.py` uses only `print()`; total new CLI LOC ~750 |
| CLI-09 | 07-01 | Output routing: progress → `stderr`, final summary → `stdout` | SATISFIED | `print_result_summary` → `print()` (stdout); `print_experiment_header` → `print(..., file=sys.stderr)`; `tqdm(file=sys.stderr)` |
| CLI-10 | 07-02 | `--quiet`: suppress progress, keep final summary. `--verbose`: add subprocess events / tracebacks. | SATISFIED | `tqdm(disable=quiet or not sys.stderr.isatty())`; `format_error(verbose=verbose)` shows traceback; `test_run_quiet_flag_accepted` verifies |
| CLI-12 | 07-02 | Exit codes: 0 (success), 1 (error), 2 (usage/config), 130 (SIGINT) | SATISFIED | `ConfigError`/`ValidationError` → `typer.Exit(2)`; `PreFlightError`/`ExperimentError`/`BackendError` → `typer.Exit(1)`; `KeyboardInterrupt` → `SystemExit(130)`; all 4 exit-code tests pass |
| CLI-13 | 07-02 | `LLEMError` → `ConfigError`, `BackendError`, `PreFlightError`, `ExperimentError`, `StudyError` | SATISFIED | All five exception types imported in `run.py`; `ConfigError` and `(PreFlightError\|ExperimentError\|BackendError)` caught; `format_error()` uses `type(error).__name__` |
| CLI-14 | 07-01 | Pydantic `ValidationError` passes through unchanged | SATISFIED | `format_validation_error(e)` calls `e.errors()` without re-wrapping; `ValidationError` NOT subclassed; `test_format_validation_error` verifies header added without altering underlying error messages |

**Orphaned requirements check:** REQUIREMENTS.md maps CLI-01 through CLI-04, CLI-06 through CLI-14 to Phase 7. CLI-05 and CLI-11 are explicitly M2. All M1 CLI requirements are claimed by plans 07-01, 07-02, or 07-03. No orphaned requirements.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None | — | — | — |

No stubs, no empty implementations, no TODO/FIXME/PLACEHOLDER, no `return null`, no `return {}`. The CLAUDE.md for `cli/` still references old v1.x module structure (`display/`, `experiment.py`, `batch.py`) but this is documentation stale-ness, not a code anti-pattern.

### Human Verification Required

#### 1. Live GPU Experiment Run

**Test:** In a GPU-enabled environment, run `llem run --model gpt2 --backend pytorch`
**Expected:** Indeterminate tqdm spinner ("Measuring...") appears on stderr during measurement; on completion, result summary with Energy/Performance/Timing sections prints to stdout; process exits 0
**Why human:** GPU not available on host; tqdm spinner rendering requires a real TTY

#### 2. Non-TTY Output Suppression

**Test:** `llem run --model gpt2 --backend pytorch > output.txt 2>/dev/null` (piped output)
**Expected:** tqdm spinner suppressed automatically; `output.txt` contains only the result summary (no spinner characters)
**Why human:** CliRunner in tests is always non-TTY; real pipe suppression needs a live terminal

#### 3. SIGINT Handling

**Test:** Start `llem run --model gpt2 --backend pytorch`, press Ctrl-C during measurement
**Expected:** stderr shows "Interrupted.", process exits with code 130 (verify with `echo $?`)
**Why human:** Signal handling requires an interactive terminal session; can't be reproduced in CliRunner

### Gaps Summary

No blocking gaps. The phase goal is achieved:

- `llem run` and `llem config` are the only entry points; both work correctly
- Plain text output confirmed by no-Rich audit (AST-level test passes)
- Exit codes verified for all 5 error paths: 0 (success), 1 (PreFlightError/ExperimentError/BackendError), 2 (ConfigError/ValidationError), 130 (SIGINT/KeyboardInterrupt)
- `--dry-run` validates via Pydantic (through `load_experiment_config`) and estimates VRAM (via `estimate_vram()`)
- 37/37 unit tests pass in 1.85s with no GPU or network access

The one partial result (CLI-03 study YAML auto-detection) is an intentional M1 design boundary: `run_study()` was verified as `NotImplementedError` in Phase 3 verification with the explicit note "Intentional M1 design — M2 implements study execution." Phase 7 correctly implements the M1 scope of `llem run`.

---

_Verified: 2026-02-27_
_Verifier: Claude (gsd-verifier)_
