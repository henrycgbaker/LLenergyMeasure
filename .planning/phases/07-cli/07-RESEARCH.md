# Phase 7: CLI - Research

**Researched:** 2026-02-27
**Domain:** Typer CLI, plain-text terminal output, tqdm progress, exit-code handling, did-you-mean suggestions
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Result summary format**
- Strictly raw metrics only — no derived ratios (no J/token, no J/request)
- Grouped sections layout: Energy, Performance, Timing (not flat key:value)
- Include FLOPs estimate (with method/confidence) in the summary
- Include measurement warnings inline in the summary
- 3 significant figures for all numeric values (consistent, avoids false precision)
- All peer tools surveyed (AIEnergyScore, vLLM bench, lm-eval, Optimum-Benchmark, mlperf) show raw metrics only; derived ratios computed externally

**Dry-run presentation**
- Single-experiment `--dry-run`: config echo + VRAM estimate (no pre-flight detail)
- VRAM breakdown shown: weights, KV cache, overhead as separate lines plus total
- `--dry-run` always exits 0 for valid configs (informational, not a gate). Exit 2 only for ConfigError.
- `--verbose --dry-run`: adds source annotations to config echo (e.g., 'bf16 (default)', 'pytorch (--backend)', '100 (experiment.yaml)')
- Standard `--dry-run`: values only, no source annotations
- Study-mode `--dry-run` grid preview already specified in decisions/cli-ux.md

**Error message style**
- Guided errors (Rust/Elm style): what went wrong, where (file:line if applicable), and a fix suggestion
- Example: `ConfigError: unknown backend 'pytorh'\n  -> experiment.yaml, line 5\n  Did you mean: pytorch?\n  Valid backends: pytorch, vllm, tensorrt`
- Did-you-mean suggestions on ALL string enum fields (backend, precision, dataset aliases, etc.)
- Python stack traces hidden by default; shown with `--verbose`
- Pydantic ValidationError: wrapped with friendly header ("Config validation failed (N errors):") but Pydantic's own error messages pass through unchanged

**Progress display**
- Standard tqdm bars for both warmup and measurement phases
- Warmup gets its own tqdm bar (replaced by measurement bar when warmup completes)
- Non-TTY / piped output: suppress all progress bars entirely, print final result summary only (tqdm auto-detects)
- NO_COLOR respected (tqdm handles natively)

### Claude's Discretion
- Experiment header line: model + backend always shown, plus any non-default parameters (Claude decides which params qualify as "non-default" worth showing)
- Exact tqdm format string customisation
- Exact indentation and spacing in output
- How `--verbose` subprocess events are formatted
- Internal error formatting utilities (difflib for did-you-mean, etc.)

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLI-01 | 2 commands + 1 flag: `llem run`, `llem config`, `llem --version` | Typer `@app.command()` + `@app.callback()` pattern; skeleton in `cli/__init__.py` |
| CLI-02 | Entry point renamed `lem` → `llem`. No backward-compat shim. | `pyproject.toml [project.scripts]` already set to `llem`; just need correct implementation |
| CLI-03 | `llem run [CONFIG] [OPTIONS]` — auto-detects single vs study YAML | `load_experiment_config()` and `load_study_config()` already exist in `config/loader.py` |
| CLI-04 | `llem run` flags: `--model`, `--backend`, `--dataset`, `-n`, `--batch-size`, `--precision`, `--output`, `--dry-run`, `--quiet`, `--verbose` | Standard Typer `Annotated[T, typer.Option(...)]` pattern |
| CLI-06 | `llem config [--verbose]` — environment display + setup guidance | `pynvml` for GPU info, `importlib.util.find_spec` for backend detection, `load_user_config()` already in `config/user_config.py` |
| CLI-07 | `--dry-run`: L1 (Pydantic validation, always) + L2 (VRAM estimate, grid preview) | VRAM estimate formula: weights (params × dtype_bytes) + KV cache + overhead; no library needed |
| CLI-08 | Plain text output (~200 LOC). tqdm for progress. No Rich dependency. | `tqdm` already in base deps (`INF-02`); plain `print()` to stdout/stderr |
| CLI-09 | Output routing: progress → `stderr`, final summary → `stdout` | `tqdm(file=sys.stderr)` + `print()` to stdout; Typer default is stdout |
| CLI-10 | `--quiet`: suppress progress, keep final summary. `--verbose`: add subprocess events. | tqdm `disable` param; thread-safe flag passed into backend |
| CLI-12 | Exit codes: 0 (success), 1 (error), 2 (usage/config), 130 (SIGINT) | `raise SystemExit(N)` or `raise typer.Exit(code=N)`; SIGINT already handled in `infra/subprocess.py` |
| CLI-13 | `LLEMError` → `ConfigError`, `BackendError`, `PreFlightError`, `ExperimentError`, `StudyError` | Exception hierarchy already in `exceptions.py`; map in CLI error handler |
| CLI-14 | Pydantic `ValidationError` passes through unchanged | Separate `except ValidationError` block before general `except LLEMError` |
</phase_requirements>

---

## Summary

Phase 7 builds the final user-visible layer of M1: the `llem run` and `llem config` commands, wiring them to the library API (`run_experiment()`) that previous phases built. The hard work — config loading, pre-flight, energy measurement, ExperimentResult — is already done. This phase is about presentation and polish: correct exit codes, tqdm progress bars, grouped result summary output, did-you-mean error messages, VRAM estimation for `--dry-run`, and the environment-display `llem config` command.

The current `cli/__init__.py` is a skeleton with only `--version`. The existing `cli/experiment.py` and `cli/config.py` are v1.x code that use Rich and the old campaign/orchestration architecture — they must be **replaced entirely** rather than adapted. The v2.0 CLI is architecturally simpler: instead of spawning subprocesses via `accelerate launch`, it calls `run_experiment()` directly (STU-05: single experiment runs in-process).

The key technical challenge in this phase is the **complete rewrite of both command modules** without carrying forward v1.x complexity. Secondary challenges are: VRAM estimation for `--dry-run` (no pre-existing utility), did-you-mean error formatting via `difflib`, and the `llem config` environment probe (GPU info, installed backends, energy backends, user config display).

**Primary recommendation:** Write `cli/run.py` and `cli/config_cmd.py` as new modules (under 200 LOC each), register them in `cli/__init__.py`, and delete all v1.x CLI code that is superseded. Use the `ExperimentResult` fields directly for output formatting.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `typer` | `>=0.9` (already in deps) | CLI framework, argument parsing | Already in `INF-02` base deps; skeleton uses it |
| `tqdm` | `>=4.66` (already in deps) | Progress bars | Already in `INF-02` base deps; auto-detects TTY; NO_COLOR support |
| `difflib` | stdlib | Did-you-mean string matching | No dependency needed; `difflib.get_close_matches()` is the standard Python approach |
| `pynvml` | `nvidia-ml-py` (already in deps) | GPU info for `llem config` | Already in `INF-02`; used by existing energy backends |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `importlib.util` | stdlib | Soft backend/package probing | For `llem config` backend detection without importing heavy modules |
| `platformdirs` | `>=3.0` (already in deps) | User config path | `get_user_config_path()` already uses it |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `difflib.get_close_matches()` | `rapidfuzz` or manual Levenshtein | stdlib is sufficient; 0 new dependencies |
| plain `print()` to stderr | `logging.getLogger()` | print is simpler for CLI display; logging is for structured backend events |
| tqdm | Rich Progress | Rich is explicitly excluded (CLI-08); tqdm already in deps |

---

## Architecture Patterns

### Recommended Project Structure

```
src/llenergymeasure/cli/
├── __init__.py          # Typer app + command registration (update skeleton)
├── run.py               # NEW: llem run command (replaces experiment.py)
├── config_cmd.py        # NEW: llem config command (replaces config.py)
├── _display.py          # NEW: shared plain-text formatting utilities (~50 LOC)
└── _vram.py             # NEW: VRAM estimation for --dry-run (~40 LOC)
```

The old `cli/experiment.py`, `cli/config.py`, and `cli/display/` are v1.x artefacts. They should be deleted or left in place (unused) during this phase — do not import from them.

### Pattern 1: Typer command with error dispatch

The entry point for each command catches `LLEMError` subclasses and maps them to exit codes.

```python
# Source: typer docs + existing cli/__init__.py skeleton
import sys
import signal
import typer
from pydantic import ValidationError
from llenergymeasure.exceptions import ConfigError, PreFlightError, ExperimentError, BackendError

@app.command()
def run(
    config: Annotated[Path | None, typer.Argument()] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    backend: Annotated[str | None, typer.Option("--backend", "-b")] = None,
    # ... other flags
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    try:
        _run_impl(config, model, backend, dry_run, quiet, verbose)
    except ConfigError as e:
        _print_error(str(e), verbose=verbose)
        raise typer.Exit(code=2)
    except (PreFlightError, ExperimentError, BackendError) as e:
        _print_error(str(e), verbose=verbose)
        raise typer.Exit(code=1)
    except ValidationError as e:
        _print_validation_error(e)
        raise typer.Exit(code=2)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
```

### Pattern 2: SIGINT → exit code 130

Per `STATE.md` decision: `SubprocessRunner` raises `SystemExit(130)`. The CLI must catch `KeyboardInterrupt` directly and exit 130 — not via `typer.Exit` (which maps to 0).

```python
# SIGINT handling in run command body
import signal

def _handle_sigint(signum, frame):
    print("\nInterrupted.", file=sys.stderr)
    raise SystemExit(130)

signal.signal(signal.SIGINT, _handle_sigint)
```

### Pattern 3: tqdm to stderr

tqdm bars write to stderr by default when `file=sys.stderr` is set. This matches CLI-09 (progress → stderr). Pass `disable=quiet or not sys.stderr.isatty()` for non-TTY suppression.

```python
from tqdm import tqdm

# Warmup bar
with tqdm(total=n_warmup, desc="Warmup", file=sys.stderr,
          disable=quiet, bar_format="{l_bar}{bar}| {n}/{total}") as pbar:
    # called back from backend via callback
    pbar.update(1)

# Measurement bar (replaces warmup bar after warmup completes)
with tqdm(total=n, desc="Measuring", file=sys.stderr,
          disable=quiet, bar_format="{l_bar}{bar}| {n}/{total} prompts") as pbar:
    ...
```

**Key detail:** tqdm auto-detects `NO_COLOR` and `TERM=dumb`. No manual check needed.

### Pattern 4: Result summary output (grouped sections)

```python
# Final summary to stdout (not stderr — this is the scientific record)
def print_result_summary(result: ExperimentResult) -> None:
    print(f"\nResult: {result.experiment_id}")
    print()
    print("Energy")
    print(f"  Total          {_sig3(result.total_energy_j)} J")
    if result.baseline_power_w is not None:
        print(f"  Baseline       {_sig3(result.baseline_power_w)} W")
    if result.energy_adjusted_j is not None:
        print(f"  Adjusted       {_sig3(result.energy_adjusted_j)} J")
    print()
    print("Performance")
    print(f"  Throughput     {_sig3(result.avg_tokens_per_second)} tok/s")
    # FLOPs estimate (from ExperimentResult.total_flops)
    print(f"  FLOPs          {_sig3(result.total_flops)}")
    print()
    print("Timing")
    dur = result.duration_sec
    print(f"  Duration       {_format_duration(dur)}")
    if result.warmup_excluded_samples is not None:
        print(f"  Warmup         {result.warmup_excluded_samples} prompts")
    # Measurement warnings inline
    for w in result.measurement_warnings:
        print(f"\nWarning: {w}")
```

### Pattern 5: Did-you-mean via difflib

```python
import difflib

def _did_you_mean(value: str, valid_options: list[str]) -> str | None:
    matches = difflib.get_close_matches(value, valid_options, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Usage in error formatting:
def _format_config_error(e: ConfigError) -> str:
    msg = str(e)
    # Intercept unknown backend, precision, dataset errors and append suggestion
    # (see Common Pitfalls for how to integrate this with Pydantic ValidationError)
    return msg
```

### Pattern 6: VRAM estimation for --dry-run

No existing VRAM estimator in the codebase. Must be built from scratch (~40 LOC):

```python
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

def estimate_vram_gb(
    param_count: int,        # From model config or HF hub metadata
    precision: str,          # ExperimentConfig.precision
    max_seq_len: int = 2048,
    batch_size: int = 1,
    n_layers: int = None,    # For KV cache estimation
    n_heads: int = None,
    head_dim: int = None,
) -> dict[str, float]:
    bytes_per_param = DTYPE_BYTES.get(precision, 2)
    weights_gb = (param_count * bytes_per_param) / 1e9
    # KV cache: 2 × n_layers × batch × seq_len × n_heads × head_dim × dtype_bytes
    kv_gb = 0.0  # estimated 0 when arch details unavailable
    if n_layers and n_heads and head_dim:
        kv_bytes = 2 * n_layers * batch_size * max_seq_len * n_heads * head_dim * bytes_per_param
        kv_gb = kv_bytes / 1e9
    overhead_gb = weights_gb * 0.15  # ~15% activation/framework overhead (empirical)
    return {"weights_gb": weights_gb, "kv_cache_gb": kv_gb, "overhead_gb": overhead_gb,
            "total_gb": weights_gb + kv_gb + overhead_gb}
```

Parameter count comes from HF Hub model card metadata (`huggingface_hub.HfApi().model_info(model_id).safetensors.total`) or falls back to `None` (display "unknown"). Avoid loading the model.

### Pattern 7: llem config environment probe

```python
def _probe_gpu() -> dict | None:
    """Return GPU info dict or None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({"name": name, "vram_gb": mem.total / 1e9})
        return {"count": count, "devices": gpus}
    except Exception:
        return None

def _probe_backend(name: str, package: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(package) is not None
```

### Anti-Patterns to Avoid

- **Importing Rich**: `cli/config.py` and `cli/experiment.py` import Rich — do not import these modules in the new CLI. They are v1.x artefacts.
- **Calling the old orchestration layer**: The new `llem run` calls `run_experiment()` directly (in-process, STU-05), not `accelerate launch` subprocess.
- **Wrapping ValidationError in ConfigError**: CLI-14 requires `ValidationError` to pass through. Catch it separately.
- **Suppressing the final summary**: `--quiet` suppresses progress bars only. The result summary always prints to stdout.
- **Exit via `raise typer.Exit(130)`**: `typer.Exit` does not produce the correct UNIX signal exit code for SIGINT. Use `raise SystemExit(130)` directly.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TTY detection for progress suppression | Custom `sys.stdout.isatty()` check | `tqdm` auto-detects non-TTY and disables | tqdm handles piped output, CI, TERM=dumb |
| NO_COLOR support | Manual check + ANSI stripping | `tqdm` reads `NO_COLOR` natively | Standard; tqdm 4.66+ supports it |
| Close-match suggestions | Levenshtein distance implementation | `difflib.get_close_matches()` | Stdlib; no dependency; sufficient for short vocab |
| XDG config path | Manual `~/.config/...` path building | `load_user_config()` + `get_user_config_path()` already built | Phase 2 delivered this |
| Backend availability check | Import attempt | `importlib.util.find_spec()` | Already pattern in `orchestration/preflight.py` |

**Key insight:** tqdm is the correct tool for all progress display concerns in this codebase. The 950-LOC Rich layer it replaces was handling progress, tables, and result display. In v2.0, tqdm handles progress, and `print()` handles result display.

---

## Common Pitfalls

### Pitfall 1: Rich still imported via existing cli/ modules

**What goes wrong:** `cli/__init__.py` currently imports from `cli/display/` which imports Rich. If the new commands import anything from the existing `cli/` package, Rich gets pulled in.

**Why it happens:** The `cli/display/__init__.py` does a star-import from `console.py` which creates a `rich.console.Console` instance at import time.

**How to avoid:** New modules (`run.py`, `config_cmd.py`, `_display.py`) must not import from `cli/display/`, `cli/experiment.py`, or `cli/config.py`. Register the new commands in `__init__.py` only (do not import the old modules).

**Warning signs:** `import rich` appears in `sys.modules` after CLI startup.

### Pitfall 2: typer.Exit(130) vs SystemExit(130) for SIGINT

**What goes wrong:** `raise typer.Exit(code=130)` causes Typer to call `sys.exit(130)` but this is caught by Typer's own exception handler and may produce a different exit code.

**Why it happens:** Typer wraps all exits. For SIGINT, the shell convention requires the process to exit via `SystemExit(130)` directly, bypassing Typer's handler.

**How to avoid:** In the SIGINT handler, use `raise SystemExit(130)`. In normal error paths, use `raise typer.Exit(code=N)`.

### Pitfall 3: ValidationError caught before LLEMError

**What goes wrong:** A broad `except Exception` or `except LLEMError` before the `except ValidationError` clause swallows Pydantic errors.

**Why it happens:** `ValidationError` is not a subclass of `LLEMError`. It must be caught in its own `except` block.

**How to avoid:** Exception handler ordering must be: `ConfigError` → `(PreFlightError, BackendError, ExperimentError)` → `ValidationError` → `Exception` (optional fallback). Never `except Exception` before `except ValidationError`.

### Pitfall 4: Did-you-mean on raw Pydantic ValidationError messages

**What goes wrong:** Pydantic's `ValidationError` contains structured `error.errors()` dicts. Trying to apply did-you-mean to the raw string representation misses the field path and value.

**Why it happens:** The error value (`'pytorh'`) and the valid options (`['pytorch', 'vllm', 'tensorrt']`) are not available from the raw string — they must be extracted from `e.errors()`.

**How to avoid:** Iterate `e.errors()`. For each error where `type == 'literal_error'` or `type == 'value_error'`, extract `loc` (field path) and `input` (bad value), then lookup valid options from `config/ssot.py` `PRECISION_SUPPORT` / `BACKEND_NAMES` to generate the suggestion.

### Pitfall 5: tqdm bar not cleaned up on exception

**What goes wrong:** If an exception is raised during measurement, the tqdm bar stays open and the next print output appears on the same line.

**Why it happens:** `tqdm` uses context manager protocol to clean up on `__exit__`. Using it without `with` leaves the bar open.

**How to avoid:** Always use `with tqdm(...) as pbar:` pattern. The `__exit__` closes the bar even on exception.

### Pitfall 6: VRAM estimation requires model param count from HF Hub

**What goes wrong:** `llem run --dry-run --model meta-llama/Llama-3.1-8B` fails or produces "unknown" VRAM estimate because HF Hub API is network-dependent.

**Why it happens:** Param count must come from HF Hub metadata (`model.safetensors.total` or `model_info.model_card_data.model_config`). This is a network call.

**How to avoid:** Wrap the HF Hub call in `try/except` with timeout. On failure, show "VRAM estimate: unavailable (model not cached locally)". This is consistent with `_check_model_accessible()` in `orchestration/preflight.py` (non-blocking on network failure).

### Pitfall 7: Progress callbacks into backend — not yet wired

**What goes wrong:** The Phase 4/5 PyTorch backend has no callback mechanism for progress updates. tqdm bars in the CLI would show "Measuring 0/100" because the backend doesn't report progress.

**Why it happens:** The `InferenceBackend.run(config)` API returns `ExperimentResult` with no intermediate events. This is a known gap for Phase 7.

**How to avoid:** For Phase 7, the CLI can show a spinner or indeterminate bar while `run_experiment()` executes (it's a synchronous blocking call). Exact per-prompt progress requires a callback protocol extension — that is a Phase 7 discretionary decision. An acceptable M1 implementation is: `tqdm` with `total=None` (spinner mode) during the run, then clean up and print the result summary.

---

## Code Examples

### 3 significant figures utility

```python
def _sig3(value: float) -> str:
    """Format float to 3 significant figures."""
    if value == 0:
        return "0"
    from math import log10, floor
    digits = -int(floor(log10(abs(value)))) + 2
    return f"{value:.{max(0, digits)}f}"

# Examples:
# _sig3(312.4)   → "312"
# _sig3(3.12)    → "3.12"
# _sig3(0.00312) → "0.00312"
# _sig3(847.0)   → "847"
```

### Duration formatting

```python
def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"
```

### ValidationError friendly wrapper

```python
from pydantic import ValidationError

def _format_validation_error(e: ValidationError) -> str:
    """Wrap Pydantic ValidationError with friendly header."""
    errors = e.errors()
    header = f"Config validation failed ({len(errors)} error{'s' if len(errors) > 1 else ''}):"
    lines = [header]
    for err in errors:
        loc = " -> ".join(str(p) for p in err["loc"])
        msg = err["msg"]
        value = err.get("input", "")
        # Attempt did-you-mean for string enums
        suggestion = _did_you_mean_for_error(err)
        line = f"  {loc}: {msg}"
        if suggestion:
            line += f"\n    Did you mean: {suggestion}?"
        lines.append(line)
    return "\n".join(lines)
```

### VRAM display for --dry-run

```python
def print_dry_run(config: ExperimentConfig, vram: dict | None, gpu_vram_gb: float | None) -> None:
    print("Config (resolved)")
    print(f"  Model          {config.model}")
    print(f"  Backend        {config.backend}")
    print(f"  Precision      {config.precision}")
    # ... other fields

    print()
    if vram is None:
        print("VRAM estimate")
        print("  (unavailable — model not in local cache and HF Hub unreachable)")
    else:
        avail = f" / {gpu_vram_gb:.0f} GB available" if gpu_vram_gb else ""
        status = "  OK" if (gpu_vram_gb is None or vram["total_gb"] < gpu_vram_gb) else "  WARNING: may not fit"
        print("VRAM estimate")
        print(f"  Weights        {vram['weights_gb']:.2f} GB ({config.precision})")
        print(f"  KV cache       {vram['kv_cache_gb']:.2f} GB")
        print(f"  Overhead       {vram['overhead_gb']:.2f} GB")
        print(f"  Total          ~{vram['total_gb']:.2f} GB{avail}{status}")

    print()
    print("Config valid. Run without --dry-run to start.")
```

---

## Codebase State: What Phase 7 Inherits

### Already built (do NOT rebuild):
- `exceptions.py`: Full `LLEMError` hierarchy (5 subclasses + v1.x aliases)
- `_api.py`: `run_experiment()` — complete, calls `_run()` → `get_backend().run(config)`
- `config/loader.py`: `load_experiment_config()`, `load_study_config()`
- `config/user_config.py`: `load_user_config()`, `get_user_config_path()`
- `config/models.py`: `ExperimentConfig` with v2.0 field names (`model`, `precision`, `n`)
- `domain/experiment.py`: `ExperimentResult` — complete v2.0 schema
- `orchestration/preflight.py`: `run_preflight()` — runs pre-flight checks
- `cli/__init__.py`: Skeleton with `--version` flag, `llem` Typer app

### Must be replaced/deleted:
- `cli/experiment.py`: v1.x — subprocess-based, uses Rich, old field names (`model_name`, `fp_precision`), campaign-aware. Do not adapt.
- `cli/config.py`: v1.x — uses Rich, old commands (`validate`, `show`, `new`, `generate-grid`). Do not adapt.
- `cli/display/`: v1.x Rich-based display package. Do not import.

### Entry point already correct:
- `pyproject.toml` already has `llem = "llenergymeasure.cli:app"` — CLI-02 satisfied at packaging level.

### What Phase 7 must build:
1. `cli/run.py` — `llem run` command
2. `cli/config_cmd.py` — `llem config` command
3. `cli/_display.py` — shared plain-text formatting (~50 LOC)
4. `cli/_vram.py` — VRAM estimator for `--dry-run` (~40 LOC)
5. Update `cli/__init__.py` to register new commands

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Rich console for all output | Plain `print()` + tqdm | Preservation audit N-X04 (2026-02-26) | Simplifies to ~200 LOC; removes Rich as base dep |
| `lem experiment` command | `llem run` command | v2.0 redesign | Entry point rename; no shim |
| Subprocess `accelerate launch` | In-process `run_experiment()` | Phase 3 architecture (STU-05) | M1 is local-only; no subprocess needed |
| Config `model_name`, `fp_precision` | `model`, `precision` | Phase 2 field renames | CLI flags and display use new names |
| 15-command CLI | 2 commands + 1 flag | v2.0 simplification | Drastically reduced surface area |

**Deprecated/outdated:**
- `cli/display/console.py` Rich-based console: superseded by `print()` to stdout/stderr
- `cli/experiment.py` campaign/orchestration flow: superseded by `run_experiment()` call
- `lem` entrypoint: removed — `llem` only

---

## Open Questions

1. **Progress callbacks from backend**
   - What we know: `backend.run(config)` is synchronous; `PyTorchBackend.run()` has no progress callback.
   - What's unclear: Can we add a `progress_callback` to the `InferenceBackend` Protocol without breaking Phase 4/5 work?
   - Recommendation: For M1, use an indeterminate tqdm spinner for the measurement phase. Add callback protocol in a later phase. This is acceptable — the CONTEXT.md only says "standard tqdm bars for warmup and measurement phases" but does not specify per-prompt granularity.

2. **Verbose --dry-run source annotations**
   - What we know: `--verbose --dry-run` should show source annotations (e.g., "bf16 (default)", "pytorch (--backend)"). Config provenance tracking (`config/provenance.py`) was built for this but it's part of the v1.x code.
   - What's unclear: Does `load_experiment_config()` return provenance information in v2.0?
   - Recommendation: Check `config/loader.py` and `config/provenance.py`. If provenance tracking is still compatible, use it. Otherwise, implement simple source annotation: explicit CLI args vs YAML-loaded values vs Pydantic defaults can be distinguished by comparing the loaded config against ExperimentConfig defaults.

3. **FLOPs display in result summary**
   - What we know: `ExperimentResult.total_flops` is a float (from Phase 4/5). `FlopsResult` has `method` and `confidence` fields (CM-27).
   - What's unclear: Where is `FlopsResult` stored on `ExperimentResult`? The current `ExperimentResult` schema has `total_flops: float` only — no `flops_method` field.
   - Recommendation: Display `total_flops` only (as a scalar) in the Phase 7 summary. If `FlopsResult` is needed, it may be added to `ExperimentResult` in Phase 6 completion or Phase 7 plan. The CONTEXT.md says "Include FLOPs estimate (with method/confidence)" — verify `ExperimentResult` has these fields before planning the display task.

---

## Sources

### Primary (HIGH confidence)
- Codebase: `src/llenergymeasure/cli/__init__.py` — skeleton with --version, Typer app
- Codebase: `src/llenergymeasure/_api.py` — `run_experiment()` and `_run()` implementation
- Codebase: `src/llenergymeasure/exceptions.py` — error hierarchy
- Codebase: `src/llenergymeasure/config/user_config.py` — `load_user_config()` API
- Codebase: `src/llenergymeasure/orchestration/preflight.py` — `_check_backend_installed()`, `_probe_gpu()` pattern
- Codebase: `src/llenergymeasure/domain/experiment.py` — `ExperimentResult` v2.0 schema
- Product design: `.product/designs/cli-commands.md` — command signatures, flags
- Product design: `.product/designs/observability.md` — output format, routing, verbosity levels
- Product context: `.planning/phases/07-cli/07-CONTEXT.md` — locked decisions
- Product requirements: `.product/REQUIREMENTS.md` — CLI-01 through CLI-14

### Secondary (MEDIUM confidence)
- `difflib.get_close_matches()` — stdlib since Python 3.0; documented stable API
- tqdm NO_COLOR behaviour — documented in tqdm README (4.66+)
- VRAM estimation formula — empirical from HuggingFace model card conventions; 15% overhead is an approximation

### Tertiary (LOW confidence)
- HF Hub `model_info().safetensors.total` for parameter counts — API field existence needs verification against current huggingface_hub version

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in deps; verified in codebase
- Architecture: HIGH — derived directly from existing `_api.py`, `exceptions.py`, and design docs
- Pitfalls: HIGH for Rich/subprocess issues (directly visible in existing code); MEDIUM for tqdm/SIGINT edge cases (well-known patterns)
- VRAM estimation: MEDIUM — formula is standard; HF Hub API field name needs verification

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable domain; tqdm/typer APIs change slowly)
