# N-X11: Resilience — Retry Logic and GPU Memory Cleanup

**Module**: `src/llenergymeasure/resilience.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: Not mentioned in any planning document. A small but important utility module. Its absence from the v2.0 module layout in `designs/architecture.md` risks omission during the rebuild.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/resilience.py`
**Key classes/functions**:

- `retry_on_error(max_retries=3, delay_seconds=1.0, backoff_factor=2.0, exceptions=(RetryableError,))` (line 16) — decorator factory; returns a decorator that wraps the target function with retry logic. The wrapped function:
  - Attempts the call up to `max_retries + 1` times total
  - On each failure (matching `exceptions`): logs at WARNING level, sleeps `delay * backoff_factor^attempt`
  - After all attempts exhausted: logs at ERROR level, re-raises the last exception
  - Uses `functools.wraps` to preserve function metadata

- `cleanup_gpu_memory()` (line 64) — calls `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`, and `gc.collect()` in sequence; guards against `ImportError` (torch not installed) and any other exception; degrades silently (warning log only); intended for use between experiments to avoid VRAM fragmentation

- `safe_cleanup(cleanup_func)` (line 80) — wraps any `Callable[[], None]` to catch and log (but not raise) exceptions during cleanup; preserves the wrapped function's name via `functools.wraps`

Total: 98 lines.

The `RetryableError` exception (imported from `llenergymeasure.exceptions`) distinguishes transient errors (network timeouts, temporary NVML failures) from permanent errors — only `RetryableError` subclasses trigger retries by default.

## Why It Matters

GPU operations are inherently fragile: NVML can return transient errors, model downloads can timeout, and distributed barriers can fail due to timing. The `retry_on_error` decorator provides a standardised retry pattern rather than each caller implementing its own loop. `cleanup_gpu_memory()` is called between experiments in study runs to release PyTorch's CUDA memory cache — without this, experiments 2–N in a study can see elevated initial VRAM usage that pollutes memory efficiency measurements. `safe_cleanup()` is a defensive wrapper that ensures cleanup code in `finally` blocks does not suppress the original exception.

## Planning Gap Details

`designs/architecture.md` module layout does not list `resilience.py`. The planned structure lists `core/`, `config/`, `domain/`, `orchestration/`, `results/`, `state/`, `cli/`, `study/`, `datasets/` — but not the cross-cutting utilities at the package root (`protocols.py`, `security.py`, `resilience.py`, `constants.py`).

This is a gap in the module layout documentation: the root-level utility files need to be explicitly listed, or the Phase 5 rebuild team may assume they are new additions rather than carries.

No planning doc specifies the retry policy for model loading, energy backend initialisation, or NVML polling. This is an implicit contract between `resilience.py` and its callers.

## Recommendation for Phase 5

Carry `resilience.py` forward unchanged at `src/llenergymeasure/resilience.py`. The module is 98 lines and correct.

Key wiring to verify in the v2.0 orchestrator:
1. `cleanup_gpu_memory()` must be called by `StudyRunner` in `_run_one()` after each subprocess completes (to clear the parent process's CUDA cache before the next experiment spawns)
2. `retry_on_error` should wrap the HuggingFace model download step in `ModelLoader.load()` — network transients are the most common failure mode for new model runs
3. `safe_cleanup()` should be used in `ExperimentOrchestrator`'s `finally` block for the energy backend `stop_tracking()` call — a failure in `stop_tracking` during cleanup should not mask the original exception

Add `resilience.py` to the module layout in `designs/architecture.md` alongside `protocols.py` and `security.py` as package-root utilities.
