---
phase: 01-measurement-foundations
plan: 03
subsystem: core-measurement
tags: [warmup, convergence, cv, latency, measurement]
dependency-graph:
  requires: ["01-01"]
  provides: ["warmup_until_converged", "create_warmup_inference_fn", "WarmupResult integration"]
  affects: ["01-04", "01-05"]
tech-stack:
  added: []
  patterns: ["CV-based convergence detection", "callable injection for testability"]
key-files:
  created:
    - src/llenergymeasure/core/warmup.py
  modified: []
decisions: []
metrics:
  duration: "4 min"
  completed: "2026-01-29"
---

# Phase 1 Plan 3: Warmup Convergence Detection Summary

CV-based warmup convergence using rolling window CV calculation with configurable threshold, safety cap, and progress reporting.

## What Was Done

### Task 1: Warmup convergence detection module
**Commit:** `b860313`

Created `core/warmup.py` with two public functions:

1. **`warmup_until_converged()`** - Main warmup function accepting:
   - `run_single_inference`: Callable returning latency ms (decoupled from inference impl)
   - `config`: WarmupConfig (cv_threshold, max_prompts, window_size, min_prompts)
   - `show_progress`: Toggle tqdm progress bar

   Algorithm: runs inference prompts in a loop, computing rolling-window CV after reaching min_prompts. Breaks when CV drops below threshold. Safety cap prevents runaway.

2. **`create_warmup_inference_fn()`** - Convenience factory for PyTorch model+tokenizer warmup.

### Modes
- **Adaptive (default)**: Stops when CV < threshold
- **Fixed** (`convergence_detection=False`): Runs all max_prompts, reports converged=True
- **Disabled** (`enabled=False`): Returns immediately with zero iterations

### Edge cases handled
- Failed inference prompts: caught, logged as warning, skipped (doesn't abort warmup)
- Non-convergence: warning log with final CV, returns `converged=False`
- Progress bar: optional, shows current CV vs target in postfix

## Verification Results

All checks passed:
- Import verification: OK
- Disabled warmup returns immediately with converged=True, 0 iterations
- Stable latencies (~2% variance) converge within 5 prompts at 10% threshold
- Noisy latencies (50-200ms range) hit safety cap at 1% threshold
- Fixed mode runs all iterations regardless of CV
- Exception in inference callable logged and skipped

## Deviations from Plan

None - plan executed exactly as written.

## Key Technical Decisions

None new - followed plan specifications.

## Files

| File | Action | Purpose |
|------|--------|---------|
| `src/llenergymeasure/core/warmup.py` | Created | CV-based warmup convergence + PyTorch helper |
