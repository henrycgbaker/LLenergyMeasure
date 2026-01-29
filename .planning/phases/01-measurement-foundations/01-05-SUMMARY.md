# Phase 01 Plan 05: Orchestrator Integration Summary

## One-liner
Wire all Phase 1 measurement components (environment, baseline, warmup, power/thermal sampler, timeseries) into the experiment lifecycle with non-fatal graceful degradation.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Wire measurement components into ExperimentOrchestrator.run() | a4c9ee4 | src/llenergymeasure/orchestration/runner.py |
| 2 | Aggregation + CLI display for schema v3 fields | c1e190e | src/llenergymeasure/results/aggregation.py, src/llenergymeasure/cli/experiment.py |

## What Was Built

### Runner Integration (runner.py)

The `ExperimentOrchestrator.run()` method now follows this lifecycle:

1. **Environment metadata** -- Calls `collect_environment_metadata()` before model load; logs summary line
2. **Baseline power** -- Calls `measure_baseline_power()` with config-driven parameters; supports cache TTL, required/optional modes
3. **Model load** -- Unchanged
4. **Warmup convergence** -- Calls `warmup_until_converged()` with CV-based detection after model load, before inference
5. **PowerThermalSampler** -- Context manager wraps inference call; samples power, temp, memory, throttle state
6. **Inference** -- Unchanged (runs inside sampler context)
7. **Energy breakdown** -- Calls `create_energy_breakdown()` to compute baseline-adjusted energy
8. **Thermal throttle info** -- Extracts from sampler; warns if throttling detected
9. **Save result** -- All new fields (environment, energy_breakdown, thermal_throttle, warmup_result) added to RawProcessResult
10. **Time-series export** -- Exports sampler data when config.timeseries.save=True; updates result with path via model_copy

All integration points wrapped in try/except with non-fatal logging. Experiment always completes even if all new features fail.

### Aggregation (aggregation.py)

Added schema v3 field aggregation to `aggregate_results()`:
- **environment**: Taken from first process (all share same GPU)
- **energy_breakdown**: raw_j summed, adjusted_j summed (when available), baseline info from first
- **thermal_throttle**: Merged via OR across processes (any throttled = throttled), max duration/temperature
- **timeseries_path**: First available path propagated

### CLI Display (experiment.py)

Added `_display_measurement_summary()` helper called after experiment completion:
- Environment summary line (GPU, CUDA, driver, temp)
- Thermal throttle warning (yellow, with duration)
- Energy breakdown (raw vs adjusted with baseline power)
- Warmup status (converged/not, iterations, CV)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Timeseries path assignment after frozen model save**
- **Found during:** Task 1
- **Issue:** RawProcessResult is frozen (Pydantic), so timeseries_path cannot be set after initial construction since it's only known after save
- **Fix:** Use `model_copy(update=...)` to create updated result with timeseries_path, then re-save
- **Files modified:** runner.py

**2. [Rule 3 - Blocking] Gitignored results/ path matching src/llenergymeasure/results/**
- **Found during:** Task 2 commit
- **Issue:** `.gitignore` pattern `results/` matches the source module path
- **Fix:** Used `git add -f` for the results module file
- **Files modified:** N/A (git operation only)

**3. [Rule 1 - Bug] Mypy type narrowing error in CLI helper**
- **Found during:** Task 2 commit
- **Issue:** mypy flagged `AggregatedResult | None` assignment incompatible with prior `AggregatedResult` type
- **Fix:** Added explicit type annotation `result: AggregatedResult | None` before the branch
- **Files modified:** experiment.py

## Decisions Made

None -- plan executed as written with only bug fixes.

## Test Results

- Unit tests: 791 passed, 0 failed
- Integration tests: 94 passed, 2 skipped, 0 failed
- All pre-commit hooks passed (ruff, mypy, formatting)

## Metrics

- Duration: 7 min
- Tasks: 2/2
- Completed: 2026-01-29
