---
phase: 01-measurement-foundations
plan: 04
subsystem: results-export
tags: [csv, timeseries, export, json, grouped-columns]
depends_on:
  requires: ["01-01"]
  provides: ["extended-csv-export", "timeseries-export"]
  affects: ["01-05", "01-06"]
tech_stack:
  added: []
  patterns: ["grouped-prefix-columns", "atomic-json-write", "compact-timeseries"]
key_files:
  created:
    - src/llenergymeasure/results/timeseries.py
  modified:
    - src/llenergymeasure/results/exporters.py
    - src/llenergymeasure/results/__init__.py
decisions:
  - id: "01-04-01"
    decision: "Renamed total_energy_j to energy_raw_j in CSV column output"
    rationale: "Grouped prefix convention (energy_*) for CSV readability; model field unchanged"
  - id: "01-04-02"
    decision: "Compact JSON keys for timeseries (t, mem_mb, sm_pct, throttle)"
    rationale: "File size management for long experiments with 100ms sampling"
metrics:
  duration: "4 min"
  completed: "2026-01-29"
---

# Phase 01 Plan 04: Extended CSV & Time-Series Export Summary

Extended CSV export with ~24 new grouped-prefix columns (energy_, thermal_, env_, gpu_, latency_, batch_, kv_cache_) and separate JSON time-series files with compact sample format and summary statistics.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Extended CSV export with grouped-prefix columns | 053d9e5 | exporters.py |
| 2 | Time-series export to separate JSON files | f83688f | timeseries.py, __init__.py |

## What Was Built

### Task 1: Extended CSV Export

Updated `_aggregated_to_row()` to include schema v3 fields:
- **energy_** group: `energy_raw_j` (renamed from `total_energy_j`), `energy_adjusted_j`, `energy_baseline_w`, `energy_baseline_method`
- **thermal_** group: `thermal_throttle_detected`, `thermal_throttle_duration_sec`, `thermal_max_temp_c`
- **env_** group: `env_gpu_name`, `env_gpu_vram_mb`, `env_cuda_version`, `env_driver_version`, `env_gpu_temp_c`, `env_power_limit_w`, `env_cpu_governor`, `env_in_container`, `env_summary`
- **Extended metrics**: `gpu_util_mean_pct`, `gpu_mem_peak_mb`, `latency_e2e_mean_ms`, `latency_e2e_p95_ms`, `batch_effective_size`, `kv_cache_hit_rate`
- **Reference**: `timeseries_path`

Updated `_order_columns()` to group columns by prefix for spreadsheet readability.

All new fields default to `None`/`False`/`0.0` when schema v3 data is absent (backwards compatible).

### Task 2: Time-Series Export

Created `results/timeseries.py` with three functions:
- `export_timeseries()` - Per-process JSON with compact samples and summary header
- `load_timeseries()` - Simple JSON loader with FileNotFoundError
- `aggregate_timeseries()` - Bundles per-process files into one aggregated file

Features: atomic writes (temp + rename), empty sample handling, summary statistics (power mean/min/max, memory mean/max, temperature mean/max, throttle count).

## Decisions Made

1. **Renamed CSV column `total_energy_j` to `energy_raw_j`**: Follows grouped-prefix convention. The Pydantic model field is unchanged; only the CSV column name changed.
2. **Compact JSON keys for timeseries**: Uses `t`, `mem_mb`, `sm_pct`, `throttle` instead of full names to keep file size manageable for long experiments at 100ms sampling intervals.

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- All 15 existing export tests pass unchanged
- CSV row contains 37 columns (up from ~15)
- Grouped prefixes verified: energy_(4), thermal_(3), env_(9), gpu_(3), latency_(2), batch_(1), kv_(1)
- Time-series export/load/aggregate all verified with mock samples
- Empty sample edge case verified
- FileNotFoundError verified

## Next Phase Readiness

No blockers. The time-series export is ready for integration with the PowerThermalSampler (Plan 02/03) and the results pipeline.
