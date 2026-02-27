---
phase: 06-results-schema-and-persistence
plan: 03
subsystem: results
tags: [results, aggregation, exporters, logging, v2.0]
dependency_graph:
  requires: [06-01]
  provides: [aggregate_results-v2.0, exporters-v2.0, aggregation-unit-tests]
  affects: [06-04-pytorch-backend-wiring]
tech_stack:
  added: []
  patterns: [stdlib-logging, late-aggregation, sum-energy-avg-throughput]
key_files:
  created:
    - tests/unit/test_aggregation_v2.py
  modified:
    - src/llenergymeasure/results/aggregation.py
    - src/llenergymeasure/results/exporters.py
decisions:
  - "aggregate_results() signature keeps backward-compatible positional params (experiment_id, measurement_config_hash) — callers can pass without keyword"
  - "warmup_result and thermal_throttle fall through from first process if not explicitly passed — avoids caller needing to extract them"
  - "_aggregate_extended_metrics_from_results returns ExtendedEfficiencyMetrics | None — matches ExperimentResult.extended_metrics field type"
metrics:
  duration_sec: 310
  completed_date: "2026-02-27"
  tasks_completed: 2
  files_modified: 3
---

# Phase 6 Plan 03: Aggregation v2.0 and Exporters Summary

aggregate_results() updated to return ExperimentResult (v2.0 schema) with all new fields; loguru replaced with stdlib logging in both aggregation.py and exporters.py; dead campaign-level code removed; CSV exporter updated to v2.0 field names; 15 unit tests pass.

## What Was Built

### aggregation.py — v2.0 aggregate_results()

Complete signature update:

```python
def aggregate_results(
    raw_results: list[RawProcessResult],
    experiment_id: str,
    measurement_config_hash: str,
    measurement_methodology: str = "total",
    steady_state_window: tuple[float, float] | None = None,
    baseline_power_w: float | None = None,
    energy_adjusted_j: float | None = None,
    energy_per_device_j: list[float] | None = None,
    energy_breakdown: EnergyBreakdown | None = None,
    multi_gpu: MultiGPUMetrics | None = None,
    environment_snapshot: EnvironmentSnapshot | None = None,
    measurement_warnings: list[str] | None = None,
    warmup_excluded_samples: int | None = None,
    warmup_result: WarmupResult | None = None,
    thermal_throttle: ThermalThrottleInfo | None = None,
    timeseries: str | None = None,
    effective_config: dict[str, Any] | None = None,
    ...
) -> ExperimentResult:
```

Key behaviours preserved or updated:
- **Energy**: summed across all processes
- **Tokens**: summed across all processes
- **Throughput**: averaged across all processes (per-GPU rate)
- **Latencies**: concatenated (late aggregation — no average of averages)
- **FLOPs**: summed
- **Time**: wall-clock range (earliest start to latest end)
- **energy_breakdown / thermal_throttle**: auto-derived from process results if not passed
- **warmup_result**: taken from first process if not passed

### Dead code removed

- `calculate_efficiency_metrics()` — v1.x utility, not in v2.0 path
- `aggregate_campaign_results()` — campaign-level, not in v2.0 path
- `aggregate_campaign_with_grouping()` — campaign-level, not in v2.0 path
- `_extract_field_value()` — helper for the above

### exporters.py — v2.0 ExperimentResult

- Import changed from `AggregatedResult` to `ExperimentResult`
- `export_aggregated_to_csv()` signature updated to `list[ExperimentResult]`
- `_aggregated_to_row()` updated to use v2.0 field names:
  - `environment_snapshot` (not `environment`)
  - `timeseries` (not `timeseries_path`)
  - `measurement_config_hash` added
  - `schema_version` added
  - `measurement_methodology` and `steady_state_window` added
  - Aggregation fields guarded for `None` (single-GPU results have `aggregation=None`)
- `ResultsExporter` class updated to use `ExperimentResult` throughout

### Unit tests (`tests/unit/test_aggregation_v2.py`)

15 tests covering:

| Test | What it verifies |
|------|-----------------|
| `test_aggregate_single_process` | Single-process produces correct metrics |
| `test_aggregate_returns_experiment_result` | Return type is ExperimentResult |
| `test_aggregate_schema_version` | schema_version == "2.0" |
| `test_aggregate_measurement_config_hash` | Hash passes through unchanged |
| `test_aggregate_measurement_methodology` | Methodology passes through |
| `test_aggregate_energy_sum` | 25 + 30 = 55 J across two processes |
| `test_aggregate_tokens_sum` | 500 + 600 = 1100 tokens across two processes |
| `test_aggregate_late_aggregation_latencies` | Latencies concatenated, not averaged |
| `test_aggregate_process_results_embedded` | Original RawProcessResult objects embedded |
| `test_aggregate_metadata_num_processes` | aggregation.num_processes == 3 |
| `test_validate_process_completeness_complete` | 2 procs + 2 markers → is_complete=True |
| `test_validate_process_completeness_missing` | 1 missing → is_complete=False |
| `test_validate_process_completeness_duplicate` | Duplicate index → is_complete=False |
| `test_no_loguru_import` | aggregation.py AST scan: no loguru |
| `test_csv_exporter_no_loguru` | exporters.py AST scan: no loguru |

## Commits

| Hash | Description |
|------|-------------|
| c3d1dac | refactor(results): update aggregation and exporters to v2.0 schema |
| e634bf7 | test(results): add 15 unit tests for v2.0 aggregation |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] v1.x field references on RawProcessResult no longer exist**

- **Found during:** Task 1 implementation — reading the current aggregation.py
- **Issue:** The v1.x aggregation.py referenced `r.energy_tracking_failed`, `r.timeseries_path`, `r.cli_overrides`, `r.config_warnings`, and `r.environment` on RawProcessResult. These fields were removed in Plan 01 when RawProcessResult was updated to v2.0 schema.
- **Fix:** Removed all references to removed fields. `energy_tracking_failed` warning is no longer generated. `timeseries`, `effective_config` are now caller-supplied parameters. `environment_snapshot` is also a caller-supplied parameter (the PyTorchBackend will collect it and pass it in).
- **Files modified:** `src/llenergymeasure/results/aggregation.py`
- **Commit:** c3d1dac

**2. [Rule 1 - Bug] exporters.py referenced result.environment (old v1.x field)**

- **Found during:** Task 1 — reading exporters.py
- **Issue:** `_aggregated_to_row()` used `result.environment` which no longer exists; v2.0 uses `result.environment_snapshot`. Similarly `result.timeseries_path` → `result.timeseries`, `result.aggregation.num_processes` used without None guard.
- **Fix:** Updated all field accesses to v2.0 names. Added None guard on `result.aggregation` (single-GPU results have `aggregation=None`).
- **Files modified:** `src/llenergymeasure/results/exporters.py`
- **Commit:** c3d1dac

## Self-Check: PASSED

- FOUND: `src/llenergymeasure/results/aggregation.py`
- FOUND: `src/llenergymeasure/results/exporters.py`
- FOUND: `tests/unit/test_aggregation_v2.py`
- FOUND: commit c3d1dac
- FOUND: commit e634bf7
- 15 tests pass, 0 failures
