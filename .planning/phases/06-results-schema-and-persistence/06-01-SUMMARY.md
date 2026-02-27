---
phase: 06-results-schema-and-persistence
plan: 01
subsystem: domain
tags: [results, schema, pydantic, domain-models]
dependency_graph:
  requires: [05-energy-measurement]
  provides: [ExperimentResult-v2.0, MultiGPUMetrics, compute_measurement_config_hash]
  affects: [06-02-persistence, 06-03-aggregation]
tech_stack:
  added: []
  patterns: [pydantic-frozen-model, sha256-config-hash, type-checking-guard-for-circular-import]
key_files:
  created:
    - tests/unit/test_experiment_result_v2.py
  modified:
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/domain/metrics.py
decisions:
  - "RawProcessResult preserved with inference_metrics, energy_metrics, compute_metrics — required by existing aggregation/CLI consumers"
  - "ExperimentConfig import uses TYPE_CHECKING guard in experiment.py to avoid circular import with config/models.py"
  - "Inner class-body imports removed from RawProcessResult — Pydantic raises PydanticUserError for non-annotated class attributes"
metrics:
  duration_sec: 220
  completed_date: "2026-02-26"
  tasks_completed: 2
  files_modified: 3
---

# Phase 6 Plan 01: ExperimentResult v2.0 Schema Summary

ExperimentResult rewritten to v2.0 schema with all 11 RES requirements: measurement_config_hash (SHA-256[:16]), measurement_methodology Literal, steady_state_window, baseline_power_w, energy_adjusted_j, energy_per_device_j, environment_snapshot, measurement_warnings, warmup_excluded_samples, reproducibility_notes, and schema_version="2.0".

## What Was Built

### ExperimentResult v2.0 (`src/llenergymeasure/domain/experiment.py`)

Complete rewrite of the `ExperimentResult` class to the v2.0 schema:

- **Identity fields:** `schema_version="2.0"`, `experiment_id`, `measurement_config_hash` (required)
- **Methodology fields (RES-03, RES-04):** `measurement_methodology: Literal["total", "steady_state", "windowed"]`, `steady_state_window: tuple[float, float] | None`
- **Energy detail (RES-06, RES-07):** `baseline_power_w`, `energy_adjusted_j`, `energy_per_device_j`, `energy_breakdown`
- **Multi-GPU:** `multi_gpu: MultiGPUMetrics | None`
- **Environment (RES-09):** `environment_snapshot: EnvironmentSnapshot | None`
- **Quality (RES-08, RES-10, RES-11):** `measurement_warnings`, `warmup_excluded_samples`, `reproducibility_notes`
- **Timeseries:** renamed `timeseries_path` → `timeseries` (relative filename, portable)
- **`aggregation` made optional** (`AggregationMetadata | None`) — not needed for single-GPU
- **`extended_metrics` changed** from `Field(default_factory=ExtendedEfficiencyMetrics)` to `ExtendedEfficiencyMetrics | None = None`
- **Model frozen:** `model_config = {"frozen": True}`
- **`AggregatedResult = ExperimentResult` alias preserved** (v1.x compat)

### MultiGPUMetrics (`src/llenergymeasure/domain/metrics.py`)

New model added before `EnergyBreakdown`:

```python
class MultiGPUMetrics(BaseModel):
    num_gpus: int
    energy_per_gpu_j: list[float]
    energy_total_j: float
    energy_per_output_token_j: float
```

### compute_measurement_config_hash() (`src/llenergymeasure/domain/experiment.py`)

Module-level utility function. SHA-256[:16] of `ExperimentConfig.model_dump()` serialised with `sort_keys=True`. Returns a deterministic 16-char hex string. Layer 3 fields excluded naturally (they are not in `ExperimentConfig`).

### Unit Tests (`tests/unit/test_experiment_result_v2.py`)

30 tests covering all v2.0 fields and behaviours:
- Schema version default, all required/optional fields
- Literal validation for `measurement_methodology`
- Tuple handling for `steady_state_window`
- Frozen model enforcement
- `duration_sec` and `tokens_per_joule` properties
- Config hash: 16-char hex, deterministic, config-sensitive
- `MultiGPUMetrics` validation
- Full JSON round-trip (datetime, tuple, nested models)

## Commits

| Hash | Description |
|------|-------------|
| 8b68ec4 | feat(results): rewrite ExperimentResult to v2.0 schema with all RES fields |
| e107165 | test(results): add 30 unit tests for ExperimentResult v2.0 schema |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Inner class-body imports in RawProcessResult**

- **Found during:** Task 1 implementation
- **Issue:** The plan's `RawProcessResult` template had `from llenergymeasure.domain.metrics import (ComputeMetrics, EnergyMetrics, InferenceMetrics)` inside the class body. Pydantic raises `PydanticUserError: A non-annotated attribute was detected` for any name introduced via an import statement inside a class body.
- **Fix:** Moved all imports to module-level (already imported by the existing code).
- **Files modified:** `src/llenergymeasure/domain/experiment.py`
- **Commit:** 8b68ec4

**2. [Rule 2 - Missing critical functionality] RawProcessResult metrics fields omitted in draft**

- **Found during:** Task 1 — running verification revealed import error, investigation showed `RawProcessResult` was missing `inference_metrics`, `energy_metrics`, `compute_metrics` fields
- **Issue:** These required fields are used by `core/implementations.py`, `orchestration/runner.py`, `cli/display/results.py`. Removing them would break existing code.
- **Fix:** Added the three fields back to `RawProcessResult` as required fields matching the original.
- **Files modified:** `src/llenergymeasure/domain/experiment.py`
- **Commit:** 8b68ec4

## Self-Check: PASSED

- FOUND: `src/llenergymeasure/domain/experiment.py`
- FOUND: `src/llenergymeasure/domain/metrics.py`
- FOUND: `tests/unit/test_experiment_result_v2.py`
- FOUND: commit 8b68ec4
- FOUND: commit e107165
