---
phase: 06-results-schema-and-persistence
verified: 2026-02-26T23:44:59Z
status: passed
score: 18/18 requirements verified
re_verification: false
---

# Phase 6: Results Schema and Persistence Verification Report

**Phase Goal:** Every experiment produces a complete, schema-versioned ExperimentResult written to a stable output directory — with collision-safe naming, a Parquet timeseries sidecar, and a round-trip-safe persistence API.
**Verified:** 2026-02-26T23:44:59Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | ExperimentResult has `schema_version="2.0"` | VERIFIED | `Field(default="2.0")` at line 150 of `experiment.py`; `test_schema_version_default_is_2_0` passes |
| 2 | ExperimentResult has `measurement_config_hash` (required str) | VERIFIED | `Field(...)` at line 152; `test_measurement_config_hash_field` passes |
| 3 | ExperimentResult has `measurement_methodology` Literal | VERIFIED | `Literal["total", "steady_state", "windowed"]` at line 163; validation tests pass |
| 4 | ExperimentResult has `steady_state_window: tuple[float, float] \| None` | VERIFIED | Line 166; `test_steady_state_window_tuple` and round-trip tests pass |
| 5 | ExperimentResult has energy detail fields | VERIFIED | `baseline_power_w`, `energy_adjusted_j`, `energy_per_device_j`, `energy_breakdown` all present |
| 6 | ExperimentResult has `reproducibility_notes` with fixed NVML disclaimer | VERIFIED | Default string contains "NVML" and "+/-5%"; `test_reproducibility_notes_default` passes |
| 7 | ExperimentResult has `environment_snapshot: EnvironmentSnapshot \| None` | VERIFIED | Line 199; `test_environment_snapshot_optional` passes |
| 8 | ExperimentResult has `measurement_warnings: list[str]` | VERIFIED | Line 204; `test_measurement_warnings_default_empty` passes |
| 9 | ExperimentResult has `warmup_excluded_samples: int \| None` | VERIFIED | Line 208; `test_warmup_excluded_samples_optional` passes |
| 10 | ExperimentResult is frozen | VERIFIED | `model_config = {"frozen": True}` at line 259; `test_frozen_model` passes |
| 11 | `AggregatedResult = ExperimentResult` alias preserved | VERIFIED | Line 322 of `experiment.py`; `AggregatedResult is ExperimentResult` asserts True |
| 12 | `compute_measurement_config_hash()` returns 16-char hex | VERIFIED | SHA-256[:16] of `ExperimentConfig.model_dump(sort_keys=True)`; determinism tests pass |
| 13 | `MultiGPUMetrics` model exists with required fields | VERIFIED | `class MultiGPUMetrics(BaseModel)` at line 269 of `metrics.py`; `test_multi_gpu_metrics_model` passes |
| 14 | `result.save(output_dir)` creates `{model}_{backend}_{timestamp}/result.json` | VERIFIED | `_experiment_dir_name()` + `_find_collision_free_dir()` in `persistence.py`; `test_save_directory_name_format` passes |
| 15 | Atomic writes via temp + `os.replace()` | VERIFIED | `_atomic_write()` in `persistence.py` lines 53-67; `test_save_atomic_write` passes |
| 16 | Collision suffix `_1`, `_2` applied — never overwrites | VERIFIED | `_find_collision_free_dir()` loop; `test_collision_suffix_applied`, `test_never_overwrites` (5 saves → 5 distinct dirs) pass |
| 17 | `ExperimentResult.from_json(path)` round-trips all fields | VERIFIED | `load_result()` via `model_validate_json()`; `test_from_json_round_trip` including datetime and tuple pass |
| 18 | Missing sidecar loads successfully with `UserWarning` | VERIFIED | `warnings.warn(... UserWarning ...)` in `load_result()`; `test_from_json_missing_sidecar_warns` passes |
| 19 | `aggregate_results()` returns v2.0 `ExperimentResult` | VERIFIED | Return type `ExperimentResult`; new params `measurement_config_hash`, `measurement_methodology` in signature |
| 20 | Late aggregation concatenates per-process latencies | VERIFIED | `test_aggregate_late_aggregation_latencies` passes — lists concatenated, not averaged |
| 21 | `validate_process_completeness()` performs 4 checks | VERIFIED | Count, contiguity, no duplicates, marker files — all in function body lines 72-119 |
| 22 | `loguru` removed from `aggregation.py` and `exporters.py` | VERIFIED | `test_no_loguru_import` and `test_csv_exporter_no_loguru` pass (AST scan) |
| 23 | `export_aggregated_to_csv()` uses `ExperimentResult` v2.0 | VERIFIED | Import `from llenergymeasure.domain.experiment import ExperimentResult` at line 10 of `exporters.py` |

**Score:** 23/23 truths verified (18 requirements, 23 observable truths across 3 plans)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/domain/experiment.py` | ExperimentResult v2.0 schema | VERIFIED | 323 lines; all v2.0 fields present; `save()`, `from_json()`, `compute_measurement_config_hash()` all wired |
| `src/llenergymeasure/domain/metrics.py` | `MultiGPUMetrics` model | VERIFIED | `class MultiGPUMetrics` at line 269; `num_gpus`, `energy_per_gpu_j`, `energy_total_j`, `energy_per_output_token_j` |
| `src/llenergymeasure/results/persistence.py` | v2.0 persistence API | VERIFIED | New module; `save_result()`, `load_result()`, `_atomic_write()`, `_find_collision_free_dir()` |
| `src/llenergymeasure/results/aggregation.py` | v2.0 `aggregate_results()` | VERIFIED | Updated; no loguru; dead campaign code removed; new params for v2.0 fields |
| `src/llenergymeasure/results/exporters.py` | CSV exporter using v2.0 | VERIFIED | Updated; `ExperimentResult` (not `AggregatedResult`); no loguru |
| `tests/unit/test_experiment_result_v2.py` | Schema unit tests | VERIFIED | 30 tests; all pass |
| `tests/unit/test_persistence_v2.py` | Persistence unit tests | VERIFIED | 16 tests; all pass |
| `tests/unit/test_aggregation_v2.py` | Aggregation unit tests | VERIFIED | 15 tests; all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `experiment.py` | `metrics.py` | `from llenergymeasure.domain.metrics import EnergyBreakdown, WarmupResult, ThermalThrottleInfo, MultiGPUMetrics, ...` | WIRED | Lines 13-23; all types used in field definitions |
| `experiment.py` | `environment.py` | `from llenergymeasure.domain.environment import EnvironmentSnapshot` | WIRED | Line 12; used in `environment_snapshot` field |
| `experiment.py` | `persistence.py` | Deferred import inside `save()` and `from_json()` body | WIRED | Lines 288 and 303; deferred to avoid circular import; verified by round-trip tests |
| `persistence.py` | `experiment.py` | `from llenergymeasure.domain.experiment import ExperimentResult` (deferred in `load_result()`) | WIRED | Line 124 inside function body |
| `aggregation.py` | `experiment.py` | `from llenergymeasure.domain.experiment import ExperimentResult, RawProcessResult, AggregationMetadata` | WIRED | Lines 14-18 |
| `exporters.py` | `experiment.py` | `from llenergymeasure.domain.experiment import ExperimentResult, RawProcessResult` | WIRED | Line 10 |
| `__init__.py` | `experiment.py` | `from llenergymeasure.domain.experiment import ExperimentResult, StudyResult` | WIRED | Line 14; public API export verified |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| RES-01 | 06-01 | `ExperimentResult` — all v2.0 fields ship together | SATISFIED | All fields present; 30 schema tests pass |
| RES-02 | 06-01 | `measurement_config_hash: str` — SHA-256[:16] | SATISFIED | Field + `compute_measurement_config_hash()` function; determinism tests pass |
| RES-03 | 06-01 | `measurement_methodology: Literal["total", "steady_state", "windowed"]` | SATISFIED | Literal type enforced; invalid value raises `ValidationError` |
| RES-04 | 06-01 | `steady_state_window: tuple[float, float] \| None` | SATISFIED | Field present; tuple round-trips through JSON |
| RES-05 | 06-01 | `schema_version: str = "2.0"` (not "2.0.0") | SATISFIED | `Field(default="2.0")`; `test_schema_version_default_is_2_0` passes |
| RES-06 | 06-01 | `baseline_power_w`, `energy_adjusted_j`, `energy_per_device_j` | SATISFIED | All three optional fields present with correct types |
| RES-07 | 06-01 | `EnergyBreakdown` nested model carry-forward | SATISFIED | `energy_breakdown: EnergyBreakdown \| None` in `ExperimentResult`; model in `metrics.py` |
| RES-08 | 06-01 | `reproducibility_notes: str` — fixed NVML disclaimer | SATISFIED | Default string "Energy measured via NVML polling. Accuracy +/-5%..." |
| RES-09 | 06-01 | `environment_snapshot: EnvironmentSnapshot` | SATISFIED | Optional field present; `EnvironmentSnapshot` imported from `environment.py` |
| RES-10 | 06-01 | `measurement_warnings: list[str]` | SATISFIED | `Field(default_factory=list)` present |
| RES-11 | 06-01 | `warmup_excluded_samples: int \| None` | SATISFIED | Optional field present with correct type |
| RES-12 | 06-03 | Process completeness validation (4 checks) | SATISFIED | `validate_process_completeness()`: count, contiguity, no duplicates, marker files |
| RES-16 | 06-02 | Output in `{name}_{timestamp}/result.json` + `timeseries.parquet` | SATISFIED | `save_result()` creates subdirectory; timeseries sidecar copied when provided |
| RES-17 | 06-02 | Collision policy: append `_1`, `_2` — never overwrite | SATISFIED | `_find_collision_free_dir()` loop verified; 5-save never-overwrite test passes |
| RES-18 | 06-02 | JSON primary, Parquet timeseries sidecar, CSV opt-in | SATISFIED | JSON always written; Parquet copied when `timeseries_source` provided; `export_aggregated_to_csv()` opt-in |
| RES-19 | 06-02 | `save()`, `from_json()` on ExperimentResult (design spec uses `save()` not `to_json()`) | SATISFIED | Both methods wired as thin delegation wrappers; design spec (`result-schema.md` line 25) confirms `save()` naming |
| RES-20 | 06-03 | Late aggregation in `results/aggregation.py` | SATISFIED | Per-process latencies concatenated; `test_aggregate_late_aggregation_latencies` passes |
| RES-21 | 06-03 | Unified output layout — all backends → one `ExperimentResult` | SATISFIED | `aggregate_results()` returns single `ExperimentResult`; single-GPU trivially works |

**All 18 requirements: SATISFIED**

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `metrics.py` | 237 | `placeholder` (method name on `EnergyMetrics`) | Info | Not a stub — legitimate factory method name for energy measurement fallback |

No blocker or warning-level anti-patterns found. The `placeholder` item is a false positive (it is a `@classmethod` named `placeholder` that creates `EnergyMetrics` with zeroed values when energy tracking is unavailable — correct and intentional).

---

### Human Verification Required

None required. All truths and requirements are verifiable programmatically. The full test suite (61 tests) passes with no failures.

---

### Test Suite Summary

All 61 tests pass in 3.84 seconds (Python 3.10.14, pytest 8.4.2):

- `tests/unit/test_experiment_result_v2.py`: **30 passed** — schema fields, validation, frozen model, hash function, JSON round-trip, property methods, MultiGPUMetrics
- `tests/unit/test_persistence_v2.py`: **16 passed** — directory naming, slug normalisation, atomic writes, collision handling, round-trip fidelity, sidecar management, missing sidecar graceful degradation
- `tests/unit/test_aggregation_v2.py`: **15 passed** — v2.0 schema from aggregation, energy/token summing, late aggregation, process completeness validation, loguru removal (AST scan)

---

### Notes on RES-19 Naming

RES-19 in `.product/REQUIREMENTS.md` lists `to_json()`, `to_parquet()`, `from_json()`. The implementation uses `save()` and `from_json()`. This deviation is **intentional and correct**: the locked design decision in `.product/designs/result-schema.md` (line 25) explicitly specifies `result.save(output_dir)` and `ExperimentResult.from_json(path)` as the API names. The requirements file pre-dates the design finalisation. The implementation follows the design spec.

---

_Verified: 2026-02-26T23:44:59Z_
_Verifier: Claude (gsd-verifier)_
