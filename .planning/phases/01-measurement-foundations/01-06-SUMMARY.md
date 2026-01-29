# Phase 01 Plan 06: Unit Tests & UAT Round 1 Summary

## One-liner
72 new unit tests covering all Phase 1 modules (power/thermal, baseline, warmup, environment, schema v3, CSV export, timeseries) plus UAT validating A100 end-to-end experiment with baseline-adjusted energy.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Unit tests for Phase 1 modules | 98ca28e | tests/unit/test_domain_schema_v3.py, test_core_power_thermal.py, test_core_baseline.py, test_core_warmup.py, test_core_environment.py, test_results_exporters_v3.py, test_results_timeseries.py |
| 2 | UAT Round 1 (checkpoint) | e753c9c | src/llenergymeasure/orchestration/runner.py (2 bug fixes found during UAT) |

## What Was Built

### Test Files (7 new, 72 tests)

**test_domain_schema_v3.py** (15 tests) -- Schema v3 domain model tests:
- EnergyBreakdown: construction, defaults, with baseline values
- ThermalThrottleInfo: defaults (all false), with throttle detected
- WarmupResult: converged, not-converged, disabled states
- EnvironmentMetadata: construction, summary_line format, container detection
- Backwards compatibility: v2-style RawProcessResult (no v3 fields) + v3-style (all fields) both work

**test_core_power_thermal.py** (12 tests) -- PowerThermalSampler tests:
- Module imports, dataclass fields with defaults/values
- Sampler start/stop with mocked NVML (no GPU required)
- Thermal throttle info returns all-false when no samples
- Context manager pattern works with and without GPU
- PowerThermalResult.from_sampler() with empty sampler

**test_core_baseline.py** (11 tests) -- Baseline measurement tests:
- `adjust_energy_for_baseline`: 100J - 10W*5s = 50J, floor at zero, zero baseline, zero duration
- `create_energy_breakdown`: without baseline (unavailable), with fresh baseline, with cached baseline
- Cache invalidation: specific device, all devices, nonexistent device
- Caching reuses result (mocked pynvml); graceful None return without pynvml

**test_core_warmup.py** (6 tests) -- Warmup convergence tests:
- Disabled warmup: returns immediately, inference never called
- Convergence: stable latencies converge before max_prompts
- Non-convergence: noisy latencies with tight threshold hit max_prompts
- Fixed mode: runs exactly N prompts
- Exception handling: continues past inference failures
- All WarmupResult fields populated correctly (including final_cv=0 for identical latencies)

**test_core_environment.py** (8 tests) -- Environment metadata tests:
- Module and domain imports
- Degraded metadata returned when pynvml unavailable (not crash)
- Full construction with all fields
- Summary line format: GPU name, CUDA version, VRAM, temperature

**test_results_exporters_v3.py** (7 tests) -- Extended CSV export tests:
- v3 columns present: energy_adjusted_j, energy_baseline_w, thermal_throttle_detected, env_gpu_name
- Backwards compat: results without v3 fields export without crash
- Column ordering: energy_* < thermal_* < env_* (grouped prefixes)
- Full CSV write integration with all v3 fields

**test_results_timeseries.py** (13 tests) -- Time-series export/load tests:
- Round-trip: export then load preserves all data
- Empty samples: file created with sample_count=0
- Summary statistics: power (mean/min/max), memory, temperature, throttle count
- Compact keys: t, power_w, mem_mb, temp_c, sm_pct, throttle
- Relative timestamps: first sample at t=0, correct intervals, duration matches span

### UAT Round 1 Results

Experiment ran on A100-PCIE-40GB with gpt2, alpaca dataset, 5 prompts:
- **Environment**: A100-PCIE-40GB 40GB | CUDA 12.2 | Driver 535.261.03 | 25C
- **Baseline**: 42.9W measured (289 samples over 30s)
- **Energy**: 2584.47J raw, 238.25J adjusted (baseline correctly subtracted)
- **Thermal**: No throttling detected
- **CLI**: Environment + energy summary displayed correctly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Power thermal tests assumed no GPU**
- **Found during:** Task 1 first run
- **Issue:** Tests asserted empty samples and is_available=False, but the test machine has a real GPU with pynvml installed
- **Fix:** Mocked pynvml at the builtins.__import__ level for no-GPU tests; made GPU-present tests assert type correctness only
- **Files modified:** tests/unit/test_core_power_thermal.py
- **Commit:** 98ca28e (included in task commit)

**2. [Rule 1 - Bug] WarmupConfig validation error (window_size > max_prompts)**
- **Found during:** Task 1 first run
- **Issue:** WarmupConfig(max_prompts=3) fails because default window_size=5 > 3
- **Fix:** Changed test to use max_prompts=5 with explicit window_size=5
- **Files modified:** tests/unit/test_core_warmup.py
- **Commit:** 98ca28e

### UAT Bugs (Fixed by User During Checkpoint)

**3. Warmup skip for backend-managed models**
- BackendModelLoaderAdapter returns (None, None) for model/tokenizer
- Warmup now skips gracefully when model is backend-managed

**4. Energy breakdown duration fix**
- CodeCarbon reports duration_sec=0.0 in some cases
- Energy breakdown now uses experiment timestamps for baseline subtraction

Both fixed in commit e753c9c.

## Decisions Made

None -- pure test/validation plan, no architectural decisions.

## Metrics

- **Duration:** ~15 min (including UAT checkpoint wait)
- **Tests added:** 72
- **Total test suite:** 863 passing
- **Regressions:** 0
- **Completed:** 2026-01-29

## Phase 1 Completion Status

With this plan complete, **all 6 plans in Phase 1 are done**:

| Plan | Name | Status |
|------|------|--------|
| 01-01 | Domain models + config extensions | Done |
| 01-02 | NVML measurement primitives | Done |
| 01-03 | Warmup convergence detection | Done |
| 01-04 | Extended CSV & time-series export | Done |
| 01-05 | Orchestrator integration | Done |
| 01-06 | Unit tests & UAT Round 1 | Done |

Phase 1 delivers: baseline power measurement, thermal throttling detection, environment metadata, warmup convergence, time-series sampling, schema v3 with backwards compatibility, extended CSV export, and full orchestrator integration with 863 passing tests.
