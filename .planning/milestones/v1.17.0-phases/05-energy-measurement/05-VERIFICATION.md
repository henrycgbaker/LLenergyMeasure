---
phase: 05-energy-measurement
verified: 2026-02-26T16:00:00Z
status: gaps_found
score: 15/18 requirements verified
gaps:
  - truth: "REQUIREMENTS.md status updated to reflect Phase 5 completions"
    status: failed
    reason: "CM-15, CM-16, CM-17, CM-18, CM-19, CM-20, CM-25 remain marked 'Pending' in REQUIREMENTS.md despite being fully implemented and tested in Plan 03"
    artifacts:
      - path: ".planning/REQUIREMENTS.md"
        issue: "7 Phase 5 requirements show 'Pending' status but are implemented and passing tests"
    missing:
      - "Update REQUIREMENTS.md lines 143-148 and 153 to 'Complete' for CM-15 through CM-20 and CM-25"
  - truth: "_prepare_prompts() uses real dataset loading, not synthetic placeholder"
    status: partial
    reason: "'M1 placeholder' prompt generation noted in comments and docstring. CONTEXT.md states Phase 5 adds dataset loading (aienergyscore + SyntheticDatasetConfig) but plan documents do not include this as a task. The CONTEXT.md comment 'Phase 5 replaces this with proper dataset loading' is inaccurate — Phase 5 plans (01-03) have no such task. However, this may be intentional scope deferral."
    artifacts:
      - path: "src/llenergymeasure/core/backends/pytorch.py"
        issue: "_prepare_prompts() generates repeated 'Hello, ' strings (L343-L348). Comment says 'M1 placeholder, Phase 5 adds dataset loading' — Phase 5 did not include this."
    missing:
      - "Clarify whether dataset loading is out-of-scope for Phase 5 or a missed task. If out-of-scope, remove the 'Phase 5 replaces this' comment to avoid misleading the next verifier."
human_verification:
  - test: "Run a real GPU experiment end-to-end and verify energy measurement populates"
    expected: "ExperimentResult.total_energy_j > 0, energy_breakdown populated with baseline_power_w, timeseries.parquet written"
    why_human: "Cannot invoke GPU inference in host environment — CUDA only available inside containers"
  - test: "Verify CUDA sync timing is correct relative to energy stop"
    expected: "_cuda_sync() called at line 139, energy_backend.stop_tracking() at line 144 — sync must precede stop"
    why_human: "Sequencing is verified in code review but only a live GPU trace can confirm the wall-clock ordering"
---

# Phase 5: Energy Measurement Verification Report

**Phase Goal:** Energy measurement integration — energy backends, warmup refinement, FLOPs estimation, timeseries, measurement warnings, and PyTorchBackend integration
**Verified:** 2026-02-26
**Status:** gaps_found (documentation drift + one ambiguous scope item)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | NVMLBackend implements EnergyBackend protocol, integrates power samples to joules via trapezoidal rule | VERIFIED | `nvml.py` L81-121 — loop over consecutive sample pairs, `total_j += avg_power * dt` |
| 2 | ZeusBackend wraps ZeusMonitor and returns EnergyMeasurement with per-GPU joule breakdown | VERIFIED | `zeus.py` L61-82 — `tracker.end_window()`, `sum(per_gpu_j.values())`, returns `EnergyMeasurement` |
| 3 | select_energy_backend('auto') returns Zeus > NVML > CodeCarbon priority | VERIFIED | `__init__.py` L168-188 — `_auto_select()` probes find_spec("zeus") then NVMLBackend.is_available() then find_spec("codecarbon") |
| 4 | select_energy_backend('zeus') raises ConfigError with install guidance when zeus not installed | VERIFIED | `__init__.py` L157-165 — `_instantiate(explicit)` + `is_available()` check raises `ConfigError` |
| 5 | select_energy_backend(None) returns None without warnings | VERIFIED | `__init__.py` L151-152 — `if explicit is None: return None` |
| 6 | pyproject.toml uses zeus>=0.13.1 | VERIFIED | `pyproject.toml` — `zeus = ["zeus>=0.13.1"]` |
| 7 | WarmupConfig.n_warmup defaults to 5 | VERIFIED | `models.py` L136+ — `n_warmup: int = Field(default=5, ge=1, ...)` |
| 8 | WarmupConfig has CV opt-in fields with correct defaults | VERIFIED | `models.py` — `convergence_detection=False`, `cv_threshold=0.05`, `max_prompts=20`, `window_size=5`, `min_prompts=5` |
| 9 | thermal_floor_seconds enforces ge=30.0 minimum | VERIFIED | `models.py` L141+ — `thermal_floor_seconds: float = Field(ge=30.0, ...)`. ValidationError raised for 29.0 (tested) |
| 10 | WarmupResult has 7 fields including thermal_floor_wait_s | VERIFIED | `metrics.py` L344-363 — all 7 fields present including `thermal_floor_wait_s: float = 0.0` |
| 11 | FlopsResult.method Literal includes 'palm_formula' | VERIFIED | `metrics.py` L183 — `Literal["calflops", "architecture", "parameter_estimate", "palm_formula"]` |
| 12 | estimate_flops_palm() implements 2 * N_non_embedding * total_tokens | VERIFIED | `flops.py` L43+ — `flops_total = 2 * n_params * batch * (n_input + n_output)`. Tested: 2 * 100 * 15 = 3000 |
| 13 | _count_non_embedding_params() excludes embedding layers | VERIFIED | `flops.py` L34-41 — `if "embed" not in name.lower()` |
| 14 | ExperimentConfig has energy.backend field | VERIFIED | `models.py` L203+, L329-330 — `class EnergyConfig`, `energy: EnergyConfig = Field(default_factory=EnergyConfig)` |
| 15 | ExperimentResult has measurement_warnings field | VERIFIED | `experiment.py` L246 — `measurement_warnings: list[str] = Field(default_factory=list)` |
| 16 | PyTorchBackend.run() follows correct energy lifecycle | VERIFIED | `backends/pytorch.py` L66-192 — 13-step lifecycle with baseline before load, energy after thermal floor, CUDA sync before stop |
| 17 | Timeseries written as 1 Hz Parquet with locked 8-column schema | VERIFIED | `timeseries.py` L17-113 — `_timeseries_schema()` defines locked schema, 1-second bucketing implemented |
| 18 | REQUIREMENTS.md status is current for all Phase 5 requirements | FAILED | CM-15, CM-16, CM-17, CM-18, CM-19, CM-20, CM-25 remain "Pending" in REQUIREMENTS.md despite being implemented |
| 19 | Prompt preparation uses dataset loading (not placeholder) | PARTIAL | `backends/pytorch.py` L330-348 — `_prepare_prompts()` is an "M1 placeholder" generating repeated "Hello, " strings. Phase 5 CONTEXT.md stated this would be replaced but Plan 03 has no task for it |

**Score:** 16/19 truths verified (85% — 2 failures, 1 partial — but the partial is likely out-of-scope)

---

## Required Artifacts

### Plan 01 Artifacts

| Artifact | Expected | Status | Details |
|----------|---------|--------|---------|
| `src/llenergymeasure/core/energy_backends/nvml.py` | NVMLBackend + trapezoidal integration | VERIFIED | 122 LOC, `class NVMLBackend`, `EnergyMeasurement` dataclass, full trapezoidal integration |
| `src/llenergymeasure/core/energy_backends/zeus.py` | ZeusBackend wrapping ZeusMonitor | VERIFIED | 83 LOC, `class ZeusBackend`, `WINDOW_NAME = "llem_measurement"`, deferred imports |
| `src/llenergymeasure/core/energy_backends/__init__.py` | select_energy_backend() + registry | VERIFIED | 231 LOC, `select_energy_backend`, `_auto_select()`, legacy registry retained |
| `tests/unit/test_energy_backends_v2.py` | Unit tests (min 9) | VERIFIED | 17 tests collected, all pass |

### Plan 02 Artifacts

| Artifact | Expected | Status | Details |
|----------|---------|--------|---------|
| `src/llenergymeasure/config/models.py` (WarmupConfig) | n_warmup=5, CV fields, EnergyConfig | VERIFIED | Contains `convergence_detection`, `cv_threshold`, `max_prompts`, `window_size`, `min_prompts`, `enabled`, `EnergyConfig` class |
| `src/llenergymeasure/domain/experiment.py` | measurement_warnings field | VERIFIED | L246 — `measurement_warnings: list[str]` |
| `src/llenergymeasure/core/flops.py` | estimate_flops_palm() primary method | VERIFIED | L34-82 — `_count_non_embedding_params()`, `estimate_flops_palm()`, legacy `FlopsEstimator` kept |
| `src/llenergymeasure/domain/metrics.py` (FlopsResult) | palm_formula in method Literal | VERIFIED | L183 — Literal includes "palm_formula" |
| `src/llenergymeasure/domain/metrics.py` (WarmupResult) | thermal_floor_wait_s field | VERIFIED | L360-363 — `thermal_floor_wait_s: float = Field(default=0.0, ge=0.0)` |
| `tests/unit/test_warmup_v2.py` | Warmup config + CV tests | VERIFIED | 31 tests collected, all pass |
| `tests/unit/test_flops_v2.py` | PaLM FLOPs + embedding exclusion | VERIFIED | 17 tests collected, all pass |

### Plan 03 Artifacts

| Artifact | Expected | Status | Details |
|----------|---------|--------|---------|
| `src/llenergymeasure/core/backends/pytorch.py` | Full energy lifecycle wired | VERIFIED | 803 LOC, select_energy_backend, estimate_flops_palm, write_timeseries_parquet, collect_measurement_warnings all present and called |
| `src/llenergymeasure/core/timeseries.py` | write_timeseries_parquet() | VERIFIED | 114 LOC, 8-column locked schema, 1 Hz bucketing |
| `src/llenergymeasure/core/measurement_warnings.py` | collect_measurement_warnings() | VERIFIED | 72 LOC, 4 warning flags with remediation text |
| `tests/unit/test_measurement_integration.py` | Integration tests (min 10) | VERIFIED | 20 tests collected, all pass |

---

## Key Link Verification

### Plan 01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `energy_backends/__init__.py` | `energy_backends/nvml.py` | `from .nvml import EnergyMeasurement, NVMLBackend` | VERIFIED | L43 in `__init__.py` |
| `energy_backends/__init__.py` | `energy_backends/zeus.py` | find_spec("zeus") probe | VERIFIED | L171 — `importlib.util.find_spec("zeus")` |

### Plan 02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `core/flops.py` | `domain/metrics.py` | `from llenergymeasure.domain.metrics import FlopsResult` | VERIFIED | Confirmed via successful PaLM FLOPs test execution |
| `core/warmup.py` | `config/models.py` | `from llenergymeasure.config.models import WarmupConfig` | VERIFIED | warmup.py references WarmupConfig fields (n_warmup, max_prompts, cv_threshold, etc.) |

### Plan 03 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `core/backends/pytorch.py` | `core/energy_backends/__init__.py` | `from ... import select_energy_backend` | VERIFIED | L121-122 (deferred import in run()) |
| `core/backends/pytorch.py` | `core/flops.py` | `from ... import estimate_flops_palm` | VERIFIED | L148 (deferred import) |
| `core/backends/pytorch.py` | `core/baseline.py` | `from ... import measure_baseline_power`, `create_energy_breakdown` | VERIFIED | L89, L733 (two deferred imports) |
| `core/backends/pytorch.py` | `core/timeseries.py` | `from ... import write_timeseries_parquet` | VERIFIED | L163 (deferred import) |
| `core/backends/pytorch.py` | `core/measurement_warnings.py` | `from ... import collect_measurement_warnings` | VERIFIED | L664-665 (deferred import in `_collect_warnings()`) |

All 9 key links: VERIFIED.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| CM-11 | Plan 01 | NVML poller always available — base install | SATISFIED | `nvml.py` — `NVMLBackend` with pynvml deferred import; no zeus dependency |
| CM-12 | Plan 01 | Zeus backend optional (`[zeus]` extra) | SATISFIED | `zeus.py` — deferred import, `is_available()` probes ImportError; `zeus>=0.13.1` in pyproject.toml extras |
| CM-13 | Plan 01 | CodeCarbon backend optional (`[codecarbon]` extra) | SATISFIED | `codecarbon.py` exists; `select_energy_backend("codecarbon")` paths verified |
| CM-14 | Plan 01 | Backend priority: Zeus > NVML > CodeCarbon | SATISFIED | `__init__.py` `_auto_select()` L168-188 — probes in order |
| CM-15 | Plan 03 | `torch.cuda.synchronize()` before every measurement stop | SATISFIED | `pytorch.py` L139 `self._cuda_sync()`, L614-628 `_cuda_sync()` implementation |
| CM-16 | Plan 03 | Timeseries: 1 Hz sampling, sidecar `timeseries.parquet` | SATISFIED | `timeseries.py` full implementation; `pytorch.py` L161-170 writes sidecar |
| CM-17 | Plan 03 | Idle GPU baseline before warmup | SATISFIED | `pytorch.py` L86-96 — baseline measurement at step 2, before model load (step 3) |
| CM-18 | Plan 03 | `baseline_power_w` stored in ExperimentResult | SATISFIED | `ExperimentResult.energy_breakdown.baseline_power_w` via `create_energy_breakdown()` |
| CM-19 | Plan 03 | `energy_adjusted_j = energy_total_j - (baseline_power_w * duration_sec)` | SATISFIED | `baseline.py` L164 `baseline_energy_j = baseline_power_w * duration_sec` used in breakdown |
| CM-20 | Plan 03 | Baseline cache with session-level TTL | SATISFIED | `baseline.py` L36 `_baseline_cache: dict[int, BaselineCache]`, L61-68 TTL check |
| CM-21 | Plan 02 | Fixed-count default: n_warmup=5 | SATISFIED | `models.py` `n_warmup: int = Field(default=5)` |
| CM-22 | Plan 02 | Thermal floor: 60s wait, configurable down to 30s | SATISFIED | `models.py` `thermal_floor_seconds = Field(default=60.0, ge=30.0)` |
| CM-23 | Plan 02 | CV-based convergence as opt-in | SATISFIED | `models.py` `convergence_detection: bool = Field(default=False)` |
| CM-24 | Plan 02 | WarmupResult with 6 fields | SATISFIED | `metrics.py` has 7 fields (6 spec + `thermal_floor_wait_s` extension). The spec says 6; code has 7. Additive, not conflicting. |
| CM-25 | Plan 03 | Primary metrics: energy_per_output_token, tokens_per_second | SATISFIED | `pytorch.py` L737-748 computes both; `experiment.py` L170-171 both fields |
| CM-26 | Plan 02 | FLOPs via PaLM formula (2 * N_params * tokens) | SATISFIED | `flops.py` L43-82 `estimate_flops_palm()` |
| CM-27 | Plan 02 | FlopsResult with method: str and confidence Literal | SATISFIED | `metrics.py` L183 — `method: Literal[...]`, `confidence: Literal["high", "medium", "low"]` |
| CM-28 | Plan 02 | Warmup tokens excluded from FLOPs | SATISFIED | `pytorch.py` L32-35 `_MeasurementData.input_tokens/output_tokens` — only measurement loop tokens accumulated |

**Note on REQUIREMENTS.md status:** All 18 requirements are implemented and verified in code. However, `.planning/REQUIREMENTS.md` lines 143-148 and 153 still show CM-15, CM-16, CM-17, CM-18, CM-19, CM-20, and CM-25 as "Pending". This is a documentation drift — the requirement tracker was not updated after Plan 03 completion.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|---------|--------|
| `core/backends/pytorch.py` | 333-348 | "M1 placeholder" comment + synthetic prompt generation | Warning | Prompts are simple repeated "Hello, " strings. Phase 5 CONTEXT.md stated this would be replaced, but Plans 01-03 contain no task for dataset loading. Out-of-scope or missed? Requires clarification. |
| `core/backends/pytorch.py` | 710 | "no more 0.0 placeholders" comment | Info | Accurate — `total_energy_j=total_energy_j`, `total_flops=total_flops`, `measurement_warnings=measurement_warnings` all use real values |

No blocker anti-patterns. The "M1 placeholder" in `_prepare_prompts()` generates actual prompts (functional synthetic data) — it is not a stub that returns null or an empty list. Energy measurement, FLOPs, and warnings all function correctly with synthetic prompts.

---

## Test Coverage Summary

| Test File | Tests | Result |
|-----------|-------|--------|
| `tests/unit/test_energy_backends_v2.py` | 17 | All passed |
| `tests/unit/test_warmup_v2.py` | 31 | All passed |
| `tests/unit/test_flops_v2.py` | 17 | All passed |
| `tests/unit/test_measurement_integration.py` | 20 | All passed |
| **Total** | **85** | **85/85 passed** |

---

## Human Verification Required

### 1. End-to-End GPU Energy Experiment

**Test:** Run `llem run config.yaml` with a small model (e.g., `gpt2`) on a GPU machine with energy.backend set to "auto"
**Expected:** `ExperimentResult.total_energy_j > 0`, `energy_breakdown.baseline_power_w` is a realistic idle wattage (e.g., 20-80W), `timeseries.parquet` written alongside `result.json`
**Why human:** CUDA is only available inside containers on this host — cannot invoke live GPU inference from host shell

### 2. CUDA Sync Timing Verification

**Test:** Add debug logging before/after sync and energy stop, run real inference
**Expected:** `torch.cuda.synchronize()` completes before `energy_backend.stop_tracking()` is called
**Why human:** Code ordering is correct (`_cuda_sync()` at L139, stop at L144) but wall-clock confirmation requires a live GPU trace

---

## Gaps Summary

### Gap 1: REQUIREMENTS.md Status Drift (Documentation Only)

Seven Phase 5 requirements remain marked "Pending" in `.planning/REQUIREMENTS.md` despite being fully implemented and verified in Phase 5 Plan 03:

- CM-15 (CUDA sync) — implemented in `pytorch.py` `_cuda_sync()`
- CM-16 (timeseries) — implemented in `timeseries.py` + `pytorch.py`
- CM-17 (baseline before load) — implemented, baseline at step 2 before model load at step 3
- CM-18 (baseline_power_w in result) — implemented via `energy_breakdown`
- CM-19 (energy_adjusted_j formula) — implemented in `baseline.py` `create_energy_breakdown()`
- CM-20 (baseline cache TTL) — implemented as module-level `_baseline_cache` dict
- CM-25 (energy_per_output_token, tokens_per_second) — implemented, both fields on `ExperimentResult`

**Fix:** Update `.planning/REQUIREMENTS.md` lines 143-148 and 153 from `Pending` to `Complete`.

### Gap 2: Prompt Generation Placeholder (Scope Clarification Needed)

`_prepare_prompts()` in `pytorch.py` is explicitly labelled "M1 placeholder" with a comment saying "Phase 5 replaces this with proper dataset loading." Phase 5 Plans 01-03 contain no task for dataset loading. The method generates functional (but synthetic) prompts — it is not a stub. This is likely intentional scope deferral to a future phase, but the stale comment creates confusion.

**Fix:** Either clarify in Phase 5 context/summary that dataset loading was deferred, OR remove the "Phase 5 replaces this" reference and update to name the actual future phase.

---

*Verified: 2026-02-26*
*Verifier: Claude (gsd-verifier)*
