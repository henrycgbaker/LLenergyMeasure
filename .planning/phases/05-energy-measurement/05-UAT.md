---
status: complete
phase: 05-energy-measurement
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md]
started: 2026-02-26T20:30:00Z
updated: 2026-02-26T20:35:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Energy backend imports resolve
expected: All key classes (NVMLBackend, ZeusBackend, EnergyMeasurement, select_energy_backend) importable without error
result: pass

### 2. Null energy disable
expected: `select_energy_backend(None)` returns `None` with no warnings or exceptions — maps from YAML `null`
result: pass

### 3. Explicit unavailable backend raises ConfigError
expected: `select_energy_backend("zeus")` raises `ConfigError` with install guidance (zeus not installed on this host)
result: pass

### 4. WarmupConfig v2 defaults
expected: `WarmupConfig()` has `n_warmup=5`, `thermal_floor_seconds=60.0`, `convergence_detection=False`, `cv_threshold=0.05`
result: pass

### 5. Thermal floor minimum enforced
expected: `WarmupConfig(thermal_floor_seconds=29.0)` raises Pydantic `ValidationError` (minimum is 30.0)
result: pass

### 6. EnergyConfig wired to ExperimentConfig
expected: `ExperimentConfig(model="gpt2").energy.backend == "auto"` — EnergyConfig defaults on ExperimentConfig
result: pass

### 7. PaLM FLOPs formula correctness
expected: `estimate_flops_palm(mock_model, 100, 50)` returns `value == 2 * non_embed_params * 150` with `method="palm_formula"`
result: pass

### 8. Measurement warnings — worst case
expected: `collect_measurement_warnings(5.0, False, 30.0, 45.0, 5)` returns exactly 4 warnings (short duration, persistence off, thermal drift, low samples)
result: pass

### 9. Measurement warnings — clean case
expected: `collect_measurement_warnings(60.0, True, 30.0, 32.0, 100)` returns 0 warnings
result: pass

### 10. Full test suite passes
expected: `python -m pytest tests/unit/test_energy_backends_v2.py tests/unit/test_warmup_v2.py tests/unit/test_flops_v2.py tests/unit/test_measurement_integration.py -v` — all 85 tests pass
result: pass

### 11. pyproject.toml zeus package fixed
expected: `grep 'zeus>=0.13.1' pyproject.toml` matches — no reference to abandoned zeus-ml
result: pass

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
