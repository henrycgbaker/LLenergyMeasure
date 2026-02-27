---
status: complete
phase: 03-library-api
source: [03-01-SUMMARY.md, 03-02-SUMMARY.md]
started: 2026-02-26T17:30:00Z
updated: 2026-02-26T17:35:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Public API imports resolve
expected: `from llenergymeasure import run_experiment, run_study, ExperimentConfig, StudyConfig, ExperimentResult, StudyResult, __version__` succeeds — all 7 names importable
result: pass

### 2. Version is 2.0.0
expected: `llenergymeasure.__version__` returns exactly `"2.0.0"`
result: pass

### 3. run_experiment with ExperimentConfig object
expected: `run_experiment(ExperimentConfig(model="gpt2"))` calls the internal `_run()` and returns an `ExperimentResult` (not a union type, not None). With `_run` stubbed, result type is exactly `ExperimentResult`.
result: pass

### 4. run_experiment with kwargs convenience form
expected: `run_experiment(model="gpt2", n=50)` builds an `ExperimentConfig` internally — the config has `model="gpt2"` and `n=50` with default `backend="pytorch"`.
result: pass

### 5. run_experiment with no args raises ConfigError
expected: `run_experiment()` with no arguments raises `ConfigError` (not TypeError, not ValidationError) with a helpful message mentioning `model=`.
result: pass

### 6. Internal names are private
expected: Accessing `llenergymeasure.load_experiment_config`, `llenergymeasure.ConfigError`, or `llenergymeasure.AggregatedResult` raises `AttributeError`. Only `__all__` names are accessible.
result: pass

### 7. run_study raises NotImplementedError
expected: `run_study(StudyConfig(experiments=[ExperimentConfig(model="gpt2")]))` raises `NotImplementedError` with a message containing "M2".
result: pass

### 8. ExperimentResult is AggregatedResult alias
expected: `from llenergymeasure.domain.experiment import ExperimentResult, AggregatedResult` — `ExperimentResult is AggregatedResult` is `True`. Existing v1.x code using `AggregatedResult` continues to work.
result: pass

### 9. StudyConfig validation
expected: `StudyConfig(experiments=[])` raises `ValidationError` (min_length=1). `StudyConfig(experiments=[ExperimentConfig(model="gpt2")], bad_field="x")` raises `ValidationError` (extra=forbid).
result: pass

### 10. Unit tests pass
expected: `python -m pytest tests/unit/test_api.py -v` runs 12 tests, all pass. No GPU required.
result: pass

### 11. Existing domain tests pass (no regression)
expected: `python -m pytest tests/unit/test_domain_experiment.py -v` runs 28 tests, all pass unchanged.
result: pass

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
