---
phase: 08-testing-and-integration
plan: 02
subsystem: testing
tags: [pytest, unit-tests, state-machine, exceptions, protocols, config, introspection, ssot]

# Dependency graph
requires:
  - phase: 08-testing-and-integration
    plan: 01
    provides: "tests/fakes.py, tests/conftest.py make_config/make_result factories"
  - phase: 02-config-system
    provides: "ExperimentConfig, loader, introspection, user_config, ssot"
  - phase: 01-measurement-foundations
    provides: "3-state machine, exceptions hierarchy, protocols"
provides:
  - "7 new unit test files covering all untested v2.0 subsystems"
  - "tests/unit/test_state_machine.py: 3-phase state machine, StateManager, compute_config_hash"
  - "tests/unit/test_exceptions.py: flat LLEMError hierarchy with InvalidStateTransitionError"
  - "tests/unit/test_protocols.py: INF-10 protocol conformance with fakes (no mock.patch)"
  - "tests/unit/test_config_schema.py: ExperimentConfig validation, field renames, cross-validators"
  - "tests/unit/test_config_loader.py: YAML loader, ConfigError/ValidationError boundary"
  - "tests/unit/test_config_introspection.py: INF-11 SSOT-driven test generation"
  - "tests/unit/test_config_user_config.py: XDG path, env var overrides, silent ignore"
affects:
  - "08-03 (CI workflows run the full clean test suite)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SSOT-driven parametric tests: @pytest.mark.parametrize with PRECISION_SUPPORT values"
    - "Protocol isinstance checks: validate fakes structurally, no mock.patch"
    - "ConfigError vs ValidationError boundary: structural errors vs field value errors"
    - "tmp_path + monkeypatch fixtures for env var and file isolation"

key-files:
  created:
    - tests/unit/test_state_machine.py
    - tests/unit/test_exceptions.py
    - tests/unit/test_protocols.py
    - tests/unit/test_config_schema.py
    - tests/unit/test_config_loader.py
    - tests/unit/test_config_introspection.py
    - tests/unit/test_config_user_config.py
  modified: []

key-decisions:
  - "test_get_shared_params docstring clarified: 'model' is not a shared param (it's a required top-level ExperimentConfig field); shared params are precision/n/decoder.*"
  - "Protocol isinstance check in test_protocols.py validated against runtime behaviour before writing — all 3 fakes satisfy their protocols"
  - "test_config_introspection SSOT test parametrises over PRECISION_SUPPORT['pytorch'] matching get_param_test_values('precision') output"

requirements-completed: [STU-05, INF-10, INF-11]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 8 Plan 02: Subsystem Unit Tests Summary

**7 new unit test files covering infrastructure and config subsystems: 147 new tests, total suite 405 passing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-27T00:43:16Z
- **Completed:** 2026-02-27T00:47:30Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Created 3 infrastructure test files: state machine (17 tests), exception hierarchy (14 tests), protocol conformance (20 tests)
- Created 4 config test files: schema validation (28 tests), YAML loader (20 tests), SSOT introspection (28 tests), user config (20 tests)
- Total new tests: 147, raising suite from 258 to 405 passing
- INF-10 compliance verified: test_protocols.py uses only fakes (FakeInferenceBackend, FakeEnergyBackend, FakeResultsRepository), zero mock.patch calls
- INF-11 compliance verified: test_config_introspection.py and test_config_schema.py use PRECISION_SUPPORT and get_param_test_values() for schema-driven test parametrisation

## Task Commits

1. **Task 1: Infrastructure unit tests (state machine, exceptions, protocols)** - `aeba2e4` (test)
2. **Task 2: Config subsystem unit tests (schema, loader, introspection, user config)** - `07e084a` (test)

## Files Created

- `tests/unit/test_state_machine.py` — ExperimentPhase enum, valid/invalid transitions, mark_failed, StateManager create/load/save roundtrip, compute_config_hash determinism and hash format
- `tests/unit/test_exceptions.py` — LLEMError base, 5 direct subclasses, InvalidStateTransitionError under ExperimentError, catchability chains, message preservation
- `tests/unit/test_protocols.py` — isinstance checks for all 3 fakes, FakeInferenceBackend.run() returns injected result and records calls, FakeEnergyBackend lifecycle, FakeResultsRepository save/load
- `tests/unit/test_config_schema.py` — ExperimentConfig minimal valid, extra=forbid, v2.0 field renames (model/precision/n), backend validation, backend section cross-validator, passthrough_kwargs collision, SSOT-parametrised precision tests
- `tests/unit/test_config_loader.py` — load valid YAML, version field stripped, ConfigError for unknown keys with did-you-mean, ValidationError passthrough for bad values, nonexistent file, CLI override merging, dotted key unflattening, deep_merge utility (8 tests)
- `tests/unit/test_config_introspection.py` — get_backend_params/get_shared_params/get_experiment_config_schema/get_param_test_values/get_all_params/list_all_param_paths/get_validation_rules; schema-driven @parametrize over PRECISION_SUPPORT
- `tests/unit/test_config_user_config.py` — get_user_config_path XDG, missing file returns defaults, partial file merges, env vars (LLEM_CARBON_INTENSITY, LLEM_DATACENTER_PUE, LLEM_NO_PROMPT, LLEM_RUNNER_*), env var precedence over file, silent ignore of invalid float env vars

## Decisions Made

- `test_get_shared_params_returns_model_field` renamed to `test_get_shared_params_contains_precision` after verifying the actual API: `model` is a required field on ExperimentConfig, not in the shared introspection section. `precision` is the correct shared param to assert.
- Protocol isinstance checks validated at runtime before writing tests — confirmed all 3 fakes satisfy their protocols structurally via duck typing.
- SSOT assertion added: `test_ssot_precision_values_match_param_test_values` confirms that `PRECISION_SUPPORT["pytorch"]` == `set(get_param_test_values("precision"))`, ensuring SSOT consistency.

## Deviations from Plan

None — plan executed exactly as written.

The plan's `test_fake_inference_backend_satisfies_protocol` description referenced `InferenceBackend` protocol from `llenergymeasure.core.backends.protocol` (not `llenergymeasure.protocols`), which was the correct import — verified before writing.

## Issues Encountered

None — all 7 test files passed on first run.

## User Setup Required

None.

## Next Phase Readiness

- 405 GPU-free unit tests pass with 0 failures and 0 collection errors
- All 7 subsystems now have unit test coverage: state machine, exceptions, protocols, config schema, config loader, config introspection, user config
- Suite ready for Plan 08-03 (CI workflows)
- No blockers

---
*Phase: 08-testing-and-integration*
*Completed: 2026-02-27*
