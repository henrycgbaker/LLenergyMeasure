---
phase: 04-pytorch-backend-pre-flight
plan: 02
subsystem: core/backends
tags: [inference, pytorch, protocol, p0-fix, model-loading, gpu-free-tests]

# Dependency graph
requires:
  - phase: 04-pytorch-backend-pre-flight
    plan: 01
    provides: collect_environment_snapshot(), EnvironmentSnapshot, ExperimentResult.environment_snapshot
  - phase: 03-library-api
    provides: ExperimentConfig, ExperimentResult, AggregationMetadata domain models
provides:
  - InferenceBackend Protocol (runtime_checkable) with name property and run() method
  - PyTorchBackend satisfying InferenceBackend Protocol
  - get_backend() factory function
  - detect_default_backend() returning 'pytorch' when transformers installed
  - P0 model_kwargs bug fixed structurally (no intermediate loader class)
  - 25 GPU-free unit tests covering Protocol, factory, detection, kwargs, precision
affects:
  - 04-03 (runner integration — PyTorchBackend.run() is the entry point)
  - Any future backends (vLLM, TRT-LLM) implement InferenceBackend Protocol

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Protocol + runtime_checkable: structural subtyping via isinstance() without inheritance"
    - "Deferred heavy imports: torch and transformers imported inside method bodies, not at module level"
    - "P0 fix structure: _model_load_kwargs() → dict passed directly to from_pretrained()"
    - "Cleanup in finally block: model deletion and CUDA cache clear always runs"

key-files:
  created:
    - src/llenergymeasure/core/backends/__init__.py
    - src/llenergymeasure/core/backends/protocol.py
    - src/llenergymeasure/core/backends/pytorch.py
    - tests/unit/test_backend_protocol.py
  modified: []

key-decisions:
  - "No intermediate loader class — from_pretrained() called directly (P0 fix CM-06)"
  - "passthrough_kwargs merged last so researcher can override any backend default"
  - "_prepare_prompts() is a M1 placeholder — Phase 5 replaces with dataset loading"
  - "Energy fields (total_energy_j, avg_energy_per_token_j, total_flops) are 0.0 — Phase 5"
  - "Cleanup in finally block — always runs even if measurement fails"
  - "OOM errors wrapped in BackendError with actionable fix suggestions"

requirements-completed: [CM-01, CM-04, CM-05, CM-06]

# Metrics
duration: 4min
completed: 2026-02-26
---

# Phase 4 Plan 02: InferenceBackend Protocol and PyTorch Backend Summary

**InferenceBackend Protocol with PyTorchBackend clean rewrite, P0 model_kwargs bug fixed structurally, and 25 GPU-free unit tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T18:26:39Z
- **Completed:** 2026-02-26T18:30:55Z
- **Tasks:** 2
- **Files created:** 4

## Accomplishments

- `InferenceBackend` Protocol defined with `@runtime_checkable` — `name` property and `run(config) -> ExperimentResult`. Same structural pattern as lm-eval's LM subclass.
- `PyTorchBackend` satisfies the Protocol and owns its full experiment lifecycle: snapshot → load → prepare → warmup → measure → build → cleanup.
- **P0 model_kwargs bug fixed structurally**: `_model_load_kwargs()` builds the complete kwargs dict; `_load_model()` passes it directly to `AutoModelForCausalLM.from_pretrained(**kwargs)`. No intermediate `HuggingFaceModelLoader` class. The v1.x bug was that `_build_model_kwargs()` built the dict but `loader.load(config)` ignored it entirely.
- `passthrough_kwargs` from config are merged into the kwargs dict (last, so they can override backend defaults).
- `detect_default_backend()` returns `'pytorch'` when transformers is installed; raises `BackendError` with install hint otherwise.
- `get_backend("pytorch")` returns a `PyTorchBackend` instance.
- 25 GPU-free tests: Protocol satisfaction, factory, detection (with monkeypatched transformers), kwargs construction (P0 fix verification), precision mapping, generate kwargs.

## Task Commits

1. **Task 1: InferenceBackend Protocol and PyTorch backend** — `740b951` (feat)
2. **Task 2: GPU-free Protocol and backend detection tests** — `be8ed84` (test)

## Files Created

- `src/llenergymeasure/core/backends/__init__.py` — Package init with `get_backend()`, `detect_default_backend()`, `InferenceBackend` re-export
- `src/llenergymeasure/core/backends/protocol.py` — `InferenceBackend` Protocol (`@runtime_checkable`, `name` property, `run()` method)
- `src/llenergymeasure/core/backends/pytorch.py` — `PyTorchBackend` clean rewrite: `_load_model()`, `_model_load_kwargs()`, `_precision_to_dtype()`, `_prepare_prompts()`, `_run_warmup()`, `_run_measurement()`, `_run_batch()`, `_build_generate_kwargs()`, `_build_result()`, `_cleanup()`
- `tests/unit/test_backend_protocol.py` — 25 GPU-free tests

## Decisions Made

- **No intermediate loader** — `from_pretrained()` called directly with `**kwargs`. This is the P0 fix, not a workaround.
- **passthrough_kwargs merged last** — researcher can override any backend default via escape hatch. This is intentional.
- **`_prepare_prompts()` is a M1 placeholder** — generates `config.n` synthetic prompts. Phase 5 replaces with dataset loading (aienergyscore + SyntheticDatasetConfig).
- **Energy fields are 0.0 placeholders** — `total_energy_j`, `avg_energy_per_token_j`, `total_flops` set to 0.0. Phase 5 adds Zeus/NVML/CodeCarbon measurement.
- **Cleanup in finally block** — model deletion and CUDA cache clear always runs, even if measurement raises.
- **OOM wrapped in BackendError** — catches both `torch.cuda.OutOfMemoryError` and `RuntimeError("cuda out of memory")` with actionable suggestions.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- `PyTorchBackend.run(config)` is the primary entry point for Plan 03 (runner integration)
- The Protocol enables future backends (vLLM, TRT-LLM) to be swapped in without changing calling code
- All pre-flight hooks from Plan 01 (`run_preflight()`, `collect_environment_snapshot()`) are called from `run()` before GPU allocation

---

*Phase: 04-pytorch-backend-pre-flight*
*Completed: 2026-02-26*
