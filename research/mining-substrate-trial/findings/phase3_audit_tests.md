# Phase 3 Audit: Engine Configs Tests Classification

**Context**: Phase 3 cleanup intends to delete `src/llenergymeasure/config/engine_configs.py` (~1100 LOC). The spike's option-A migration already replaced old hand-curated `TransformersConfig`, `VLLMConfig`, `TensorRTConfig` classes with codegen-emitted `engines.<e>.Config` classes from `src/llenergymeasure/engines/<e>/config.py`. New classes use NESTED shape (`Config.engine_params` + `Config.sampling_params`).

4 test files still import from `engine_configs`. Below: classification of each test that actually touches engine_configs (tests without imports/references are skipped).

---

## File 1: `tests/unit/config/test_config_schema.py`

Tests schema validation, dtype support, engine composition, and field renames. Many tests touch ExperimentConfig (which references the engine config sections), but only those that directly import/assert OLD engine_configs classes are listed.

| Line | Test Name | Classification | Justification | Notes |
|------|-----------|---|---|---|
| 147 | `test_pytorch_config_has_no_num_processes_field` | **DELETE** | Asserts "num_processes" not in TransformersConfig.model_fields—tests OLD class curation detail. New generated class has different fields; assertion meaningless. | OLD TransformersConfig only |
| 154 | `test_pytorch_config_num_processes_not_a_declared_field` | **DELETE** | Tests OLD extra='allow' passthrough; verifies num_processes is extra, not typed. Tests the OLD curation boundary, not behavior. | OLD TransformersConfig |
| 351 | `test_pytorch_config_tp_plan_accepts_auto` | **DELETE** | Tests TransformersConfig.tp_plan field (Literal enforcement). New generated class does not have tp_plan—field is gone. | OLD TransformersConfig.tp_plan |
| 359 | `test_pytorch_config_tp_plan_rejects_invalid` | **DELETE** | Tests Literal("auto") validation on OLD tp_plan. New class doesn't have field; assertion obsolete. | OLD Literal validation |
| 367 | `test_pytorch_config_tp_size_accepts_positive` | **DELETE** | Tests tp_size field with ge=1 constraint. New class may not have this field. Tests old curation. | OLD TransformersConfig |
| 376 | `test_pytorch_config_tp_size_rejects_zero` | **DELETE** | Tests ge=1 validation on tp_size. New structure different. | OLD validation |
| 384 | `test_pytorch_config_tp_plan_device_map_exclusive` | **DELETE** | Tests @model_validator for mutual exclusion (tp_plan, device_map). New class doesn't have this validator. | OLD model_validator |
| 392 | `test_pytorch_config_tp_plan_without_device_map_ok` | **DELETE** | Tests OLD field absence behavior. New class structure different. | OLD field interaction |
| 401 | `test_pytorch_config_device_map_without_tp_plan_ok` | **DELETE** | Tests OLD field absence behavior. New class structure different. | OLD field interaction |
| 415 | `test_vllm_dtype_float32_rejected` | **DELETE** | Tests OLD VLLMConfig.dtype Literal that rejects float32. New generated class accepts broader set (mined as str). Old narrow bounds obsolete. | OLD VLLMConfig.dtype Literal |
| 563 | `test_trt_dtype_float32_rejected` | **DELETE** | Tests OLD TensorRTConfig.dtype Literal rejection. New class accepts broader set. Tests old narrow curation. | OLD TensorRTConfig |

**File 1 Summary**: 11 DELETE. `test_config_schema.py` is primarily ExperimentConfig validation (which crosses all engines); most tests don't import engine_configs directly. Only TP/dtype/field-existence tests directly assert OLD class structure—all should be deleted.

---

## File 2: `tests/unit/config/test_tensorrt_config.py`

Tests TensorRT config validation at all levels: compile params, quantization, KV cache, scheduler, sampling. Imports OLD `TensorRTConfig`, `TensorRTKvCacheConfig`, `TensorRTQuantConfig`, `TensorRTSamplingConfig`.

| Line | Test Name | Classification | Justification | Migrate Strategy |
|------|-----------|---|---|---|
| 38 | `test_tensorrt_compile_params_accepted` | **MIGRATE** | Tests constructor with compile params. New generated class has similar params; acceptance behavior still relevant. | Swap to `engines.tensorrt.Config` |
| 55 | `test_tensorrt_dtype_literal_validation` | **DELETE** | Tests OLD dtype Literal rejection of float32. New class accepts broader set (mined schema). Old narrow validation obsolete. | OLD Literal only |
| 60 | `test_tensorrt_dtype_bfloat16_accepted` | **MIGRATE** | Tests dtype='bfloat16' acceptance. New class still accepts this. | Swap to `engines.tensorrt.Config` |
| 65 | `test_tensorrt_tensor_parallel_size_ge_1` | **MIGRATE** | Tests ge=1 constraint. Still relevant in new class. | Swap to new class |
| 70 | `test_tensorrt_max_batch_size_ge_1` | **MIGRATE** | Tests ge=1 constraint. Still relevant. | Swap to new class |
| 75 | `test_tensorrt_max_input_len_ge_1` | **MIGRATE** | Tests ge=1 constraint. Still relevant. | Swap to new class |
| 80 | `test_tensorrt_max_seq_len_ge_1` | **MIGRATE** | Tests ge=1 constraint. Still relevant. | Swap to new class |
| 106 | `test_valid_quant_algo_accepted` | **MIGRATE** | Tests all QuantAlgo Literal values. New class should test same (quant values schema-driven). | Swap to new class, parameterize over all algos |
| 112 | `test_invalid_quant_algo_rejected` | **MIGRATE** | Tests invalid quant_algo rejection. Still meaningful with new class. | Swap to new class |
| 117 | `test_kv_cache_quant_algo_accepted` | **MIGRATE** | Tests kv_cache_quant_algo validation. Behavior preserved. | Swap to new class |
| 123 | `test_invalid_kv_cache_quant_algo_rejected` | **MIGRATE** | Tests rejection of invalid kv_cache_quant_algo. Still relevant. | Swap to new class |
| 143 | `test_kv_cache_config_accepted` | **MIGRATE** | Tests KV cache section validation. Behavior still relevant. | Swap to new class |
| 154 | `test_kv_cache_free_gpu_memory_fraction_range` | **MIGRATE** | Tests 0.0-1.0 range validation. Still relevant. | Swap to new class |
| 186 | `test_scheduler_config_accepted` | **MIGRATE** | Tests scheduler section validation. Still relevant. | Swap to new class |
| 193 | `test_valid_scheduler_policies` | **MIGRATE** | Tests valid scheduler policies. Still meaningful. | Swap to new class, test all policies |
| 199 | `test_invalid_scheduler_policy_rejected` | **MIGRATE** | Tests rejection of invalid policy. Still relevant. | Swap to new class |
| 217 | `test_sampling_config_accepted` | **MIGRATE** | Tests sampling section validation. Behavior preserved. | Swap to new class |
| 228 | `test_sampling_return_perf_metrics_is_extra_allow` | **MIGRATE** | Tests extra='allow' passthrough. New class has same contract. | Swap to new class |
| 234 | `test_sampling_n_ge_1` | **MIGRATE** | Tests n >= 1 constraint. Still relevant. | Swap to new class |
| 248 | `test_experiment_config_with_full_tensorrt` | **MIGRATE** | Tests ExperimentConfig integration with full tensorrt section. New shape (nested engine_params/sampling_params) still needs this test. | Update to nest properly under new shape |
| 297 | `test_tensorrt_extra_allow_forwards_unknown` | **MIGRATE** | Tests extra='allow' passthrough on TensorRTConfig. New generated class has same contract. | Swap to new class |
| 312 | `test_tensorrt_none_defaults` | **MIGRATE** | Tests None defaults. New generated class similar structure (may have mined defaults too). Behavior worth verifying. | Swap to new class, adjust for mined defaults |

**File 2 Summary**: 1 DELETE, 21 MIGRATE. `test_tensorrt_config.py` is a comprehensive validation test suite. Nearly all tests exercise behavior that should still work with new class—just point at `engines.tensorrt.Config` instead of old hand-written class. Only the dtype Literal rejection test (old narrow bound) should be deleted.

---

## File 3: `tests/unit/engines/test_vllm_engine.py`

Tests VLLMEngine protocol compliance, build methods, sampling params. Imports OLD `VLLMBeamSearchConfig` on lines 24-26 and 690.

| Line | Test Name | Classification | Justification | Notes |
|------|-----------|---|---|---|
| 676 | `test_beam_search_config_triggers_beam_path` | **KEEP_AS_IS** | Tests beam_width routing via sampling_params extras. OLD class imported but test intent (beam search routing) is orthogonal; behavior independent of class deletion. | Import swap only: `VLLMBeamSearchConfig` still exists |
| 686 | `test_beam_search_mutual_exclusion_with_sampling` | **DELETE** | Tests OLD VLLMConfig.beam_search/sampling mutual exclusion @model_validator. New architecture routes beam via sampling_params extras (no class-level mutual exclusion). Test assertion is about OLD validator. | OLD model_validator only |
| 703 | `test_beam_search_config_accepts_all_fields` | **MIGRATE** | Tests VLLMBeamSearchConfig construction (beam_width, length_penalty, early_stopping, max_tokens). VLLMBeamSearchConfig still exists in new codebase; behavior still relevant. | Keep, verify against new class |
| 713 | `test_beam_search_config_extra_allow` | **MIGRATE** | Tests VLLMBeamSearchConfig.extra='allow' contract. Class still exists; behavior still relevant. | Keep, verify extra='allow' |
| 718 | `test_beam_search_beam_width_ge_1` | **MIGRATE** | Tests beam_width >= 1 constraint. VLLMBeamSearchConfig still enforces this; behavior relevant. | Keep, verify constraint |

**File 3 Summary**: 1 DELETE, 3 MIGRATE, 1 KEEP_AS_IS. `VLLMBeamSearchConfig` survives engine_configs deletion (still needed for beam search config), so most beam search tests migrate trivially. Only the OLD VLLMConfig mutual exclusion validator (which is gone) should be deleted.

---

## File 4: `tests/integration/test_codegen_tracer_bullet.py`

Tracer-bullet acceptance test for Phase 2-T codegen pipeline. Demonstrates architectural payoff: generated classes accept values that old hand-written classes reject. Imports OLD `TransformersConfig`, `TransformersSamplingConfig` (lines 35-38) for CONTRAST TESTS ONLY.

| Line | Test Name | Classification | Justification | Notes |
|------|-----------|---|---|---|
| 72 | `test_handwritten_rejects_temperature_above_two` | **DELETE** | Contrast test: OLD TransformersSamplingConfig rejects temperature > 2.0 (Field(le=2.0)). Once class deleted, contrast test is pointless. | Contrast-only, old class gone |
| 114 | `test_handwritten_rejects_half` | **DELETE** | Contrast test: OLD TransformersConfig rejects dtype="half" (Literal['float32','float16','bfloat16']). Once class deleted, contrast is gone. | Contrast-only, old class gone |
| 140 | `test_handwritten_rejects_novel_backend` | **DELETE** | Contrast test: OLD TransformersConfig rejects unknown attn_implementation (narrow Literal). Once class deleted, contrast is pointless. | Contrast-only, old class gone |
| 390 | `test_handwritten_uses_none_default` | **DELETE** | Contrast test: OLD TransformersSamplingConfig uses None defaults vs new class mined defaults. Pure contrast; once old class deleted, test is meaningless. | Contrast-only, old class gone |

**File 4 Summary**: 4 DELETE. All engine_configs references in this file are CONTRAST TESTS showing the architectural difference between old hand-written (narrow bounds, None defaults) and new generated (broad bounds, mined defaults). Once the old class is deleted, these contrast tests have no meaning. All other tests in the file (TestPublicAPISurface, positive tracer-bullet assertions, ExperimentConfig integration) already test the NEW classes and do NOT import engine_configs—those stay.

---

## OVERALL SUMMARY

| File | DELETE | MIGRATE | KEEP_AS_IS | Total | Fate |
|------|--------|---------|-----------|-------|------|
| `test_config_schema.py` | 11 | 0 | 0 | 11 | **DELETE all engine_configs tests** — File has 100+ tests; only 11 directly assert OLD class structure. Keep all other ExperimentConfig schema tests. |
| `test_tensorrt_config.py` | 1 | 21 | 0 | 22 | **MIGRATE 21 tests to new class** — Comprehensive validation suite; swap OLD TensorRTConfig imports to `engines.tensorrt.Config`. Delete only dtype Literal rejection test. |
| `test_vllm_engine.py` | 1 | 3 | 1 | 5 | **MIGRATE 3 beam search config tests, keep 1 import-only test** — VLLMBeamSearchConfig survives deletion. Delete only mutual-exclusion validator test (old cross-validator). |
| `test_codegen_tracer_bullet.py` | 4 | 0 | 0 | 4 | **DELETE all 4 contrast tests** — These demonstrate old vs new; once old class gone, contrast is pointless. Keep all other tests (80+) that test NEW classes. |

**Grand Totals**: **17 DELETE, 24 MIGRATE, 1 KEEP_AS_IS across all 4 files.**

**Headlines**:
- **`test_config_schema.py`**: Delete 11 tests testing old TP/dtype fields and model_validator. Keep 90+ ExperimentConfig tests.
- **`test_tensorrt_config.py`**: Migrate 21 validation tests to new `engines.tensorrt.Config`; delete 1 dtype Literal test.
- **`test_vllm_engine.py`**: Migrate 3 VLLMBeamSearchConfig tests; keep 1 import-only test; delete 1 mutual-exclusion validator test.
- **`test_codegen_tracer_bullet.py`**: Delete 4 contrast tests showing old vs new; keep 80+ tests on new classes (no changes needed).
