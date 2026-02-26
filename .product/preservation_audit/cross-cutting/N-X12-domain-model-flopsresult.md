# N-X12: Domain Model — FlopsResult and Related Types

**Module**: `src/llenergymeasure/domain/metrics.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: `FlopsResult`, `NormalisedMetrics`, `PrecisionMetadata`, and `ThermalThrottleInfo` are not mentioned in any planning document. `designs/result-schema.md` (not read but referenced) is expected to cover the result schema — but the presence of `PrecisionMetadata` with a `precision_factor` property and cross-backend normalisation logic is significant and may not be captured there.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/metrics.py`
**Key classes/functions**:

- `FlopsResult` (line 172) — Pydantic model:
  - `value: float` — estimated FLOPs count
  - `method: Literal["calflops", "architecture", "parameter_estimate"]` — three estimation methods in decreasing accuracy order
  - `confidence: Literal["high", "medium", "low"]` — mirrors method: calflops=high, architecture=medium, parameter_estimate=low
  - `precision: str` — compute precision string (e.g., "fp16", "fp32")
  - `notes: str | None` — contextual warnings (e.g., "BitsAndBytes: FLOPs are FP16 because compute happens at FP16 after dequantization")
  - `is_valid` property (line 194) — `self.value > 0`

- `PrecisionMetadata` (line 19) — Pydantic model tracking per-layer precision:
  - `weights: Literal["fp32","fp16","bf16","fp8","int8","int4","mixed"]`
  - `activations: Literal["fp32","fp16","bf16","fp8","int8"]`
  - `compute: Literal["fp32","fp16","bf16","fp8","int8","tf32"]`
  - `mixed_precision_breakdown: dict[str, float] | None`
  - `quantization_method: str | None` — "bitsandbytes", "gptq", "awq", "trt_ptq", etc.
  - `perplexity_degradation: float | None`
  - `precision_factor` property (line 64): `fp32/fp16/bf16/tf32=1.0`, `fp8/int8=0.5`, `int4=0.25`, `mixed=0.75`

- `NormalisedMetrics` (line 88) — Pydantic model for cross-backend comparison:
  - `tokens_per_joule: float`, `tokens_per_effective_pflop: float`, `tokens_per_second_per_watt: float`
  - `theoretical_flops: float`, `effective_flops: float` (= theoretical * precision_factor)
  - `precision: PrecisionMetadata | None`
  - `from_metrics()` classmethod (line 127) — computes all fields from raw measurements

- `ThermalThrottleInfo` (line 299) — Pydantic model:
  - `detected: bool`, `thermal: bool`, `power: bool`, `sw_thermal: bool`, `hw_thermal: bool`, `hw_power: bool`
  - `throttle_duration_sec: float`, `max_temperature_c: float | None`
  - `throttle_timestamps: list[float]`

- `WarmupResult` (line 344) — Pydantic model tracking convergence:
  - `converged: bool`, `final_cv: float`, `iterations_completed: int`, `target_cv: float`, `max_prompts: int`, `latencies_ms: list[float]`

## Why It Matters

`FlopsResult` with its `method` and `confidence` fields is essential for result provenance — calflops and parameter-estimation give FLOPs numbers that can differ by 2-5x depending on model architecture. Users need to know which method was used and how confident the estimate is. `PrecisionMetadata.precision_factor` enables the cross-backend efficiency normalisation in `NormalisedMetrics`: comparing throughput between a FP16 experiment and an INT4 experiment without precision adjustment would be misleading. `ThermalThrottleInfo` flags when hardware throttling invalidates measurements — a crucial data quality signal for overnight studies. `WarmupResult` documents whether convergence was achieved before measurement began — another data quality field.

## Planning Gap Details

`designs/result-schema.md` (referenced in planning but not read in this audit) is expected to cover `ExperimentResult` schema. However:
- `NormalisedMetrics` (cross-backend precision-adjusted efficiency) is a complex type with a specific use case — it may not be in the planning docs
- `PrecisionMetadata` with layer-level breakdown and `quantization_method` is sophisticated and likely code-only
- `ThermalThrottleInfo` (lines 299–342) with NVML throttle reason flags (`hw_thermal`, `hw_power`, `sw_thermal`) is not referenced in `designs/energy-backends.md` or `designs/observability.md`

## Recommendation for Phase 5

Carry all four types forward unchanged into the v2.0 domain model in `domain/results.py` or alongside the metrics types.

Key decisions needed:
1. Is `NormalisedMetrics` part of `ExperimentResult`? It is not currently in `AggregatedResult` but `FlopsResult` is (via `ComputeMetrics`). If cross-backend efficiency comparison is a v2.0 feature, `NormalisedMetrics` needs to be in the result schema.
2. `ThermalThrottleInfo` is confirmed in `aggregate_results()` (aggregation.py lines 322–337) — it aggregates thermal throttle across processes. Ensure it appears in the v2.0 `ExperimentResult` schema.
3. `WarmupResult` — verify it is stored in `RawProcessResult` and confirm whether it should appear in `ExperimentResult` or only in raw results.

The `FlopsResult.method` values ("calflops", "architecture", "parameter_estimate") must match whatever the FLOPs estimator in `core/flops.py` produces — do not rename these without updating the estimator.
