# N-C05: Compute Metrics and FLOPs Confidence (ComputeMetrics + FlopsResult)

**Module**: `src/llenergymeasure/domain/metrics.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: Planning docs do not describe the two-class FLOPs architecture (`ComputeMetrics` vs `FlopsResult`). `FlopsResult` includes a `confidence` field with a three-level enum and a `method` enum with three estimation strategies — neither is documented in any design or decision file.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/metrics.py`
**Key classes/functions**:
- `ComputeMetrics` (line 362) — legacy Pydantic BaseModel with 7 fields; used in `RawProcessResult` and `AggregatedResult`
- `FlopsResult` (line 172) — newer Pydantic BaseModel with 5 fields and a `is_valid` property; FLOPs-specific result with provenance tracking
- `NormalisedMetrics` (line 88) — precision-adjusted efficiency metrics that consume `FlopsResult`-style values
- `PrecisionMetadata` (line 19) — precision factors for effective FLOPs calculation

### `ComputeMetrics` (line 362) — currently wired into result models

All fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `flops_total` | `float` | Required | Total FLOPs for the inference |
| `flops_per_token` | `float` | `0.0` | FLOPs per output token |
| `flops_per_second` | `float` | `0.0` | FLOPs throughput |
| `peak_memory_mb` | `float` | `0.0` | Peak GPU memory usage in MB |
| `model_memory_mb` | `float` | `0.0` | Model weight memory footprint in MB |
| `flops_method` | `str` | `"unknown"` | Estimation method: `"calflops"`, `"architecture"`, or `"parameter"` |
| `flops_confidence` | `str` | `"unknown"` | Confidence level: `"high"`, `"medium"`, or `"low"` |
| `compute_precision` | `str` | `"fp16"` | Compute precision used |

`flops_method` and `flops_confidence` are plain strings (not enums) in `ComputeMetrics`. `RawProcessResult.compute_metrics: ComputeMetrics` (line 67) and `AggregatedResult` reference this class directly. The `CombinedMetrics` class (line 378) bundles `InferenceMetrics`, `EnergyMetrics`, and `ComputeMetrics` together with two computed properties: `efficiency_tokens_per_joule` and `efficiency_flops_per_watt`.

### `FlopsResult` (line 172) — newer model with typed enums

All fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `value` | `float` | Yes | Estimated FLOPs count |
| `method` | `Literal["calflops", "architecture", "parameter_estimate"]` | Yes | Estimation method used |
| `confidence` | `Literal["high", "medium", "low"]` | Yes | Confidence level of estimate |
| `precision` | `str` | Yes | Compute precision (e.g. `"fp16"`, `"fp32"`) |
| `notes` | `str | None` | No | Additional context or warnings |

**Property**: `is_valid: bool` (line 193) — returns `True` if `value > 0`.

The method enum documents the three estimation strategies:
- `"calflops"` — uses the `calflops` library (highest accuracy, architecture-dependent)
- `"architecture"` — derives from model architecture metadata (medium accuracy)
- `"parameter_estimate"` — the standard `2 × params × tokens` approximation (lowest accuracy)

The docstring explicitly notes: "For BitsAndBytes quantization, FLOPs = FP16 FLOPs because computation happens at FP16 after dequantization." This is a non-obvious correctness constraint that must be preserved in Phase 5.

### The Two-Class Situation

`FlopsResult` appears to be a more principled redesign of the FLOPs portion of `ComputeMetrics`, with typed enums instead of raw strings and a `notes` field for non-obvious caveats. However, `FlopsResult` is not currently wired into `RawProcessResult` or `AggregatedResult` — those models still use `ComputeMetrics`. It is unclear whether `FlopsResult` was intended to replace `ComputeMetrics.flops_*` fields or to be added as an additional field.

### `PrecisionMetadata` and `NormalisedMetrics`

`PrecisionMetadata` (line 19) tracks weight, activation, and compute precision separately (`weights`, `activations`, `compute` — all typed literals) plus `mixed_precision_breakdown: dict[str, float] | None`, `quantization_method: str | None`, and `perplexity_degradation: float | None`. Its `precision_factor` property (line 63) implements the effective FLOPs scaling: FP32/FP16/BF16 = 1.0, FP8/INT8 = 0.5, INT4 = 0.25, mixed = 0.75.

`NormalisedMetrics` (line 88) uses `PrecisionMetadata.precision_factor` to compute `effective_flops = theoretical_flops * precision_factor`, and derives `tokens_per_joule`, `tokens_per_effective_pflop`, and `tokens_per_second_per_watt` via the `from_metrics()` classmethod (line 127). Neither `PrecisionMetadata` nor `NormalisedMetrics` is referenced in `RawProcessResult` or `AggregatedResult` — they appear unused in the result pipeline, potentially built for a planned feature.

## Why It Matters

FLOPs are the core comparative metric of the tool. Two experiments with the same throughput but different FLOPs counts reveal hardware utilisation efficiency. The `confidence` field is uniquely important: without it, a user cannot distinguish a high-fidelity calflops measurement from a rough parameter estimate — they would look identical in the output. Confidence propagation is also how `NormalisedMetrics.tokens_per_effective_pflop` maintains interpretive honesty: a "low" confidence FLOPs estimate should be flagged when used in cross-backend comparisons.

The BitsAndBytes dequantisation note in `FlopsResult` prevents a common analysis error: treating INT4 BitsAndBytes experiments as having 1/4 the FLOPs of FP16 when they actually have FP16 FLOPs (the computation happens at FP16 post-dequant). This needs to survive in Phase 5.

## Planning Gap Details

Neither `designs/result-schema.md` nor `designs/experiment-config.md` mentions `FlopsResult`, `PrecisionMetadata`, or `NormalisedMetrics`. The peer comparison table in `result-schema.md` shows "FLOPs: ✓" for v2.0 but does not specify the confidence mechanism.

`designs/experiment-config.md` documents `PRECISION_SUPPORT` and `DECODING_SUPPORT` SSOT dicts but makes no connection to `PrecisionMetadata.precision_factor` or the effective FLOPs calculation.

The `ComputeMetrics.flops_method` and `flops_confidence` fields use plain strings rather than the typed `Literal` enums in `FlopsResult` — an inconsistency that will cause analysis code to need string comparisons rather than enum equality checks. No planning doc identifies this as a known inconsistency or states that `FlopsResult` was intended to replace those fields.

`NormalisedMetrics` and `PrecisionMetadata` are fully implemented but appear unwired from the result models. No planning doc describes their intended integration point.

## Recommendation for Phase 5

**Consolidate to `FlopsResult` as the canonical FLOPs model**. Replace `ComputeMetrics.flops_*` string fields with a nested `FlopsResult` field on `ComputeMetrics`, or replace `ComputeMetrics` entirely:

```python
class ComputeMetrics(BaseModel):
    flops: FlopsResult                       # replaces flops_total + method/confidence strings
    flops_per_token: float = 0.0
    flops_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    model_memory_mb: float = 0.0
```

Keep the typed `Literal` enums in `FlopsResult` — they are safer than plain strings. Preserve the BitsAndBytes dequantisation note in the docstring.

For `NormalisedMetrics` and `PrecisionMetadata`: decide whether to wire them into `ExperimentResult` at v2.0 (adds cross-backend normalised metrics) or defer to v2.1. If deferring, keep the classes — they represent substantial design work. If wiring at v2.0, add `normalised_metrics: NormalisedMetrics | None` to `AggregatedResult`.

Document `FlopsResult.confidence` semantics explicitly in a design doc:
- `"high"` = calflops library with architecture support
- `"medium"` = architecture-based derivation
- `"low"` = `2 × params × tokens` estimation
