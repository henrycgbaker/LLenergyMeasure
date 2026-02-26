# N-X02: Inference Metrics Computation

**Module**: `src/llenergymeasure/core/inference.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: Not mentioned in planning docs, but this is a pure computation utility with no architectural concerns. The key gap is that the calculation has a known limitation — latency is summed not wall-clocked — which should be documented in the result schema design.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/core/inference.py`
**Key classes/functions**:
- `calculate_inference_metrics()` (line 8) — takes `num_prompts: int`, `latencies_ms: list[float]`, `total_input_tokens: int`, `total_generated_tokens: int`; returns `InferenceMetrics`

The function computes:
- `total_time_sec = sum(latencies_ms) / 1000.0` — sum of batch latencies (not wall-clock time)
- `tokens_per_sec = total_generated_tokens / total_time_sec` — throughput
- `latency_per_token_ms = 1000.0 / tokens_per_sec` — inverse of throughput
- Returns `InferenceMetrics` with: `total_tokens`, `input_tokens`, `output_tokens`, `inference_time_sec`, `tokens_per_second`, `latency_per_token_ms`

The function handles zero-division gracefully: `tokens_per_sec = 0.0` when `total_time_sec == 0`, and `latency_per_token_ms = 0.0` when `tokens_per_sec == 0`.

The file is 37 lines total — a minimal, focused utility.

## Why It Matters

This is the central throughput calculation. `tokens_per_second` is one of the three primary metrics the tool reports (alongside energy in Joules and FLOPs). The function is the single source of truth for how latency measurements from the inference loop are converted into the normalised per-token metrics stored in results. Without it, or with a different formulation, throughput numbers would be inconsistent across backends.

The important architectural detail: `latencies_ms` is a list of *batch* latencies (not per-token), which means `sum(latencies_ms)` is the total inference time across all batches processed sequentially. This is correct for PyTorch static batching where batches run serially. For vLLM (continuous batching), the wiring into this function needs verification — vLLM may report total elapsed time differently.

## Planning Gap Details

The function is a pure computation utility without architectural implications — planning docs are not expected to describe it. However:
- The `latency_per_token_ms` field is computed as the inverse of throughput, not as a direct per-request measurement. This is a methodological choice that should be documented in the result schema to avoid confusion with `LatencyStatistics.itl_mean_ms` (which measures actual inter-token latency from streaming callbacks). These are different concepts.
- No planning doc distinguishes `InferenceMetrics.latency_per_token_ms` (derived from throughput) from `LatencyStatistics.itl_mean_ms` (measured from streaming). Both appear in results — users may conflate them.

## Recommendation for Phase 5

Carry `calculate_inference_metrics()` forward as-is. The function is trivially simple and correct.

One naming note: the planning decisions (MEMORY.md, session 3) confirmed renaming `num_input_prompts` → `n`. Verify the calling code passes the correct argument when the config field is renamed — the function signature uses `num_prompts` internally which is fine (it's an implementation detail).

Add a note to the result schema design distinguishing the two latency concepts:
- `InferenceMetrics.latency_per_token_ms` = `1000 / tokens_per_second` (throughput-derived)
- `LatencyStatistics.itl_mean_ms` = mean inter-token gap from streaming callbacks (measured)

The former is always available; the latter only when `streaming=True`.
