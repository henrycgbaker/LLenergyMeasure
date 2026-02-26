# P-23: Streaming Latency — TTFT and ITL

**Module**: `src/llenergymeasure/domain/metrics.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: The `LatencyStatistics` and `LatencyMeasurements` types are confirmed to exist in planning. However, the wiring of these types into the v2.0 orchestrator is undocumented — specifically, which backends collect per-token timestamps, what the streaming callback mechanism looks like, and how warmup requests are excluded from measurement.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/metrics.py`
**Key classes/functions**:
- `LatencyMeasurementMode` (line 547) — enum with three values: `TRUE_STREAMING` ("true_streaming"), `PER_REQUEST_BATCH` ("per_request_batch"), `PROPORTIONAL_ESTIMATE` ("proportional"). Distinguishes measurement accuracy across backends.
- `LatencyMeasurements` (line 581) — raw dataclass storing: `ttft_ms: list[float]`, `itl_full_ms: list[float]`, `itl_trimmed_ms: list[float]`, `request_count: int`, `total_output_tokens: int`, `excluded_tokens: int`, `streaming_mode: bool`, `warmup_requests_excluded: int`, `measurement_mode: LatencyMeasurementMode`
- `LatencyStatistics` (line 619) — computed statistics dataclass: TTFT (mean, median, p95, p99, min, max, samples) and ITL (mean, median, p95, p99, samples) for trimmed ITL, plus full ITL mean and p99 for comparison
- `collect_itl_measurements()` (line 650) — utility function; takes per-request token timestamp lists, computes ITL diffs via `numpy.diff()`, trims first/last intervals per request (to exclude warmup effect and EOS anomalies), returns `(itl_full, itl_trimmed, excluded_count)`

The `LatencyMeasurements` are stored on `InferenceMetrics.latency_measurements` (line 212 of metrics.py), typed as `Any | None` to avoid circular dependency. The aggregation layer in `aggregation.py` (lines 237–258) correctly concatenates raw TTFT and ITL samples across all processes before computing percentiles — the statistically correct approach (not averaging per-process percentiles).

## Why It Matters

TTFT (Time to First Token) and ITL (Inter-Token Latency) are the primary latency metrics for streaming LLM inference. TTFT measures perceived responsiveness — users experience this as "how long until I see the first word". ITL measures the generation speed after the first token. These metrics are completely invisible from bulk throughput measurements (tokens/second). Without them, latency comparisons between backends (e.g., does vLLM's continuous batching actually reduce TTFT vs static PyTorch batching?) cannot be made. The trimmed ITL approach (excluding first/last interval per request) is methodologically sound and follows MLPerf inference benchmark conventions.

## Planning Gap Details

The `designs/result-schema.md` (not read but referenced in planning docs) is expected to confirm `LatencyStatistics` in the `ExperimentResult` schema. The `core/CLAUDE.md` internal note correctly documents the streaming callback approach (PyTorch via `TextIteratorStreamer`, vLLM/TensorRT via proportional estimation).

What is missing from planning docs:
- `designs/architecture.md` — `metrics.py` in `core/` is listed as handling "FLOPs calculation, throughput, latency statistics" but the streaming callback wiring is not described
- `designs/observability.md` — the standard output example shows `Latency TTFT 142ms ITL 28ms` — confirming these metrics are expected in v2.0 output, but the design does not say which backends produce them or how
- No planning doc specifies that `warmup_requests_excluded` must match `config.streaming_warmup_requests` (currently `DEFAULT_STREAMING_WARMUP_REQUESTS = 5` in constants)
- The `LatencyMeasurementMode` enum (which backends use which mode) is code-only knowledge

## Recommendation for Phase 5

Carry `LatencyMeasurements`, `LatencyStatistics`, `LatencyMeasurementMode`, and `collect_itl_measurements()` forward unchanged into the v2.0 domain model. These are clean, well-structured types.

Wire the measurement mode explicitly into each backend's implementation notes:
- PyTorch: `TRUE_STREAMING` via `TextIteratorStreamer` callback — capture per-token arrival time
- vLLM: check whether async streaming gives true per-token timestamps or requires `PROPORTIONAL_ESTIMATE`
- TensorRT: likely `PROPORTIONAL_ESTIMATE`

The `aggregate_latency_measurements()` function in `aggregation.py` (lines 514–592) is the correct aggregation implementation and must be preserved. The key invariant: concatenate all `ttft_ms` and `itl_trimmed_ms` lists from all processes first, then compute percentiles — never average the per-process percentiles.

Add a note to `designs/architecture.md` specifying which backends produce `TRUE_STREAMING` latency vs estimated latency, and confirming that `llem run --streaming` is the flag that enables collection.
