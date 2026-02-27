# N-C03: Warmup Convergence Result (WarmupResult)

**Module**: `src/llenergymeasure/domain/metrics.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: `designs/result-schema.md` references `WarmupResult` in several places but never specifies its fields. `designs/experiment-config.md` defines `WarmupConfig` (the input) but does not document the output model. Phase 5 must not accidentally drop fields when implementing from the design doc alone.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/metrics.py`
**Key classes/functions**:
- `WarmupResult` (line 344) — Pydantic BaseModel with 6 fields recording warmup phase outcome
- `WarmupConfig` (defined in `designs/experiment-config.md` as the configuration counterpart)

All fields of `WarmupResult`:

| Field | Type | Required | Description |
|---|---|---|---|
| `converged` | `bool` | Yes | Whether CV threshold was reached before `max_prompts` |
| `final_cv` | `float` | Yes | Coefficient of variation at the end of warmup |
| `iterations_completed` | `int` | Yes | Actual number of warmup prompts run |
| `target_cv` | `float` | Yes | The configured CV threshold (`WarmupConfig.cv_target`) |
| `max_prompts` | `int` | Yes | The configured maximum (`WarmupConfig.max_prompts`) |
| `latencies_ms` | `list[float]` | No (default `[]`) | Raw warmup latency samples in ms, for debugging |

All five non-list fields are required (no default). `latencies_ms` defaults to an empty list — it is explicitly labelled "for debugging" in the description, suggesting it may not always be populated in production runs. The relationship to `WarmupConfig` is direct: `target_cv` mirrors `WarmupConfig.cv_target` and `max_prompts` mirrors `WarmupConfig.max_prompts`, making the result self-contained for interpretation without cross-referencing the config.

The model is referenced in `domain/experiment.py`:
- `RawProcessResult.warmup_result: WarmupResult | None` (line 122) — labelled "Schema v3"
- Note: `AggregatedResult` does not have a `warmup_result` field — this is correct, since warmup results are per-process, not aggregated

The `WarmupResult` is also the trigger for `measurement_methodology` in `ExperimentResult` per `designs/result-schema.md`: `warmup_result is None → TOTAL`, `warmup_result.converged → STEADY_STATE`.

## Why It Matters

`WarmupResult` is the evidence trail for a critical measurement decision: whether the reported energy and throughput numbers reflect steady-state operation or a transient ramp-up period. Without it, a result showing `measurement_methodology = STEADY_STATE` is an unsupported assertion. With `WarmupResult`, the result file documents exactly how many prompts were run, what the latency stability looked like (`final_cv`), and whether the system actually converged. The `latencies_ms` list enables visual inspection of the warmup curve (useful for debugging slow convergence or oscillating systems). The `iterations_completed` field is the count later exposed as `warmup_excluded_samples` in the v2.1 schema additions.

## Planning Gap Details

`designs/result-schema.md` mentions `WarmupResult` in two places:
1. The `measurement_methodology` section (lines describing `warmup_result.converged` trigger)
2. The `steady_state_window` section (`"Already partially implicit via WarmupResult.warmup_duration_sec"`)

Neither reference matches the actual implementation: the code has no `warmup_duration_sec` field. The design doc assumes a field that does not exist in `WarmupResult`. To compute `steady_state_window = (warmup_result.warmup_duration_sec, timestamps.duration_sec)`, a `warmup_duration_sec` field would need to be added, or computed from `latencies_ms` (sum of warmup latency samples).

`designs/experiment-config.md` documents `WarmupConfig` (the input) correctly:
```python
class WarmupConfig(BaseModel):
    n_prompts: int = 20
    cv_target: float = 0.05
    max_prompts: int = 100
```
But makes no mention of `WarmupResult` (the output). A Phase 5 developer reading only the design docs would know how to configure warmup but not what the result looks like.

`designs/reproducibility.md` does not mention warmup at all.

The `v2.1` additions in `result-schema.md` include `warmup_excluded_samples: int | None`. This maps to `WarmupResult.iterations_completed` — the design doc should reference that field explicitly.

## Recommendation for Phase 5

Keep `WarmupResult` with its current 6 fields. Add one field to resolve the `steady_state_window` gap:

```python
class WarmupResult(BaseModel):
    converged: bool
    final_cv: float
    iterations_completed: int
    target_cv: float
    max_prompts: int
    latencies_ms: list[float] = []
    warmup_duration_sec: float = 0.0   # ADD: sum of latencies_ms / 1000, or wall-clock
```

Update `designs/result-schema.md` to:
1. Document the full `WarmupResult` field set
2. Correct the reference to `warmup_result.warmup_duration_sec` (currently a forward reference to a field that does not exist)
3. Clarify that `warmup_excluded_samples` in v2.1 = `warmup_result.iterations_completed`

Decide whether `latencies_ms` is always populated (useful but memory-heavy for large warmup runs) or only in debug mode (flag on `WarmupConfig`). The current implementation does not clarify this.
