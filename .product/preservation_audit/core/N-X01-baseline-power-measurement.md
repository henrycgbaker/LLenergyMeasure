# N-X01: Baseline Power Measurement

**Module**: `src/llenergymeasure/core/baseline.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0
**Planning Gap**: `designs/energy-backends.md` mentions baseline as a concept (idle vs inference power) but does not document the implementation: NVML-based polling, session-level cache with TTL, the `BaselineCache` dataclass, or the floor-at-zero adjustment logic.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/core/baseline.py`
**Key classes/functions**:
- `BaselineCache` (line 22) — dataclass with fields: `power_w: float`, `timestamp: float`, `device_index: int`, `sample_count: int`, `duration_sec: float`
- `measure_baseline_power()` (line 37) — primary measurement function; parameters: `device_index=0`, `duration_sec=30.0`, `sample_interval_ms=100`, `cache_ttl_sec=3600.0`. Uses `pynvml.nvmlDeviceGetPowerUsage()` in a polling loop; computes mean of all samples collected. Returns `None` on any failure (graceful degradation).
- Module-level `_baseline_cache: dict[int, BaselineCache]` (line 34) — session-level cache keyed by device index; survives across experiment runs within the same Python process. TTL checked against `time.time()`.
- `invalidate_baseline_cache()` (line 129) — clears cache for one device or all; takes `device_index: int | None`
- `adjust_energy_for_baseline()` (line 143) — computes `total_energy_j - (baseline_power_w * duration_sec)`; floors at `0.0` (line 161: `return max(0.0, adjusted)`) — negative adjustment is physically impossible and indicates measurement noise
- `create_energy_breakdown()` (line 164) — produces an `EnergyBreakdown` domain object; includes `baseline_method` field: `"cached"` if cache age > 1.0 second, `"fresh"` if just measured, `"unavailable"` if no baseline

## Why It Matters

Idle GPU power is not zero. A modern data-centre GPU at rest draws 50–100W+ of idle power. Without baseline subtraction, energy measurements conflate this constant idle draw with the inference workload. For short experiments (30–60 second runs), idle power can constitute 20–40% of the total measured energy, making comparisons between experiments that vary in duration meaningless. The floor-at-zero in `adjust_energy_for_baseline()` is not a simplification — negative adjusted energy would indicate the inference experiment consumed *less* energy than idle, which is physically impossible and would corrupt results. The session-level cache (1-hour TTL) is important for study runs: measuring baseline once per session rather than once per experiment avoids adding 30 seconds overhead to every experiment in a study.

## Planning Gap Details

`designs/energy-backends.md` references the baseline concept:
> "Energy measurements reflect parent GPU total, not per-instance consumption"

and describes the accuracy table for NVML polling (~10-15%) but does not document:
- That a 30-second idle measurement is required before each session
- The `BaselineCache` TTL mechanism
- The `EnergyBreakdown` domain object fields (`raw_j`, `adjusted_j`, `baseline_method`, `baseline_cache_age_sec`)
- The floor-at-zero invariant

`designs/architecture.md` mentions `core/energy/nvml.py` as a target file but does not say baseline measurement lives there vs separately.

The `domain/metrics.py` `EnergyBreakdown` model (line 269–296) is the downstream consumer of this module's output. The planning docs confirm `EnergyBreakdown` exists in the result schema but do not document where it is computed.

## Recommendation for Phase 5

Carry `baseline.py` forward unchanged into `core/` in the v2.0 structure. The implementation is correct and complete.

In the v2.0 orchestrator (`orchestration/orchestrator.py`), ensure `measure_baseline_power()` is called once at experiment initialisation (before model load), with the result stored and passed to `create_energy_breakdown()` after measurement completes. The `ExperimentOrchestrator` should cache the result internally for the session — do not call it again on retry.

Add to `designs/energy-backends.md`:

```
## Baseline Measurement
- 30s idle poll at 100ms intervals via NVML before inference starts
- Session-level cache: 1-hour TTL per device (avoids 30s overhead per study experiment)
- Adjustment: adjusted_j = raw_j - (baseline_w * duration_sec), floor 0.0
- Stored in EnergyBreakdown.adjusted_j (None if baseline unavailable)
```

The `baseline_method: "cached" | "fresh" | "unavailable"` field in the result provides full provenance — important for scientific reproducibility.
