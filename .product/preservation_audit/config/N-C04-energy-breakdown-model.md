# N-C04: Energy Breakdown Model (EnergyBreakdown)

**Module**: `src/llenergymeasure/domain/metrics.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: `designs/result-schema.md` defers baseline-corrected energy (`energy_adjusted_j`, `baseline_power_w`) to v2.1, but `EnergyBreakdown` already implements all of these fields — plus additional provenance fields (`baseline_method`, `baseline_timestamp`, `baseline_cache_age_sec`) that the design doc does not mention anywhere.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/metrics.py`
**Key classes/functions**:
- `EnergyBreakdown` (line 269) — Pydantic BaseModel with 6 fields for detailed energy attribution

All fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `raw_j` | `float` | Yes | Total measured energy in Joules (unmodified) |
| `adjusted_j` | `float | None` | No | Baseline-adjusted energy: `raw_j - (baseline_power_w * duration_sec)` |
| `baseline_power_w` | `float | None` | No | Measured idle baseline power in Watts |
| `baseline_method` | `str | None` | No | How baseline was obtained: `'cached'`, `'fresh'`, or `'unavailable'` |
| `baseline_timestamp` | `datetime | None` | No | When the baseline power measurement was taken |
| `baseline_cache_age_sec` | `float | None` | No | Age of a cached baseline in seconds |

The model is referenced in `domain/experiment.py`:
- `RawProcessResult.energy_breakdown: EnergyBreakdown | None` (line 114) — labelled "Schema v3"
- `AggregatedResult.energy_breakdown: EnergyBreakdown | None` (line 231) — labelled "Schema v3"

The baseline methodology string (`'cached'`, `'fresh'`, `'unavailable'`) captures how the baseline was sourced — a fresh 30-second idle measurement taken immediately before the experiment, a cached measurement from a previous experiment in the same study, or unavailable (baseline measurement skipped or failed). The `baseline_cache_age_sec` field enables filtering on baseline staleness: a baseline measured 3 hours ago is less trustworthy than one measured 30 seconds ago.

The relationship to `core/baseline.py` (referenced in the task brief): the `EnergyBreakdown` model is the output type of the baseline adjustment logic. The `baseline_method` and `baseline_cache_age_sec` fields are produced by the baseline measurement module, which decides whether to take a fresh measurement or use a cached one.

## Why It Matters

Baseline subtraction is critical for short experiments. GPU idle power ranges from ~15 W (consumer GPU) to ~60 W (A100 SXM). For a 10-second experiment at 200 W active power with a 42 W idle baseline, raw energy is 2000 J but inference-attributable energy is only (200 - 42) × 10 = 1580 J — a 26% overstatement. Without `EnergyBreakdown`, batch=1 experiments (which are short and run at lower power) appear systematically worse relative to batch=32 experiments (which run longer at higher power), introducing a spurious effect that obscures the true efficiency relationship.

`designs/result-schema.md` confirms this importance: "For short experiments, idle power can account for 15–30% of raw measured energy — systematically overstating the cost of actual inference work." The `EnergyBreakdown` model is the concrete implementation of this correction.

The `baseline_method` field is particularly valuable in multi-experiment studies: it records whether each experiment's baseline was fresh (most accurate), shared from cache (slightly stale), or missing (uncorrected) — enabling study-level quality filtering in analysis.

## Planning Gap Details

`designs/result-schema.md` treats baseline correction as a v2.1 addition:

> **v2.1 Additions:**
> `baseline_power_w: float | None` — Idle GPU power measured during a 30-second window
> `energy_adjusted_j: float | None` — Baseline-subtracted energy

However, `EnergyBreakdown` already implements both of these (as `baseline_power_w` and `adjusted_j` respectively) plus three additional provenance fields that the design doc does not mention anywhere:
- `baseline_method: str | None` — `'cached'`, `'fresh'`, or `'unavailable'`
- `baseline_timestamp: datetime | None` — when baseline was measured
- `baseline_cache_age_sec: float | None` — staleness of cached baseline

The design doc's v2.1 flat fields (`baseline_power_w`, `energy_adjusted_j` directly on `ExperimentResult`) are architecturally different from the code's nested `EnergyBreakdown` sub-model. Phase 5 must choose one approach.

Additionally, the design doc mentions `BaselineConfig` (from `experiment-config.md`: `enabled: bool = True`, `duration_seconds: float = 30.0`) but never connects it to `EnergyBreakdown`. The connection is: `BaselineConfig.enabled` determines whether `EnergyBreakdown.baseline_method` is `'unavailable'` or actually measured.

The peer comparison table in `result-schema.md` confirms "No peer tool publishes baseline-corrected energy" — this field set is a genuine differentiator.

## Recommendation for Phase 5

Keep `EnergyBreakdown` as a nested sub-model on `ExperimentResult` (not flat fields). The nested approach is superior to the v2.1 flat-field design because:
- It groups all baseline-related data together (raw, adjusted, provenance)
- `baseline_timestamp` and `baseline_cache_age_sec` have no natural place as flat fields
- The `adjusted_j: float | None` naming is clearer than `energy_adjusted_j`

Update `designs/result-schema.md` to:
1. Move `EnergyBreakdown` from v2.1 to v2.0 — it is already implemented
2. Document all 6 fields of `EnergyBreakdown` (not just the 2 currently mentioned)
3. Update the peer comparison table row "Baseline-corrected energy: ✗ — ✓ (NEW)" to v2.0

Update `designs/experiment-config.md` to explicitly link `BaselineConfig.enabled = False` → `EnergyBreakdown.baseline_method = 'unavailable'` and `EnergyBreakdown.adjusted_j = None`.

Verify that `core/baseline.py` exists and implements the caching logic that populates `baseline_method`, `baseline_timestamp`, and `baseline_cache_age_sec` — this is the only module named in the task brief that was not directly read.
