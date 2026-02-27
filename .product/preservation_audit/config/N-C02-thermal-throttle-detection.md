# N-C02: Thermal Throttle Detection (ThermalThrottleInfo)

**Module**: `src/llenergymeasure/domain/metrics.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: No planning doc describes the throttle detection model at field level. `designs/result-schema.md` lists `thermal_throttle` as an existing schema field (in the peer comparison table) but does not specify the `ThermalThrottleInfo` structure or the six distinct throttle type flags.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/metrics.py`
**Key classes/functions**:
- `ThermalThrottleInfo` (line 299) — Pydantic BaseModel with 9 fields capturing GPU throttling state

All fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `detected` | `bool` | `False` | Any throttling occurred during experiment |
| `thermal` | `bool` | `False` | GPU thermal throttling (NVML `nvmlClocksThrottleReasonGpuIdle` family) |
| `power` | `bool` | `False` | Power brake throttling |
| `sw_thermal` | `bool` | `False` | Software thermal slowdown |
| `hw_thermal` | `bool` | `False` | Hardware thermal slowdown |
| `hw_power` | `bool` | `False` | Hardware power brake slowdown |
| `throttle_duration_sec` | `float` | `0.0` | Estimated total duration of throttling |
| `max_temperature_c` | `float | None` | `None` | Peak GPU temperature during experiment (Celsius) |
| `throttle_timestamps` | `list[float]` | `[]` | Seconds from experiment start when throttle was detected |

The `detected` field is a summary flag (True if any of thermal, power, sw_thermal, hw_thermal, or hw_power is True). The four granular flags (`sw_thermal`, `hw_thermal`, `hw_power`, `power`) map directly to NVML's `nvmlClocksThrottleReasons` bitmask values. `throttle_timestamps` enables post-hoc analysis of when during an experiment throttling occurred, which is critical for understanding whether the measurement window was corrupted.

The model is referenced in `domain/experiment.py`:
- `RawProcessResult.thermal_throttle: ThermalThrottleInfo | None` (line 118) — labelled "Schema v3"
- `AggregatedResult.thermal_throttle: ThermalThrottleInfo | None` (line 234) — also "Schema v3"

Both are optional fields defaulting to `None`, indicating throttle detection was added after the initial schema (hence the "v3" annotation).

## Why It Matters

Thermal throttling directly invalidates energy and throughput measurements. When a GPU throttles, its clock speed drops below its rated frequency — the same workload takes longer and draws different power than under non-throttled conditions. Without `ThermalThrottleInfo`, there is no way to flag which experimental results are potentially unreliable. The `throttle_timestamps` list enables a particularly important use case: determining whether throttling occurred during the measurement window (bad) or only during warmup (acceptable). The `max_temperature_c` field — combined with `ThermalEnvironment.temperature_c` from `EnvironmentMetadata` — gives a before/after temperature picture for the experiment.

This is the only peer tool in the space that captures throttling information (confirmed in `result-schema.md` peer comparison table: all peers show "✗" for thermal throttle).

## Planning Gap Details

`designs/result-schema.md` acknowledges `thermal_throttle` exists as a schema field in the peer comparison table row "Thermal throttle: ✗ ✗ ✗ ✗ ✓" but does not describe the `ThermalThrottleInfo` model structure anywhere in the document. There is no section specifying what fields it has, how the NVML bitmask is mapped to the six boolean flags, or how `throttle_timestamps` is populated.

`designs/reproducibility.md` does not mention thermal throttle at all — it is listed only in the "What IS NOT Controlled" table as "GPU boost clock speed / Thermal state at measurement time", with the implicit acknowledgement that the tool cannot prevent throttling. However, it does not document that the tool *detects* throttling, which is what `ThermalThrottleInfo` provides.

No planning doc specifies the expected collection mechanism (NVML background polling), the polling interval, or what action is taken when `detected=True` (warn only? invalidate result?).

## Recommendation for Phase 5

Keep `ThermalThrottleInfo` exactly as specified, with the following additions to the design docs:

1. Add a `ThermalThrottleInfo` section to `designs/reproducibility.md` documenting all 9 fields with the NVML bitmask mapping for the four granular flags.

2. Decide and document the action policy when `detected=True`:
   - Option A: Warn in CLI output, flag in result JSON, do not abort
   - Option B: Abort the experiment run (too aggressive — throttling may be brief)
   - Recommendation: Option A. The result is still valid data; users should be aware.

3. Specify the collection mechanism: NVML background polling at N-second intervals during the measurement window, sampling `nvmlDeviceGetCurrentClocksThrottleReasons()`. The polling interval determines the resolution of `throttle_timestamps`.

4. Consider adding `throttle_fraction` (throttle_duration_sec / total_duration_sec) as a derived property — useful for quick filtering of "significantly throttled" experiments (e.g., >10% throttled) in study post-processing.
