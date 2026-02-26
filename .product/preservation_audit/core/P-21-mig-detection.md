# P-21: MIG Instance Detection

**Module**: `src/llenergymeasure/results/aggregation.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: Not mentioned in any planning document. Energy measurement accuracy on A100/H100 GPUs with MIG enabled is a known problem domain; the current warning logic is the only mitigation, and it has no planning-level counterpart.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/results/aggregation.py`
**Key classes/functions**:
- MIG detection block (lines 199–207) in `aggregate_results()` — checks `r.gpu_is_mig` on each `RawProcessResult` and accumulates `gpu_mig_profile` values
- Warning string generated (line 203–206): `"Experiment ran on N MIG instance(s) (profile_str). Energy measurements reflect parent GPU total, not per-instance consumption."`
- Warning is appended to `AggregationMetadata.warnings` list, which propagates into `AggregatedResult.aggregation.warnings`

The detection is passive: `gpu_is_mig: bool` and `gpu_mig_profile: str | None` are fields on `RawProcessResult` (populated at experiment time, presumably from `nvidia-smi` or NVML at model load). The aggregator does not attempt to correct or adjust the energy measurement — it flags the limitation. The profile string (e.g., `"1g.10gb"`, `"4g.40gb"`) is collected from all MIG results and deduplicated before inclusion in the warning.

The warning surfaces in the CLI via `show_raw_result()` in `cli/display/results.py` (line 33–34): if `result.gpu_is_mig` is true, the profile is displayed inline with the per-process result output.

## Why It Matters

Multi-Instance GPU (MIG) partitioning on A100 and H100 GPUs presents a fundamental energy measurement problem: NVML reports power consumption for the **entire physical GPU**, not for the individual MIG instance. A measurement on a `1g.10gb` partition (1/7th of an A100) will report the power draw of all seven MIG instances combined, making the energy figure meaningless for efficiency comparisons. Without this flag, results from MIG instances would silently appear to show vastly higher energy consumption than the experiment actually consumed, corrupting any downstream efficiency analysis. The warning is essential for result integrity.

## Planning Gap Details

None of the following docs reference MIG detection, A100/H100 MIG behaviour, or the energy measurement implications:
- `decisions/architecture.md`
- `designs/architecture.md`
- `designs/energy-backends.md` (this is the most notable gap — energy accuracy is the topic of the document, and MIG is a major accuracy caveat)
- `decisions/cli-ux.md`
- `designs/observability.md`

The `core/CLAUDE.md` internal note (line: "MIG instances report parent GPU power, not per-instance") captures the gotcha correctly — but a gotcha note is not a planning decision. The v2.0 result schema should formally document that `energy_measurement_warning` on `RawProcessResult` carries this flag when MIG is detected.

## Recommendation for Phase 5

The detection logic in `aggregate_results()` (lines 199–207) is correct and should be carried forward verbatim. It is three lines of code with a meaningful warning.

Additionally: when rebuilding `RawProcessResult` in Phase 5, ensure `gpu_is_mig: bool` and `gpu_mig_profile: str | None` are preserved as fields. These are populated in the inference backends at GPU detection time (check `domain/model_info.py` or the backend initialisation code for where `gpu_is_mig` is set — that path must be preserved too).

The planning gap should be closed: add a note to `designs/energy-backends.md` accuracy table explicitly stating "MIG: parent GPU power reported, not per-instance — flag in results, no correction applied."
