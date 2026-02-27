# Phase 6: Results Schema and Persistence - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Every experiment produces a complete, schema-versioned `ExperimentResult` written to a stable
output directory — with collision-safe naming, a Parquet timeseries sidecar, and a round-trip-safe
persistence API. This phase implements the result schema, persistence layer, and late aggregation.

**Backend-agnostic requirement:** All schema and persistence code must work for PyTorch (M1),
vLLM, and TensorRT-LLM (M3) without modification. No backend-specific assumptions in result
models or save/load logic.

Study-level results (`StudyResult`, `StudyManifest`) are M2 scope — not this phase.

</domain>

<decisions>
## Implementation Decisions

### Persistence API Surface
- **Methods on model** — `result.save(output_dir)` and `ExperimentResult.from_json(path)`
- `save()` handles full directory lifecycle: creates `{name}_{timestamp}/`, writes `result.json`
  + `timeseries.parquet`, applies collision suffixes (`_1`, `_2`) — one call does everything
- `from_json()` auto-discovers `timeseries.parquet` from the same directory as the loaded
  `result.json` — single path gives you the full result
- **Missing sidecar on load:** `from_json()` loads successfully with `timeseries=None` + emits
  a warning (graceful degradation, not an error). The JSON result is always valid on its own.
- Round-trip guarantee: `ExperimentResult.from_json(result.save(path))` produces identical data

### Multi-GPU Raw File Visibility
- Per-process raw results are **temp files during the run** — written to a temp directory,
  aggregated into the single `ExperimentResult`, then discarded
- `ExperimentResult.process_results: list[RawProcessResult]` embeds per-process data directly
  in the JSON — no separate files in the output directory
- Users see only: `result.json` + `timeseries.parquet` (clean output directory)
- **Late aggregation** for top-level metrics: concatenate all per-process raw data (e.g.
  per-request latencies from all processes), compute statistics once. Avoids "average of
  averages" bias for latency percentiles.

### Measurement Warnings
- `measurement_warnings: list[str]` — actionable human-readable suggestions
- Each warning includes what was detected AND what to do about it
  - Example: "Short experiment duration (8.2s < 60s recommended). Consider increasing n or
    using longer prompts for stable energy readings."
- **Generated at measurement time** by Phase 5 energy/backend code (has access to live GPU
  state, thermal readings, timing) — passed into ExperimentResult as data. Result model is
  passive, does not compute warnings.
- **All known quality signals** trigger warnings:
  - Short duration (<60s)
  - Thermal drift (>10C during measurement)
  - GPU persistence mode off
  - Low sample count (<30 prompts)
  - No baseline measurement taken
  - ECC memory disabled
- No `--strict` mode — warnings are informational only (decided in Phase 5 context)

### Timeseries Sidecar
- `result.timeseries: str | None` — relative filename only (`"timeseries.parquet"`)
  - Portable: works if the whole directory is moved/zipped
  - `None` when energy measurement is disabled or no energy backend active
- **Minimal raw columns** (peer-validated — no peer tool stores derived cumulative energy):
  - `timestamp_s` (float) — elapsed seconds from measurement start
  - `gpu_power_w` (float) — instantaneous GPU power reading
  - `gpu_temperature_c` (float) — GPU temperature at same sample
  - `gpu_index` (int) — which GPU (multi-GPU ready from day one)
- Derived metrics (cumulative energy = `np.trapz(power, time)`) computed at analysis time,
  not stored. Decouples storage from integration method.
- **Only written when energy is active** — no empty Parquet files. `result.timeseries = None`
  when no energy backend produced samples.

### Claude's Discretion
- Atomic write strategy (write to temp + rename, or direct write)
- Parquet compression codec (snappy vs zstd vs none)
- Internal aggregation module structure (`results/aggregation.py` vs methods on result)
- JSON serialisation details (indent level, datetime format, enum encoding)
- CSV export implementation details (column order, header format)

</decisions>

<specifics>
## Specific Ideas

- The `ls results/` idiom from cli-ux.md is the design principle — output directories should
  be self-explanatory without a separate command
- Collision suffix pattern matches Hydra (`_1`, `_2`) — familiar to ML researchers
- Per the preservation audit (N-R03), CSV export carries forward from v1.x as `--export-csv`
  opt-in alongside primary JSON + Parquet
- Schema and persistence must be backend-agnostic from day one — vLLM and TensorRT-LLM
  backends arrive in M3 and must work without modifying any Phase 6 code

</specifics>

<deferred>
## Deferred Ideas

- `StudyResult` and `StudyManifest` persistence — M2 (study/sweep scope)
- Bootstrap confidence intervals (`ConfidenceIntervals` model) — v2.1
- Parquet export of flattened ExperimentResult for cross-experiment analysis — M2
- JSONL format for high-frequency per-request timeseries — v2.x
- Docker image digest in results — M3 Docker milestone

</deferred>

---

*Phase: 06-results-schema-and-persistence*
*Context gathered: 2026-02-26*
