# Result Schema Design

**Last updated**: 2026-02-25
**Source research**: [../research/12-result-schemas.md](../research/12-result-schemas.md)
**Status**: Confirmed (session 3, naming updated session 4; v2.1 fields confirmed session 5)

## Summary

The existing `RawProcessResult` / `ExperimentResult` schema is already best-in-class
among peer tools (Zeus, CodeCarbon, lm-eval, ML.ENERGY). No peer tracks FLOPs, thermal
throttle, warmup status, distributed aggregation, or per-process breakdown.

> **Note (2026-02-25):** Research recommends demoting FLOPs from "primary metric" to
> "reference metadata" — FLOPs are deterministic for a given model+input and do not vary
> between deployment configurations. The primary metrics that vary are `energy_per_output_token`
> and `tokens_per_second`. See [decisions/flops-estimation.md](../decisions/flops-estimation.md)
> for the research annotation.

All fields ship at **v2.0** (decided 2026-02-25, decision #50 — collapse v2.0/v2.1 split).
Study-level schema (new) and export formats also v2.0.

---

## On-Disk Format: Always `ExperimentResult`

The user-visible file is always `ExperimentResult`. `RawProcessResult` is internal
plumbing — collected per-process (or per-GPU for tensor parallel), then aggregated.

```
results/
  llama-3.1-8b_pytorch_2026-02-18T14-30/        ← single experiment (always subdirectory)
    result.json                                    ← ExperimentResult
    timeseries.parquet                             ← power, energy, throughput, temperature at 1 Hz
  batch-size-effects_2026-02-18T15-00/           ← study
    study_summary.json                            ← StudyResult
    llama-3.1-8b_pytorch_bf16_batch1_2026-...json ← ExperimentResult (per experiment)
    llama-3.1-8b_pytorch_bf16_batch8_2026-...json
    ...
```

**Single-process experiment** (common case in v2.0):
`ExperimentResult.process_results` has exactly one item. Aggregation is trivial.

**Tensor-parallel experiment** (e.g. 70B across 4 GPUs):
`ExperimentResult.process_results` has N items. Late aggregation runs across all N.
Late aggregation = collect all per-request latencies from all processes, compute
statistics once. Avoids "average of averages" bias (Phase 4 audit confirmed as justified).

---

## Confirmed Additions to Existing Schema

Three fields added to `ExperimentResult` (and propagated to `RawProcessResult`):

### 1. `measurement_config_hash: str`

SHA-256 of the fully-resolved `ExperimentConfig` as a canonical JSON string.
Enables exact reproducibility checking: two experiments with the same `measurement_config_hash`
used identical measurement configurations.

```python
import hashlib, json

measurement_config_hash = hashlib.sha256(
    json.dumps(config.model_dump(), sort_keys=True).encode()
).hexdigest()[:16]  # 16 hex chars = 64-bit collision resistance, sufficient
```

**Layer 3 exclusion — DECIDED 2026-02-19:** `datacenter_pue` and `grid_carbon_intensity_gco2_kwh`
are **excluded from the hash**. Rationale: these fields have already moved from `ExperimentConfig`
(Layer 2) to user config (Layer 1) as defaults, and are stored in results as Layer 3 scientific
context. They describe the environment the measurement was taken in, not what was measured.
Two researchers running identical experiments in Germany (350 gCO2/kWh) and the US (386 gCO2/kWh)
produce the same measurement — the hash should reflect this. CO2 calculations differ but the
energy measurement is identical.

These fields are not in `ExperimentConfig` — they live in user config and env vars only.
`experiment.yaml` is infrastructure-agnostic by design (decided 2026-02-20). See
[../decisions/architecture.md — Layer 3](../decisions/architecture.md).

**Field name rationale:** `measurement_config_hash` over `config_hash` (too ambiguous — could
imply user config or infrastructure context) and over `experiment_hash` (too broad — would
imply hashing runtime environment too). The name makes the scope explicit.

**Peer reference:** lm-eval-harness `task_hashes` dict (per-task hashes for reproducibility).

### 2. `measurement_methodology: str`

Explicit field declaring what was measured. Makes interpretation unambiguous.

```python
class MeasurementMethodology(str, Enum):
    TOTAL = "total"            # Entire run including warmup
    STEADY_STATE = "steady_state"  # Excludes warmup (warmup_result not None)
    WINDOWED = "windowed"      # Explicit time window (rare, future use)
```

Value determined automatically from `warmup_result`:
- `warmup_result is None` → `TOTAL`
- `warmup_result.converged` → `STEADY_STATE`
- `warmup_result.skipped` → `TOTAL` (warmup not applicable)

**Peer reference:** ML.ENERGY uses steady-state implicitly (`steady_state_duration`,
`steady_state_energy` fields) — we make it explicit.

### 3. `steady_state_window: tuple[float, float] | None`

`(start_sec, end_sec)` relative to experiment start — the portion actually measured.
`None` when `measurement_methodology == TOTAL`.

Already partially implicit via `WarmupResult` fields. Making it explicit
means the result file is self-contained: you can interpret the energy/latency numbers
without cross-referencing `WarmupResult`.

```python
# When steady-state measurement used:
steady_state_window = (warmup_result.warmup_duration_sec, timestamps.duration_sec)
# NOTE (2026-02-26): warmup_duration_sec does not exist in v1.x WarmupResult.
# Must be added (sum of latencies_ms / 1000, or wall-clock). See preservation audit N-C03.
# e.g., (12.3, 67.8) → measured seconds 12.3–67.8 of a 67.8-second experiment
```

---

## Multi-GPU Result Fields

For experiments using multiple GPUs (TP/PP parallelism, or `device_map="auto"`).

```python
class MultiGPUMetrics(BaseModel):
    num_gpus: int
    energy_per_gpu_j: list[float]        # per-device breakdown (imbalance diagnosis)
    energy_total_j: float                 # sum across all devices (primary for comparison)
    energy_per_output_token_j: float      # primary efficiency metric; normalises for
                                           # GPU count, batch size, and output length

class ExperimentResult(BaseModel):
    ...
    multi_gpu: MultiGPUMetrics | None = None   # None for single-GPU runs
```

`multi_gpu=None` for single-GPU: energy goes to top-level `energy_total_j`. No special-casing
in analysis code — presence of `multi_gpu` signals a multi-GPU run; absence does not.

`energy_per_output_token_j` is the **primary cross-configuration efficiency metric** — it
normalises for GPU count, batch size, and output length simultaneously. See
[decisions/multi-gpu.md](../decisions/multi-gpu.md) for rationale.

---

## Additional `ExperimentResult` Fields (v2.0)

> **Updated (2026-02-25):** These fields were originally planned for v2.1 but have been
> moved to v2.0 (decision #50). All result fields ship together — no micro-versioned rollout.

### 1. `schema_version: str`

Explicit version on `ExperimentResult` (already present on `StudyResult`). Makes standalone
result files self-describing — no cross-referencing the tool version that produced them.

```python
schema_version: str = "2.0"   # bumped to "2.1" when v2.1 fields are populated
```

### 2. `baseline_power_w: float | None`

Idle GPU power (Watts) measured during a 30-second window immediately before the experiment
starts (as part of pre-flight). Used to compute `energy_adjusted_j`.

`None` if baseline measurement was skipped (e.g. `energy.baseline: false` in config).

```python
# Pre-flight: 30s idle measurement
baseline_power_w: float | None = None   # e.g. 42.3 W
```

**Why it matters:** GPU idle power ranges from ~15 W (consumer) to ~60 W (A100). For short
experiments, idle power can account for 15–30% of raw measured energy — systematically
overstating the cost of the actual inference work. Without correction, batch=1 appears far
worse than it is relative to batch=32.

### 3. `energy_adjusted_j: float | None`

Baseline-subtracted energy — the energy attributable to inference work alone.

```python
energy_adjusted_j = energy_total_j - (baseline_power_w * duration_sec)
```

`None` when `baseline_power_w is None`. `energy_total_j` (the raw measurement) is preserved
unchanged — `energy_adjusted_j` is additive, never a replacement.

**Peer reference:** No peer tool publishes baseline-corrected energy. ML.ENERGY measures
"active power" but does not subtract an idle baseline from the total energy integral.

### 4. `energy_per_device_j: list[float] | None`

Per-GPU energy breakdown when the Zeus backend is active (Zeus reads per-device NVML
energy counters directly). For single-GPU experiments the list has one element.

```python
energy_per_device_j: list[float] | None = None   # e.g. [234.1, 241.8, 239.5, 237.2]
```

`None` for non-Zeus backends (NVML polling and CodeCarbon aggregate across devices).

**Peer reference:** Zeus `ZeusMonitor.end_window()` returns per-GPU energy. We surface this
directly rather than flattening it, so tensor-parallel experiments show per-device balance.

### 5. `confidence_intervals: ConfidenceIntervals | None`

> **Deferred to v2.1 (2026-02-26):** Preservation audit P-17 confirmed CIs are downstream
> analysis, not primary measurement. The field and schema below are retained for v2.1
> implementation. The bootstrap infrastructure exists in v1.x (`results/bootstrap.py`) and
> will be wired up at v2.1. Note: v1.x `BootstrapResult` conflicts with `MetricCI` naming
> below — resolve at v2.1. See [../preservation_audit/INDEX.md](../preservation_audit/INDEX.md) decision #16.

Bootstrap 95% confidence intervals for key metrics, computed from per-request samples
within the experiment. Requires `n >= 30` to be meaningful.

```python
class MetricCI(BaseModel):
    mean: float
    ci_lower: float     # 2.5th percentile of bootstrap distribution
    ci_upper: float     # 97.5th percentile
    n_bootstrap: int = 1000

class ConfidenceIntervals(BaseModel):
    energy_total_j: MetricCI
    energy_adjusted_j: MetricCI | None   # None if no baseline
    tokens_per_second: MetricCI
    ttft_ms: MetricCI | None             # None for batch-only backends
    itl_ms: MetricCI | None
```

`None` when `n < 30`. Bootstrap resampling runs on `RawProcessResult.per_request_latencies`.

### 6. `warmup_excluded_samples: int | None`

Explicit count of prompts excluded from the measurement window during warmup.
Complements `steady_state_window` (which gives the time window) with the sample count.

```python
warmup_excluded_samples: int | None = None   # e.g. 12
# None when measurement_methodology == TOTAL (no warmup exclusion)
```

---

## New: Study Result Schema

> **`StudyResult` vs `StudyManifest` — Disambiguation (item 7):**
> `StudyResult` is the **completed return value of `run_study()`** — written once, at end.
> `StudyManifest` (in `study_manifest.json`) is the **in-progress checkpoint** — written
> incrementally during the run for recovery. They share some fields (`study_name`, `started_at`)
> but are separate Pydantic models with different purposes. See `designs/study-resume.md`
> for the `StudyManifest` schema and the full disambiguation table.

Study-level summary written to `results/study-{id}/study_summary.json`.

```python
class StudySkipped(BaseModel):
    config: dict[str, Any]          # The raw config dict that failed validation
    reason: str                     # Pydantic ValidationError message (human-readable)

class StudyFailed(BaseModel):
    config: dict[str, Any]          # The ExperimentConfig that was attempted
    exception_type: str             # e.g. "RuntimeError", "torch.cuda.OutOfMemoryError"
    error_message: str              # Exception message

class StudySummary(BaseModel):
    total_generated: int            # Total configs from grid expansion
    ran: int                        # Successfully completed experiments
    skipped: int                    # L1 invalid (Pydantic ValidationError)
    failed: int                     # L3 runtime failure (subprocess crash, OOM, etc.)

class MeasurementProtocol(BaseModel):
    """Execution settings used for this study run. NOT in study_design_hash — stored as metadata.
    See decisions/study-execution-model.md Decision B."""
    n_cycles: int
    cycle_order: str                      # "sequential" | "interleaved" | "shuffled"
    config_gap_seconds_used: float        # actual gap used (from user config or study override)
    cycle_gap_seconds_used: float


class StudyResult(BaseModel):
    schema_version: str = "2.0"    # Fixed 2026-02-19: v2.0 tool → schema "2.0"
    study_id: str                   # e.g. "study-2026-02-18T14-30"
    study_yaml: str                 # Path to source study.yaml
    study_design_hash: str          # SHA-256[:16] of sweep+experiments only (execution block excluded)
                                    # Same study with different n_cycles → same hash ("topping up")
                                    # Replaces study_yaml_hash (see rejected design below)
    started_at: datetime
    completed_at: datetime
    runner: str                     # "local" | "docker"

    measurement_protocol: MeasurementProtocol   # execution settings used — NOT in hash

    summary: StudySummary

    skipped: list[StudySkipped]     # All L1 invalids with reasons
    failed: list[StudyFailed]       # All L3 runtime failures

    result_files: list[str]         # Paths to individual ExperimentResult JSON files
    # Results as paths, NOT embedded — studies can have hundreds of experiments

# Rejected (2026-02-20): study_yaml_hash: str (SHA-256 of full study.yaml contents)
# Rejected because: hashing the full YAML file is fragile (whitespace, comments, field ordering
# all change the hash even when the experimental design is identical). study_design_hash hashes
# the canonical model_dump() of the experimental design portion only, excluding the execution
# block — two runs of the same study at different rigour levels share the same hash.
# See decisions/study-execution-model.md Decision B.
```

**Key design choices:**
- `result_files` contains paths, not embedded results — keeps the summary file small
  and readable regardless of study size
- `skipped` and `failed` are separate arrays with full config + reason — enough to
  diagnose patterns and update `PRECISION_SUPPORT` / `DECODING_SUPPORT` SSOT dicts
- `study_design_hash` identifies the experimental design (sweep + experiment list, execution block
  excluded). Two runs of the same study with different `n_cycles` share the same hash — same study,
  different rigour. Enables "topping up" a study (3 cycles → 5) without creating a new identity.
- `measurement_protocol` stores the execution settings used (not hashed) — provides full
  provenance for how rigorously the study was run alongside the results.

---

## Export Formats

| Format | When written | Use case |
|--------|-------------|----------|
| **JSON** | Always (primary) | Human-readable, single-experiment inspection, API responses |
| **Parquet** | On request (`--format parquet` or always alongside JSON for studies) | Study analysis across many experiments in Pandas/DuckDB/Polars |

**Parquet schema** (flattened from `ExperimentResult` for analysis):

```
experiment_id, schema_version, measurement_config_hash, backend, backend_version, model, precision,
batch_size, dataset, n_prompts,
energy_total_j, energy_adjusted_j, baseline_power_w, energy_per_token_j, gpu_power_w,
tokens_per_second, ttft_mean_ms, itl_mean_ms, itl_p99_ms,
tokens_per_second_ci_lower, tokens_per_second_ci_upper,
energy_total_j_ci_lower, energy_total_j_ci_upper,
flops_total, flops_per_token, peak_memory_mb,    # FLOPs = reference metadata, not primary metric (see decisions/flops-estimation.md)
# peak_memory_mb: inference window only — torch.cuda.reset_peak_memory_stats() called after
# model load, before first inference. Captures KV cache + activations + batch buffers, not
# model weights. Semantics under review — see todo: revisit-peak-memory-mb-measurement-semantics
measurement_methodology, warmup_excluded_samples,
gpu_name, cuda_version,
started_at, duration_sec
```

Per-device energy (`energy_per_device_j`) and full CI objects stay in JSON — too wide for
flat Parquet. The flattened CI columns above cover the most-compared metrics.

Nested fields (per-process, per-request latencies, extended metrics) stay in JSON.
Parquet is for the flat metrics that researchers compare across experiments.

> **Updated (2026-02-26):** CSV export kept for v2.0 (preservation audit N-R03). CSV is the
> only exporter that exists in v1.x — carries forward as `--export-csv` opt-in alongside
> the primary JSON + Parquet sidecar. Rationale: accessibility for Excel, R, `cat`.
> See [../preservation_audit/INDEX.md](../preservation_audit/INDEX.md) decision #3.

**Not in v2.0:**
- JSONL: defer — useful for per-request timeseries in v2.x when timeseries grows
- Arrow/HDF5: overkill, heavy dependencies

---

## Schema Comparison: Peer Tools vs LLenergyMeasure

| Field | Zeus | CodeCarbon | lm-eval | ML.ENERGY | LLenergyMeasure v2.0 |
|-------|------|-----------|---------|-----------|---------------------|
| Energy (Joules) | ✓ | ✗ (kWh) | ✗ | ✓ | ✓ |
| TTFT/ITL | ✗ | ✗ | ✗ | ✓ (raw) | ✓ (statistics + raw) |
| FLOPs (reference metadata) | ✗ | ✗ | ✗ | ✗ | ✓ |
| Thermal throttle | ✗ | ✗ | ✗ | ✗ | ✓ |
| Warmup status | ✗ | ✗ | ✗ | Implicit | ✓ (explicit) |
| Measurement methodology | ✗ | ✗ | ✗ | Implicit | ✓ |
| Config hash | ✗ | ✗ | ✗ | ✗ | ✓ |
| Steady-state window | ✗ | ✗ | ✗ | Implicit | ✓ |
| Per-process breakdown | ✗ | ✗ | ✗ | ✗ | ✓ |
| Backend version | ✗ | ✗ | ✓ | Implicit | ✓ |
| Study summary | ✗ | ✗ | ✗ | ✗ | ✓ |
| Parquet export | ✗ | ✗ | ✗ | ✗ | ✓ |
| Schema version | ✗ | ✗ | ✗ | ✗ | ✓ |
| Baseline-corrected energy | ✗ | ✗ | ✗ | ✗ | ✓ |
| Per-device energy breakdown | ✓ (Zeus only) | ✗ | ✗ | ✗ | ✓ |
| Confidence intervals | ✗ | ✗ | ✗ | ✗ | ✓ |
| Warmup excluded count | ✗ | ✗ | ✗ | ✗ | ✓ |
