# Phase 6: Results Schema and Persistence - Research

**Researched:** 2026-02-26
**Domain:** Pydantic result model redesign, JSON/Parquet persistence, directory lifecycle, SHA-256 hashing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Persistence API Surface:**
- Methods on model — `result.save(output_dir)` and `ExperimentResult.from_json(path)`
- `save()` handles full directory lifecycle: creates `{name}_{timestamp}/`, writes `result.json` + `timeseries.parquet`, applies collision suffixes (`_1`, `_2`) — one call does everything
- `from_json()` auto-discovers `timeseries.parquet` from the same directory as the loaded `result.json` — single path gives you the full result
- Missing sidecar on load: `from_json()` loads successfully with `timeseries=None` + emits a warning (graceful degradation, not an error)
- Round-trip guarantee: `ExperimentResult.from_json(result.save(path))` produces identical data

**Multi-GPU Raw File Visibility:**
- Per-process raw results are temp files during the run — written to a temp directory, aggregated into the single `ExperimentResult`, then discarded
- `ExperimentResult.process_results: list[RawProcessResult]` embeds per-process data directly in the JSON — no separate files in the output directory
- Users see only: `result.json` + `timeseries.parquet` (clean output directory)
- Late aggregation for top-level metrics: concatenate all per-process raw data, compute statistics once (avoids "average of averages" bias)

**Measurement Warnings:**
- `measurement_warnings: list[str]` — actionable human-readable suggestions
- Generated at measurement time by Phase 5 energy/backend code — passed into ExperimentResult as data (result model is passive)
- All six quality signals trigger warnings: short duration (<60s), thermal drift (>10C), GPU persistence mode off, low sample count (<30 prompts), no baseline measurement taken, ECC memory disabled
- No `--strict` mode — warnings informational only

**Timeseries Sidecar:**
- `result.timeseries: str | None` — relative filename only (`"timeseries.parquet"`) for portability
- `None` when energy measurement is disabled or no energy backend active
- Minimal raw columns only: `timestamp_s` (float), `gpu_power_w` (float), `gpu_temperature_c` (float), `gpu_index` (int)
- Only written when energy is active — no empty Parquet files

### Claude's Discretion

- Atomic write strategy (write to temp + rename, or direct write)
- Parquet compression codec (snappy vs zstd vs none)
- Internal aggregation module structure (`results/aggregation.py` vs methods on result)
- JSON serialisation details (indent level, datetime format, enum encoding)
- CSV export implementation details (column order, header format)

### Deferred Ideas (OUT OF SCOPE)

- `StudyResult` and `StudyManifest` persistence — M2 (study/sweep scope)
- Bootstrap confidence intervals (`ConfidenceIntervals` model) — v2.1
- Parquet export of flattened ExperimentResult for cross-experiment analysis — M2
- JSONL format for high-frequency per-request timeseries — v2.x
- Docker image digest in results — M3 Docker milestone
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RES-01 | `ExperimentResult` (renamed from `AggregatedResult`). All v2.0 fields ship together. | Schema rewrite section; field inventory from design doc |
| RES-02 | `measurement_config_hash: str` — SHA-256[:16], environment snapshot excluded | Hashing pattern (hashlib.sha256 + model_dump); exclusion strategy |
| RES-03 | `measurement_methodology: Literal["total", "steady_state", "windowed"]` | Enum-as-Literal pattern; value derivation from warmup_result |
| RES-04 | `steady_state_window: tuple[float, float] \| None` | Pydantic tuple field; None semantics |
| RES-05 | `schema_version: str = "2.0"` | Simple default field; already in codebase as SCHEMA_VERSION |
| RES-06 | `baseline_power_w`, `energy_adjusted_j`, `energy_per_device_j` | Fields from Phase 5; EnergyBreakdown already exists |
| RES-07 | `EnergyBreakdown` nested model (raw_j, adjusted_j, baseline provenance) | Already implemented in domain/metrics.py |
| RES-08 | `reproducibility_notes: str` — fixed disclaimer about NVML accuracy | Fixed-string field with default value |
| RES-09 | `environment_snapshot: EnvironmentSnapshot` | Already implemented in domain/environment.py |
| RES-10 | `measurement_warnings: list[str]` — quality flags | Added in Phase 5 Plan 02 to ExperimentResult; verify it's wired |
| RES-11 | `warmup_excluded_samples: int \| None` | New field on ExperimentResult; derived from WarmupResult |
| RES-12 | Process completeness validation (marker files + 4-check) — scoped to PyTorch multi-GPU internal | Existing validate_process_completeness() in aggregation.py covers this |
| RES-16 | Output always in subdirectory: `{name}_{timestamp}/result.json` + `timeseries.parquet` | Directory lifecycle in save() method |
| RES-17 | Collision policy: append `_1`, `_2` counter — never overwrite | Suffix logic in save() |
| RES-18 | JSON = always primary. Parquet = timeseries sidecar. CSV = opt-in `--export-csv`. | Persistence module structure; CSV delegated to exporters.py |
| RES-19 | `to_json()`, `to_parquet()`, `from_json()` in `results/persistence.py` | New module — does not exist yet |
| RES-20 | Late aggregation (per-process → ExperimentResult) in `results/aggregation.py` | Existing aggregation.py has this; needs v2.0 API alignment |
| RES-21 | Unified output layout: all backends → one `ExperimentResult`. PyTorch multi-GPU raw files internal (`.state/`). | save() writes single clean directory; aggregation is internal |
</phase_requirements>

---

## Summary

Phase 6 has two distinct workstreams: (1) rewriting the `ExperimentResult` schema to v2.0 with all required fields, and (2) building the persistence API (`results/persistence.py`) that handles directory creation, collision avoidance, JSON serialisation, and Parquet sidecar writing.

The existing codebase has substantial v1.x machinery that needs replacement, not extension. `domain/experiment.py` contains the old `AggregatedResult`/`RawProcessResult` models (frozen Pydantic, complex field mix of v1/v2). `results/repository.py` implements the old `raw/` + `aggregated/` directory split with `FileSystemRepository` — this entire pattern is replaced by the v2.0 `{name}_{timestamp}/` subdirectory layout. `results/aggregation.py` is ~760 lines of aggregation logic that partially applies but uses the old model structure, loguru, and some v1.x patterns. The `results/timeseries.py` exports JSON, not Parquet — this is also replaced.

Phase 5 (not yet implemented) will produce: (a) `measurement_warnings: list[str]` already added to `ExperimentResult` in Plan 02, (b) `EnergyMeasurement` dataclass from NVMLBackend/ZeusBackend, (c) `write_timeseries_parquet()` in `core/timeseries.py`, (d) `EnergyBreakdown` populated via `create_energy_breakdown()`. Phase 6 consumes these as inputs — the result schema is the container that holds Phase 5 outputs.

**Primary recommendation:** Write `results/persistence.py` as a clean new module (no inheritance from the old repository pattern). Keep `results/aggregation.py` but strip v1.x logic and align to v2.0 models. The `ExperimentResult` schema rewrite is the highest-risk task because the existing tests depend on the old field structure — plan for a test migration in Wave 0.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | >=2.0 (already installed) | `ExperimentResult` and `RawProcessResult` model definition | Already the project standard; v2.0 features (model_dump, model_validate_json, frozen models) used throughout |
| pyarrow | >=14.0 (already installed, base dep) | Parquet timeseries sidecar read/write | Already in `pyproject.toml` base deps; `pq.write_table()` / `pq.read_table()` used in Phase 5 `core/timeseries.py` |
| hashlib | stdlib | SHA-256 for `measurement_config_hash` | No external dep; `hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest()[:16]` |
| json | stdlib | JSON serialisation/deserialisation | Pydantic's `model_dump_json()` / `model_validate_json()` handle the heavy lifting |
| pathlib | stdlib | Directory lifecycle, path construction | Project standard per CLAUDE.md |
| tempfile + os.replace | stdlib | Atomic writes (temp file → rename) | Crash-safe; already used in `results/timeseries.py` `_atomic_write_json()` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| datetime | stdlib | ISO 8601 timestamps in directory names | Pydantic serialises datetime as ISO 8601 by default |
| warnings | stdlib | Emit warning on missing timeseries sidecar during load | Avoid logging dep in persistence layer |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `os.replace()` (atomic rename) | Direct `path.write_text()` | `write_text` is not atomic — crash mid-write corrupts the file. `os.replace()` is atomic on POSIX. |
| pyarrow (already base dep) | pandas + pyarrow | pandas adds ~50MB dependency; pyarrow directly avoids this. Phase 5 already uses pyarrow directly. |
| `hashlib.sha256` (stdlib) | `hashlib.blake2b` | SHA-256 is the peer standard (lm-eval uses it for task hashes). No speed benefit needed at this scale. |
| snappy Parquet compression | zstd or none | Snappy is pyarrow's default and has good read speed. zstd has better ratio but slower decompression. Snappy is correct default. |

**Installation:** No new dependencies — pyarrow is already a base dep (`pyproject.toml` INF-02).

---

## Architecture Patterns

### Recommended Module Structure

Phase 6 creates one new module and updates two existing ones:

```
src/llenergymeasure/
├── domain/
│   └── experiment.py          # REWRITE: ExperimentResult v2.0 schema
├── results/
│   ├── persistence.py         # NEW: save(), from_json(), to_parquet()
│   ├── aggregation.py         # UPDATE: strip v1.x, align to v2.0 models
│   ├── exporters.py           # UPDATE: CSV exporter uses new ExperimentResult fields
│   ├── repository.py          # KEEP AS-IS (v1.x compat) or mark deprecated
│   └── timeseries.py          # KEEP AS-IS (v1.x JSON timeseries, now superseded)
```

The `results/persistence.py` module is the single source of all disk I/O for v2.0 results. `repository.py` stays for backward compatibility (it's not actively called by the v2.0 path). `timeseries.py` (JSON format) stays for the same reason.

### Pattern 1: ExperimentResult v2.0 Schema

**What:** Complete rewrite of `ExperimentResult` in `domain/experiment.py` with all v2.0 fields.

**Key structural decisions:**
- `frozen=True` stays — result is immutable once constructed
- `AggregatedResult = ExperimentResult` alias stays (v1.x compat, removed in v3.0)
- `measurement_config_hash` computed at construction time via `model_validator(mode="before")`
- `schema_version` has `default="2.0"` (not `SCHEMA_VERSION` constant which is `"2.0.0"` — the design doc specifies `"2.0"`)
- `measurement_methodology` uses `Literal` not an `Enum` (consistent with other Literal fields in the codebase)
- `reproducibility_notes` has a fixed default string — no user input

**Example schema (from design doc):**

```python
# Source: .product/designs/result-schema.md
class ExperimentResult(BaseModel):
    # Identity
    schema_version: str = Field(default="2.0")
    experiment_id: str = Field(...)
    measurement_config_hash: str = Field(...)  # set by validator

    # Backend
    backend: str = Field(default="pytorch")
    backend_version: str | None = Field(default=None)

    # Config
    effective_config: dict[str, Any] = Field(default_factory=dict)

    # Methodology
    measurement_methodology: Literal["total", "steady_state", "windowed"] = Field(...)
    steady_state_window: tuple[float, float] | None = Field(default=None)

    # Core metrics
    total_tokens: int = Field(...)
    total_energy_j: float = Field(...)
    total_inference_time_sec: float = Field(...)
    avg_tokens_per_second: float = Field(...)
    avg_energy_per_token_j: float = Field(...)
    total_flops: float = Field(...)

    # Energy detail (RES-06, RES-07)
    baseline_power_w: float | None = Field(default=None)
    energy_adjusted_j: float | None = Field(default=None)
    energy_per_device_j: list[float] | None = Field(default=None)
    energy_breakdown: EnergyBreakdown | None = Field(default=None)

    # Environment (RES-09)
    environment_snapshot: EnvironmentSnapshot | None = Field(default=None)

    # Quality (RES-10, RES-11, RES-08)
    measurement_warnings: list[str] = Field(default_factory=list)
    warmup_excluded_samples: int | None = Field(default=None)
    reproducibility_notes: str = Field(
        default="Energy measured via NVML polling. Accuracy ±5%. "
                "Results may vary with thermal state and system load."
    )

    # Timeseries sidecar reference
    timeseries: str | None = Field(default=None)  # relative filename

    # Multi-GPU (from design doc)
    multi_gpu: MultiGPUMetrics | None = Field(default=None)

    # Timestamps
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)

    # Per-process breakdown (embedded, not separate files)
    process_results: list[RawProcessResult] = Field(default_factory=list)
    aggregation: AggregationMetadata = Field(...)

    model_config = {"frozen": True}
```

### Pattern 2: measurement_config_hash Computation

**What:** SHA-256[:16] of `ExperimentConfig.model_dump()` with `sort_keys=True`. Layer 3 fields (datacenter_pue, grid_carbon_intensity) excluded because they're in user config, not ExperimentConfig.

**Implementation:** Use Pydantic `model_validator(mode="before")` to compute the hash before model construction, accepting `config: ExperimentConfig | dict` parameter.

```python
# Source: .product/designs/result-schema.md
import hashlib, json

def _compute_config_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(
        json.dumps(config.model_dump(), sort_keys=True).encode()
    ).hexdigest()[:16]
```

**Note:** The hash is computed from `ExperimentConfig`, not from `ExperimentResult` itself. The caller (PyTorchBackend) must pass the config to the result constructor. In `_build_result()`, the config is available as a parameter.

### Pattern 3: Directory Lifecycle in save()

**What:** `result.save(output_dir: Path) -> Path` creates the full output directory, writes files, handles collisions.

```python
# Source: .product/decisions/output-storage.md + CONTEXT.md
def save(self, output_dir: Path) -> Path:
    """Save result to {output_dir}/{name}_{timestamp}/ directory.

    Returns the path to result.json (for from_json() round-trip).
    """
    from datetime import datetime

    # Build directory name
    name = self._experiment_name()  # slugified from model + backend
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    base = output_dir / f"{name}_{timestamp}"

    # Collision suffix
    target = base
    suffix = 0
    while target.exists():
        suffix += 1
        target = Path(f"{base}_{suffix}")

    target.mkdir(parents=True)

    # Write result.json (atomic)
    result_path = target / "result.json"
    _atomic_write(self.model_dump_json(indent=2), result_path)

    # Write timeseries.parquet (if timeseries data present)
    # NOTE: timeseries data is not stored in the model — it's passed separately
    # or the model stores the path reference and caller handles Parquet writing
    # See Open Question #1

    return result_path
```

**Note on timeseries:** The CONTEXT.md specifies `result.timeseries: str | None` as a relative filename. The actual Parquet data is produced by Phase 5's `write_timeseries_parquet()` in `core/timeseries.py`. Phase 6's `save()` needs to either:
- Accept a `timeseries_path: Path | None` parameter pointing to the temp Parquet file and copy/move it to the output directory, OR
- Accept raw samples and write the Parquet internally

The cleaner approach: `save(output_dir, timeseries_source: Path | None = None)` — if the caller has already written a temp Parquet (as Phase 5 plans do), `save()` moves it to the output dir and sets `result.timeseries = "timeseries.parquet"`. This keeps the Parquet writing in Phase 5 and the directory management in Phase 6. See Open Question #1.

### Pattern 4: Round-trip via from_json()

**What:** `ExperimentResult.from_json(path: Path) -> ExperimentResult` — class method.

```python
@classmethod
def from_json(cls, path: Path) -> "ExperimentResult":
    """Load ExperimentResult from result.json path.

    Auto-discovers timeseries.parquet from the same directory.
    If sidecar missing, loads successfully with timeseries=None + emits warning.
    """
    content = path.read_text()
    result = cls.model_validate_json(content)

    # Auto-discover sidecar
    sidecar = path.parent / "timeseries.parquet"
    if not sidecar.exists() and result.timeseries is not None:
        import warnings
        warnings.warn(
            f"Timeseries sidecar not found: {sidecar}. "
            "Loading result without timeseries data.",
            stacklevel=2,
        )
        # Return with timeseries field already set from JSON (relative path preserved)

    return result
```

**Round-trip guarantee:** `model_dump_json()` → `model_validate_json()` is Pydantic's standard path. Frozen models with only JSON-serialisable fields round-trip cleanly. The key risk is `datetime` fields — Pydantic v2 serialises these as ISO 8601 strings by default and deserialises them correctly.

**Tuple serialisation:** `steady_state_window: tuple[float, float] | None` — Pydantic v2 serialises tuples as JSON arrays and deserialises JSON arrays back to tuples. This round-trips correctly.

### Pattern 5: Late Aggregation in aggregation.py

**What:** The existing `aggregate_results()` in `results/aggregation.py` has the right structure but uses the wrong models (v1.x `AggregatedResult`, loguru, numpy). Phase 6 needs to update it to produce v2.0 `ExperimentResult`.

The core aggregation logic (sum energy, average throughput, concatenate latencies, OR thermal flags) is correct and should be preserved. The changes are:
1. Replace `AggregatedResult` with `ExperimentResult` as return type
2. Replace `loguru` with `import logging; logger = logging.getLogger(__name__)`
3. Add computation of `measurement_methodology` from aggregated `WarmupResult`
4. Add `warmup_excluded_samples` from aggregated warmup data
5. Remove the old campaign-level aggregation functions (dead code for v2.0)

**Multi-GPU late aggregation:** The existing `_aggregate_extended_metrics_from_results()` correctly concatenates per-process latencies before computing statistics. This pattern is valid and should be preserved.

### Pattern 6: Atomic Writes

**What:** Write to a temp file in the same directory, then `os.replace()` to the final path.

```python
# Source: existing results/timeseries.py _atomic_write_json() — verified working pattern
import os, tempfile

def _atomic_write(content: str, path: Path) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.stem)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        import contextlib
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
```

This pattern already exists in `results/timeseries.py` — reuse it in `results/persistence.py`.

### Anti-Patterns to Avoid

- **Storing computed fields in the model:** `measurement_config_hash` should be computed once at construction from the config, not on every access. Use `model_validator(mode="before")` or compute in the caller (`_build_result()`) and pass as a constructor argument.
- **Importing loguru in persistence layer:** Base package uses stdlib logging only. `results/aggregation.py` currently imports loguru — this must be replaced.
- **Non-atomic writes:** `path.write_text()` directly is not crash-safe. Always use temp + rename.
- **Embedding timeseries bytes in JSON:** The `timeseries` field stores a relative path (`"timeseries.parquet"`), not the bytes. Parquet goes in a sidecar file.
- **Using numpy in aggregation without handling the import:** `results/aggregation.py` imports numpy at module level. This is acceptable (numpy is available in the ML environment) but should be guarded for edge cases where it might not be installed.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON serialisation of Pydantic models | Custom JSON encoder | `model.model_dump_json()` + `Model.model_validate_json()` | Pydantic v2 handles datetime, tuple, nested models, unions correctly |
| Parquet schema definition | Manual column building | `pa.schema([pa.field(...)])` + `pa.Table.from_pylist()` | pyarrow handles nullable columns, type coercion, compression |
| Atomic file writes | Try/except around write_text | `tempfile.mkstemp()` + `os.replace()` | Already implemented in codebase; OS-level atomicity |
| SHA-256 hashing | Any custom hash | `hashlib.sha256(...).hexdigest()[:16]` | stdlib; 16 hex chars = 64-bit collision resistance |
| Collision suffix logic | Custom naming scheme | Simple while loop checking `Path.exists()` | One-liner; no edge cases beyond what the loop handles |

**Key insight:** The hardest part of this phase is not any individual operation — it is correctly wiring the schema rewrite into the existing codebase without breaking the existing v1.x tests that depend on `AggregatedResult` field structure.

---

## Common Pitfalls

### Pitfall 1: SCHEMA_VERSION Constant vs "2.0" String

**What goes wrong:** The codebase uses `SCHEMA_VERSION = "2.0.0"` (three-part) in `constants.py`. The design doc specifies `schema_version: str = "2.0"` (two-part). If `schema_version` defaults to `SCHEMA_VERSION`, the version string will be wrong.

**Why it happens:** The constant was set early in the project to match PEP 440 versioning conventions; the design doc uses a schema-specific format.

**How to avoid:** Set `schema_version: str = Field(default="2.0")` directly — do not import `SCHEMA_VERSION`. The `RawProcessResult.schema_version` field already uses `SCHEMA_VERSION`, so there will be a discrepancy between `RawProcessResult` (old constant) and `ExperimentResult` (new literal). This is acceptable — `RawProcessResult` is internal.

**Warning signs:** Tests asserting `result.schema_version == "2.0.0"` will catch this immediately.

### Pitfall 2: frozen=True + model_validator for hash

**What goes wrong:** `ExperimentResult` uses `model_config = {"frozen": True}`. A `model_validator(mode="after")` on a frozen model cannot set fields because the model is already immutable at that point.

**Why it happens:** Pydantic v2 freezes the model after `__init__`, including after `mode="after"` validators. `model_validator(mode="before")` receives a raw dict and can add/modify keys before the model is constructed.

**How to avoid:** Either:
1. Compute `measurement_config_hash` externally (in `_build_result()` in `pytorch.py`) and pass it as a constructor argument — **simplest, recommended**
2. Use `model_validator(mode="before")` which receives a mutable dict before Pydantic constructs the frozen model

Option 1 is cleaner: the config is available in `_build_result()` so the hash can be computed there and passed as `measurement_config_hash=_compute_config_hash(config)`.

### Pitfall 3: timeseries field name clash

**What goes wrong:** The existing `ExperimentResult` has `timeseries_path: str | None` (old name). The v2.0 schema uses `timeseries: str | None`. If the old field name is preserved, JSON round-trips from v1.x files will fail silently (field not found → None).

**Why it happens:** The CONTEXT.md locked the field name as `timeseries` (relative filename only). The existing code uses `timeseries_path`. They must be consolidated.

**How to avoid:** The v2.0 rewrite replaces the field. The old `timeseries_path` is on `RawProcessResult` (internal) — it can be renamed there too, or kept for backward compat (it's not user-visible).

### Pitfall 4: Circular import between domain/experiment.py and results/persistence.py

**What goes wrong:** `results/persistence.py` imports `ExperimentResult` from `domain/experiment.py`. If `ExperimentResult` is given a `save()` method that imports from `results/persistence.py`, a circular import results.

**Why it happens:** Methods on the model (`result.save()`) that import from `results/` at call time are fine if the import is deferred to the function body. Module-level imports create the circular dependency.

**How to avoid:** `save()` and `from_json()` as methods on `ExperimentResult` must use deferred imports:
```python
def save(self, output_dir: Path) -> Path:
    from llenergymeasure.results.persistence import _save_result  # deferred
    return _save_result(self, output_dir)
```
Alternatively, put all disk I/O logic directly in `persistence.py` as module-level functions and add thin `save()` / `from_json()` wrappers on the model that call them.

### Pitfall 5: datetime serialisation in JSON round-trip

**What goes wrong:** `datetime` fields serialised by Pydantic v2 include timezone info if the datetime is timezone-aware. If `start_time` is naive (no tzinfo), Pydantic serialises it as `"2026-02-26T14:30:00"`. On deserialisation, this becomes a naive datetime again — round-trip is safe. But if the codebase mixes naive and aware datetimes, the round-trip may produce type mismatches.

**How to avoid:** Use `datetime.now()` (naive) consistently throughout the codebase, or `datetime.now(UTC)` (aware) consistently. Do not mix. Check `start_time = datetime.now()` in `PyTorchBackend.run()` — it's currently naive. Keep it naive.

### Pitfall 6: v1.x test breakage from schema rewrite

**What goes wrong:** `test_results_aggregation.py`, `test_domain_schema_v3.py`, `test_api.py` and others construct `AggregatedResult` / `RawProcessResult` instances with the old field set. Rewriting `ExperimentResult` will break these tests.

**Why it happens:** The schema rewrite removes fields (`config_warnings`, `cli_overrides`, `parameter_provenance`, `preset_chain`, `extended_metrics` as nested model), renames fields (`timeseries_path` → `timeseries`), and adds required fields (`measurement_methodology`, `measurement_config_hash`).

**How to avoid:** Wave 0 task must update or replace the test fixtures. Since `AggregatedResult = ExperimentResult` alias is preserved, tests that use `AggregatedResult` by name will still compile but must be updated to use the new field set. Plan for a dedicated test migration task in Wave 1 or Wave 0.

---

## Code Examples

### Config Hash Computation

```python
# Source: .product/designs/result-schema.md
import hashlib
import json

def compute_measurement_config_hash(config: "ExperimentConfig") -> str:
    """SHA-256[:16] of ExperimentConfig. Layer 3 fields (env/infra) excluded.

    Layer 3 fields (datacenter_pue, grid_carbon_intensity) are not in
    ExperimentConfig — they live in user config only — so model_dump()
    naturally excludes them. No special exclusion logic needed.
    """
    canonical = json.dumps(config.model_dump(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

### Directory Name Generation

```python
# Based on .product/decisions/output-storage.md filename format
def _experiment_dir_name(result: "ExperimentResult") -> str:
    """Generate `{model_slug}_{backend}_{timestamp}` directory name."""
    # model slug: HF ID with / → - and lowercase
    model = result.effective_config.get("model", "unknown")
    slug = model.replace("/", "-").lower()
    backend = result.backend
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    return f"{slug}_{backend}_{ts}"
```

### Collision-safe Directory Creation

```python
# Source: .product/decisions/output-storage.md J3 decision
def _find_collision_free_dir(base: Path) -> Path:
    """Return base or base_1, base_2, etc. — never overwrites."""
    target = base
    counter = 0
    while target.exists():
        counter += 1
        target = Path(f"{base}_{counter}")
    target.mkdir(parents=True)
    return target
```

### Parquet Sidecar Read (via pyarrow)

```python
# Source: pyarrow documentation (verified against installed version 14+)
import pyarrow.parquet as pq

def load_timeseries_parquet(path: Path) -> "pa.Table":
    """Load timeseries sidecar. Returns pyarrow Table."""
    return pq.read_table(path)
```

### ExperimentResult.from_json() Class Method

```python
@classmethod
def from_json(cls, path: Path) -> "ExperimentResult":
    """Load from result.json. Auto-discovers timeseries sidecar."""
    import warnings
    content = Path(path).read_text(encoding="utf-8")
    result = cls.model_validate_json(content)

    # Graceful sidecar check
    sidecar = Path(path).parent / "timeseries.parquet"
    if result.timeseries is not None and not sidecar.exists():
        warnings.warn(
            f"Timeseries sidecar missing at {sidecar}. "
            "result.timeseries field preserved but file is not present.",
            stacklevel=2,
        )
    return result
```

### measurement_methodology Derivation

```python
# Source: .product/designs/result-schema.md
def derive_measurement_methodology(
    warmup_result: "WarmupResult | None",
) -> Literal["total", "steady_state", "windowed"]:
    if warmup_result is None:
        return "total"
    if warmup_result.converged:
        return "steady_state"
    return "total"  # warmup ran but didn't converge = total
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `raw/` + `aggregated/` two-tier directory split | Single `{name}_{timestamp}/result.json` subdirectory | v2.0 design (2026-02-25) | Old `FileSystemRepository` is superseded; new `save()` method handles everything |
| JSON timeseries sidecar (`process_0_timeseries.json`) | Parquet timeseries sidecar (`timeseries.parquet`) | v2.0 design (Phase 5 plan) | `results/timeseries.py` (JSON) is superseded by `core/timeseries.py` (Parquet); old file stays for compat |
| `AggregatedResult` model name | `ExperimentResult` model name | v2.0 schema (2026-02-25) | `AggregatedResult = ExperimentResult` alias maintained until v3.0 |
| `schema_version = "2.0.0"` | `schema_version = "2.0"` | v2.0 design doc | Change the default value in ExperimentResult; leave `SCHEMA_VERSION` constant as-is |
| loguru in results/aggregation.py | stdlib logging | v2.0 code standard | Base package uses stdlib logging only (decided Phase 01) |

**Deprecated/outdated:**
- `FileSystemRepository`: The old `save_raw()` / `save_aggregated()` / `load_aggregated()` pattern is superseded by `result.save()` in `results/persistence.py`. Keep `repository.py` for backward compat during transition, do not call it from v2.0 paths.
- `results/timeseries.py` `export_timeseries()`: Superseded by `core/timeseries.py` `write_timeseries_parquet()` (Phase 5). Keep for backward compat.

---

## Open Questions

1. **timeseries Parquet handoff between Phase 5 and Phase 6**
   - What we know: Phase 5 Plan 03 writes `timeseries.parquet` to a temp location during measurement, sets `timeseries_path` on the result. Phase 6 `save()` needs to write this sidecar to the output directory.
   - What's unclear: Does Phase 5 write to a temp dir and Phase 6 moves it? Or does Phase 5 write directly to the final output dir (which it can't know ahead of time)? Or does Phase 5 pass raw samples to Phase 6?
   - Recommendation: Phase 5 should write timeseries Parquet to a temp path (e.g., `Path(tempfile.mkdtemp()) / "timeseries.parquet"`) and Phase 6's `save()` accepts `timeseries_source: Path | None` — if provided, it copies/moves the file to the output dir. This keeps Phase 5 and 6 decoupled. The `result.timeseries` field is set to `"timeseries.parquet"` only if the sidecar is actually written.

2. **ExperimentResult constructor: frozen + config hash**
   - What we know: `frozen=True` prevents post-construction mutation. `measurement_config_hash` must be computed from `ExperimentConfig`, which is available in `_build_result()`.
   - What's unclear: Whether to pass hash as a constructor argument or compute it inside a `model_validator(mode="before")`.
   - Recommendation: Pass as constructor argument from `_build_result()`. Simplest, most explicit, no Pydantic validator magic.

3. **aggregation.py: how much v1.x code to remove**
   - What we know: `results/aggregation.py` is ~760 lines. About 200 lines are `aggregate_campaign_results()` and `aggregate_campaign_with_grouping()` — v1.x campaign functions that are not used in v2.0. Another 100 lines are `calculate_efficiency_metrics()`, also not in the v2.0 path.
   - What's unclear: Whether to remove dead code in Phase 6 or leave it for Phase 8 (infrastructure/testing cleanup).
   - Recommendation: Remove the campaign-level functions and `calculate_efficiency_metrics()` in Phase 6 (they use `AggregatedResult` type that will be aliased anyway, but they reference v1.x patterns). This keeps the module clean. The core `aggregate_results()` and its helpers stay.

---

## Validation Architecture

`workflow.nyquist_validation` is not set in `.planning/config.json` — skipping this section.

---

## Sources

### Primary (HIGH confidence)

- `.product/designs/result-schema.md` — full ExperimentResult v2.0 field inventory, hash strategy, multi-GPU metrics, export formats
- `.product/decisions/output-storage.md` — directory layout, collision policy, filename format decisions
- `.planning/phases/06-results-schema-and-persistence/06-CONTEXT.md` — locked API decisions, timeseries sidecar spec, warning signals
- `.planning/phases/05-energy-measurement/05-01-PLAN.md` — EnergyMeasurement dataclass produced by NVMLBackend/ZeusBackend
- `.planning/phases/05-energy-measurement/05-02-PLAN.md` — measurement_warnings field added to ExperimentResult; EnergyConfig added
- `.planning/phases/05-energy-measurement/05-03-PLAN.md` — timeseries.py write_timeseries_parquet() lifecycle; measurement integration
- `src/llenergymeasure/domain/experiment.py` — existing v1.x ExperimentResult/RawProcessResult/AggregationMetadata
- `src/llenergymeasure/results/aggregation.py` — existing aggregate_results() and late aggregation logic
- `src/llenergymeasure/results/repository.py` — old FileSystemRepository (superseded)
- `src/llenergymeasure/results/timeseries.py` — existing _atomic_write_json() pattern (reuse)
- `src/llenergymeasure/protocols.py` — ResultsRepository Protocol (save/load only in v2.0)
- `.planning/REQUIREMENTS.md` — RES-01 through RES-21 requirement definitions

### Secondary (MEDIUM confidence)

- Python 3.10 `hashlib` docs — `sha256().hexdigest()[:16]` confirmed stdlib API
- Pydantic v2 docs (training knowledge, HIGH for stable Pydantic v2 patterns) — `model_dump_json`, `model_validate_json`, `frozen=True`, `model_validator(mode="before")`
- pyarrow docs (training knowledge, HIGH) — `pq.write_table`, `pq.read_table`, `pa.Table.from_pylist`, schema construction

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project deps, APIs verified against existing codebase usage
- Architecture: HIGH — design docs are authoritative; patterns derived directly from locked decisions
- Pitfalls: HIGH — identified by reading actual existing code (frozen model issue, SCHEMA_VERSION mismatch, circular import risk, test breakage all verified against actual files)

**Research date:** 2026-02-26
**Valid until:** 2026-03-28 (stable domain — Pydantic v2, pyarrow, stdlib)
