# N-C01: Environment Metadata (EnvironmentMetadata)

**Module**: `src/llenergymeasure/domain/environment.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: `designs/reproducibility.md` describes a different class (`EnvironmentSnapshot`) with a different field set. The actual implementation (`EnvironmentMetadata`) is more structured and captures more data. Phase 5 must decide which model to implement; currently they are incompatible.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/domain/environment.py`
**Key classes/functions**:
- `GPUEnvironment` (line 12) — GPU hardware sub-model
- `CUDAEnvironment` (line 31) — CUDA runtime sub-model
- `ThermalEnvironment` (line 42) — GPU thermal state at experiment start
- `CPUEnvironment` (line 63) — CPU governor, model string, OS platform
- `ContainerEnvironment` (line 77) — container detection flags
- `EnvironmentMetadata` (line 90) — top-level composite model
- `EnvironmentMetadata.summary_line` (line 111) — property returning one-line log string

The implementation uses a **five-sub-model composition** rather than a flat model. `GPUEnvironment` captures: `name: str`, `vram_total_mb: float`, `compute_capability: str | None`, `pcie_gen: int | None`, `mig_enabled: bool`. `CUDAEnvironment` captures: `version: str`, `driver_version: str`, `cudnn_version: str | None`. `ThermalEnvironment` captures: `temperature_c: float | None`, `power_limit_w: float | None`, `default_power_limit_w: float | None`, `fan_speed_pct: float | None`. `CPUEnvironment` captures: `governor: str` (default `"unknown"`), `model: str | None`, `platform: str`. `ContainerEnvironment` captures: `detected: bool`, `runtime: str | None`. The composite `EnvironmentMetadata` assembles these five sub-models plus `collected_at: datetime`. The `summary_line` property produces a formatted string like `"A100 80GB | CUDA 12.4 | Driver 535.104 | 42C | container"`.

The model is referenced in `domain/experiment.py` at line 110 (`RawProcessResult.environment: EnvironmentMetadata | None`) and line 226 (`AggregatedResult.environment: EnvironmentMetadata | None`), stored as optional fields in both result types, labelled "Schema v3".

## Why It Matters

Environment metadata is the primary mechanism for reproducibility analysis. Without it, two experiments that produced different energy readings cannot be diagnosed — was it a different GPU, a different CUDA version, a different thermal state, or genuine measurement variance? The `ThermalEnvironment` sub-model (temperature at experiment start, power limit vs factory default) is particularly important: it captures exactly the environmental factors that are "not controlled" per `designs/reproducibility.md`. The MIG detection flag (`mig_enabled`) in `GPUEnvironment` is directly relevant to energy measurement accuracy warnings documented in `RawProcessResult.energy_measurement_warning`.

## Planning Gap Details

`designs/reproducibility.md` describes a class called `EnvironmentSnapshot` (not `EnvironmentMetadata`) with a flat structure:

```python
class EnvironmentSnapshot(BaseModel):
    python_version: str
    cuda_version: str | None
    driver_version: str | None
    llenergymeasure_version: str
    installed_packages: list[str]   # pip freeze
    timestamp_utc: str
    gpu_name: str | None
    gpu_vram_gb: float | None
    gpu_count: int | None
```

The actual implementation differs in three significant ways:

1. **Structure**: The code uses a five-sub-model composition; the design uses a flat model. The code's sub-models expose fields the design omits entirely: `compute_capability`, `pcie_gen`, `mig_enabled`, `cudnn_version`, `power_limit_w`, `default_power_limit_w`, `fan_speed_pct`, `cpu_governor`, `cpu_model`, `container.runtime`.

2. **Missing from code vs design**: The design includes `python_version`, `llenergymeasure_version`, `installed_packages` (pip freeze), and `gpu_count` — none of these exist in `EnvironmentMetadata`. The design's `installed_packages` (pip freeze for full software environment capture) is particularly important for reproducibility and is absent from the code.

3. **Class name**: `EnvironmentSnapshot` in design vs `EnvironmentMetadata` in code. `result-schema.md` references `environment_snapshot` as the field name; the code uses `environment`.

No planning doc acknowledges or reconciles this divergence. The design doc appears to be a forward-looking proposal that was never reconciled with the existing implementation.

## Recommendation for Phase 5

Keep `EnvironmentMetadata` and its five-sub-model structure — it is more informative than the flat `EnvironmentSnapshot` design. Reconcile the two by:

1. Rename `EnvironmentMetadata` → `EnvironmentSnapshot` to align with planning docs and the `result-schema.md` field name `environment_snapshot`, OR update planning docs to use `EnvironmentMetadata`. Choose one name and apply it everywhere.

2. Add the fields present in the design but missing from the code:
   - `python_version: str`
   - `llenergymeasure_version: str`
   - `installed_packages: list[str]` (pip freeze output)
   - `gpu_count: int` (derive from number of `GPUEnvironment` instances, or add explicit field)

3. The `collected_at: datetime` field in the code (line 108) maps to `timestamp_utc: str` in the design — keep the `datetime` type, not a string.

4. For multi-GPU environments (tensor-parallel at v2.1), the sub-model approach is correct: `gpu: list[GPUEnvironment]` rather than a flat `gpu_names: list[str]`. This is noted as a TODO in the design doc.

5. Add a `capture_environment()` function (described in design doc as pseudocode, not yet implemented in the domain module) alongside the model in `domain/environment.py`.
