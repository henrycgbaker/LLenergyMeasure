# domain/ - Domain Models

Pydantic models for metrics, experiment results, and model metadata.

## Purpose

Defines the data structures used throughout the framework for metrics collection and result persistence. All models are immutable Pydantic BaseModels.

## Key Files

### metrics.py
Metrics collected during experiments.

**EnergyMetrics** - Energy consumption:
```python
EnergyMetrics(
    total_energy_j=150.0,
    gpu_energy_j=140.0,
    cpu_energy_j=10.0,
    gpu_power_w=280.0,
    duration_sec=2.5,
    emissions_kg_co2=0.00015,
)
```

**FlopsResult** - FLOPs estimation with provenance:
```python
FlopsResult(
    value=1.5e12,
    method="calflops",  # calflops | architecture | parameter_estimate
    confidence="high",   # high | medium | low
    dtype="float16",
)
```

**LatencyStatistics** - Computed percentiles:
```python
LatencyStatistics(
    mean_ms=46.5,
    median_ms=45.8,
    p95_ms=52.1,
    p99_ms=58.3,
    min_ms=38.2,
    max_ms=62.1,
    sample_count=95,
)
```

### experiment.py
Experiment result models.

**ExperimentResult** - The user-visible output of a single-process measurement
run. Holds the final metrics (energy, throughput, FLOPs, latency) directly.

## Schema Version

Results include a `schema_version` field on `ExperimentResult` (see its default in
`experiment.py`) for forward compatibility.

## Related

- See `../results/README.md` for result persistence
