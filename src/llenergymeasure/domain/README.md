# domain/ - Domain Models

Pydantic models for metrics, experiment results, and model metadata.

## Purpose

Defines the data structures used throughout the framework for metrics collection and result persistence. All models are immutable Pydantic BaseModels.

## Key Files

### metrics.py
Metrics collected during experiments.

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

## Bundle Version

`ExperimentResult` carries a `bundle_version` field (see its default in
`experiment.py`, sourced from `bundle_artefacts.BUNDLE_VERSION`). It is the single
version for the whole per-experiment bundle - the same value is stamped into
`config.json` and `environment.json` - and it replaces the retired per-artefact
`schema_version` counters.

## Related

- See `../results/README.md` for result persistence
