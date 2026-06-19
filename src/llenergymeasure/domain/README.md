# domain/ - Domain Models

Pydantic models for metrics, experiment results, and model metadata.

## Purpose

Defines the data structures used throughout the framework for metrics collection and result persistence. All models are immutable Pydantic BaseModels.

## Key Files

### metrics.py
Metrics collected during experiments.

**InferenceMetrics** - Throughput and latency:
```python
InferenceMetrics(
    total_tokens=1024,
    input_tokens=512,
    output_tokens=512,
    inference_time_sec=2.5,
    tokens_per_second=204.8,
    latency_per_token_ms=4.88,
)
```

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

**LatencyMeasurements** - Streaming latency data:
```python
LatencyMeasurements(
    ttft_ms=[45.2, 48.1, 42.8, ...],      # Time to first token samples
    itl_full_ms=[12.1, 11.8, 13.2, ...],  # All inter-token latencies
    itl_trimmed_ms=[12.1, 11.8, ...],     # ITL with first/last tokens removed
    request_count=95,
    total_output_tokens=12350,
    excluded_tokens=190,                   # First/last tokens excluded from ITL
    streaming_mode=True,
    warmup_requests_excluded=5,
    measurement_method="streaming",        # streaming | per_request_batch | proportional_estimate
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

Results include `schema_version` (currently "2.0.0") for forward compatibility:
```python
from llenergymeasure.utils.constants import SCHEMA_VERSION
```

## Related

- See `../results/README.md` for result persistence
