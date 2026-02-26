Excellent. Now I have comprehensive information. Let me compile the findings into a structured schema document:

## Schema Extraction Report: Peer Tools & LLenergyMeasure

This report extracts result output schemas from peer tools and compares them with LLenergyMeasure's existing domain models.

---

## 1. Peer Tool Result Schemas

### 1.1 Zeus (ML.ENERGY Initiative)

**Energy Measurement Output: `Measurement` Dataclass**

```python
@dataclass
class Measurement:
    time: float                              # Elapsed seconds
    gpu_energy: dict[int, float]             # GPU index -> Joules
    cpu_energy: dict[int, float] | None      # CPU index -> Joules (via RAPL)
    dram_energy: dict[int, float] | None     # DRAM index -> Joules
    soc_energy: SoCMeasurement | None        # Apple Silicon / Jetson
    
    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules)."""
        return sum(self.gpu_energy.values())
```

**CSV Logging Format** (from `log_file` parameter):
```
Header: start_time, window_name, elapsed_time, gpu{i}_energy
Example: 1708345123.456, inference, 45.2, 12345.67
```

**Power Timeline Output** (PowerMonitor):
```python
dict[int, list[tuple[float, float]]]  # gpu_idx -> [(timestamp, power_watts), ...]
```

**Provenance Tracking:**
- Named measurement windows (`begin_window("name")` / `end_window("name")`)
- Window metadata stored internally; CSV logs capture start_time, window_name
- No explicit run ID or version tracking in measurement itself

**Hardware Metadata:**
- GPU detection: NVML API returns GPU name, compute capability, VRAM
- Power limits in milliwatts (mW)
- Implemented via `zeus.device.get_gpus()` abstraction

---

### 1.2 CodeCarbon (v3.2.2)

**Emissions Tracking Output**

```python
# Tracker.stop() returns dict-like structure (internal CodeCarbonData):
{
    "duration": float,                    # Seconds
    "emissions": float,                   # kg CO2e
    "energy_consumed": float,             # kWh
    "timestamp": datetime,
    "cpu_power": float | None,            # Watts
    "gpu_power": float | None,            # Watts
    "ram_power": float | None,            # Watts
}
```

**CSV Output** (`emissions.csv`):
```
timestamp,duration,emissions,energy_consumed,cpu_power,gpu_power,ram_power
2026-02-18 10:30:15,0.5,0.00001234,0.000000567,12.5,85.3,4.2
```

**Cloud API Output** (to `api.codecarbon.io`):
```json
{
  "api_version": "3.2.2",
  "organization_id": "uuid",
  "project_id": "uuid",
  "experiment_id": "uuid",
  "run_id": "uuid",
  "emissions_data": {
    "duration": 45.2,
    "energy_consumed": 0.00156,
    "emissions": 0.000234,
    "cpu_power": 25.3,
    "gpu_power": 120.5,
    "ram_power": 8.2
  },
  "metadata": {
    "country_iso_code": "US",
    "region": "California",
    "timestamp": "2026-02-18T10:30:15Z"
  }
}
```

**Provenance Tracking:**
- Configuration hierarchy: Script > Env vars > `.codecarbon.config` > `~/.codecarbon.config`
- Country/region metadata for carbon intensity
- Timestamp-based tracking
- No explicit version tracking of inference backend or model

**Hardware Metadata:**
- CPU model via `/proc/cpuinfo`
- GPU info via nvidia-ml-py (name, memory)
- Platform detection (OS, container detection)
- CPU frequency governor state

---

### 1.3 lm-evaluation-harness (EleutherAI)

**Results Schema: `EvalResults` TypedDict**

```python
# results_YYYY-MM-DDTHH-MM-SS.xxxxx.json
{
    "results": dict[str, _TaskMetrics],     # task_name -> metrics
    "groups": dict,                          # aggregated group-level metrics
    "group_subtasks": dict[str, list[str]],
    "configs": dict,                         # full YAML task configurations
    "versions": dict,                        # task versions
    "n-shot": dict,                          # few-shot counts per task
    "higher_is_better": dict,                # metric direction
    "n-samples": dict[str, object],          # _SampleCount(original, effective)
    "samples": dict[str, list],              # SampleResult if log_samples=True
    
    # Configuration metadata
    "config": _EvalConfig,
    {
        "model": str,
        "model_args": str,
        "batch_size": int,
        "device": str,
        "seed": int,
        "git_hash": str,
        "date": str
    },
    
    # Environment metadata
    "pretty_env_info": str,
    "transformers_version": str,
    "lm_eval_version": str,
    
    # Tokenizer metadata
    "pad_token": str | None,
    "eos_token": str | None,
    "bos_token": str | None,
    "eot_token_id": int | None,
    "max_length": int | None,
    
    # Model identity
    "model_source": str,
    "model_name": str,
    "model_name_sanitized": str,
    
    # Chat fields
    "system_instruction": str | None,
    "chat_template": str | None,
    "fewshot_as_multiturn": bool,
    
    # Task metadata
    "task_hashes": dict,
    "total_evaluation_time_seconds": float
}
```

**Per-Task Metrics** (`_TaskMetrics`):
```python
{
    "name": str,
    "alias": str,
    "sample_len": int,
    # Dynamic metric keys: "acc,none", "f1,macro_weighted", etc.
    "acc,none": float,
    "f1,macro_weighted": float,
    ...
}
```

**Per-Sample Results** (JSONL, `samples_{task_name}_YYYY-MM-DDTHH-MM-SS.xxxxx.jsonl`):
```python
{
    "doc_id": str,
    "doc": dict,                    # Original document
    "target": str,
    "arguments": list,
    "resps": list[str],             # Model responses
    "filtered_resps": list[str],    # Post-processing filtered
    "filter": str,
    "metrics": dict,
    "doc_hash": str,
    "prompt_hash": str,
    "target_hash": str
}
```

**Provenance Tracking:**
- Explicit version fields: `lm_eval_version`, `transformers_version`
- Git hash of eval repo
- Timestamp (ISO format)
- Task hashes for reproducibility
- Model source tracking

**Hardware Metadata:**
- Device type (cuda, cpu, mps)
- Pretty environment info (human-readable string from dataclasses)
- Batch size
- Tokenizer info

---

### 1.4 ML.ENERGY Benchmark (v3.0)

**Benchmark Results** (`results.json`):

```json
{
  "model_id": "Qwen/Qwen3-8B",
  "gpu_model": "H100",
  "num_gpus": 1,
  "max_num_seqs": 128,
  "num_prompts": 1024,
  "completed": 1024,
  "duration": 135.2,
  "steady_state_duration": 97.6,
  "steady_state_energy": 54631.48,
  "steady_state_energy_per_token": 0.093545,
  "output_throughput": 4890.37,
  "total_output_tokens": 661234,
  "results": [
    {
      "ttft": 0.156,
      "itl": [0.012, 0.011, ...],
      "latency": 8.234,
      "energy": 52.1,
      "input_len": 245,
      "output_len": 646,
      "success": true
    }
  ]
}
```

**Leaderboard Data** (`public/data/tasks/{task}.json`):

```typescript
{
  "task": "gpqa",
  "task_display_name": "GPQA",
  "architecture": "llm",
  "configurations": [
    {
      "model_id": "Qwen/Qwen3-14B",
      "nickname": "Qwen 3 14B",
      "gpu_model": "B200",
      "num_gpus": 1,
      "total_params_billions": 14.0,
      "activated_params_billions": 14.0,
      "architecture": "Dense Transformer",
      "weight_precision": "bfloat16",
      "max_num_seqs": 128,
      
      "tensor_parallel": 1,
      "expert_parallel": 0,
      "data_parallel": 1,
      
      "energy_per_token_joules": 0.183,
      "energy_per_request_joules": 118.2,
      "avg_power_watts": 508,
      
      "avg_output_len": 646,
      "median_itl_ms": 3.2,
      "p90_itl_ms": 5.1,
      "p95_itl_ms": 6.8,
      "p99_itl_ms": 12.5,
      "output_throughput_tokens_per_sec": 2771,
      "avg_batch_size": 87
    }
  ]
}
```

**Provenance Tracking:**
- Explicit model_id (HuggingFace path)
- GPU model name and count
- Deployment parameters (max_num_seqs, TP degree, etc.)
- Benchmark version implicit (v3.0)

**Hardware Metadata:**
- GPU model (H100, B200)
- Number of GPUs
- Implicit: steady-state measurement (excludes startup)
- Implicit: single GPU per benchmark (no communication variance)

---

## 2. LLenergyMeasure Current Result Schemas

### 2.1 RawProcessResult (Per-GPU Measurement)

**File:** `/home/h.baker@hertie-school.lan/workspace/llm-efficiency-measurement-tool/src/llenergymeasure/domain/experiment.py`

```python
class RawProcessResult(BaseModel):
    """Raw metrics from a single process - never aggregated inline."""
    
    schema_version: str                     # e.g., "3.1"
    experiment_id: str                      # Unique identifier
    backend: str                            # "pytorch", "vllm", "tensorrt"
    backend_version: str | None             # e.g., "0.6.2"
    process_index: int                      # Rank in distributed setup
    gpu_id: int                             # Device index
    gpu_name: str                           # e.g., "NVIDIA A100-SXM4-80GB"
    gpu_is_mig: bool                        # Multi-Instance GPU flag
    gpu_mig_profile: str | None             # e.g., "1g.6gb"
    energy_measurement_warning: str | None  # Energy measurement caveats
    energy_tracking_failed: bool            # Placeholder metrics if failed
    
    config_name: str                        # Config identifier
    model_name: str                         # Model path/name
    timestamps: Timestamps                  # start, end, duration_sec
    
    # Core metrics
    inference_metrics: InferenceMetrics     # tokens, throughput, latency
    energy_metrics: EnergyMetrics           # joules, power, emissions
    compute_metrics: ComputeMetrics         # FLOPs, memory
    
    # Configuration provenance (new in schema v2)
    effective_config: dict[str, Any]        # Full resolved config
    cli_overrides: dict[str, Any]           # CLI-overridden parameters
    config_warnings: list[str]              # Validation warnings
    parameter_provenance: dict[str, dict[str, Any]]  # Per-param: value, source, source_detail
    preset_chain: list[str]                 # Presets applied in order
    
    # Extended metrics (always present, fields null when not computable)
    extended_metrics: ExtendedEfficiencyMetrics
    
    # Raw data for late aggregation
    per_request_latencies_ms: list[float]   # E2E per-request latencies
    gpu_utilisation_samples: list[float]    # GPU SM % samples
    
    # Schema v3: Environment, energy breakdown, thermal
    environment: EnvironmentMetadata | None # Hardware/software environment
    energy_breakdown: EnergyBreakdown | None # Raw vs baseline-adjusted
    thermal_throttle: ThermalThrottleInfo | None  # Throttle detection
    warmup_result: WarmupResult | None      # Warmup convergence status
    timeseries_path: str | None             # Path to time-series data
```

**Metric Classes:**

```python
class InferenceMetrics(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    inference_time_sec: float
    tokens_per_second: float
    latency_per_token_ms: float
    time_to_first_token_ms: float | None
    latency_measurements: Any | None        # LatencyMeasurements (raw TTFT/ITL)

class EnergyMetrics(BaseModel):
    total_energy_j: float
    gpu_energy_j: float
    cpu_energy_j: float
    ram_energy_j: float
    gpu_power_w: float
    cpu_power_w: float
    duration_sec: float
    emissions_kg_co2: float
    energy_per_token_j: float

class ComputeMetrics(BaseModel):
    flops_total: float
    flops_per_token: float
    flops_per_second: float
    peak_memory_mb: float
    model_memory_mb: float
    flops_method: str              # "calflops" | "architecture" | "parameter"
    flops_confidence: str          # "high" | "medium" | "low"
    compute_precision: str         # "fp16", "fp32", etc.
```

**Environment Metadata:**

```python
class EnvironmentMetadata(BaseModel):
    gpu: GPUEnvironment                     # name, vram_total_mb, compute_capability
    cuda: CUDAEnvironment                   # version, driver_version, cudnn_version
    thermal: ThermalEnvironment             # temperature_c, power_limit_w, fan_speed_pct
    cpu: CPUEnvironment                     # governor, model, platform
    container: ContainerEnvironment         # detected, runtime
    collected_at: datetime
```

### 2.2 AggregatedResult (Multi-Process Aggregation)

```python
class AggregatedResult(BaseModel):
    """Aggregated experiment result from multiple processes."""
    
    schema_version: str
    experiment_id: str
    backend: str
    backend_version: str | None
    aggregation: AggregationMetadata    # method, num_processes, warnings
    
    # Aggregated metrics
    total_tokens: int
    total_energy_j: float
    total_inference_time_sec: float
    avg_tokens_per_second: float
    avg_energy_per_token_j: float
    total_flops: float
    
    # Per-process breakdown (for debugging)
    process_results: list[RawProcessResult]
    
    # Timestamps
    start_time: datetime
    end_time: datetime
    
    # Configuration provenance
    effective_config: dict[str, Any]
    cli_overrides: dict[str, Any]
    config_warnings: list[str]
    parameter_provenance: dict[str, dict[str, Any]]
    preset_chain: list[str]
    
    # Latency statistics (computed at aggregation time)
    latency_stats: LatencyStatistics | None
    
    # Extended metrics (aggregated from per-process)
    extended_metrics: ExtendedEfficiencyMetrics
    
    # Environment & thermal
    environment: EnvironmentMetadata | None
    energy_breakdown: EnergyBreakdown | None
    thermal_throttle: ThermalThrottleInfo | None
    timeseries_path: str | None
```

### 2.3 Extended Efficiency Metrics

```python
class ExtendedEfficiencyMetrics(BaseModel):
    """Extended efficiency metrics - consistent schema, null when not computable."""
    
    # Core efficiency
    tpot_ms: float | None                   # Time Per Output Token (ITL mean, streaming)
    token_efficiency_index: float | None    # Throughput x tokens_per_joule x precision_factor
    
    # Grouped metrics (always present as objects)
    memory: MemoryEfficiencyMetrics         # VRAM usage, tokens/GB
    gpu_utilisation: GPUUtilisationMetrics  # SM %, memory bandwidth %
    batch: BatchEfficiencyMetrics           # Padding overhead, batch utilisation
    kv_cache: KVCacheEfficiencyMetrics      # Prefix cache hit rate (vLLM only)
    request_latency: RequestLatencyMetrics  # E2E latency percentiles
```

### 2.4 Latency Measurement Types

```python
@dataclass
class LatencyMeasurements:
    """Raw latency samples for late aggregation."""
    ttft_ms: list[float]
    itl_full_ms: list[float]
    itl_trimmed_ms: list[float]              # Excluding first/last per request
    request_count: int
    total_output_tokens: int
    excluded_tokens: int
    streaming_mode: bool
    warmup_requests_excluded: int
    measurement_mode: LatencyMeasurementMode  # TRUE_STREAMING | PER_REQUEST_BATCH | PROPORTIONAL

@dataclass
class LatencyStatistics:
    """Computed statistics from LatencyMeasurements."""
    ttft_mean_ms: float
    ttft_median_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_min_ms: float
    ttft_max_ms: float
    ttft_samples: int
    
    itl_mean_ms: float | None              # Trimmed (primary)
    itl_median_ms: float | None
    itl_p95_ms: float | None
    itl_p99_ms: float | None
    itl_samples: int
    
    itl_full_mean_ms: float | None         # Full (for comparison)
    itl_full_p99_ms: float | None
```

---

## 3. Schema Comparison Matrix

| Aspect | Zeus | CodeCarbon | lm-eval-harness | ML.ENERGY Benchmark | LLenergyMeasure |
|--------|------|-----------|-----------------|-------------------|-----------------|
| **Primary Output Type** | `Measurement` dataclass | CSV + API dict | JSON TypedDict | JSON (per-request + leaderboard) | Pydantic BaseModel |
| **Energy Units** | Joules | kWh | N/A | Joules (steady-state) | Joules + kgCO2e |
| **Per-GPU Granularity** | dict[int, float] | Aggregated | N/A | Per-model aggregate | dict[int, float] in breakdown |
| **Energy Attribution** | GPU, CPU, DRAM, SoC | GPU, CPU, RAM (combined) | N/A | GPU only | GPU, CPU, RAM (via CodeCarbon) |
| **Latency Details** | None | None | N/A | TTFT, ITL arrays | TTFT, ITL, trimmed ITL, per-request |
| **FLOPs Tracking** | None | None | N/A | None | Yes, with method + confidence |
| **Throughput** | None (application-level) | None | N/A | Output tokens/sec | Tokens/sec derived |
| **Compute Metrics** | None | None | N/A | None | Memory, utilisation, precision |
| **Model Metadata** | None | None | Full (model_name, source) | Explicit (model_id, params) | ModelInfo (params, layers, type) |
| **Hardware Metadata** | GPU name via NVML | CPU/GPU name | Environment string | GPU model, num_gpus | EnvironmentMetadata (full) |
| **Configuration Provenance** | Implicit (named windows) | Configuration path | Explicit (config dict + git_hash) | Explicit (deployment params) | Explicit (config_name + parameter_provenance) |
| **Backend Tracking** | N/A | N/A | model source | Implicit (vLLM) | Explicit (backend, version) |
| **Version Tracking** | Implicit (ZeusMonitor class) | Implicit (CodeCarbon version) | Explicit (lm_eval_version) | Implicit (v3.0) | Explicit (schema_version, backend_version) |
| **Timestamp Format** | Unix float | ISO datetime | ISO string | ISO (implicit) | ISO (datetime object) |
| **Warmup Handling** | Not tracked | Not tracked | Not tracked | Implicit (steady-state window) | Explicit (WarmupResult) |
| **Distributed Aggregation** | Per-window (implicit) | Not designed for | Not designed for | Not designed for | Explicit (AggregationMetadata) |
| **Throttling Detection** | Not tracked | Not tracked | Not tracked | Not tracked | Explicit (ThermalThrottleInfo) |
| **Multi-Request Breakdown** | Per-window (user-managed) | Not tracked | Per-sample detailed | Per-request array | Per-request latencies + late aggregation |

---

## 4. Schema Design Insights

### 4.1 Provenance Patterns

**Zeus:** Minimal provenance. Relies on caller to track experiment metadata.
- Named windows provide implicit run identification
- No version tracking

**CodeCarbon:** Configuration-path-based provenance.
- Hierarchical config (script > env > files)
- Regional metadata (carbon intensity)
- No backend version tracking

**lm-eval-harness:** Explicit, comprehensive provenance.
- Config + git hash + timestamps
- Task hashes for reproducibility
- Per-sample detailed provenance

**ML.ENERGY:** Deployment-parameter provenance.
- Model ID + GPU + batch size as primary identifier
- Architecture specifications (TP degree, etc.)
- No backend version tracking

**LLenergyMeasure:** Hybrid approach (best of breed).
- `experiment_id` for grouping
- `schema_version` + `backend_version` for reproducibility
- `parameter_provenance` (new in v2) with source tracking
- Preset chain for configuration inheritance

### 4.2 Distributed/Aggregation Patterns

**Zeus:** Per-window measurement. Aggregation left to caller.

**CodeCarbon:** Single tracker per process. No multi-GPU design.

**lm-eval-harness:** Per-sample results. Aggregation via result schema itself.

**ML.ENERGY:** Centrally run (no distributed design). Per-request breakdown for later averaging.

**LLenergyMeasure:** Purpose-built distributed pattern.
- `RawProcessResult` per GPU/process (never inline-aggregated)
- `AggregatedResult` with per-process breakdown
- `AggregationMetadata` with warnings
- Late aggregation of raw samples (correct statistical handling)

### 4.3 Measurement Mode Metadata

**Zeus:** Implicit in API (begin_window/end_window)

**CodeCarbon:** Implicit (tracker.start/stop)

**lm-eval-harness:** N/A (evaluation only, not measurement)

**ML.ENERGY:** Implicit (steady-state window)

**LLenergyMeasure:** Explicit `LatencyMeasurementMode` enum:
- `TRUE_STREAMING` (per-token timestamps via callbacks)
- `PER_REQUEST_BATCH` (total timing, estimated ITL)
- `PROPORTIONAL_ESTIMATE` (fallback)

This makes downstream interpretation explicit and correct.

### 4.4 Null-Handling Strategy

**Zeus:** Separate result types (Measurement, SoCMeasurement)

**CodeCarbon:** Optional fields in dict/dataclass

**lm-eval-harness:** Dynamic fields in result dict

**ML.ENERGY:** Aggregated values (assumes data exists)

**LLenergyMeasure:** Explicit null fields with consistent schema.
- `ExtendedEfficiencyMetrics` always present
- Individual fields null when not computable
- No errors on missing data (graceful degradation)

This is the most user-friendly approach for dynamic configuration handling.

---

## 5. Recommendations for LLenergyMeasure

### 5.1 Adopt from Peer Schemas

| Pattern | Source | Application |
|---------|--------|-------------|
| Named measurement windows | Zeus | Consider for explicit measurement phase tracking |
| Per-token energy attribution | ML.ENERGY | Already done via `energy_per_token_j` |
| Measurement mode metadata | lm-eval-harness | Already done via `LatencyMeasurementMode` |
| Parameter provenance with source | lm-eval-harness | Already done; enhance with source_detail |
| Task/config hashes | lm-eval-harness | Consider adding for reproducibility |
| Prometheus metrics | Zeus | Future integration point |
| Leaderboard data format | ML.ENERGY | Align energy fields for cross-tool comparison |

### 5.2 Fields to Add/Enhance

```python
# Consider adding:
- task_hash: str | None          # Hash of workload config (dataset, prompt, etc.)
- measurement_methodology: str   # "steady-state" | "total" | "windowed"
- steady_state_window: tuple[float, float] | None  # (start_sec, end_sec) exclusions

# Enhance existing:
- parameter_provenance: add git_hash, code_hash for config files
- latency_measurements: add measurement_overhead_ms estimate
```

### 5.3 Export Formats to Support

Currently: JSON (Pydantic `.model_dump_json()`)

Consider:
1. **ML.ENERGY Leaderboard compatible** JSON (for cross-tool comparison)
2. **Zeus CSV** format (for integration with Zeus ecosystem)
3. **Parquet** (for analysis in Pandas/DuckDB)

---

## Summary Table: Key Schema Fields by Tool

| Field | Zeus | CodeCarbon | lm-eval | ML.ENERGY | LLenergyMeasure |
|-------|------|-----------|--------|-----------|-----------------|
| Energy (Joules) | ✓ | ✗ (kWh) | ✗ | ✓ | ✓ |
| Power (Watts) | Implicit | ✓ | ✗ | ✓ | ✓ |
| TTFT/ITL | ✗ | ✗ | ✗ | ✓ (raw) | ✓ (statistics) |
| FLOPs | ✗ | ✗ | ✗ | ✗ | ✓ |
| Precision | ✗ | ✗ | ✗ | ✓ (implicit) | ✓ (explicit) |
| GPU Memory | ✗ | ✗ | ✗ | ✗ | ✓ |
| Thermal Throttle | ✗ | ✗ | ✗ | ✗ | ✓ |
| Warmup Status | ✗ | ✗ | ✗ | ✗ | ✓ |
| Environment | ✗ | Basic | ✓ | Basic | ✓ (comprehensive) |
| Backend Version | ✗ | ✗ | ✓ | Implicit | ✓ |
| Per-Process Raw | ✗ | ✗ | ✗ | ✗ | ✓ |
| Aggregation Meta | ✗ | ✗ | ✗ | ✗ | ✓ |

**Conclusion:** LLenergyMeasure's schema is the most complete and production-ready for inference energy research, with explicit tracking of measurement methodology, distributed aggregation, and comprehensive environment metadata.