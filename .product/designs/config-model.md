# Configuration Model — SSOT

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/study-execution-model.md](../decisions/study-execution-model.md),
                      [../decisions/architecture.md](../decisions/architecture.md),
                      [../decisions/experiment-study-architecture.md](../decisions/experiment-study-architecture.md)
**Status**: Authoritative — all field placement questions resolved here.

This document is the single source of truth for what goes where, what gets hashed, and
how values flow from config files through to stored results. All other docs defer to this.

---

## 1. Overall Model

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  Four Concerns — two config sources, one auto-capture, one measurement protocol ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  USER CONFIG                EXPERIMENT/STUDY YAML     AUTO-DETECTED
  ──────────                 ──────────────────        ─────────────
  ~/.config/llem/            experiment.yaml           at runtime
  config.yaml                study.yaml                ↓
  ↓                          ↓                         environment snapshot
  (machine-local)            (portable, shareable)     (stored with results)

                             ┌─────────────────────┐
                             │   study.yaml         │
                             │                      │
                             │  [experimental       │ ← study_design_hash
                             │   design section]    │   covers this only
                             │  sweep: ...          │
                             │  experiments: ...    │
                             │                      │
                             │  [execution block]   │ ← NOT in hash
                             │  execution:          │   stored as metadata
                             │    n_cycles: 5       │
                             │    cycle_order: ...  │
                             └─────────────────────┘

  User config provides          experiment.yaml
  defaults for gap_seconds  ──→ (one measurement        environment snapshot
  used by the execution         atom)                   stored in result JSON
  block                         measurement_config_hash
                                covers this
```

---

## 2. Data Flow Diagram

How each field travels from definition to stored result:

```
                    ┌──────────────────────────────────────────────────────┐
                    │              STORED RESULT (JSON)                     │
                    │                                                       │
  user config ─────►  measurement_protocol.config_gap_seconds_used         │
  (gap defaults)       measurement_protocol.cycle_gap_seconds_used         │
                    │                                                       │
  study.yaml ──────►  measurement_protocol.n_cycles                        │
  execution:           measurement_protocol.cycle_order                    │
                    │                                                       │
  study.yaml ──────►  study_design_hash  (SHA-256[:16] of sweep+expts)     │
  sweep / expts        (execution block EXCLUDED from this hash)            │
                    │                                                       │
  experiment.yaml ─►  measurement_config_hash  (SHA-256[:16] of all        │
  (or inline)          ExperimentConfig fields)                             │
                    │                                                       │
  user config / ───►  environment.carbon_intensity_gco2_kwh                │
  env var              environment.datacenter_pue                           │
                    │                                                       │
  auto-detected ───►  environment.gpu_names, gpu_vram_gb, gpu_count        │
  at runtime           environment.cuda_version, driver_version, cpu_model │
                    │                                                       │
                    └──────────────────────────────────────────────────────┘
```

---

## 3. Field Tables

### 3.1 User Config — Machine/Infrastructure

`~/.config/llenergymeasure/config.yaml` — machine-local, never versioned.

```yaml
# Full user config schema
runners:
  pytorch: local                         # or docker:<image>
  vllm: local
  tensorrt: local

output:
  results_dir: ./results
  model_cache_dir: ~/.cache/huggingface  # or $HF_HOME

measurement:
  energy_backend: nvml                   # nvml | zeus | codecarbon
  carbon_intensity_gco2_kwh: 350         # gCO₂/kWh — or null (lookup by location)
  datacenter_pue: 1.0                    # power usage effectiveness
  datacenter_location: DE               # ISO-3166-1 code for carbon lookup

execution:
  config_gap_seconds: 60                 # thermal gap between experiment configs
  cycle_gap_seconds: 300                 # thermal gap between full cycles

ui:
  verbosity: normal                      # quiet | normal | verbose
  prompt: true                           # interactive backend selection
```

| Field | Type | Default | Env var | Notes |
|-------|------|---------|---------|-------|
| `runners.<backend>` | `str` | `local` | `LLEM_RUNNER_<BACKEND>` | `local` or `docker:<image>` |
| `output.results_dir` | `Path` | `./results` | — | |
| `output.model_cache_dir` | `Path` | HF default | `HF_HOME` | |
| `measurement.energy_backend` | `str` | `nvml` | — | `nvml \| zeus \| codecarbon` |
| `measurement.carbon_intensity_gco2_kwh` | `float \| None` | `None` | `LLEM_CARBON_INTENSITY` | Looks up by location if None |
| `measurement.datacenter_pue` | `float` | `1.0` | `LLEM_DATACENTER_PUE` | |
| `measurement.datacenter_location` | `str \| None` | `None` | `LLEM_DATACENTER_LOCATION` | ISO-3166-1 code |
| `execution.config_gap_seconds` | `float` | `60` | — | Can be overridden in study.yaml |
| `execution.cycle_gap_seconds` | `float` | `300` | — | Can be overridden in study.yaml |
| `ui.verbosity` | `str` | `normal` | — | |
| `ui.prompt` | `bool` | `true` | — | Set false for CI |

### 3.2 Measurement Protocol (in study.yaml, excluded from hash)

```yaml
# study.yaml execution block
execution:
  n_cycles: 5                     # how many times to repeat full experiment set
  cycle_order: shuffled            # sequential | interleaved | shuffled
  config_gap_seconds: 120          # optional: override user config machine default
  cycle_gap_seconds: 600           # optional: override user config machine default
```

| Field | Pydantic default | CLI effective default | In `study_design_hash`? | Notes |
|-------|-----------------|----------------------|------------------------|-------|
| `n_cycles` | `1` | `3` | **No** | Explicit study file value always wins |
| `cycle_order` | `"sequential"` | `"interleaved"` | **No** | |
| `config_gap_seconds` | from user config | from user config | **No** | Override of user config default |
| `cycle_gap_seconds` | from user config | from user config | **No** | Override of user config default |

**CLI shortcuts** (expand to flag combinations, not a config layer):

| Flag | Expands to |
|------|-----------|
| `--profile quick` | `--cycles 1 --no-gaps` |
| `--profile publication` | `--cycles 5 --order shuffled` |

### 3.3 ExperimentConfig — Single Measurement Atom

`experiment.yaml` or inline within study.yaml. `measurement_config_hash` covers all of these.

```yaml
# experiment.yaml
model: meta-llama/Llama-3.1-8B
backend: pytorch
precision: bf16
dataset: aienergyscore
n: 100
random_seed: 42

decoder:
  decoding_strategy: greedy
  temperature: 1.0

warmup:
  n_warmup: 5
  thermal_floor_seconds: 30.0

baseline:
  enabled: true
  duration_seconds: 30.0

pytorch:
  batch_size: 8
  attn_implementation: flash_attention_2

lora:                               # optional
  adapter_id: owner/my-adapter
  merge_weights: false

extra:                              # escape hatch
  torch_compile_mode: max-autotune
```

| Field | Type | Default | In `measurement_config_hash`? |
|-------|------|---------|------------------------------|
| `model` | `str` | required | ✓ |
| `backend` | `Literal[...]` | required | ✓ |
| `precision` | `Literal[...]` | `"bf16"` | ✓ |
| `dataset` | `str \| SyntheticDatasetConfig` | `"aienergyscore"` | ✓ |
| `n` | `int` | `100` | ✓ |
| `random_seed` | `int` | `42` | ✓ |
| `decoder.*` | `DecoderConfig` | defaults | ✓ |
| `warmup.*` | `WarmupConfig` | defaults | ✓ |
| `baseline.*` | `BaselineConfig` | defaults | ✓ |
| `pytorch.*` | `PyTorchConfig \| None` | `None` | ✓ |
| `vllm.*` | `VLLMConfig \| None` | `None` | ✓ |
| `tensorrt.*` | `TensorRTConfig \| None` | `None` | ✓ |
| `lora.*` | `LoRAConfig \| None` | `None` | ✓ |
| `extra` | `dict \| None` | `None` | ✓ |
| infrastructure fields | — | — | **N/A** — not in ExperimentConfig |

### 3.4 StudyConfig — Parameter Space

`study.yaml`. `study_design_hash` covers experimental design only (not execution block).

```yaml
# batch-size-effects.yaml
model: meta-llama/Llama-3.1-8B
backend: [pytorch, vllm]
precision: bf16

sweep:
  pytorch.batch_size: [1, 4, 8, 16]   # backend-scoped grid
  vllm.max_num_seqs: [64, 256]

# OR explicit list:
experiments:
  - backend: pytorch
    pytorch: {batch_size: 1}
  - backend: pytorch
    pytorch: {batch_size: 8}

execution:
  n_cycles: 5
  cycle_order: shuffled
```

| Field | In `study_design_hash`? | Notes |
|-------|------------------------|-------|
| `sweep:` (dotted grid) | **Yes** | Defines parameter space |
| `experiments:` (list) | **Yes** | Explicit experiment definitions |
| `execution:` block | **No** | Measurement protocol — stored as metadata |

### 3.5 Environment Snapshot — Infrastructure Context (auto-captured)

Written to result JSON at measurement time. Never in any config file. Immutable once stored.

| Field | Source | Type |
|-------|--------|------|
| `gpu_names` | `nvidia-smi` | `list[str]` |
| `gpu_vram_gb` | `nvidia-smi` | `list[float]` |
| `gpu_count` | `nvidia-smi` | `int` |
| `cuda_version` | `torch.version.cuda` → `/usr/local/cuda/version.txt` → `nvcc` → `None` | `str \| None` |
| `driver_version` | `nvidia-smi` | `str \| None` |
| `cpu_model` | `/proc/cpuinfo` | `str` |
| `carbon_intensity_gco2_kwh` | from user config / env var (at measurement time) | `float \| None` |
| `datacenter_pue` | from user config / env var (at measurement time) | `float` |
| `datacenter_location` | from user config / env var (at measurement time) | `str \| None` |

---

## 4. Precedence Chains

### 4.1 Environment snapshot fields

```
Lowest                                                              Highest
  │                                                                    │
  ▼                                                                    ▼
user config                                               env var override
~/.config/llem/config.yaml                               LLEM_CARBON_INTENSITY
measurement.carbon_intensity_gco2_kwh      ──────────►   LLEM_DATACENTER_PUE
measurement.datacenter_pue                               LLEM_DATACENTER_LOCATION
measurement.datacenter_location
                                                              │
                                                              ▼
                                                     Stored in environment
                                                     snapshot in result JSON
                                                     (immutable after)
```

### 4.2 Measurement protocol fields

```
Lowest                                                              Highest
  │                                                                    │
  ▼                                                                    ▼
Pydantic defaults        user config          study.yaml          CLI flag
n_cycles=1               execution.           execution:          --cycles N
cycle_order=sequential   config_gap_seconds   n_cycles: 5         --no-gaps
                         cycle_gap_seconds    cycle_order: ...    --order X
                                              config_gap_seconds  --profile quick
                                              cycle_gap_seconds   --profile publication
```

Gap seconds note: Pydantic defaults → user config → study.yaml override → CLI flag.
n_cycles/cycle_order: Pydantic default → CLI effective default (3/interleaved) → study.yaml → CLI flag.

### 4.3 ExperimentConfig fields

```
Lowest                              Highest
  │                                    │
  ▼                                    ▼
Pydantic defaults    experiment.yaml   CLI flags
model=required       or inline in      --model, --backend
n=100                study.yaml        --precision, --batch-size
precision="bf16"     definitions       etc.
```

---

## 5. Hash Semantics

### 5.1 `measurement_config_hash` (per experiment)

```python
def compute_measurement_config_hash(config: ExperimentConfig) -> str:
    """SHA-256[:16] of ExperimentConfig. Answers: 'Did you run the same experiment?'
    Environment snapshot is not in ExperimentConfig — no exclusion needed.
    """
    return hashlib.sha256(
        json.dumps(config.model_dump(), sort_keys=True).encode()
    ).hexdigest()[:16]
```

Same model + backend + params → same hash. Same hash → results are directly comparable.

### 5.2 `study_design_hash` (per study)

```python
def compute_study_design_hash(config: StudyConfig) -> str:
    """SHA-256[:16] of experimental design only. Answers: 'Did you test the same space?'
    Execution block excluded — same study, more cycles = same hash.
    """
    return hashlib.sha256(
        json.dumps(config.model_dump(exclude={"execution"}), sort_keys=True).encode()
    ).hexdigest()[:16]
```

Same sweep grammar + same experiment list → same hash, regardless of n_cycles.
This enables "topping up" a study (3 cycles → 5 cycles) without creating a new study identity.

---

## 6. Result JSON Structure (summary)

```json
{
  "schema_version": "2.0",
  "study_design_hash": "a3f2b8c1...",      // hash of sweep + experiments only
  "measurement_protocol": {                 // NOT hashed — stored as metadata
    "n_cycles": 5,
    "cycle_order": "shuffled",
    "config_gap_seconds_used": 120,
    "cycle_gap_seconds_used": 600
  },
  "experiments": [
    {
      "measurement_config_hash": "d7e4...", // hash of ExperimentConfig
      "config": { /* ExperimentConfig fields */ },
      "cycles": [
        { /* ExperimentResult for cycle 1 */ },
        { /* ExperimentResult for cycle 2 */ }
      ],
      "aggregated": { /* mean/std across cycles */ }
    }
  ],
  "environment": {                          // environment snapshot — auto-captured
    "gpu_names": ["NVIDIA A100-SXM4-80GB"],
    "gpu_vram_gb": [80.0],
    "gpu_count": 1,
    "cuda_version": "12.4",
    "driver_version": "535.86",
    "cpu_model": "Intel Xeon Gold 6326",
    "carbon_intensity_gco2_kwh": 350.0,    // from user config at measurement time
    "datacenter_pue": 1.2,
    "datacenter_location": "DE"
  }
}
```

---

## Related

- [../decisions/study-execution-model.md](../decisions/study-execution-model.md): Decisions A, B, C (why)
- [../decisions/architecture.md](../decisions/architecture.md): Two sources + auto-capture model overview
- [experiment-config.md](experiment-config.md): Full ExperimentConfig schema
- [study-yaml.md](study-yaml.md): Full StudyConfig schema
- [user-config.md](user-config.md): Full user config schema
- [result-schema.md](result-schema.md): Full result JSON schema
