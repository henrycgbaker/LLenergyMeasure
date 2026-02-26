# ExperimentConfig Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/config-architecture.md](../decisions/config-architecture.md)
**Status**: Decided — Composition. Sweep grammar decided (dotted notation, Option A).

---

## Decision: Composition (Single Type with Optional Backend Sections)

`ExperimentConfig` is a single Pydantic model with optional nested backend sections.
The library API takes one type regardless of backend: `run_experiment(config: ExperimentConfig)`.

Inheritance was rejected: it propagates a union type (`PyTorchConfig | VLLMConfig | TensorRTConfig`)
through every layer — `isinstance()` checks everywhere, union types in `StudyConfig`. The sweep
grammar requires one config type that can represent any backend for Cartesian product generation.

---

## Full Schema

```python
# src/llenergymeasure/config/models.py

from typing import Any, Literal
from pydantic import BaseModel, model_validator, Field


class DecoderConfig(BaseModel):
    model_config = {"extra": "forbid"}

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    decoding_strategy: Literal["greedy", "sampling", "beam"] = "greedy"
    num_beams: int | None = None   # beam search only
    speculative_decoding: bool = False
    draft_model: str | None = None

    @model_validator(mode="after")
    def validate_decoder(self) -> "DecoderConfig":
        if self.num_beams is not None and self.decoding_strategy != "beam":
            raise ValueError("num_beams requires decoding_strategy='beam'")
        if self.speculative_decoding and not self.draft_model:
            raise ValueError("speculative_decoding=True requires draft_model to be set")
        return self


class WarmupConfig(BaseModel):
    model_config = {"extra": "forbid"}

    n_warmup: int = 3                    # fixed run count at full sequence length
    thermal_floor_seconds: float = 60.0  # min wait (60s MLPerf-aligned; configurable down to 30)
    # Full-length warmup warms KV cache + decode path + thermal state
    # See decisions/warmup-strategy.md for rationale


class BaselineConfig(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = True
    duration_seconds: float = 30.0   # idle baseline measurement before model load


class SyntheticDatasetConfig(BaseModel):
    model_config = {"extra": "forbid"}

    n: int
    input_len: int = 512     # approximate tokens (padded/truncated to target)
    output_len: int = 128    # sets max_new_tokens
    seed: int = 42


# ─── LoRA / adapter config (optional — v2.0) ────────────────────────────────

class LoRAConfig(BaseModel):
    model_config = {"extra": "forbid"}

    adapter_id: str | None = None         # HuggingFace Hub ID (e.g. "peft-internal-testing/llama-lora")
    adapter_path: str | None = None       # local path (alternative to adapter_id)
    merge_weights: bool = False           # False = unmerged (measures adapter overhead)
                                          # True  = merged (measures deployed cost)

    @model_validator(mode="after")
    def validate_adapter_source(self) -> "LoRAConfig":
        if self.adapter_id is None and self.adapter_path is None:
            raise ValueError("LoRAConfig requires either adapter_id or adapter_path")
        if self.adapter_id is not None and self.adapter_path is not None:
            raise ValueError("Specify adapter_id OR adapter_path, not both")
        return self


# ─── Backend-specific config sections ───────────────────────────────────────
# All fields None by default — None means "use backend's own default".
# Backend implementation resolves None → its effective default at execution time.

class PyTorchConfig(BaseModel):
    model_config = {"extra": "forbid"}

    batch_size: int | None = None             # None → backend default (1)
    batching_strategy: Literal["static", "dynamic"] | None = None   # None → static
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = None
    torch_compile: bool | None = None         # None → False
    load_in_4bit: bool | None = None          # BitsAndBytes quantization
    load_in_8bit: bool | None = None
    num_processes: int | None = None          # data parallel processes


class VLLMConfig(BaseModel):
    model_config = {"extra": "forbid"}

    max_num_seqs: int | None = None           # None → vLLM default (256)
    tensor_parallel_size: int | None = None   # None → 1
    gpu_memory_utilization: float | None = None   # None → 0.9
    enable_prefix_caching: bool | None = None
    block_size: int | None = None
    quantization: Literal["awq", "gptq", "fp8"] | None = None
    # TODO: Complete vLLM parameter list — see PARAM-04 backend parameter completeness audit (v2.3)


class TensorRTConfig(BaseModel):
    model_config = {"extra": "forbid"}

    max_batch_size: int | None = None         # compile-time constant (not runtime)
    tp_size: int | None = None                # tensor parallel size
    pp_size: int | None = None                # pipeline parallel size
    max_seq_len: int | None = None            # max sequence length (affects engine)
    kv_cache_type: Literal["paged"] | None = None
    quantization: Literal["int8_sq", "int4_awq", "fp8"] | None = None
    builder_opt_level: int | None = None      # 0-5; higher = slower build, faster inference
    engine_path: str | None = None            # pre-compiled engine path (skip compilation)
    # TODO: Complete TRT parameter list — see PARAM-04 backend parameter completeness audit (v2.3)


# ─── SSOT: backend compatibility constraints ─────────────────────────────────
# Single source of truth for what IS valid per backend.
# Anything absent is implicitly invalid. Update here when backends release new versions.
# See: src/llenergymeasure/config/ssot.py

PRECISION_SUPPORT: dict[str, list[str]] = {
    "pytorch":  ["fp32", "fp16", "bf16"],
    "vllm":     ["fp16", "bf16", "int8", "int4"],
    "tensorrt": ["fp16", "int8", "fp8", "int4"],   # fp32/bf16 not supported
}

DECODING_SUPPORT: dict[str, list[str]] = {
    "pytorch":  ["greedy", "sampling", "beam"],
    "vllm":     ["greedy", "sampling", "beam"],
    "tensorrt": ["greedy", "sampling"],              # no beam search
}


# ─── Main ExperimentConfig ───────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    model_config = {"extra": "forbid"}   # reject unknown YAML keys — typos fail loudly

    # ─── Shared params (universal across all backends) ───────────────────────
    model: str                                       # HuggingFace model ID
    backend: Literal["pytorch", "vllm", "tensorrt"]
    precision: Literal["fp32", "fp16", "bf16", "int8", "fp8", "int4"] = "bf16"
    dataset: str | SyntheticDatasetConfig = "aienergyscore"
    n: int = 100                                     # number of prompts from dataset
    random_seed: int = 42

    # ─── Sub-configs ─────────────────────────────────────────────────────────
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)

    # ─── Backend sections (optional — None = use backend defaults) ───────────
    pytorch: PyTorchConfig | None = None
    vllm: VLLMConfig | None = None
    tensorrt: TensorRTConfig | None = None

    # ─── LoRA / adapter (optional — v2.0) ────────────────────────────────────
    lora: LoRAConfig | None = None        # None = base model only

    # ─── Escape hatch (declared field — compatible with extra="forbid") ──────
    passthrough_kwargs: dict[str, Any] | None = None   # pass arbitrary kwargs to backend runner

    # ─── Validators ──────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def validate_backend_compatibility(self) -> "ExperimentConfig":
        # Precision constraint
        if (valid := PRECISION_SUPPORT.get(self.backend)) and self.precision not in valid:
            raise ValueError(
                f"{self.backend} does not support precision={self.precision!r}. "
                f"Valid: {valid}"
            )
        # Decoder constraint
        if (valid := DECODING_SUPPORT.get(self.backend)) and \
                self.decoder.decoding_strategy not in valid:
            raise ValueError(
                f"{self.backend} does not support decoding_strategy="
                f"{self.decoder.decoding_strategy!r}. Valid: {valid}"
            )
        # Backend section vs backend field consistency
        if self.vllm is not None and self.backend != "vllm":
            raise ValueError(
                f"vllm: config section provided but backend={self.backend!r}. "
                "Remove the vllm: section or set backend: vllm."
            )
        if self.pytorch is not None and self.backend != "pytorch":
            raise ValueError(
                f"pytorch: config section provided but backend={self.backend!r}."
            )
        if self.tensorrt is not None and self.backend != "tensorrt":
            raise ValueError(
                f"tensorrt: config section provided but backend={self.backend!r}."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
```

---

## YAML Examples

### Single-backend experiment

```yaml
# experiment.yaml
model: meta-llama/Llama-3.1-8B
backend: pytorch
precision: bf16
n: 100

pytorch:
  batch_size: 8
  attn_implementation: flash_attention_2
```

### Environment snapshot (auto-captured)

Environment snapshot fields (`carbon_intensity`, `datacenter_pue`) are **not in
`ExperimentConfig`**. They are machine-level facts, not experiment parameters. `experiment.yaml`
must remain portable and infrastructure-agnostic so it can be shared and reproduced on any machine.

Set defaults in `~/.config/llenergymeasure/config.yaml` (user-local, never versioned):

```yaml
measurement:
  carbon_intensity_gco2_kwh: 350   # Germany national grid
  datacenter_pue: 1.2              # university cluster
  datacenter_location: DE          # ISO-3166-1 code
```

For one-off overrides (e.g. running on a cloud GPU in a different region) use env vars — they
take effect without touching the experiment file or user config:

```bash
LLEM_CARBON_INTENSITY=386 llem run experiment.yaml
LLEM_DATACENTER_PUE=1.5 llem run experiment.yaml
```

### Escape hatch for niche/undocumented params

```yaml
model: meta-llama/Llama-3.1-8B
backend: pytorch
precision: bf16

passthrough_kwargs:
  torch_compile_mode: max-autotune          # undocumented niche param passed through to backend
  use_bettertransformer: true
```

> **Renamed (2026-02-25):** `extra:` → `passthrough_kwargs:`. The old name `extra` clashed
> with Pydantic's `extra = "forbid"` model config, causing UX confusion ("why can I set
> `extra:` when the model forbids extras?"). The new name makes the pass-through intent explicit.

`passthrough_kwargs:` is a declared Pydantic field (not a wildcard). `extra = "forbid"` prevents
`bachsize: 8` (a typo) from silently being treated as an extra — it would raise `ValidationError`.
Only the explicitly declared `passthrough_kwargs:` field bypasses this.

---

## Sweep Grammar (Confirmed: Dotted Notation — Option A)

**Decided 2026-02-19.** See [decisions/config-architecture.md](../decisions/config-architecture.md) § "Sweep Grammar Decision" for full rationale, algorithm, and edge-case table.

```yaml
# ✓ Universal param sweep — applies to all backends
backend: pytorch
sweep:
  precision: [fp16, bf16]

# ✓ Backend-scoped sweep — dotted notation adds dimension to that backend's grid only
backend: [pytorch, vllm]
sweep:
  precision: [fp16, bf16]          # universal — both backends
  pytorch.batch_size: [1, 8, 16]  # scoped — 3 dimensions for pytorch only
  vllm.max_num_seqs: [64, 256]    # scoped — 2 dimensions for vllm only
# → pytorch: precision × batch_size = 2×3 = 6 experiments
# → vllm:    precision × max_num_seqs = 2×2 = 4 experiments
# → total: 10 experiments (independent grids per backend)
```

`experiments:` list for fully explicit, non-Cartesian combinations:

```yaml
experiments:
  - backend: pytorch
    pytorch: {batch_size: 1}
  - backend: vllm
    vllm: {max_num_seqs: 64}
```

Both modes compose: `sweep:` generates a grid; `experiments:` adds explicit entries on top.
Full grid generation algorithm: [designs/study-yaml.md](study-yaml.md) § "Mode A — Grid Sweep".

---

## Renamed Fields (v2.0 breaking changes)

| Old name | New name | Reason |
|---|---|---|
| `model_name` | `model` | shorter, aligns with HF Hub convention |
| `fp_precision` | `precision` | consistent with peer tools |
| `num_input_prompts` | `n` | shorter, aligns with lm-eval `--num_fewshot` brevity |

No aliases, no shims. v2.0 is a clean break.

---

## Removed Fields

These were removed from `ExperimentConfig` in v2.0:

| Field | Reason for removal |
|---|---|
| `schedule` | OS/cron concern; no peer has this in experiment config |
| `io` | No clear purpose post-refactor |
| `query_rate` | Load testing concern, not efficiency measurement |
| `streaming` | Backend-specific concern, not a universal config field |
| `streaming_warmup_requests` | ditto |
| `save_outputs` | Library is side-effect-free; CLI handles persistence |
| `decode_token_to_text` | Unnecessary with modern tokenizers |

---

## LoRA / Adapter Support (v2.0)

Add an optional `lora:` block to `ExperimentConfig`. Decision rationale:
[decisions/adapter-support.md](../decisions/adapter-support.md).

```yaml
# Unmerged — measures adapter inference overhead (~10–12% throughput penalty)
model: meta-llama/Llama-3.1-8B
backend: pytorch
lora:
  adapter_id: peft-internal-testing/llama-lora   # HF Hub ID
  merge_weights: false                            # default

# Merged — measures deployed cost (single fused tensor, zero overhead)
lora:
  adapter_path: ./my-finetuned-adapter
  merge_weights: true
```

**Backend compatibility:**

| Backend | Support | Notes |
|---------|---------|-------|
| PyTorch | ✓ | `PeftModel.from_pretrained()` |
| vLLM | ✓ | `LoRARequest` API; requires `vllm.enable_lora: true` in VLLMConfig |
| TensorRT-LLM | ✗ | Must pre-merge before TRT engine compilation |

**TRT pre-merge workflow** (for users who want to benchmark a fine-tuned model on TRT):

```python
from peft import PeftModel
merged = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()
merged.save_pretrained("./merged-model")
```

Then use `model: ./merged-model` in the experiment config with no `lora:` block.
If `lora:` is specified with `backend: tensorrt`, a `ValidationError` is raised at parse time.

**FLOPs:** Adapter matrices (rank 8–64) are <0.1% of base model FLOPs. `N_params` in the
estimation formula uses base model params unchanged. See [decisions/flops-estimation.md](../decisions/flops-estimation.md).

**Config hash:** `measurement_config_hash` includes `lora.adapter_id`/`lora.adapter_path`
and `lora.merge_weights` — two runs with the same base model but different adapters get
different hashes.

---

## None-as-Sentinel Pattern (Backend Defaults)

Backend section fields use `None` as "not specified — use backend default":

```python
class PyTorchConfig(BaseModel):
    batch_size: int | None = None
    # None = backend decides (effective default: 1)
    # Docstring documents effective default for discoverability
```

**Why not embed defaults in the Pydantic model**:
- `config_hash` in ExperimentResult should reflect only explicit user choices, not baked-in defaults
- Backend implementation version-controls its own defaults (not stale values in config model)
- Users can see exactly what they explicitly set vs what the backend will decide

The backend runner resolves `None` → its own default at execution time.

---

## `config_hash` Computation

Used in `ExperimentResult` for reproducibility checking.

```python
import hashlib, json

def compute_config_hash(config: ExperimentConfig) -> str:
    """16-char SHA-256 of the fully-resolved config (measurement identity only).

    Environment snapshot (carbon_intensity, datacenter_pue) is not in ExperimentConfig —
    it lives in user config / env vars and is stored in the environment snapshot alongside
    results. The hash therefore reflects only the measurement config.
    """
    config_dict = config.model_dump()
    canonical = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

---

## Related

- [../decisions/config-architecture.md](../decisions/config-architecture.md): Decision rationale
- [study-yaml.md](study-yaml.md): StudyConfig (sweep grammar, execution block)
- [architecture.md](architecture.md): Module placement (`config/models.py`)
- [result-schema.md](result-schema.md): `config_hash` in ExperimentResult
