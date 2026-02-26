# Research: Config Architecture Patterns in Peer Tools

**Date**: 2026-02-19 (extracted from decisions/config-architecture.md)
**Cited by**: [decisions/config-architecture.md](../decisions/config-architecture.md)
**Question**: How do peer ML benchmarking tools structure backend-specific configuration?
**Confidence**: HIGH (direct source inspection)

---

## Optimum Benchmark (HuggingFace) — INHERITANCE

https://github.com/huggingface/optimum-benchmark

```python
@dataclass
class BenchmarkConfig:
    name: str
    backend: str
    launcher: str

@dataclass
class TorchrunConfig(BenchmarkConfig):
    # torch.distributed specific params
    nproc_per_node: int = 1
    rdzv_backend: str = "c10d"

@dataclass
class PyTorchConfig(BenchmarkConfig):
    # PyTorch-specific params
    torch_dtype: str = "float32"
    bettertransformer: bool = False
```

Config files are backend-specific. No composition. Launcher and model type are determined
by which subclass is instantiated. Flat YAML per backend — no nesting required.

**Assessment for llem**: Closest peer to our use case, but the inheritance approach
propagates a union type through every layer that takes a config. Our cross-backend study
grammar (`sweep:` across multiple backends) requires a single unified type — inheritance
makes this very difficult.

---

## lm-eval-harness (EleutherAI) — FLAT STRING ARGS

```python
# All model args as a flat string (no type safety, no nesting)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b,dtype=float16,parallelize=True
```

No config model at all. Backend-specific params are positional string kwargs.
Works for a single backend but doesn't compose across backends.

**Assessment for llem**: No type safety, no IDE support, no validation. Not suitable
for a tool whose primary output is research data — config errors must be caught at
parse time, not runtime.

---

## W&B Sweeps — FLAT DICT, NAMING CONVENTION

```yaml
parameters:
  learning_rate: {values: [0.001, 0.01]}
  batch_size: {values: [16, 32]}
  model: {values: ["pytorch", "tensorflow"]}
```

All params flat. Backend differences handled by framework code, not config model.
No type safety. Intended for hyperparameter optimization, not multi-backend benchmarking.

**Assessment for llem**: W&B sweeps are for optimisation (find the best), not measurement
(record each). The `sweep:` name was rejected for llem precisely because of this association.

---

## MLflow Projects — NO UNIFIED CONFIG MODEL

Each backend (`mlflow.pytorch`, `mlflow.transformers`) is a separate Python module.
No cross-backend config schema. Not comparable — MLflow is an experiment tracker,
not a benchmarking framework.

---

## Zeus (ML.ENERGY) — NO CONFIG MODEL

`ZeusMonitor` is an energy measurement API, not a config-driven benchmarking framework.
Users instantiate it programmatically. No YAML config. Irrelevant for config structure decision.

---

## Hydra (Meta) — STRUCTURED CONFIGS + COMPOSITION

```yaml
# conf/model/pytorch.yaml
_target_: MyModel
batch_size: 8
attn_implementation: flash_attention_2

# conf/model/vllm.yaml
_target_: VLLMModel
max_num_seqs: 256
tensor_parallel_size: 1

# experiment.yaml — compose them
defaults:
  - model: pytorch
model:
  batch_size: 16  # override
```

Hydra uses composition via config groups. Different model YAML files compose into an
experiment. Each backend gets its own config file, selected by name.

**Assessment for llem**: Closest to our composition approach, but Hydra's mechanism
is composition-by-file-selection (defaults: list) rather than nested sections within
a single file. Requires Hydra as a dependency and a specific project layout. Not a
direct match, but the "composition over inheritance" principle is shared.

---

## Key Finding

| Tool | Structure | Type safety | Multi-backend composition |
|------|-----------|-------------|--------------------------|
| Optimum Benchmark | Inheritance (subclasses) | ✓ (dataclass) | Difficult |
| lm-eval | Flat string args | ✗ | N/A |
| W&B Sweeps | Flat dict | ✗ | N/A (optimization, not benchmarking) |
| MLflow | Per-backend modules | N/A | N/A |
| Hydra | File-based composition | ✓ (structured configs) | Via config groups |
| **llem (chosen)** | **Single type + nested sections** | **✓ (Pydantic)** | **✓ (single sweep: grammar)** |

No peer tool supports a `sweep:` grammar that generates a Cartesian product across
multiple backends from a single file. This is llem's primary differentiator in config
design — the backend namespace (`pytorch:`, `vllm:`) makes it explicit that these params
are not interchangeable, while still allowing universal params (precision, n) to compose.
