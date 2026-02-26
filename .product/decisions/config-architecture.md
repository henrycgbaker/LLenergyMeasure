# Config Architecture: Composition vs Inheritance

**Status:** Accepted
**Date decided:** 2026-02-18
**Last updated:** 2026-02-26
**Research:** [../research/15-config-architecture-patterns.md](../research/15-config-architecture-patterns.md)

## Decision

| Sub-decision | Resolution |
|---|---|
| **C1. Composition vs inheritance** | Composition — single `ExperimentConfig` with optional backend sections (`pytorch:`, `vllm:`, `tensorrt:`). |
| **C2. Sweep grammar** | Dotted notation: `pytorch.batch_size: [1, 8, 32]`. Per-backend Cartesian product. |

Additional: `extra = "forbid"` on all configs (typos fail loudly). Backend fields use `None`-as-sentinel (backend owns defaults). Explicit `extra: dict` field as escape hatch for unlisted params.

---

## Context

LLenergyMeasure supports three inference backends (PyTorch, vLLM, TensorRT-LLM). Each backend has a substantial set of unique parameters (batching strategy, quantisation scheme, parallelism API, KV cache settings) that have different names, different semantics, and different valid values. A meaningful but limited set of parameters is truly universal (model, dataset, n, precision, decoder config, warmup, gpus).

The core question: how should backend-specific parameters relate to the shared experiment configuration? This decision determines the type signature of `run_experiment()`/`run_study()`, the structure of study YAML files, and the sweep grammar used to generate experiment grids.

Two structural decisions were required:

- **C1**: Composition vs inheritance for `ExperimentConfig`
- **C2**: Sweep grammar for backend-specific parameters

### The "Little Commonality" Problem

The user's insight: "between backends there's actually little commonality (shared parameters), so these YAMLs aren't really interchangeable when you dig into it."

What IS genuinely shared across all three backends: `model`, `dataset`, `n`, `precision`,
`decoder: {temperature, top_p, top_k}`, `warmup:`, `baseline:`, `timeseries:`, `gpus:`.

What is NOT shared (unique per backend):

| Parameter | PyTorch | vLLM | TensorRT |
|-----------|---------|------|----------|
| Batching | `batch_size`, `batching_strategy` | `max_num_seqs` (continuous) | `max_batch_size` (compile-time) |
| Quantization | `load_in_4bit`, `load_in_8bit` (BnB) | `quantization: awq\|gptq\|fp8` | `quantization: int8_sq\|int4_awq` |
| Parallelism | `num_processes` (data parallel) | `tensor_parallel_size` (tensor) | `tp_size` + `pp_size` |
| KV cache | `use_cache`, `cache_implementation` | `enable_prefix_caching`, `block_size` | `kv_cache_type: paged` |
| Compilation | `torch_compile` | (handled internally) | `engine_path`, `builder_opt_level` |
| Speculative | `assisted_generation` (HF pattern) | `speculative:` (vLLM pattern) | `draft_model` + `num_draft_tokens` |

A `batch_size=8` for PyTorch is static batching; vLLM's equivalent is `max_num_seqs=256`
continuous batching — completely different concepts, not interchangeable sweep dimensions.

---

## C1 — Composition vs Inheritance

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Composition — single `ExperimentConfig` with optional backend sections (chosen)** | `run_experiment(config)` takes one type regardless of backend; sweep grammar works with one model; library API alignment; Pydantic discriminated union familiar in Python ecosystem; `sweep: {precision: [fp16, bf16]}` works across backends | Backend-specific params still present in the model even when irrelevant backend; nesting (`pytorch: {batch_size: 8}`) vs flat |
| Inheritance — `PyTorchExperimentConfig`, `VLLMExperimentConfig`, `TensorRTExperimentConfig` | Type safety: `PyTorchExperimentConfig` cannot have `max_num_seqs`; flat YAML (no nesting); IDE autocomplete knows valid params; Optimum Benchmark precedent | `run_experiment(config: Py\|VL\|TRT)` — union type pollutes every layer; `StudyConfig.experiments` becomes `list[Py \| VL \| TRT]`; sweep grid generator cannot produce a single type; `isinstance()` checks everywhere |

**Rejected (2026-02-18):** Inheritance — `run_experiment(config)` would need`config: PyTorchExperimentConfig | VLLMExperimentConfig | TensorRTExperimentConfig`. `StudyConfig.experiments` becomes a union list. The sweep grid generator cannot produce a single type — it must produce different types per backend. Every internal function needs `isinstance()` checks. The ergonomic benefit (flat YAML per backend) does not outweigh this pervasive type pollution.

```python
# Rejected inheritance shape — for reference:
class BaseExperimentConfig(BaseModel):
    model: str; dataset: str; n: int
    precision: Literal["fp16", "bfloat16"]
    decoder: DecoderConfig; warmup: WarmupConfig

class PyTorchExperimentConfig(BaseExperimentConfig):
    backend: Literal["pytorch"] = "pytorch"
    batch_size: int = 1
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"

class VLLMExperimentConfig(BaseExperimentConfig):
    backend: Literal["vllm"] = "vllm"
    max_num_seqs: int = 256
    tensor_parallel_size: int = 1
```

### Decision

We will use composition. A single `ExperimentConfig` type with optional backend sections (`pytorch: PyTorchConfig | None`, `vllm: VLLMConfig | None`, `tensorrt: TensorRTConfig | None`).

Rationale: The library API and cross-backend study design are the decisive factors. A single config type flows cleanly from YAML → `run_experiment(config)` → orchestrator → results. Inheritance propagates a union type through every layer.

The flat-YAML ergonomic benefit of inheritance does not outweigh this for a tool where complex
configs are primarily written for cross-backend studies — where the backend namespace (`pytorch:`,
`vllm:`) is useful signal that these params are not interchangeable concepts.

### Consequences

Positive: Clean single-type API; sweep grammar works uniformly; no union-type complexity.
Negative / Trade-offs: YAML nesting for backend params (`pytorch: {batch_size: 8}`); a config
with `backend: pytorch` carries unused `vllm:` and `tensorrt:` sections (always `None`).
Neutral / Follow-up decisions triggered: Sweep grammar for backend-scoped params (C2 below).

---

## C2 — Sweep Grammar for Backend-Specific Parameters

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Option A — dotted notation: `pytorch.batch_size: [1, 8, 32]` (chosen)** | Compact; single sweep block; backend-scoped keys clearly signal they apply only to that backend; composes with universal keys | Requires split-on-first-dot parsing logic; nested dots (`pytorch.attn.implementation`) need a convention |
| Option B — backend lock: `sweep:` only valid for single-backend; multi-backend requires `experiments:` | Explicit; no ambiguity | Forces verbose `experiments:` list for any multi-backend study; defeats the grid syntax for the primary use case |
| Option C — silent ignore: skip params silently for backends that lack them | Simple | Silent data loss; impossible to debug why a backend's grid is smaller than expected |

**Rejected (2026-02-19):** Option B — forces verbose `experiments:` list for multi-backend
studies, which is the primary use case for sweep grammar.

**Rejected (2026-02-19):** Option C — silent ignore makes it impossible to diagnose why a
backend's experiment grid is smaller than expected.

### Decision

We will use Option A: dotted notation in `sweep:`. Backend-scoped keys use
`{backend}.{param}: [values]`. Universal keys have no prefix.

**Rule:** For each backend, take the Cartesian product of `(universal keys) × (that backend's
scoped keys)`. Backends have independent grids. A backend-scoped key adds a dimension only to
its own backend's slice of the grid — not to other backends'.

Grid generation algorithm:

```python
def generate_grid(config: StudyConfig) -> list[ExperimentConfig]:
    universal = {k: v for k, v in config.sweep.items() if "." not in k}
    scoped: dict[str, dict] = {}
    for key, values in config.sweep.items():
        if "." in key:
            backend, param = key.split(".", 1)  # split on FIRST dot only
            scoped.setdefault(backend, {})[param] = values

    backends = config.backend if isinstance(config.backend, list) else [config.backend]

    experiments = []
    for backend in backends:
        # Each backend gets: universal keys + its OWN scoped keys only
        dims = {**universal, **scoped.get(backend, {})}
        for combo in itertools.product(*dims.values()):
            params = dict(zip(dims.keys(), combo))
            experiments.append(ExperimentConfig(backend=backend, **params))

    return experiments
```

**All edge cases — resolved:**

| Scenario | Behaviour |
|----------|-----------|
| Universal key only, multi-backend | Each backend gets `universal × [...]` — same-shaped grid for all |
| Scoped key for backend A, no scoped key for backend B | Backend B gets universal-only grid |
| Scoped key references backend not in `backend:` list | `ValidationError` at parse time: `"sweep key 'vllm.max_num_seqs' references backend 'vllm' which is not in the experiment's backend list"` |
| Universal key that matches a backend-specific param name | Passes parse; per-experiment `ValidationError` at L1 Pydantic (ExperimentConfig rejects unknown fields) — with hint: `"Did you mean 'vllm.max_num_seqs'?"` |
| Scoped key with no `backend:` field set at top level | Backend inferred from scoped keys; error if multiple backends implied and no explicit list |
| Nested dots: `pytorch.attn.implementation` | Split on FIRST dot only → backend=`pytorch`, param=`attn.implementation` (passed as nested kwarg to PyTorchConfig) |

**Option A vs Option B — concrete comparison (same study):**

```yaml
# Option A — 4 lines in sweep:
backend: [pytorch, vllm]
sweep:
  precision: [bf16, fp16]
  pytorch.batch_size: [1, 8, 32]
  vllm.max_num_seqs: [64, 256]
# → 10 experiments (6 pytorch + 4 vllm)

# Option B equivalent — 12 lines in experiments: + sweep:
sweep:
  precision: [bf16, fp16]
experiments:
  - backend: pytorch
    pytorch: { batch_size: 1 }
  - backend: pytorch
    pytorch: { batch_size: 8 }
  - backend: pytorch
    pytorch: { batch_size: 32 }
  - backend: vllm
    vllm: { max_num_seqs: 64 }
  - backend: vllm
    vllm: { max_num_seqs: 256 }
# → same 10 experiments, 4× more YAML
```

The `experiments:` list is retained for fully explicit, non-Cartesian combinations (specific
pairings that do not form a grid). Both modes compose: `sweep:` generates its grid and
`experiments:` adds explicit entries on top.

### Consequences

Positive: Compact grammar for multi-backend studies; clear scope signal; composable with
`experiments:` list.
Negative / Trade-offs: Slightly non-obvious for new users; requires documentation of the
split-on-first-dot convention.
Neutral: Cross-backend comparison studies with non-shared params still require explicit
`experiments:` list (dotted notation solves backend-param sweeps, not semantic translation
between backends).

---

## Additional Implementation Decisions

### `extra = "forbid"` and the `extra:` escape hatch

These two uses of "extra" are different things and are fully compatible:

- **`model_config = {"extra": "forbid"}`** — Pydantic config setting. Rejects any YAML key
  not declared as a field. Prevents silent typos (`bachsize: 8` passing silently).

- **`extra: dict[str, Any] | None = None`** — an explicitly declared field. Because it is
  declared, Pydantic does not reject it. Deliberate escape hatch for arbitrary backend kwargs
  (undocumented or niche parameters not yet in the schema).

The `extra = "forbid"` setting prevents `bachsize: 8` (typo) from silently becoming
`extra: {bachsize: 8}`. Only the explicitly declared `extra:` field bypasses this —
intentionally.

### Backend defaults: None-as-sentinel pattern

Backend config fields use `None` as "not specified — use backend default":

```python
class PyTorchConfig(BaseModel):
    batch_size: int | None = None        # None = backend decides (effective default: 1)
    attn_implementation: str | None = None  # None = backend decides (effective default: "sdpa")
```

Not embedded defaults in the Pydantic model. Reasons:
- `config_hash` in ExperimentResult reflects only explicit user choices, not defaults
- Backend implementation version controls its own defaults (not stale values in config model)
- Users can clearly see what they explicitly set vs what the backend will default

### Required changes for implementation

1. **`extra = "forbid"`** — remove `model_config = {"extra": "allow"}`. Typos in YAML must fail loudly.
2. **Backend section fields: `T | None = None`** — keep as optional; `None` means "use backend defaults".
3. **Sweep grammar: dotted notation (Option A)** — `sweep:` supports backend-scoped keys.
4. **Naming** — use "shared params" and "backend sections" everywhere. Drop any "Tier 1/2" or "Layer 1/2/3" jargon.
5. **Rename fields:** `model_name` → `model`, `fp_precision` → `precision`, `num_input_prompts` → `n`.

---

## Peer Codebase References

| Tool | Config structure | Type safety | Multi-backend composition |
|------|-----------------|-------------|--------------------------|
| Optimum Benchmark | Inheritance (subclasses) | (dataclass) | Difficult |
| lm-eval | Flat string args | None | N/A |
| W&B Sweeps | Flat dict | None | N/A (optimisation, not benchmarking) |
| MLflow | Per-backend modules | N/A | N/A |
| Hydra | File-based composition | (structured configs) | Via config groups |
| **llem (chosen)** | **Single type + nested sections** | **(Pydantic)** | **(single sweep: grammar)** |

Full peer code samples (verbatim): [research/15-config-architecture-patterns.md](../research/15-config-architecture-patterns.md)

---

## Summary: All Config Architecture Decisions

| Decision | Resolution | Date |
|----------|-----------|------|
| Composition vs Inheritance | **Composition** — single `ExperimentConfig` with optional backend sections | 2026-02-18 |
| Sweep grammar for backend-specific params | **Option A — dotted notation** (`pytorch.batch_size: [1, 8, 32]`) | 2026-02-19 |
| Required vs optional backend sections | **Optional (`T \| None = None`)** — backend owns its defaults, not the config model | 2026-02-18 |
| Tier naming | **Drop "Tier 1 / Tier 2"** — use "shared params" and "backend sections" | 2026-02-18 |
| `extra = "forbid"` with escape hatch | **Compatible** — declared `extra: dict` field bypasses the Pydantic setting | 2026-02-18 |
| `_extends` YAML inheritance | **Cut from v2.0** — composition + sweep grammar replaces the use case. See §below. | 2026-02-26 |

---

## `_extends` YAML File Inheritance — Cut (2026-02-26)

v1.x implements `_extends: base_config.yaml` with deep merge and cycle detection in `config/loader.py` (see `preservation_audit/N-X13`). This allowed configs to inherit from a base file and override specific fields.

**Decision: Cut from v2.0.** The `_extends` mechanism is not carried forward.

**Rationale:** The composition config model (single `ExperimentConfig` with optional backend sections) and the sweep grammar (`sweep:` block with dotted notation) together cover the use cases that `_extends` served in v1.x:

- **Sharing base config across variants** → use `sweep:` grammar (generates grid from shared base)
- **Machine-specific overrides** → use `~/.config/llenergymeasure/config.yaml` (user config layer)
- **Explicit non-grid variants** → use `experiments:` list in study YAML

`_extends` adds implementation complexity (deep merge semantics, cycle detection, path traversal security boundary — see `preservation_audit/N-X13`) for a feature whose use cases are now covered by other mechanisms. No v2.0 design doc references it.

**What happens to the v1.x code:** `resolve_inheritance()`, `deep_merge()`, and `_extends` handling in `config/loader.py` are not carried forward to v2.0. The preservation audit entry N-X13 is updated accordingly.

---

## Related

- [architecture.md](architecture.md): Library-first architecture (affects how config maps to API)
- [experiment-study-architecture.md](experiment-study-architecture.md): Option C architecture, sweep resolution at parse time
- [../designs/study-yaml.md](../designs/study-yaml.md): Sweep grammar (sweep: + experiments:)
- [../designs/experiment-config.md](../designs/experiment-config.md): Full `ExperimentConfig` schema
- [../designs/config-model.md](../designs/config-model.md): SSOT for field placement and data flow
- [../research/15-config-architecture-patterns.md](../research/15-config-architecture-patterns.md): Peer code samples
- `src/llenergymeasure/config/models.py`: Current ExperimentConfig implementation
- `src/llenergymeasure/config/backend_configs.py`: Current backend config models
