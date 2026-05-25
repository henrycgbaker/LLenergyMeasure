# Walker validation set

**Date opened:** 2026-05-24
**Purpose:** A growing list of engine-field shapes we have established by
**direct introspection** of the engine's runtime API. These are the
**test oracles** that future walker improvements should reproduce
mechanically. If a walker enhancement closes a gap, the corresponding
field's shape here should be its post-improvement output.

Use this as:
1. A target for Move 1 walker deepening (each entry is a walker task).
2. A validation set: re-mine, diff against these expected shapes, score
   walker effectiveness.
3. A drift detector: if the introspected ground truth changes on engine
   bump, this file's entry needs updating (or the engine genuinely
   changed and that's a real diff).

Each entry records:
- **Engine + version** (when the introspection was done)
- **Field name + path** (where it lives in upstream)
- **Ground truth shape** (from direct introspection: signature, type,
  default, docstring)
- **Current mined state** (what `schema.discovered.json` says today)
- **Walker gap** (what the walker would need to do to surface it)
- **Overlay completion?** (whether we cover it via `overlay.yaml` in the
  interim)

Format is markdown for readability; if this grows, convert to YAML.

---

## transformers v4.57.3

### `compile_config` (sampling_params)

```yaml
field_path: sampling_params.compile_config
upstream: transformers.GenerationConfig.compile_config
upstream_type: transformers.generation.configuration_utils.CompileConfig | None
upstream_default: None
ground_truth_shape:
  type: object
  description: "torch.compile config (set by HF inside model.generate())"
  properties:
    fullgraph: { type: boolean, default: false }
    dynamic: { type: boolean, nullable: true, default: null }
    backend: { type: string, default: "inductor" }
    mode: { type: string, default: "reduce-overhead" }
    options: { type: object, nullable: true, default: null }
mined_today: NO (mining missed; nested CompileConfig dataclass not walked)
walker_gap: |
  Need to recognise that GenerationConfig.compile_config has type annotation
  pointing at the CompileConfig dataclass, then traverse the dataclass and
  emit its fields as a $defs entry with a $ref from compile_config.
overlay_completion: YES (overlay.yaml sampling_params.compile_config above)
x_completion_reason: "nested CompileConfig dataclass not walked yet (Move 1 gap)"
verified_at: 2026-05-24 via `transformers.generation.configuration_utils.CompileConfig`
```

### `low_cpu_mem_usage` (engine_params) - REMOVED BY UPSTREAM

```yaml
field_path: engine_params.low_cpu_mem_usage
upstream: transformers.PreTrainedModel.from_pretrained kwargs
status_2026_05_24: |
  Source inspection of PreTrainedModel.from_pretrained shows
  `_ = kwargs.pop("low_cpu_mem_usage", None)` at L283 - HF accepts the
  kwarg from kwargs and DISCARDS it. Effectively deprecated; no-op
  field. Also removed from the from_pretrained docstring (was present
  in earlier versions per existing curated.yaml comment).
ground_truth_decision: |
  Do NOT add to engine_params - the engine no longer acts on it.
  Walker is correctly reflecting upstream reality (not mining a
  documented-but-discarded field).
overlay_completion: NO
implication_for_migration: |
  Old hand-written TransformersConfig.low_cpu_mem_usage is dropped
  entirely. Existing user configs that set it are accepted via
  extra='allow' (the value is preserved on the dict but never used).
verified_at: 2026-05-24 via inspect.getsource(PreTrainedModel.from_pretrained) L283
```

### `disable_compile` (sampling_params)

```yaml
field_path: sampling_params.disable_compile
upstream: transformers.GenerationConfig.disable_compile
upstream_type: bool
upstream_default: false
ground_truth_shape:
  type: boolean
  default: false
  description: "If true, GenerationConfig will not attempt to torch.compile generate."
mined_today: YES (already in sampling_params with type=boolean, default=false)
walker_gap: none
overlay_completion: not needed
verified_at: 2026-05-24
```

### `tp_size` (engine_params) - DOCSTRING-vs-USAGE NARROWING

```yaml
field_path: engine_params.tp_size
upstream: transformers.PreTrainedModel.from_pretrained kwargs
docstring_says: L167 - "tp_size (`str`, *optional*):"
source_usage:
  L263: "tp_size = kwargs.pop(\"tp_size\", None)"
  L326: "tp_size = device_mesh.size()  # int"
  L329-330: "if tp_size is None: tp_size = torch.distributed.get_world_size()  # int"
  L643: "model = distribute_model(model, distributed_config, device_mesh, tp_size)"
ground_truth_shape:
  type: integer
  minimum: 1
  description: "Number of tensor parallel ranks (None -> WORLD_SIZE)"
mined_today: |
  Mined as type=string (walker faithfully reflects the docstring which is
  upstream-incorrect)
walker_gap: |
  Walker correctly extracts the docstring. No automated fix unless walker
  cross-references docstring claim against actual source-code usage (e.g.
  detect that tp_size flows into device_mesh.size() return value). That's
  a Phase 3+ enhancement.
overlay_completion: YES (overlay.yaml narrowings.engine_params.tp_size)
x_narrowing_reason: "from_pretrained docstring claims str; HF code at L326/L330 uses it as int (device_mesh.size() / torch.distributed.get_world_size() returns int)"
verified_at: 2026-05-24 via inspect.getsource(PreTrainedModel.from_pretrained)
```

---

## vllm v0.7.3

### To populate when option-A migration touches vllm in detail.

---

## tensorrt v0.21.0

### To populate when option-A migration touches tensorrt in detail.

---

## How to add an entry

1. Direct introspection: `inspect.signature(...)`, `cls.__doc__`,
   `cls.__dataclass_fields__`, `cls.model_fields` (for Pydantic).
2. Diff against mined: open `engine_versions/<e>/v<safe>/outputs/
   schema.discovered.json` and locate the field (or its absence).
3. Add YAML block to this file. Lower priority on prose; the YAML is
   the contract.
4. If walker improvement closes the gap later: don't delete the entry;
   mark `mined_today: YES` and keep the rest as historical evidence
   that this used to require overlay.
