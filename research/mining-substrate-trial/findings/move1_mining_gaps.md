# Move 1 mining gaps surfaced by Phase 2-T pilot

**Date:** 2026-05-24
**Source:** spike Phase 2-T pilot for transformers (task #5)

When bootstrapping `engine_versions/transformers/v4_57_3/outputs/curated.yaml`
from the existing hand-written `TransformersConfig` +
`TransformersSamplingConfig` (`src/llenergymeasure/config/engine_configs.py`),
the following fields are exposed today but NOT present in the freshly-mined
`schema.discovered.json`. These represent the **Move 1 deepening targets**
the spec anticipated.

## What's in the existing hand-written class but NOT mined

| Field | Where it lives in upstream | Why it's not mined today |
|---|---|---|
| `batch_size` | llem-domain | Not a transformers API field; llem wrapper concept. Stay hand-written. |
| `dtype` | `from_pretrained(torch_dtype=)` kwarg | `from_pretrained`'s `**kwargs` are not in the signature, so `inspect.signature` introspection misses it. Static-miner pattern needed. |
| `attn_implementation` | `from_pretrained(attn_implementation=)` kwarg | Same kwargs-not-in-signature issue. |
| `torch_compile`, `torch_compile_mode`, `torch_compile_backend` | `from_pretrained(torch_compile=)` kwargs | Same. |
| `load_in_4bit`, `load_in_8bit` | `BitsAndBytesConfig` | Not introspected (would need a separate landmark for `BitsAndBytesConfig` dataclass). V3-V5 validators target this surface. |
| `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant` | `BitsAndBytesConfig` fields | Same. |
| `device_map`, `max_memory` | `from_pretrained(device_map=)` / `(max_memory=)` kwargs | Same kwargs-not-in-signature issue. |
| `allow_tf32`, `autocast_enabled`, `autocast_dtype` | torch global state, not transformers | Outside transformers' surface entirely; llem-domain. |
| `low_cpu_mem_usage` | `from_pretrained(low_cpu_mem_usage=)` kwarg | Same kwargs issue. |
| `tp_plan`, `tp_size` | `from_pretrained(tp_plan=)` kwargs (HF >=4.50) | Same kwargs issue. V5 validator targets this. |

20 fields total. Several patterns:

1. **`from_pretrained` `**kwargs` documented in class docstring only**
   (`dtype`, `attn_implementation`, `torch_compile*`, `device_map`,
   `max_memory`, `low_cpu_mem_usage`, `tp_plan`, `tp_size` -
   ~12 of the 20). The introspector's `discovery_limitations` already
   flags this:
   `"from_pretrained accepts **kwargs; kwargs are not in the signature"`.
   Mining fix: add a docstring walker to lift `**kwargs` documented in
   the class docstring (HuggingFace's `from_pretrained` has a richly
   documented kwargs section). This is the largest single mining-surface
   gap.

2. **Companion-config classes not traversed** (`BitsAndBytesConfig` ->
   `load_in_4bit`, `load_in_8bit`, `bnb_4bit_*` - 5 fields). Mining fix:
   add `BitsAndBytesConfig` to LANDMARKS + traverse its dataclass fields
   (the schema introspector already has `signature_param_to_spec`; add
   a sibling for dataclass-shaped companion configs).

3. **llem-domain wrappers** (`batch_size`, `allow_tf32`, `autocast_*` -
   4 fields). NOT mining gaps; these stay in hand-written llem-domain
   config. The codegen produces *engine-API-shaped* config; llem wraps
   that with the wider experiment shape.

## Implications for the spike pilot

- Generated config.py emits 14 fields (Phase 2-T pilot scope).
- Hand-written `TransformersConfig` remains the user-facing API for now;
  the spike validates the codegen pipeline end-to-end on the subset
  that's mineable.
- Cascade test (task #6) exercises: change `current.toml` -> cells
  re-mine -> regen_engine_corpus.py mirrors archive to shadow ->
  regen_engine_configs.py regenerates config.py -> CI confirms parity.
- Migration of TransformersConfig to the generated class is **not**
  attempted in the spike; it requires Move 1 deepening first to close
  the kwargs + companion-config gaps.

## Cross-refs

- `.product/designs/engine-knowledge-as-data.md` § Move 1 (the deepening targets).
- `.product/designs/engine-knowledge-as-data.md` § Move 4 (V1-V8 validators -
  V1, V2, V3, V5 of the eight target this same kwargs/companion surface).
- `engine_versions/transformers/v4_57_3/outputs/curated.yaml` (the pilot
  curation).
- `src/llenergymeasure/config/engine_configs.py` (the hand-written class
  the spike compares against).
