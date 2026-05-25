# H9 cross-engine diagnosis summary

**Pattern:** Tier 3, cheap-batch. LLM reads (a)'s output + engine
source excerpts + the known-gap inventory; produces STRUCTURED
diagnoses of structural gap categories. No output mutation. Pure
diagnostic intelligence.

**Backend:** container Ollama @ port 11435, model `llama3.1:70b`
(q4_K_M), num_ctx=32768, temperature=0, output_format=json.

**Cells run:** transformers v4.57.3, vllm v0.7.3, tensorrt v0.21.0.

**Wall-clock:** 29.7 + 55.4 + 37.9 = 123s LLM time total.

## Diagnoses per engine

| Engine | Diagnoses | Already-known | New-here | Engine-specific |
|---|---:|---:|---:|---:|
| transformers v4.57.3 | 1 | 1 | 0 | 0 |
| vllm v0.7.3 | 4 | 3 | 1 | 0 |
| tensorrt v0.21.0 | 3 | 2 | 1 | 0 |
| **Aggregate** | **8** | **6** | **2** | **0** |

## Category breakdown across engines

| Category | transformers | vllm | tensorrt | Total |
|---|:-:|:-:|:-:|---:|
| branch-descent | - | 2 | 1 | 3 |
| local-var-alias | - | 1 | - | 1 |
| normalisation-only | - | 1 | - | 1 |
| type-blindness | - | - | 1 | 1 |
| nested-config | - | - | 1 | 1 |
| defensive-import | 1 | - | - | 1 |

## What's correctly identified (yes-already-known)

The 6 yes-already-known diagnoses re-confirm the
`post_trial_a_gap_closure.md` inventory:

| Engine | Diagnosis | Maps to gap-inventory entry |
|---|---|---|
| transformers | defensive-import on bumped versions | G-trf-1 |
| vllm | normalisation-only EngineArgs.__post_init__ | G-vllm-1 |
| vllm | local-var-alias ModelConfig._verify_quantization | G-vllm-2 |
| vllm | branch-descent CacheConfig._verify_cache_dtype | G-vllm-3 |
| tensorrt | type-blindness max_batch_size probe synth | G-trt-1 |
| tensorrt | nested-config scheduler_config.capacity_scheduler_policy | G-trt-3 |

The category labels match the inventory. The example_field values are
correct citations of fields that exhibit each gap. The severity ratings
(`blocks-correctness` vs `reduces-recall` vs `minor`) align with the
inventory's expected severities.

**This re-confirms H4's diagnostic ability finding**: the LLM is
EXCELLENT at categorising structural gaps even without doing the
fix-implementation work.

## What's new (yes-new-here)

Two new diagnoses NOT present in the prior inventory:

### N1: vllm SamplingParams._verify_args branch-descent (new)

LLM identifies branch-descent gap in `SamplingParams._verify_args`
beyond the inventory's `CacheConfig._verify_cache_dtype` instance.
This expands G-vllm-3 from "one method" to "a pattern that recurs across
multiple validators in vllm".

**Mergeable into spike refactor:** the structural fix is the same as
G-vllm-3 (handle if/elif/else properly). One walker patch closes both
sites. Cross-engine cluster value: the fix is single-implementation
even though it now closes a wider blast radius.

### N2: tensorrt model_config.arbitrary_types_allowed branch-descent (new, minor)

LLM identifies a `branch-descent` gap on
`tensorrt.model_config.arbitrary_types_allowed`. Severity rated `minor`
in the diagnosis; example field is itself a Pydantic-config-only knob
(not a runtime invariant). The yes-new-here marking is correct - this
field isn't in the inventory - but the operational value is limited
(Pydantic config attributes don't drive runtime errors).

## Cross-engine cluster pattern (re-confirmed from H4)

H9 + H4 together confirm:

- **branch-descent gaps** cluster on vllm (CacheConfig + SamplingParams)
  + tensorrt (Pydantic configs). Transformers walker already handles
  branches via the nested-dataclass walker (spike commit `15f34240`).
- **nested-config gaps** appear in tensorrt; analogous absent-recursion
  patterns in vllm CacheConfig companions; transformers handles via
  BNB inlining. Cross-engine: single `_NestedConfigWalker` abstraction
  closes all three.
- **type-blindness** is tensorrt-specific in H9's read (it's where the
  probe-synthesis happens via `_value_satisfying`). The other engines
  use the validation infrastructure at a different layer.
- **defensive-import** is transformers-specific in H9's read (the
  `import` block hits hard ImportError on bumped versions).
- **normalisation-only** is vllm-specific in H9's read (EngineArgs
  has zero raises).
- **local-var-alias** is vllm-specific in H9's read (ModelConfig's
  `_verify_*` methods alias `self.X` to local vars).

## What H9 missed vs the inventory

The inventory has 5 gap entries this batch could potentially have
surfaced. H9 hit 5 of them. The misses:

- **G-trt-2 (DeprecationWarning poisoning)**: NOT a walker gap; the
  H9 prompt correctly excluded this from scope.

No false-positive new-gap claims. H9's "yes-new-here" entries are
genuinely additive (or genuine new instances of known categories).

## Recommendation

H9's diagnoses-only output is the cheapest, highest-quality artefact
of the cheap-patterns batch. Backlog items to file:

1. **N1**: extend the if/elif/else fix planned for G-vllm-3 to also
   cover `SamplingParams._verify_args`. Same walker patch.

2. **N2**: park as low-priority; Pydantic config attributes don't
   need invariant capture.

The 6 already-known confirmations validate the inventory + add
explicit categorical labels (the inventory was loosely categorised;
H9 produces strict category strings the spike refactor can drive off).

For Phase 4 synthesis: H9 establishes that the LLM produces
diagnoses-correctness scores of 8/8 on the active row (no fabricated
gaps; categories all match the known inventory's structural reads).
This validates "LLM as diagnostic accelerator" as a defensible Phase 5
component of the production substrate.

## Artefacts

Per-engine: `research/mining-substrate-trial/findings/hybrid_experiments/h9_diagnose/<engine>/`
- `diagnoses.yaml` (structured gap entries)
- `observations.md` (engine-specific notes + category breakdown)
- `prompt.md` (locked prompt input)
- `raw_response.txt` (LLM raw output for audit)

Cross-engine: this file.
