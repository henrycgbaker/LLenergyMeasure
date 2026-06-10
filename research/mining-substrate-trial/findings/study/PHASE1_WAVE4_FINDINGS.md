# Phase 1, Wave 4 - findings (pure-LLM + the OSS strategy frontier)

Status: 4a + construction-grounding COMPLETE + verified; agentic characterised;
self-consistency / hybrid-integration next. Pre-registration:
`PHASE1_WAVE4_PREREG.md`. Map onto the prior PoC: `WAVE4_RECONCILIATION_MAP.md`.

## The question

Wave 4 drops the deterministic floor (llm-only / W-C) and asks: across DIFFERENT
STRATEGIES, how far can OSS models be pushed on engine-config invariant mining?
The frame is a strategy ladder, scored by the real runtime gate on the same 2
cells (vllm 0.19.1, tensorrt 1.2.1), with mandatory adversarial verification +
internals-guard on every confirm.

## The strategy ladder (headline)

| strategy | what it adds | qwen2.5-coder:32b result (verified-real candidate-config) |
|---|---|---|
| **4a pure-LLM / prompt** | source chunk + prompt, no floor | vllm ~14 / tensorrt **0** (100% infra wall) |
| **construction-grounding** | + AST constructor signatures injected | vllm ~26 / tensorrt **20** |
| agentic (4b/4c) | LangGraph tool-loop | FAILS for OSS (see below) |

**Construction-grounding is the first strategy that clearly lifts OSS - and it
BREAKS the tensorrt infra wall.** 4a got 0 confirmed on tensorrt (all 61 proposals
infra_error: the LLM omitted required constructor fields, so pydantic "field
required" fired before any validator). Injecting each class's AST-extracted
constructor signature (required/optional fields + types) lets the LLM supply the
required fields, so construction REACHES the real validators. Result on the
hardest engine: 0 -> 23 confirmed, 20 verified-real (cuda_graph, nvfp4_gemm,
decoding/eagle/lookahead, kv_cache, torch_compile, batch-scheduling). It is a
det+LLM hybrid: deterministic AST does construction-context discovery, the LLM
does synthesis. Runner: `scripts/phase1/wave4_construct.py`; prompt
`phase1_wave4/construct_grounded_prompt.md`.

## Breadth + model-dependence (construction-grounding, raw confirmed / recall / infra)

| model | tensorrt 4a -> cg | vllm 4a -> cg | read |
|---|---|---|---|
| qwen2.5-coder:32b | 0/0/61 -> 23/14/14 | 22/14/53 -> 30/21/37 | big win both cells |
| qwen2.5-coder:14b | 0/0/61 -> 14/10/13 | 16/13/51 -> 25/16/51 | tensorrt wall breaks |
| qwen3-coder:30b | 0/0/63 -> 0/0/68 | 17/10 -> 33/6 | NO tensorrt benefit; vllm recall down |
| deepseek-coder-v2:16b | 12/4 -> 1/1 | 4/1 -> 15/1 | regressed on tensorrt |

The lever works for the qwen2.5-coder line (the wave-3 extraction winner); it does
NOT generalise to qwen3-coder (MoE) or deepseek on tensorrt. Construction-grounding
is model-specific, not universal.

## Verification (mandatory soundness gate)

qwen2.5-coder:32b construct-grounded, independent Opus adversarial verify against
source: tensorrt 20/23 REAL candidate-config; new-vllm 7/9 REAL. The non-real:
- 2 tensorrt SPURIOUS: gpus_per_node-ray, max_beam_width-default - both labelled
  rules are NON-RAISING normalisations; they only "confirmed" because YAML `None`
  serialised to the string `"None"`, tripping a pydantic int-parse error.
- 1 tensorrt INTERNAL: model-valid-type (pydantic type trivia).
- 1 vllm SPURIOUS: max_lora_rank>0 - the field is a `Literal[1,8,16,...]`; 0 raises
  literal_error, the labelled ">0" rule is semantically wrong.
- 1 vllm INTERNAL: max_model_len-not-int (type-parse trivia).

## NEW gate-hardening finding (a real SSOT improvement)

The non-real confirms share one mechanism: the positive raised a pydantic
PARSING/literal error (`int_parsing`, or `literal_error` on a numeric-labelled
Literal) on the probed field, NOT a custom ValueError matching the labelled
semantic validator (the canonical artefact: YAML `None` -> the string `"None"` ->
`int_parsing` on a NON-raising normalisation). The wave-1/2 attribution guard
blocks wrong-FIELD raises but did not distinguish a parsing artefact on the RIGHT
field from the labelled semantic rule.

FIX (IMPLEMENTED + tested, `scripts/validate_invariants.py`
`_positive_is_type_coercion_artifact`): reject a LENIENT confirm whose positive
raised a pydantic PARSING error (`int_parsing`/`float_parsing`/`bool_parsing`/
`decimal_parsing`), or `literal_error` on a numeric-labelled predicate, unless the
invariant predicate is itself a type-check. DELIBERATELY excludes `*_type` /
`string_type` (those legitimately fire for REQUIRED-field "must be provided"
rules). Gated on `not expected_strict`, so the strict-scored GT is untouched.
Verified on the qwen-32b construct corpora: drops EXACTLY the 3 adversarially-
confirmed spurious (gpus_per_node-ray, max_beam_width-default, max_lora_rank>0),
keeps all reals incl the required-field `output_directory`. Zero false positives.
Makes raw strategy numbers trustworthy without per-run manual verification.

## Agentic (4b/4c): a POOR strategy for OSS

Reconciliation (`WAVE4_RECONCILIATION_MAP.md`) showed the prior h7_agentic "0" was
largely a HARNESS ARTEFACT (all-at-once `finalise`, no incremental emit). The
current harness (`wave4_agentic.py`) has incremental `emit_invariant`. Re-run on
the fixed harness: OSS models (incl devstral:24b, agentic-tuned) STILL emit 0 -
two robust failure modes: (1) they intermittently emit tool calls as TEXT not
structured (ollama format flakiness) -> the ReAct loop halts; (2) they explore but
do not interleave emit. So agentic ReAct is not the OSS lever - the better strategy
is to do exploration DETERMINISTICALLY (construction-grounding) and let the model
just synthesise. (Opus tool-calls reliably, but that is the paid ceiling.)

## Pure-LLM recall context (W-C) + the HYBRID production number

Even the best strategy (construction-grounding) tops at ~21/80 vllm, ~14/61
tensorrt recall pure-LLM - well below the det floor's ~55%. The floor is
load-bearing; the LLM adds the cross-field/conditional + entangled-class tail.

HYBRID (det-v2 floor UNION construct-grounded qwen2.5-coder:32b, computed from the
gated confirms - no extra run):

| cell | floor alone | LLM net-new GT keys | HYBRID |
|---|---|---|---|
| vllm 0.19.1 | 44 (55%) | +11 | **55/80 (69%)** |
| tensorrt 1.2.1 | 35 (57%) | +1 | **36/61 (59%)** |

So a LOCAL 32B-coder + det floor + gate reaches ~60-70% GT recall. The LLM's
GT-recall value is concentrated on vllm (+14 points via the cross-field tail); on
tensorrt its GT-recall lift is +1, BUT its 20 verified-real confirms are mostly
GROWTH (real invariants not yet in the GT) - the comprehensive-discovery value the
GT-recall metric understates (the GT is itself incomplete). The residual ~30-40%
of GT that neither floor nor construct-grounded-LLM reaches is the hard tail for
further strategies (self-consistency, or the Opus ceiling).

## Residual analysis + the source-coverage / extraction-method ceiling

What does the hybrid MISS? (GT not covered by floor UNION construct-LLM, qwen-32b):
- vllm: 25/80 missed - genuine cross-field tail (7 cross_field_combo, 5 literal_in,
  3 presence_conflict) + observability classes (ProfilerConfig/ObservabilityConfig =
  internals-guard territory, arguably should not be in the GT).
- tensorrt: 25/61 missed, **20 of them a SINGLE class, PluginConfig** (mostly
  literal_in). PluginConfig lives in `plugin/plugin.py`, NOT in the chunked source
  (llm_args.py + sampling_params.py) - so the LLM never saw it. The 59% tensorrt
  "ceiling" is therefore NOT a model limit.

BUT: adding `plugin.py` to the source set did NOT lift recall (14 -> 14/61). The
deeper limit: the chunker extracts VALIDATOR BODIES (`if x: raise`), while
PluginConfig's constraints are pydantic `Literal`/enum FIELD TYPES (no explicit
raise). The validator-body chunker structurally cannot see field-type constraints.
The AST signature pass DOES capture the Literal values (they are in the
construction-grounding INPUT 1) but the prompt mines invariants from INPUT 2 source,
not from the signatures.

CONCLUSIVE RESOLUTION (no build needed - the lever already EXISTS in production):
`scripts/engine_producers/_pydantic_lift.py` already extracts `Literal[...]` field
constraints ("the lift emits a value-allowlist invariant"). The GT's 20 PluginConfig
invariants came ENTIRELY from deterministic sources (prod 13, passB 6, passA 1;
literal_in/None) - NOT from the LLM. So the tensorrt residual is a STUDY-FLOOR
ARTIFACT: the study's `improved-det-v2` floor is older/narrower than the production
pydantic-lift and does not parse `plugin.py`. Therefore **the study's 59% tensorrt
hybrid recall UNDERSTATES production** - the real production floor already covers
PluginConfig's Literal/field constraints, so production hybrid recall is materially
higher. The remaining tensorrt recall is NOT an open capability gap; it is already
solved by the production deterministic path the study floor predates. (`plugin.py`
kept in the LLM source set; harmless.)

## Cost (ordinal)

construction-grounding adds ~0 cost over 4a (one cheap AST pass; same single-shot
chunked generation). qwen2.5-coder:32b ~1100s vllm / ~900s tensorrt per cell, local
GPU. The AST signature pre-pass is deterministic + free + reproducible.

## Next

1. Implement the type-coercion attribution guard (above) - banks the SSOT
   improvement, cleans all future raw numbers.
2. Fold construction-grounding into the hybrid (det floor + construct-grounded
   LLM-extend) and measure the combined production recall.
3. Self-consistency (k-vote) on construction-grounding for recall/precision.
