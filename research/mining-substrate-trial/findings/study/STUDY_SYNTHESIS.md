# Mining engine-config invariants: cross-wave synthesis (Phase 1, waves 1-4)

A synthesis of the systematic study of LLM + deterministic hybrid workflows for
mining engine-config invariants, scored by the real runtime gate. Per-wave detail:
`PHASE1_WAVE{1,2,3,4}_FINDINGS.md`. Prior-PoC map: `WAVE4_RECONCILIATION_MAP.md`.

## The problem + north star

llem must keep engine-config knowledge (schema + invariants) CURRENT across
upstream version bumps (vLLM / TensorRT-LLM / transformers), cheaply enough to run
on every bump in CI. An "invariant" is a rule the engine enforces at config
construction (`if <pred>: raise/warn`, a Literal/enum field, a cross-field
relation). North-star principles:
- the ENGINE owns its SSOT; a RUNTIME GATE validates mined knowledge in-container
  ("observe, don't re-encode" - construct a violating config, observe the raise);
- cost is understood ORDINALLY: deterministic (~free) < small-OSS < mid-OSS < Opus;
- mine COMPREHENSIVELY (full surface), expose a subset; the gate is the SSOT, the
  LLM is only a candidate generator.

## The method

Candidate generators (deterministic miner, OSS LLM, Opus) propose invariants; the
runtime gate (`scripts/validate_invariants.py`, in the engine's own container)
adjudicates each by constructing a probe and observing whether the engine raises.
The design space (assembly x call-shape x model-tier) was explored fractionally,
with mandatory adversarial source-verification of every cross-field confirm and an
internals-guard (drop private/underscore fields, type-trivia, observability,
launch-state). Cells: vllm 0.19.1 + tensorrt 1.2.1 (N=2; directional).

## What each wave established

**Wave 1 - the bottleneck is VALIDATION, not LLM recall.** det-then-llm-extend
(match-only) yielded 0 gate-confirmed lift over the deterministic floor on both
cells x both rungs (gemma3:12b AND Opus). Not because the LLM found nothing - it
surfaced ~50 real cross-field relations - but because the single-field
auto-synthesis gate could not PROBE them. The validation path, not recall, was
binding.

**Wave 2 - the kwargs-emission lever unlocks the cross-field tail.** Having the
LLM also emit constructible `kwargs_positive/negative` turned the tail
gate-confirmable: Opus 0 -> 8 verified-real cross-field confirms (folded into GT).
gemma3:12b FAILED the lever (17/25 failed; 0 verified-real). So the tail is REAL
but reachable only at Opus cost on this shape. A gate soundness gap (cross-field
confirm attribution) was found + fixed.

**Wave 3 - scale is the threshold; code-tuning sharpens (the size x tuning 2x2).**
Verified-real cross-field confirms: small models (gemma-12b general, qwen-7b code)
= 0; mid/large (qwen-coder-32b = 5, llama-70b general = 3) reach it; Opus = 8.
SCALE is the threshold to reach the tail at all (between 12B and 32B); CODE-TUNING
sharpens within the capable regime - a code-tuned 32B BEAT a general 70B on
coverage, precision, speed, and cleanliness. Agentic-tuned/general models surface
internal noise the code models don't.

**Wave 4 - the OSS strategy frontier.** Dropping the floor (llm-only) and varying
STRATEGY:
- pure-LLM / prompt: tensorrt 100% infra-blocked (LLM omits required ctor fields).
- CONSTRUCTION-GROUNDING (inject AST constructor signatures) is THE OSS lever:
  breaks the tensorrt infra wall (qwen-coder-32b 0 -> 20 verified-real), lifts
  vllm precision ~60%. A det+LLM hybrid (det discovers construction context, LLM
  synthesises). Model-specific (qwen2.5-coder line; not qwen3-coder/deepseek).
- AGENTIC (LangGraph) is a POOR strategy for OSS (tool-call flakiness; no
  incremental synthesis; even devstral). The prior "agentic=0" was an
  all-at-once-harness artefact, not a model finding.
- a second gate soundness fix: reject type-coercion-artifact confirms.
- residual analysis: the tensorrt "ceiling" is a STUDY-FLOOR artefact (the
  production pydantic-lift already covers the PluginConfig Literal constraints the
  study's older improved-det-v2 floor misses) - so production recall is HIGHER
  than the study's numbers.
- self-consistency (k-vote) for the vllm cross-field tail: <PLACEHOLDER - result pending>.

## The synthesised answer

**Production design = deterministic floor (production pydantic-lift) + a
construction-grounded LOCAL mid code-model (~32B) + the runtime gate + the two
soundness guards + the internals-guard.** This stack:
- reaches ~69% (vllm) / >=59% (tensorrt, understated by the study floor) GT recall
  measured; production floor pushes tensorrt materially higher;
- runs locally + reproducibly + cheaply (one AST pass is free; one chunked
  generation per bump on a local GPU);
- captures the cross-field/conditional/entangled-class tail the deterministic
  floor structurally cannot reach;
- needs Opus only as a small, expensive, non-reproducible TOPPING for the residual
  cross-field tail - NOT as the workhorse. "Just call Opus per bump" keeps the
  expensive half (the gate is mandatory regardless), pays to re-derive what det
  does free + reproducibly, and sacrifices reproducibility.

**Cost story (ordinal, the deliverable - NOT a plotted $-frontier):**
deterministic (~free, reproducible) < local OSS code-32B (minutes/bump, local GPU)
< Opus (API tokens/bump, non-reproducible). The cheap rungs own the bulk; the
research question "does the cheap rung suffice?" answers: largely YES, with a
local 32B-coder, for everything but the hardest cross-field tail.

## What is VALIDATED vs OPEN

Validated: the gate<->harness integration; validation-path-is-the-bottleneck; the
kwargs lever; the scale-threshold + code-tuning tier story; construction-grounding
as the OSS infra-wall lever; the hybrid recall; two gate soundness fixes; agentic
is wrong for OSS.

Open / future: self-consistency upside (pending); generalising construction-
grounding beyond the qwen-coder line; closed-loop/gate-as-tool with Opus (reliable
tool-calling) if the cross-field tail is worth the per-bump gate-call cost; the
north-star self-update / degradation-signal binary on a MAJOR bump (tensorrt
0.21->1.0) - the actual product property, never yet tested.

## Caveats

N=2 cells, mostly single-shot, directional. The study floor (improved-det-v2)
understates the production deterministic miner. Cross-field confirms always
require adversarial source-verification (the inflation class recurs). The
internals-guard is applied in analysis, not yet in the miner.
