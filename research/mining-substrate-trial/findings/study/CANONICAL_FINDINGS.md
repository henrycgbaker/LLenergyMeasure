# Engine-config invariant mining: canonical findings (Phase 1, waves 1-4)

The one-page answer to: can a CHEAP, CI-affordable workflow keep engine-config
INVARIANTS current across upstream version bumps (vLLM / TensorRT-LLM /
transformers)? Per-wave detail: `PHASE1_WAVE{1,2,3,4}_FINDINGS.md`. Prior-PoC
reconciliation: `WAVE4_RECONCILIATION_MAP.md`. The deterministic-baseline /
bump-robustness layer that this builds on: `STUDY_RESULTS.md` + `FULL_MATRIX.md`.

## The question

An "invariant" is a rule the engine enforces at config construction (`if <pred>:
raise/warn`, a Literal/enum field, a cross-field relation). llem must keep this
knowledge current across upstream bumps, cheaply enough to run on every bump in
CI. North-star principles: the ENGINE owns its SSOT; a RUNTIME GATE validates
mined knowledge in-container ("observe, don't re-encode" - construct a violating
config, observe the raise); cost is understood ORDINALLY (deterministic ~free <
small-OSS < mid-OSS < Opus), NOT as a plotted $-frontier; mine COMPREHENSIVELY,
expose a subset (the gate is the SSOT, the LLM is only a candidate generator).

## The method

Candidate generators (deterministic miner, OSS LLM, Opus) propose invariants; the
runtime gate (`scripts/validate_invariants.py`, in the engine's own container)
adjudicates each by constructing a probe and observing whether the engine raises.
The design space (assembly x call-shape x model-tier) was sampled fractionally,
with MANDATORY adversarial source-verification of every cross-field confirm (the
inflation class recurs) and an internals-guard (drop private/underscore fields,
type-trivia, observability, launch-state). Cells: vllm 0.19.1 + tensorrt 1.2.1
(N=2; directional, single-shot). The deterministic floor is the study's
`improved-det-v2`; the GT denominator is each cell's runtime-gated
`ground_truth/<engine>/<vslug>/invariants/PILOT_GT.yaml`.

## What each wave established

**Wave 1 - the bottleneck is VALIDATION, not LLM recall.** det-then-llm-extend
(match-only) yielded 0 gate-confirmed lift over the deterministic floor on both
cells x both rungs (gemma3:12b AND Opus). Not because the LLM found nothing - Opus
surfaced ~50 real net-new cross-field relations - but because the single-field
auto-synthesis gate could not PROBE them. The validation PATH, not recall, was
binding.

**Wave 2 - the kwargs-emission lever unlocks the cross-field tail.** Having the
LLM also emit constructible `kwargs_positive/negative` made the tail
gate-confirmable: Opus 0 -> 8 verified-real cross-field confirms (folded into the
GT - the first cross-field constraints in it). gemma3:12b FAILED the lever (17/25
proposals failed; 0 verified-real). So the tail is REAL but reachable only at Opus
cost on this shape. Surfaced + fixed a gate soundness gap (cross-field confirm
attribution by error locus).

**Wave 3 - scale is the threshold; code-tuning sharpens (the size x tuning 2x2).**
Verified-real cross-field confirms: small models (gemma-12b general, qwen-coder-7b)
= 0; mid/large (qwen-coder-32b = 5, llama-70b = 3) reach it; Opus = 8. SCALE is the
threshold to reach the tail at all (between 12B and 32B); CODE-TUNING sharpens
within the capable regime - a code-tuned 32B BEAT a general 70B on coverage (5 vs
3 cross-field), precision (14/14 vs noise), speed (~2680s vs ~4436s wall), and
cleanliness (zero internal-noise vs the 70B's `_api_process_rank`). Validated the
internals-guard (general 70B needs it most).

**Wave 4 - the OSS strategy frontier (drop the floor, vary STRATEGY).**
- pure-LLM / prompt (4a): tensorrt 100% infra-blocked (0/61 - the LLM omits
  required ctor fields, so pydantic "field required" fires before any validator).
- CONSTRUCTION-GROUNDING is THE OSS lever: inject each class's AST-extracted
  constructor signature (required/optional fields + types) so construction REACHES
  the real validators. Breaks the tensorrt infra wall (qwen-coder-32b 0 -> 20
  verified-real) and lifts vllm precision. It is a det+LLM hybrid: deterministic
  AST does construction-context discovery, the LLM synthesises. Model-specific to
  the qwen2.5-coder line - does NOT generalise to qwen3-coder (MoE) or deepseek.
- AGENTIC (LangGraph) is a POOR strategy for OSS: ollama tool-call flakiness +
  no incremental synthesis, even with devstral. The prior "agentic=0" was an
  all-at-once-harness artefact, not a model finding (re-diagnosed in the
  reconciliation map); exploration is better done DETERMINISTICALLY.
- a second gate soundness fix: reject type-coercion-artifact confirms (a pydantic
  parsing/literal error on the probed field, not the labelled semantic rule).
- residual analysis: the tensorrt "ceiling" is a STUDY-FLOOR ARTEFACT - 20 of the
  25 missed tensorrt GT keys are one class (PluginConfig) whose Literal/enum field
  constraints the study's validator-body floor cannot see, but the PRODUCTION
  `_pydantic_lift.py` already extracts. So production tensorrt recall is materially
  HIGHER than the study's number.
- 70B-vs-32B construction-grounding head-to-head: <PLACEHOLDER - result pending>.

## The synthesised answer

**Production design = deterministic floor (production pydantic-lift) + a
construction-grounded LOCAL mid code-model (~32B, qwen2.5-coder line) + the runtime
gate + the two soundness guards + the internals-guard.** Measured recall:

| cell | floor alone | + construct-grounded LLM net-new | HYBRID |
|---|---|---|---|
| vllm 0.19.1 | 44 (55%) | +11 | **55/80 (69%)** |
| tensorrt 1.2.1 | 35 (57%) | +1 | **36/61 (59%, understated)** |

This stack: reaches ~69% (vllm) / >=59% (tensorrt, the production floor pushes it
materially higher) GT recall; runs locally, reproducibly, cheaply (one free AST
pass + one chunked generation per bump on a local GPU); captures the
cross-field/conditional/entangled-class tail the deterministic floor structurally
cannot reach; and needs Opus only as a small, expensive, non-reproducible TOPPING
for the residual cross-field tail - NOT as the workhorse. "Just call Opus per bump"
keeps the expensive half (the gate is mandatory regardless), pays to re-derive what
det does free + reproducibly, and sacrifices reproducibility.

## Cost story (ordinal - the deliverable, NOT a plotted $-frontier)

deterministic (~free, reproducible) < local OSS code-32B (minutes/bump, local GPU;
qwen2.5-coder:32b ~1100s vllm / ~900s tensorrt per cell) < Opus (API tokens/bump,
non-reproducible). The cheap rungs own the bulk. The research question "does the
cheap rung suffice?" answers: largely YES, with a local 32B-coder, for everything
but the hardest cross-field tail. The code-tuned 32B is the efficiency winner among
OSS - Opus-comparable coverage at local-GPU cost.

## What is VALIDATED vs OPEN

Validated: the harness<->gate integration; validation-path-is-the-bottleneck; the
kwargs lever; the scale-threshold + code-tuning tier story; construction-grounding
as the OSS infra-wall lever; the hybrid recall; two gate soundness fixes; agentic
is wrong for OSS.

Open / future: the 70B-vs-32B construct head-to-head (pending); self-consistency
(k-vote) upside; generalising construction-grounding beyond the qwen-coder line;
closed-loop/gate-as-tool with Opus if the cross-field tail justifies the per-bump
gate-call cost; the north-star self-update / degradation-signal binary on a MAJOR
bump (tensorrt 0.21->1.0) - the actual product property, never yet tested.

## Caveats (honest)

N=2 cells, mostly single-shot, directional - not a frontier point. The study floor
(improved-det-v2) understates the production deterministic miner, so the tensorrt
hybrid number is a lower bound. Cross-field confirms always require adversarial
source-verification. The internals-guard is applied in analysis, not yet in the
miner. The residual ~30-40% of GT that neither floor nor construct-grounded-LLM
reaches is the hard tail for further strategies or the Opus ceiling; some of it
(observability config classes) is arguably internals-guard territory that should
not be in the GT.
