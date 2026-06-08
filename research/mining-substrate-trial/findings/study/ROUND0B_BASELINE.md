# Round 0b: deterministic baseline (in progress)

The cheap deterministic miner (improved-det-v2 + Round 0b primitives) measured
against the gate-confirmed Round-0 GT. Builds on improved-det-v2 - does NOT
restart. Authoritative GT denominator: each cell's committed
`ground_truth/<engine>/<vslug>/invariants/PILOT_GT.yaml` `confirmed` list.

## Primitives added (this round)

- (d) **Generalised subpackage glob** - `_expand_files` de-pins `defs_files` /
  `validator_files` to globs; reaches vllm `config/*.py` after the v0.19.1
  subpackage split (pinned `config.py` found 0 there). Commit 2b58dede.
- (e) **Lever-1 plugin-literal fold** - scan tensorrt `plugin/plugin.py` +
  module-level `Literal` alias resolution; **capture the Literal allowed VALUES**
  so membership candidates are gate-probeable (an empty allowlist is unprobeable
  -> skipped). Commits 2b58dede, 4d1c3b13.
- (b) **Per-platform `check_and_update_config` walker (Primitive 10, vllm)** -
  emit an invariant per VllmConfig sub-config field a platform silently overrides
  or names in a conditional raise. Commit 39eee6ad.
- RE-SCOPE (evidence-based, adversary-validated): (c) validator-body predicate is
  already covered by the existing p5/p6 walk (top_p/top_k/temperature/... all
  emitted) - skipped to avoid re-derivation. (a) default-indirection is
  schema-side (default values), zero invariant-recall cost - deferred.

## Two recall metrics (do not conflate)

1. **Surfacing-recall (offline, no container):** does the cheap method SURFACE
   the GT-confirmed constraints? `|mech tolerant-keys INTERSECT GT-confirmed
   tolerant-keys| / |GT-confirmed|`. This is deliverable-A recall - the method
   need not self-gate (the union already validated each constraint).
2. **Self-confirm recall (gated, container):** does the mech candidate
   independently re-gate (construct + probe fires)? Stricter; the
   deterministic-ONLY ceiling. Requires the runtime gate.

## Surfacing-recall - all 15 cells (offline, post-Round-0b)

| cell | GT-conf (tolerant) | mech surfaced | recovered | recall % | precision % |
|---|---|---|---|---|---|
| tensorrt 1.2.1 | 60 | 115 | 54 | 90.0 | 47.0 |
| tensorrt 0.20  | 11 | 23  | 8  | 72.7 | 34.8 |
| tensorrt 0.21  | 14 | 48  | 9  | 64.3 | 18.8 |
| tensorrt 1.0   | 17 | 55  | 15 | 88.2 | 27.3 |
| tensorrt 1.1   | 23 | 63  | 21 | 91.3 | 33.3 |
| vllm 0.18.1    | 75 | 227 | 52 | 69.3 | 22.9 |
| vllm 0.19.1    | 66 | 232 | 46 | 69.7 | 19.8 |
| vllm 0.20      | 92 | 257 | 66 | 71.7 | 25.7 |
| vllm 0.21      | 97 | 264 | 68 | 70.1 | 25.8 |
| vllm 0.22      | 93 | 275 | 68 | 73.1 | 24.7 |
| transformers 5.6.2  | 61 | 73 | 26 | 42.6 | 35.6 |
| transformers 5.7.0  | 58 | 73 | 20 | 34.5 | 27.4 |
| transformers 5.8.1  | 63 | 73 | 29 | 46.0 | 39.7 |
| transformers 5.9.0  | 75 | 74 | 32 | 42.7 | 43.2 |
| transformers 5.10.2 | 64 | 75 | 30 | 46.9 | 40.0 |

**Mean surfacing-recall (TOLERANT, leaf+bucket): tensorrt 81.3%, vllm 70.8%,
transformers 42.5%.** This measures FIELD COVERAGE - does the cheap method point
at the right field, in the right coarse bucket?

**Value-aware companion (STRICT, leaf+bucket+canonical predicate value):
tensorrt 33.2%, vllm 24.8%, transformers 2.9%** (per cell: trt 1.2.1 55.4, 0.20
28.6, 0.21 27.8, 1.0 33.3, 1.1 20.8; vllm 21-29; tf 0-9.5). This measures whether
the EXACT constraint (field + value/bound/allowlist) is captured.

Reading (corrected after adversarial review): the headline is **field coverage,
not exact-constraint capture**. The cheap deterministic method is strong at
finding WHICH fields are constrained (tolerant 70-90% for trt/vllm) but weak at
WHAT the constraint is (strict 25-33%) - it captures the surface, the exact
values/bounds/allowlists largely need the gate (to confirm) or LLM (to encode).
The strict number is a lower bound (also penalised by mech-vs-Opus predicate
ENCODING variance - the same under-merge caveat as FULL_MATRIX Section 3; do not
read it as pure miss). transformers is low on BOTH grains (tolerant 42, strict 3)
- its miss surface is presence-dominated + semantically conditional + absent from
the mechanical source, i.e. the genuine LLM tail. Precision is 19-47% (mech
surfaces 2-5x the GT count); the recall-cost/precision frontier is Phase 1.

## Pre/post lift (surfacing-recall vs the pre-Round-0b strategy at 0d679c22)

| engine | OLD recall | NEW recall | lift |
|---|---|---|---|
| tensorrt | 75.0% | 81.3% | +6.3 (plugin-literal fold) |
| vllm | 55.7% | 70.8% | +15.1 (config-subpackage glob) |
| transformers | 42.5% | 42.5% | +0.0 (no applicable primitive) |

Corrected reading (adversarial review): **vllm +15 is broad and SOUND** - every
vllm cell gains +13 to +16 from the generalised glob reaching the `config/*.py`
subpackage. **tensorrt "+6" is a single-cell effect, not an engine figure**: 4 of
5 tensorrt cells are +0; the entire lift is 1.2.1 (+31.7), the only cell with a
substantial plugin-literal surface. transformers is flat on both grains (no
applicable primitive). So the durable Round 0b primitive win is the vllm
subpackage glob; the plugin fold helps one tensorrt cell (and shows up more in
its self-confirm, below).

## Bump-delta-recovery (deliverable B)

Of the constraints NEW in B's gate-confirmed GT vs A's, the fraction the
UNCHANGED miner recovers when run on B - no edit to its code or landmark list.
Tolerant grain.

| bump | added | recovered | recovery % |
|---|---|---|---|
| tensorrt 0.20->0.21 | 4 | 1 | 25.0 |
| tensorrt 0.21->1.0 | 7 | 6 | 85.7 |
| tensorrt 1.0->1.1 | 8 | 7 | 87.5 |
| tensorrt 1.1->1.2.1 | 37 | 33 | 89.2 |
| vllm 0.18->0.19 | 9 | 3 | 33.3 |
| vllm 0.19->0.20 | 28 | 20 | 71.4 |
| vllm 0.20->0.21 | 12 | 8 | 66.7 |
| vllm 0.21->0.22 | 9 | 7 | 77.8 |
| transformers 5.6->5.7 | 9 | 1 | 11.1 |
| transformers 5.7->5.8 | 15 | 9 | 60.0 |
| transformers 5.8->5.9 | 12 | 3 | 25.0 |
| transformers 5.9->5.10 | 1 | 1 | 100.0 |

**Mean bump-delta-recovery (TOLERANT / FIELD grain): 61.1% over 12 bumps**
(tensorrt/vllm minors 67-89%). This is FIELD-level tracking: does the unchanged
miner surface the newly-appearing FIELDS across a bump, with no landmark edit -
the self-updating property the generalised glob provides. Driver:
`scripts/round0b/bump_delta.py`.

CAVEATS (adversarial review): (1) read this as field-tracking only - the
value-aware version drops to ~19% AND reintroduces the same-leaf-new-bucket
ENCODING-CHURN confound that got the "survivor re-bound" metric RETRACTED
(FULL_MATRIX Section 3), so value-grain bump-delta is deliberately NOT claimed.
(2) several pairs are small-N (N<=4, N=1), so the per-pair percentages are noisy
and the mean is unweighted; the tolerant-key denominator also deflates the true
added count (e.g. vllm 0.18->0.19 collapses 27 real new constraints to 9). Treat
61% as "field-level tracking is good on minors," not a precise value-recovery
rate.

## Self-confirm recall (gated) - validated on tensorrt 1.2.1 + vllm 0.19.1

- tensorrt 1.2.1 mech-only self-confirm: **15 -> 40** of 60 (the value-capture
  fold). All 19 plugin candidates now confirm (was 0/19, skipped on empty
  allowlist).
- `new_confirmed_vs_Round0 = 0` (tensorrt 1.2.1): every mech self-confirm is
  already inside the adversarially-reviewed Round-0 GT (trustworthy recovery,
  not unvetted new confirms). The research strategy now recovers the
  plugin-literal surface standalone, no separate production miner needed (the
  lever-1 goal).
- GATE FIX (vllm), RESOLVED: the gate previously could not construct vllm
  subpackage `native_type`s - the miner tags every class `vllm.<Class>` but
  after the 0.19.1 config-subpackage split they live at `vllm.config.*`, so the
  naive `getattr(vllm, X)` infra-errored. Fixed with a `_construct_vllm`
  module-probe resolver (mirrors `_construct_trtllm`; commit on
  `spike/engine-knowledge-as-data`). On vllm 0.19.1 mech (315 candidates):
  **infra_error 100 -> 4, confirmed 4 -> 32**. The residual 4 are classes
  needing semantically entangled required args (SchedulerConfig,
  SpeculativeConfig, StructuredOutputsParams) deliberately NOT fabricated.
- vllm 0.19.1 GT-growth (adversarially verified, honest wording): of the 11
  net-new gate-confirmed constraint-keys vs Round-0 GT, **8 are real pydantic
  `Literal[...]` allowlists** verified against source and reproduced
  in-container (cache_dtype, mamba_ssm_cache_dtype, moe_backend,
  seq_pooling_type, tok_pooling_type, logprobs_mode, mm_tensor_ipc, plus
  publisher which is a value-encoding variant of an existing GT entry);
  `max_pattern_size < 0` is a real numeric bound. Two of the eleven are
  bucket-mislabels of bool fields (cudagraph_mm_encoder tagged numeric,
  disable_additional_properties tagged membership) that confirm via pydantic
  bool-type rejection rather than the labelled constraint, and one sibling
  invariant (`max_pattern_size > 0`) is a spurious confirm that slips past
  attribution on an unrelated min_count error message. **The net-new ckey count
  is not inflated** (the spurious sibling collapses to the real `<0` key), but
  per-invariant verdicts include two type-vs-predicate mislabels and one
  misattributed confirm. Do NOT claim "~10 net-new validated constraints"
  unqualified - that is a mild overclaim.
- DEFECTS EXPOSED by unblocking the vllm gate (pre-existing, were hidden while
  every vllm candidate infra-errored; carried forward, NOT fixed here):
  1. ATTRIBUTION too weak - `validate_invariants.py` positive-confirm
     attribution is a bare substring test (`probe_leaf in haystack`), so a raise
     whose message merely mentions the field name for an unrelated reason passes
     (the `max_pattern_size > 0` spurious confirm). Affects confirm
     trustworthiness across ALL engines, not just vllm.
  2. MINER bucket-mislabel - bare `bool` fields get numeric/membership buckets
     (cudagraph_mm_encoder, disable_additional_properties); they should be
     `type` constraints so their confirms aren't mislabelled.

## Methodology decisions

- **P10 platform overrides = separate GT-growth bucket** (user-confirmed): they
  are real recall the current GT under-counts, dormant + not cheaply gateable
  (fire only inside `check_and_update_config(vllm_config)`); excluded from the
  main precision denominator so they do not bias the baseline.
- Gating is for GT-GROWTH (net-new confirms); the new primitives added ~0 new
  confirmed (they RECOVER existing GT cheaply - the cost-frontier win - rather
  than grow it).

## Status (2026-06-08): SURFACING baseline locked; self-confirm + value-grain pending

Scoped honestly after adversarial review (verdict: the directional conclusions
hold - deterministic mining is cheap and strong at FIELD coverage for trt/vllm,
the subpackage glob is the real win, transformers is the LLM-tail engine - but
the value-blind headlines were inflated; corrected above).

DONE: field-coverage (tolerant) surfacing-recall baseline + bump-delta field
curve + pre/post lift (all 15 cells, offline), lever-1 self-confirm validated on
tensorrt 1.2.1, and the vllm subpackage-resolver gate fix validated on vllm
0.19.1 (infra_error 100 -> 4, self-confirm 4 -> 32; adversarially reviewed).

NOT yet done (so this is NOT the full "deterministic baseline locked" of
STUDY_DESIGN Section 7):
- VALUE-AWARE (strict) metrics are the weaker, truer picture (recall 33/25/3) and
  rest on value-grain capture the miner is poor at; the p5 two-sided-range
  collapse (`0<=top_p<=1` keeps only one bound) is a direct cause and is NOT
  cosmetic - it is exactly the value-grain the baseline's strict number needs.
- SELF-confirm exists for 2 of 15 cells (tensorrt 1.2.1 + vllm 0.19.1). The vllm
  gate is now UNBLOCKED (resolver fix above), so fan-out across the remaining 13
  cells is mechanically possible; P10 platform candidates are 0/16 confirmed
  (dormant, not cheaply gateable - the GT-growth bucket); tensorrt mech also
  shows 25 gate FAILS worth a look. Fan-out is gated on whether to first harden
  the attribution check (see DEFECTS above) for trustworthy per-invariant
  confirms.

Carried forward (to finish the baseline / Phase-1 prereqs):
- VALUE-GRAIN ALIGNMENT (the real blocker, attempted + reverted 2026-06-08):
  lifting strict recall is NOT a quick p5 fix. mech records the FIRING op; the
  GT records the VALID constraint, AND `gt_adapter.canonical_predicate_value`
  renders mech's `{'>=':0,'<=':1}` as the malformed `{<==1,>==0}` (matches
  neither the Opus `{ge=0,le=1}` nor the bare `0`). Single bounds already match
  only via the GT's lossy bare-value duplicate. Properly fixing this = the
  deferred "mechanical miner emits operator-ful canonical values" producer item
  (STUDY_DESIGN 15.4 item 6): align mech's predicate encoding to the Opus/GT
  canonical form. A naive firing->valid inversion regressed tensorrt 55->53 and
  was reverted. Also p6 `validate_dtype` GPU-gate false-positive.
- Extend the gate's vllm subpackage `native_type` resolution: DONE (the
  `_construct_vllm` module-probe resolver). Remaining: full self-confirm fan-out
  across the other 13 cells.
- HARDEN the gate attribution check (bare-substring -> structured: require the
  raise to name the probed field as the constraint subject, e.g. pydantic
  `loc`/`input_value`). Pre-existing; surfaced by the vllm unblock. Affects
  confirm trustworthiness across all engines.
- FIX miner bucket assignment for bare `bool` fields (numeric/membership ->
  type), so confirms are not mislabelled (cudagraph_mm_encoder,
  disable_additional_properties).
- Investigate the 25 tensorrt gate-fails + the 19 plugin self-confirms' value
  fidelity.
- Producer-porting into the production per-version miners (STUDY_DESIGN 15.4
  item 6, milestone-end).

## Reproduce

- Surfacing-recall: `scripts/round0b/recall.py` (offline, all cells).
- Self-confirm gate (non-destructive): `scripts/round0b/gate.py --engine E
  --vslug V --version X [--src PATH] [--image IMG]`.
