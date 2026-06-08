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

**Mean surfacing-recall: tensorrt 81.3%, vllm 70.8%, transformers 42.5%.**

Reading: tensorrt and vllm now recover ~70-90% of the gate-confirmed surface
deterministically (cheaply), driven by the glob (vllm subpackage) and the
plugin-literal value-capture (tensorrt). transformers sits at ~42% - it received
NO new primitive (the glob/plugin/platform surfaces are tensorrt/vllm-specific;
transformers membership is enum-class-typed, not inline Literal), so its ~58%
gap is the structural tail that needs LLM mining (the study's core thesis).
Precision is 19-47% (mech surfaces 2-5x the GT-confirmed count); the
recall-cost/precision frontier is the next axis.

## Self-confirm recall (gated) - validated on tensorrt 1.2.1

- mech-only self-confirm: **15 -> 40** of 60 (the value-capture fold). All 19
  plugin candidates now confirm (was 0/19, skipped on empty allowlist).
- `new_confirmed_vs_Round0 = 0`: every mech self-confirm is already inside the
  adversarially-reviewed Round-0 GT (trustworthy recovery, not unvetted new
  confirms). The research strategy now recovers the plugin-literal surface
  standalone, no separate production miner needed (the lever-1 goal).
- GATE LIMITATION (vllm): self-confirm gating of vllm 0.19.1 mech produced 100
  infra_errors + only 4 self-confirm, because the gate cannot construct
  subpackage `native_type`s (e.g. `vllm.LoadConfig` really lives at
  `vllm.config.*`). These candidates still SURFACE GT constraints (recall
  unaffected); they just cannot self-gate. Self-confirm fan-out is blocked on
  extending the gate's vllm native_type resolution; recall (above) is not.

## Methodology decisions

- **P10 platform overrides = separate GT-growth bucket** (user-confirmed): they
  are real recall the current GT under-counts, dormant + not cheaply gateable
  (fire only inside `check_and_update_config(vllm_config)`); excluded from the
  main precision denominator so they do not bias the baseline.
- Gating is for GT-GROWTH (net-new confirms); the new primitives added ~0 new
  confirmed (they RECOVER existing GT cheaply - the cost-frontier win - rather
  than grow it).

## Open items

- Rigorous pre/post surfacing-recall lift (run the pre-Round-0b strategy for the
  delta; documented old tensorrt-1.2.1 self-confirm ceiling was 25%).
- Bump-delta-recovery curve (deliverable B): run the frozen baseline across each
  bump-pair, score recovered-vs-changed.
- Extend the gate's vllm native_type resolution to un-block vllm self-confirm.
- Deferred pre-existing fixes: p5 two-sided range collapse (`0<=top_p<=1` loses
  the upper bound, strict-score only); p6 `validate_dtype` GPU-gate false-positive.

## Reproduce

- Surfacing-recall: `scripts/round0b/recall.py` (offline, all cells).
- Self-confirm gate (non-destructive): `scripts/round0b/gate.py --engine E
  --vslug V --version X [--src PATH] [--image IMG]`.
