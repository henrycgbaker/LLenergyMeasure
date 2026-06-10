# Engine-config mining study: results (5-version window)

Standalone synthesis of the 5-version-window study for the milestone record.
Companion to `STUDY_DESIGN.md` (objective + method + execution log, Section 15)
and `findings/study/FANOUT_FINDINGS.md` (chronological detail). Builds on the
predecessor bake-off in `RESEARCH_WRITEUP.md`, whose central recommendation -
"LLM proposes, a deterministic runtime gate disposes" - this study operationalises
and stress-tests with a real runtime gate and a runtime-anchored ground truth.

## Abstract

We ask whether a CHEAP CI workflow can keep well-validated upstream engine-config
knowledge (parameter SCHEMA + validation INVARIANTS) current across version bumps
of three inference engines, using a HEAVY-DETERMINISTIC miner plus a CHEAP
runtime GATE that re-validates mined knowledge against the live engine in-container
("observe, don't re-encode"), reserving the LLM for the residual. Two findings.
(1) The cost frontier: schema discovery is ~1.0 deterministic; invariant mining
is not - bare deterministic mining confirms 25% of the runtime-verified invariant
GT, a cheap walk-surface lever lifts it to ~47%, and a structural tail resists
cheap determinism. (2) Bump robustness: across the tensorrt 0.21->1.0->1.1->1.2.1
window, a MAJOR version bump churns ~8x as much mined knowledge as a minor one
and silently re-bounds ~40% of the knobs that survive, whereas minor bumps leave
~92% intact. The re-bounding is the load-bearing result: stale knowledge across a
major bump is WRONG, not merely incomplete, so the cheap runtime gate is necessary,
not optional. Ground truth was built from two independent Opus source-reading
passes per cell, runtime-gated, and adversarially source-reviewed (62/63 confirmed
entries verified real across the three new cells).

## 1. What was tested

The predecessor bake-off established that the productive division of labour is
LLM-as-proposer + deterministic-gate-as-disposer. The open product question it
left: can that architecture actually keep knowledge CURRENT cheaply across upstream
bumps, or is heavy per-bump LLM unavoidable? That decomposes into:

- COST FRONTIER: how much of the real config-validation knowledge can the CHEAP
  deterministic half mine + verify, and where does the LLM become unavoidable?
- BUMP ROBUSTNESS: how much mined knowledge survives a version bump (cheap to keep
  current) vs must be re-mined or silently goes wrong (the gate's job)?

## 2. Method in brief

Two product gates, both runtime, both engine-agnostic in shape:
- SCHEMA gate (`scripts/validate_schema.py`, link 1->2): re-runs the engine's own
  introspector in-container and diffs discovered-vs-live {exists, type, default}
  on a semantic-type basis, plus enum construct-probes.
- INVARIANT gate (`scripts/validate_invariants.py`, link 1->2, the canonical
  product gate): constructs each config type in the engine's container with a
  positive (must-fire) and negative (must-pass) probe; an invariant is CONFIRMED
  only if it behaves as declared. Probes are synthesised from the declared
  predicate when no kwargs are authored; an attribution guard forces non-confirm
  unless the raised message names the field under test.

Ground truth (the study denominator) is the gate-confirmed UNION of independent
methods - two Opus source-reading passes (entry-point call-graph + class-hierarchy
walk), the mechanical miner, and the PoC GT - NOT the mechanical output, so
"mechanical recall vs GT" is non-tautological. Identity is at the CONSTRAINT grain
`(leaf_field, coarse_bucket, canonical_predicate_value)`. All GPU work runs in the
engine's release container via `docker run --gpus all`.

## 3. Finding 1 - the cost frontier (deterministic ceiling)

Measured on tensorrt 1.2.1 against the frozen runtime-confirmed GT (60 confirmed /
212 union constraints), at constraint grain:

| split | constraints | % of confirmed GT |
|---|---|---|
| mechanical (improved-det-v2) CONFIRMS | 15 | 25% |
| mechanical SURFACES (strict ckey) | 15 | 25% |
| surfaced-but-unconfirmed gap | 0 | 0% |
| + one cheap walk-surface lever (PluginConfig walk) | 28 | 47% |

- The probe-synthesis gap is ZERO at constraint grain: everything the miner
  surfaces, the gate confirms. The earlier "74% surfaced" was a tolerant-identity
  collapse artefact (it reproduces exactly as the tolerant-key reach, 34/46 =
  73.9%). The entire deterministic deficit is MINING SCOPE, not probing.
- One cheap deterministic lever (walk one more module surface, PluginConfig) lifts
  confirmed recall 25% -> 47% AND grows the GT by +13 runtime-confirmed constraints
  the prior union lacked (the non-tautology mechanism: a new method surfaces real
  invariants, so GT itself grows; re-gated to 74 confirmed / 228 union).
- SCHEMA discovery is ~1.0 deterministic (reflection is engine truth): transformers
  4.57.3 107/107 clean; vllm 0.7.3 116/135; tensorrt 0.21.0 92/107 - and the
  divergences are shipped-schema staleness vs refactored introspectors, not engine
  drift.

So the heavy-deterministic-mine half holds outright for schema and for ~half of
invariants cheaply; the invariant remainder has a structural tail (cross-field,
abstract-config, context-dependent) that needs LLM mining. The VERIFY half is cheap,
deterministic, AND reliable (zero probe gap) throughout.

## 4. Finding 2 - bump robustness (the gradient)

> SUPERSEDED + CORRECTED by `findings/study/FULL_MATRIX.md` (the full 15-cell,
> 3-engine matrix). Two changes there: (a) the gradient is now well-powered across
> 14 bumps and the major-vs-minor PERSIST contrast holds (major 53% vs minors
> 76-100%); (b) the "survivor re-bound rate" headlined below (42% / 36%) is
> RETRACTED - it is confounded by predicate-encoding variance across agents/sources
> (transformers 5.8->5.9, byte-identical source, shows 0% rebound while a
> PoC-folded cell shows 80%). Only field-level PERSIST is trustworthy; the
> runtime-gate argument rests on field-level churn, not a rebound rate.

Full tensorrt window, GT per cell from 2 Opus passes (+ mech/PoC where present),
runtime-gated:

| cell | union | confirmed |
|---|---|---|
| 0.21.0 | 164 | 18 |
| 1.0.0 | 123 | 21 |
| 1.1.0 | 84 | 24 |
| 1.2.1 | 228 | 74 |

Cross-version knob delta on the OPUS basis (passA+passB, present in every cell -
apples-to-apples, free of the per-cell mech/PoC source-availability confound that
distorts the raw union):

| bump | persist | dropped | re-bounded survivors |
|---|---|---|---|
| 0.21 -> 1.0 (MAJOR) | 43/81 = 53% | 38 | 18/43 = 42% |
| 1.0 -> 1.1 (minor) | 60/64 = 94% | 4 | 12/60 = 20% |
| 1.1 -> 1.2.1 (minor) | 65/71 = 92% | 6 | 9/65 = 14% |

- The gradient ISOLATES the major-boundary spike: the major bump drops or changes
  ~47% of mined knobs; both minor bumps persist ~92-94%. The major churns ~8x a
  minor.
- RE-BOUNDED knobs (same field, CHANGED valid-set/bound) are the silent-staleness
  cases - a knob still present but with a different constraint, where stale mined
  knowledge is WRONG, not just incomplete. Even among SURVIVING knobs, the major
  bump re-bounds 42% vs 14-20% on minors. These are exactly what the runtime gate
  catches by re-validating each carried-over constraint against the live engine.
- 1.1.0 sits with 1.0 structurally (pre-pydantic PluginConfig, no SamplingParams
  range checks, `validate_build_config` raises rather than warns); the big 1.2.x
  feature additions land after 1.1, so 1.1 -> 1.2.1 is addition-dominated on a
  stable base.

CROSS-ENGINE (vllm 0.18.1 -> 0.19.1; the window has no non-tensorrt major
boundary, so this probes minor-bump behaviour on a second engine). Two cells, 2
Opus passes each, runtime-gated CPU-only, fully source-reviewed (vllm 0.18.1 =
145 union / 94 confirmed; 0.19.1 = 249 / 90). Opus-basis persistence is 78%
(31/87 = 36% of survivors re-bounded). Two reads:
- The 78% does NOT mean vllm churns more "for the same kind of bump": vllm uses
  0.x versioning where the minor digit is the BREAKING-change position, so
  0.18->0.19 is effectively a feature release, not a tensorrt-style 1.x minor.
  vllm's minor landing between tensorrt's minor (92-94%) and major (53%) is what
  the versioning conventions predict.
- The robust ENGINE-INDEPENDENT signal is survivor RE-BOUNDING: 36% of persisting
  vllm knobs changed bound/allowlist, close to the tensorrt MAJOR's 42% and far
  above tensorrt minors'. Silent re-bounding is not a tensorrt quirk, so the
  runtime gate's necessity generalises across engines.

## 5. Methodology validation

- GT INTEGRITY: every confirmed entry across six cells was opened at its cited
  source line by an independent adversarial reviewer instructed to refute. Result:
  **243/247 REAL** - tensorrt 62/63 (0.21 17/18, 1.0 21/21, 1.1 24/24) + vllm
  181/184 (0.18.1 91/94, 0.19.1 90/90), zero false-confirms, zero fabrications
  anywhere. The 4 non-real are a redundant mis-encoding (0.21) and 3 imprecise
  recorded predicate_values (vllm 0.18.1) - all REAL rules, none wrong invariants.
  (1.2.1 separately: 100% real on a ~22-entry sample; the vllm 0.19.1 reviewer
  re-ran 50+ entries end-to-end in-container to verify fire-for-the-right-reason.)
- GATE SOUNDNESS: reviewers independently confirmed the attribution-hardening
  closes the "CUDA-incidental-error" hole (args-model construction touches CUDA
  unconditionally via `validate_dtype`, but the gate only confirms when the raised
  message names the field), and that the gate conservatively excludes lazy /
  not-auto-invoked / GPU-gated rules rather than mis-confirming them.
- NON-TAUTOLOGY: GT is the Opus+runtime union, not the mechanical output; the Opus
  passes are GT contributors (denominator), the mechanical miner is the cheap
  method under evaluation.

## 6. Product implication (north star)

The architecture - heavy-deterministic MINE + cheap runtime-GATE verify, LLM for
the residual - holds, with a sharp split:
- VERIFY is the cheap, reliable, always-on half. The runtime gate is necessary on
  MAJOR bumps (it is the only thing that catches the ~40% silent re-bounding of
  surviving knobs) and remains cheap on minors (~92% of knowledge is stable).
- MINE is where cost lives. Deterministic mining gets schema outright (~1.0) and
  ~half of invariants cheaply (walk-surface widening); a major bump additionally
  requires RE-MINING the dropped/changed surface; the structural invariant tail
  and the per-major reorganisation are where LLM mining is unavoidable.
- A minor bump is cheap (mostly mine-new on a stable base + a cheap gate sweep); a
  major bump is the expensive event (re-mine + heavier LLM) but is also rare.

## 7. Limitations and caveats

- IDENTITY under-merge (a COUNT-PRECISION caveat, NOT a validity threat; MEASURED).
  Confirmed ENTRY counts overstate distinct RULES because the same field+rule can
  split across sources by ENCODING variance. This introduces no wrong invariant
  (every confirmed entry was source-verified real) and changes neither headline
  finding (the ceiling conclusion is qualitative; the 53%-vs-92% gradient is far
  too large to flip on a count wobble, and matches on the tolerant key which
  re-merges most variants). Measured on the four tensorrt cells (join confirmed
  ckeys back to source citations): the genuine duplicates are ~5 behind 0.21's 18
  (-> ~13 distinct), fewer elsewhere, and are DOMINATED by the mechanical miner's
  LOSSY bare-value encoding (it emits `0` for `max_ngram_size > 0`, dropping the
  operator) not matching the Opus passes' operator-ful form (`{gt=0}`); the same
  bare value cannot be safely merged with `{gt=0}` because it could equally have
  meant `>=0` or `==0`.
  No safe identity-layer fix exists, and the citation-keyed merge once mooted is
  REJECTED: distinct constraints frequently share one source function (the three
  Lookahead fields under a single `validate_positive_values`; several SamplingParams
  rules under one `_validate`), so keying identity on file+qualname would
  re-introduce the OVER-collapse the original re-base fixed. The over-split is the
  correct fail-safe. The real root cause is upstream - the mechanical miner's
  encoding - so the right fix is PRODUCER-side (emit operator-ful canonical values),
  folded into the deferred miner-porting item (Section 8), not an identity-layer
  change. A precise distinct-rule count, if ever needed, is a per-cell manual
  reconciliation.
- GATE SCOPE (what "confirmed" guarantees): the runtime gate verifies binary
  fire/pass BEHAVIOUR (a bad value fires, a good value passes) but does NOT
  cross-check the recorded `predicate_value` against source. So an entry can
  confirm correctly while its recorded allowlist/bound is slightly off, whenever
  the probe kwargs straddle the true boundary (the 3 vllm 0.18.1 mis-stated entries
  were exactly this: e.g. a 6-value allowlist recorded for an 8-value Literal still
  confirms). Confirmation establishes the constraint EXISTS and is roughly located,
  not that its boundary is exact; a precise-boundary guarantee would need the gate
  to probe AT the recorded edges.
- Confirmed-level percentages are small-N (14-60 knobs); the Opus-basis gradient
  (64-111 knobs) is the robust signal.
- ONE major boundary in the window (tensorrt 0.21->1.0); vllm and transformers
  have no major boundary in the locked window. Cross-engine generalisation is
  therefore tested for MINOR-bump behaviour only - done for vllm (0.18->0.19),
  with the caveat that vllm 0.x minors are semver-breaking (feature releases), not
  comparable to tensorrt 1.x minors.
- The deterministic ceiling used different mechanical miners per cell; the headline
  25%/47% is the 1.2.1 figure (improved-det-v2 + production widening).
- Replayability gating: GPU/model-dir/engine-dir-gated invariants are real but
  excluded from the confirmed denominator; field renames show as drop+add, so true
  semantic re-bounding is slightly understated.

## 8. Open items

1. Cross-engine: vllm 0.18->0.19 DONE (Section 4). Remaining: transformers
   5.6->5.10 minor-bump delta; and, if a non-tensorrt MAJOR is ever wanted, it
   must come from outside the locked window.
2. Per-version producers exist for ~6 versions; the locked window needs ~9 more
   (overlaps the engine-knowledge-as-data refactor).
3. Port the trial improved-det-v2 primitives into the production per-version miners
   (deferred to milestone end).
4. OPTIONAL POLISH (not validity-bearing, see Section 7): citation-keyed identity
   canonicalisation to make confirmed counts equal distinct rules; would need
   adversarial re-validation of the identity layer first. Measure the twin count
   before deciding it is worth the risk.

## 9. Cross-references

- Objective, method, locked parameters, execution log: `STUDY_DESIGN.md` (esp.
  Section 15).
- Chronological findings + per-number detail: `findings/study/FANOUT_FINDINGS.md`.
- Per-cell GT, reports, metrics: `findings/study/ground_truth/tensorrt/<v>/invariants/`.
- Predecessor strategy bake-off + the LLM-role-split recommendation:
  `RESEARCH_WRITEUP.md`.
- Gates: `scripts/validate_schema.py`, `scripts/validate_invariants.py`; union+gate
  driver: `scripts/study_gt_pilot.py`.
