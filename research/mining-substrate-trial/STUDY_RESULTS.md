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

## 5. Methodology validation

- GT INTEGRITY: every confirmed entry in the three new cells was opened at its
  cited source line by an independent adversarial reviewer instructed to refute.
  Result: 62/63 REAL (0.21 17/18, 1.0 21/21, 1.1 24/24), zero false-confirms, zero
  fabrications; the one non-real entry is a redundant mis-encoding of a rule its
  twin captures correctly. (1.2.1 separately: 100% real on a ~22-entry sample.)
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

- IDENTITY under-merge: constraint grain does not merge cross-source re-encodings
  of the same rule (divergent predicate values), so confirmed ENTRY counts run
  ~30% over distinct RULES (~12 behind 0.21's 18). A citation-keyed canonicalisation
  pass would tighten this; filed open. (Distinct from the original re-base, which
  fixed the opposite OVER-collapse.)
- Confirmed-level percentages are small-N (14-60 knobs); the Opus-basis gradient
  (64-81 knobs) is the robust signal.
- ONE major boundary in the window (tensorrt 0.21->1.0); vllm and transformers
  have no major boundary in the locked window, so cross-engine generalisation is
  tested only for minor-bump stability (not yet run).
- The deterministic ceiling used different mechanical miners per cell; the headline
  25%/47% is the 1.2.1 figure (improved-det-v2 + production widening).
- Replayability gating: GPU/model-dir/engine-dir-gated invariants are real but
  excluded from the confirmed denominator; field renames show as drop+add, so true
  semantic re-bounding is slightly understated.

## 8. Open items

1. Identity-layer canonicalisation (citation-keyed) to reconcile cross-source
   re-encodings; fix the one mis-encoded 0.21 `lora_ckpt_source` numeric predicate.
2. Cross-engine minor-bump deltas (vllm 0.18->0.22, transformers 5.6->5.10) to test
   whether minor-bump stability generalises across engines.
3. Per-version producers exist for ~6 versions; the locked window needs ~9 more
   (overlaps the engine-knowledge-as-data refactor).
4. Port the trial improved-det-v2 primitives into the production per-version miners
   (deferred to milestone end).

## 9. Cross-references

- Objective, method, locked parameters, execution log: `STUDY_DESIGN.md` (esp.
  Section 15).
- Chronological findings + per-number detail: `findings/study/FANOUT_FINDINGS.md`.
- Per-cell GT, reports, metrics: `findings/study/ground_truth/tensorrt/<v>/invariants/`.
- Predecessor strategy bake-off + the LLM-role-split recommendation:
  `RESEARCH_WRITEUP.md`.
- Gates: `scripts/validate_schema.py`, `scripts/validate_invariants.py`; union+gate
  driver: `scripts/study_gt_pilot.py`.
