# Phase 1, Wave 2 - pre-registration (validation-path lever)

Status: PRE-REGISTERED, NOT yet executed (awaiting sign-off). Discipline per
STUDY_DESIGN Section 9. Builds directly on wave 1
(`PHASE1_WAVE1_FINDINGS.md`): wave 1 showed det-then-llm-extend yields ZERO
gate-confirmed lift over the deterministic floor on both cells x both rungs,
because the LLM's net-new invariants are dominated by cross-field / conditional /
presence forms the single-field auto-synthesis gate cannot probe (and even
hand-authored cross-field kwargs hit construction infra). Wave 1's localized
conclusion: **the bottleneck is the validation path, not LLM recall, nor model
tier.**

## Objective

Test whether upgrading the VALIDATION PATH unlocks gate-confirmation of the LLM's
cross-field / conditional / presence tail that wave 1 left at 0 confirmed. This
isolates the lever wave 1 identified, holding the model rung and cells fixed.

Decision this wave informs: does the LLM's rich tail become VALIDATED knowledge
(real GT-growth) once the gate can probe it - or is the auto-gate ceiling the
binding constraint on cheap LLM value? Only if the tail unlocks is a tier sweep
(32B/70B) or broader assembly/role sweep worth the spend.

## Locked design point

- Cells: vllm 0.19.1 + tensorrt 1.2.1 (SAME as wave 1 - direct comparison).
- Rung: Opus (Agent tool, whole-source single call/cell). Wave 1 showed tier is
  not the lever, so isolate on the ceiling rung that surfaces the richest tail.
  gemma3:12b is a SECONDARY cross-check (can a cheap model emit gateable kwargs
  at all?), not the headline.
- Role / assembly / call-shape: extract / det-then-llm-extend / single (same).
- Locked prompt: `phase1_wave2/wg_extend_kwargs_prompt.md`
  (sha256 prefix `7cd74960eab09e19`) - wave-1 prompt + a requirement to emit
  constructible `kwargs_positive`/`kwargs_negative` per invariant.

## Arms (staged; attribution-clean)

- BASELINE = wave-1 Opus result (match-only, single-field auto-synth): 0
  confirmed lift. Same cells/rung - direct comparison.
- ARM A - kwargs-emission (NO gate code change): Opus re-prompted with the
  locked wave-2 prompt to emit constructible probe pairs. The existing gate
  honours hand-authored kwargs, so this tests how much of the tail unlocks from
  the LLM simply PROVIDING the probe (solving the cross-field synthesis-scope
  gap). Gate = current `validate_invariants.py`.
- ARM B - construction-robust gate (CONDITIONAL on Arm A residual): if Arm A
  still `infra_error`s on a material fraction (the entangled-config construction
  problem - the same residual seen in the self-confirm fan-out and the wave-1
  spot-test), build the deferred required-args injection (a per-engine
  `_REQUIRED_KWARGS` map, analogous to the TRT-LLM model placeholder) so the
  entangled config classes construct, and RE-gate Arm A's proposals. This is the
  one code change wave 2 may require; it is scoped to construction, not new
  predicate semantics.

## Metrics (per cell x arm)

- gate-confirmed count of the wave-1 0-confirmed tail; decomposed:
  - kwargs-emission lift = confirmed(Arm A) - confirmed(wave-1)
  - construction-robustness lift = confirmed(Arm B) - confirmed(Arm A)
- residual after each arm: skipped (still unsynthesizable) + infra_error (still
  unconstructible) + failed (probed, did not hold) - the failed count is the
  hallucination/over-claim signal now that probes are runnable.
- GT-growth of newly-confirmed, CLASSIFIED genuine-new-field / encoding-variance
  / bucket-mislabel (the wave-fan-out method); only genuine-new is growth.
- cost: Opus tokens (kwargs-emission enlarges output); gate wall.

## MANDATORY soundness gate (load-bearing)

The gate's positive-confirm attribution is single-field; CROSS-FIELD confirms
BYPASS it (a multi-field `match` makes `_leaf_field` None). So a cross-field
positive can fire for the WRONG reason and confirm. Therefore: EVERY cross-field
confirm in wave 2 is adversarially verified against source (an independent Opus
reviewer confirms the raised error is the labelled rule, not an unrelated one)
BEFORE it is counted as real or folded. Non-negotiable - this is the exact
inflation class prior rounds caught.

## Readout / decision gate

- If Arm A (and/or B) confirms a MATERIAL, adversarially-verified cross-field
  tail absent from the floor: the LLM's value is REAL and the production design
  is "kwargs-emitting LLM + construction-robust gate"; proceed to a tier sweep
  (cheapest rung that clears the floor on this path) and the secondary gemma
  cross-check becomes load-bearing.
- If it does NOT unlock (still ~0 confirmed, or confirms collapse under
  adversarial review): the auto-gate ceiling is the binding constraint; the LLM
  cross-field tail is not cheaply validatable, and the cost-frontier conclusion
  is that deterministic + gate owns the validatable surface. Record and stop the
  det-then-llm-extend line; reconsider assembly (e.g. llm-as-diagnose/diff-review
  where validation is human-facing, not gate-confirmed).

## Discipline / provenance

- Pinned: locked prompt hash (above); model = Opus 4.8 via Agent tool +
  gemma3:12b (digest `f4031aab637d1ffa37b4`); engine container digests.
- Deviation log appended here; no mid-wave prompt/gate changes (Arm B is
  pre-registered, not a mid-wave improvisation).
- Spend: still lean - Opus on 2 cells (re-prompt is one call/cell), gemma cheap.
  32B/70B NOT provisioned. Arm B is bounded code + a re-gate, no new model spend.

## Out of scope (later waves)

Tier sweep (32B/70B); other roles (gate, diagnose, diff-review, curate); other
assemblies (closed-loop, self-consistency, ensemble-vote); other call-shapes;
the remaining 13 cells. Those follow only if wave 2 shows the tail is reachable.
