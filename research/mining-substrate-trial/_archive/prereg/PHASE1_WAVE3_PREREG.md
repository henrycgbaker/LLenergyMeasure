# Phase 1, Wave 3 - pre-registration (tier-bridge: does bigger OSS / better instructions reach the cross-field tail?)

Status: PRE-REGISTERED, NOT yet executed (awaiting sign-off). Discipline per
STUDY_DESIGN Section 9. Builds directly on waves 1-2
(`PHASE1_WAVE1_FINDINGS.md`, `PHASE1_WAVE2_FINDINGS.md`).

## The question this wave answers (and why it comes first)

Wave 2 localized the bottleneck to the VALIDATION PATH and then showed the
kwargs-emission lever unlocks the cross-field/conditional tail - but ONLY at
Opus cost. The small OSS rung (gemma3:12b) FAILED the lever: 0 verified-real
cross-field confirms across both cells, 20 of 28 deduped proposals failed or
infra-errored, precision 0.08. The wave-2 readout named the open question
explicitly: **does a mid (~32B) or large (~70B) OSS model bridge the gemma->Opus
gap, or is correct cross-field kwargs-emission an Opus-tier capability cliff?**

This wave answers THAT, before opening the broader size x shape matrix. It is the
single open question on the one shape (det-then-llm-extend + kwargs) we have
already validated end-to-end. The answer is load-bearing for the whole phase: if
a cheap-ish OSS rung bridges the gap, the production design for the tail is cheap;
if not, the tail is Opus-only and the matrix's tier axis is largely settled on
this shape. Either way it directly serves the north-star cost question ("does the
cheap rung suffice, or do you need the big model?") at full tier resolution.

No agentic framework (LangGraph/LangChain) is needed or used: the shape is
unchanged from wave 2 (single-shot det-then-llm-extend + kwargs). Framework setup
is deferred to the later shapes (closed-loop / self-consistency / agentic) that
actually require it.

## Locked design point

- Cells: vllm 0.19.1 + tensorrt 1.2.1 (SAME as waves 1-2 - direct comparison to
  the gemma baseline and the Opus ceiling already measured on these cells).
- Tiers swept THIS wave (the new datapoints): qwen2.5-coder:32b (mid, code-tuned)
  and llama3.1:70b (large). gemma3:12b (small, DONE wave 2) and Opus 4.8 (ceiling,
  DONE wave 2) are the fixed anchors of the 4-point curve.
- Role / assembly / call-shape: extract / det-then-llm-extend / single - UNCHANGED
  from wave 2. This isolates the TIER axis: everything but the model is held fixed.
- Locked prompt: `phase1_wave2/wg_extend_kwargs_prompt.md` (sha256 prefix
  `7cd74960eab09e19`) - the EXACT wave-2 kwargs-emission prompt. No change, so a
  lift is attributable to scale alone (Arm A).
- Harness: `scripts/phase1/wave1.py --rung oss --model <tier> --prompt-file <locked
  prompt>`. OSS source is chunked (same chunker as gemma); the gate is the real
  production gate via `study_gt_pilot` load+gate. Floor = improved-det-v2.
- Held-constant confound (logged, not corrected this wave): OSS rungs are chunked
  (16k-ctx chunker, same as gemma); Opus saw whole-source. This is consistent
  WITHIN the OSS tier sweep (gemma/32B/70B all chunked), so the OSS-internal scale
  comparison is clean; Opus remains the whole-source ceiling anchor as in wave 1.

## Arms (staged; attribution-clean per the conditional decision)

- ARM A - TIER SWEEP, locked prompt (PRIMARY): run qwen2.5-coder:32b and
  llama3.1:70b on the locked wave-2 prompt, both cells. NO prompt change, NO gate
  change. A confirm lift over gemma is attributable to scale alone. This is the
  headline: where on the gemma(0) -> Opus(8) curve do 32B and 70B land?
- ARM B - BETTER INSTRUCTIONS (CONDITIONAL on Arm A residual): run ONLY if the
  best OSS tier (70B) still falls materially short of Opus on verified-real
  cross-field confirms. Then draft an improved kwargs-construction prompt (more
  worked construction examples / clearer step-by-step / few-shot correct
  kwargs_positive-negative pairs - NO new predicate semantics) and re-run the OSS
  tiers. A lift here is attributable to instructions, cleanly separated from scale
  because Arm A held the prompt fixed. The improved prompt is drafted+hashed+logged
  here before it runs (no mid-wave improvisation); if Arm B fires, this prereg is
  amended with the new prompt hash before execution.

## Metrics (per cell x tier)

Primary (the capability signal):
- verified-real cross-field confirms: gate-confirmed cross-field proposals that
  PASS mandatory adversarial source-verification (the only count that means "the
  model produced a constructible probe that fires the LABELLED rule"). This is the
  apples-to-apples number vs gemma(0) and Opus(7 vllm + 1 tensorrt).
- failed count: probe ran, rule did not fire = the over-claim / hallucination
  signal (gemma's was high: 17/25 vllm).
- infra_error count: probe unconstructible (the entangled-config residual).
- gate-confirmed precision = confirmed / gateable (gemma vllm 0.08, the precision
  floor to beat).

Secondary:
- total gate-confirmed (incl single-field) and llm_lift_over_floor (most single-
  field surface is floor-owned; expected ~0, as in waves 1-2).
- GT-growth: the 8 wave-2 cross-field confirms are ALREADY folded into the GT, so
  a tier re-confirming them scores as recall, not growth; any NET-NEW verified-real
  cross-field confirm (a rule Opus did NOT surface) is genuine GT-growth and is
  folded by the same wave-2 procedure (cross-field attribution fix in place).
- cost: OSS wall-sec per cell (GPU-energy proxy, single A100, single-tenant);
  no token cost for OSS. The cost curve: det(~0) < gemma(~500s, 0 real) < 32B(?)
  < 70B(?) < Opus(~206k tok, 8 real).

## MANDATORY soundness gate (load-bearing, non-negotiable)

The gate's positive-confirm attribution is single-field; CROSS-FIELD confirms
BYPASS it. The wave-2 error-locus attribution fix (commit `fix(gate): attribute
cross-field confirms by error locus`) now rejects a cross-field confirm whose
positive raised a FIELD-level pydantic error - but that fix is a guard, NOT a
substitute for review. Therefore: EVERY cross-field confirm at EVERY tier is
adversarially verified by an independent Opus reviewer against source (the raised
error IS the labelled rule, not an unrelated one) BEFORE it is counted as real or
folded. This is the exact inflation class prior rounds caught (1 spurious in wave
2). A tier's headline number is its VERIFIED-real count, never its raw confirm
count.

## Readout / decision gate

Let R(tier) = verified-real cross-field confirms summed over both cells.
Anchors: R(gemma)=0, R(Opus)=8.

- BRIDGED-CHEAP: R(32B) ~= R(Opus) at acceptable precision -> the cross-field tail
  is reachable at mid-OSS cost. Production design for the tail can use a 32B local
  model; large cost win over Opus. Record cheapest sufficient rung = 32B.
- BRIDGED-LARGE: R(32B) short but R(70B) ~= R(Opus) -> the tail needs 70B-class
  capability (cheaper than Opus API but needs the local GPU). Record cheapest
  sufficient rung = 70B.
- CLIFF (Arm A) -> Arm B: both OSS tiers fall materially short on the locked
  prompt. Run Arm B (better instructions). If Arm B closes it: the gap was
  instruction-shaped, not capability - record the prompt as the lever. If Arm B
  does NOT close it: correct cross-field kwargs-emission is an Opus-tier capability
  cliff; the tail is Opus-only on this shape. This CONFIRMS+sharpens the wave-2
  conclusion at full tier resolution and CLOSES the det-then-llm-extend tier
  question. "Move on" then = the broader size x shape matrix (other assemblies /
  pure-LLM / richer call-shapes), or accept Opus-only for the cross-field tail.

In all branches: record findings (PHASE1_WAVE3_FINDINGS.md), fold any net-new
verified-real growth, then decide the next move WITH the user (do not auto-launch
the broader matrix).

## Discipline / provenance

- Pinned: locked prompt hash `7cd74960eab09e19`; models qwen2.5-coder:32b +
  llama3.1:70b (digests recorded at run time in the findings); gemma3:12b digest
  `f4031aab637d1ffa37b4`; Opus 4.8 via Agent tool; engine container digests as in
  wave 2 (vllm/vllm-openai:v0.19.1 CPU; nvcr.io tensorrt-llm/release:1.2.1 --gpus).
- Deviation log appended to the findings; no mid-wave prompt/gate changes (Arm B is
  pre-registered + re-hashed before it runs, not a mid-wave improvisation).
- Spend: lean. 2 OSS models x 2 cells = 4 chunked Ollama runs (Arm A); Arm B is
  conditional and adds at most 4 more. No Opus model spend (the Opus anchor is the
  wave-2 result, reused); Opus tokens are spent only on the mandatory adversarial
  verification of OSS cross-field confirms.

## Out of scope (this wave; later in the phase)

Other assemblies (pure-LLM / llm-only, llm-then-det-gate, closed-loop,
self-consistency, ensemble-vote); other call-shapes (k-vote, chunked-as-variable,
chained, agentic); other roles (gate, diagnose, diff-review, curate); the
remaining 13 cells; the agentic-orchestration framework. Those are the broader
size x shape matrix, opened only AFTER this tier-bridge question is answered and
re-confirmed with the user. The strongest parallel north-star item (the tensorrt
0.21->1.0 self-update / degradation-signal binary) is also out of scope here and
remains available as an alternative next move.
