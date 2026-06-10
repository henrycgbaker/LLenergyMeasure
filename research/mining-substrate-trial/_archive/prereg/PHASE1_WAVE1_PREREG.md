# Phase 1, Wave 1 - pre-registration (minimal integration probe)

Status: PRE-REGISTERED, NOT yet executed (awaiting sign-off). Per the study
discipline (STUDY_DESIGN Section 9: per-phase pre-registration; locked prompts;
pinned digests; deviation log; no mid-wave architectural changes).

## Objective

The smallest wave that (a) validates the PoC LLM-extend harness integrates with
the NEW runtime-gated 15-cell GT + the production gate (not the PoC's
hallucination proxy), and (b) produces the first cost-recall point + GT-growth
reading for the `det-then-llm-extend` assembly. Deliberately narrow so the
expensive model gradient (32B/70B) is provisioned only after the harness and
metrics are proven on a cheap rung.

This wave does NOT attempt to map the frontier; it de-risks the harness and
calibrates the precision-floor / epsilon inputs for later waves.

## Locked design-space point

- Cells (2, representative of the two pydantic engines + the recall extremes):
  - `vllm 0.19.1` (large mech surface, self-confirm 31)
  - `tensorrt 1.2.1` (plugin-rich, self-confirm 40)
- Tiers (lean, OSS-first):
  - OSS: `gemma3:12b` (Ollama, digest `f4031aab637d1ffa37b4`) - the cheap rung.
  - Opus: via the Agent tool (Anthropic-side, no GPU) - SMALL usage this wave
    (anchor the quality ceiling on the 2 cells only; not a broad sweep).
- Role: `extract` (LLM proposes invariants from the engine source).
- Assembly: `det-then-llm-extend` (det floor = improved-det-v2 mech output; LLM
  proposes residual invariants the det floor missed).
- Call-shape: `single` (one shot per cell per model; no k-vote/agentic).
- Locked prompt: `findings/wave2_locked_prompts/wg_extend_prompt.md`
  (sha256 prefix `18251cb6979bde77`). No prompt edits mid-wave; a change = a new
  wave.

## Integration changes required (the actual wave-1 work, before any model call)

1. RE-POINT scoring to the new GT: score recall/precision/GT-growth against each
   cell's committed `ground_truth/<engine>/<vslug>/invariants/PILOT_GT.yaml`
   (the runtime-gated, now-folded GT), via `gt_scoring.py`.
2. REPLACE the hallucination proxy with the REAL gate: LLM-proposed invariants
   are confirmed by `scripts/validate_invariants.py` in-container (reuse the
   `round0b/gate.py` / `trial_scoring.runtime_validate_invariants_dispatch`
   path - tensorrt GPU, vllm CPU). A "confirmed" LLM invariant = the gate's
   positive-raises + negative-doesn't + (hardened) attribution, same bar as the
   self-confirm fan-out.
3. Ollama endpoint: harness hardcodes `:11435`; live Ollama is `:11434` - fix.
4. Cell map: add `tensorrt 1.2.1` (harness currently lists vllm 0.19.1 +
   tensorrt 0.21.0).
5. Opus rung: invoke via the Agent tool with the same locked prompt; capture the
   proposed invariants and gate them identically.

## Metrics (recorded per cell x model)

- Recall: tolerant (headline) + strict, vs the cell GT (`gt_scoring`).
- Precision / hallucination: gate-confirmed proposed / total proposed
  (gate-rejected = hallucination rate). This is the REAL gate, not the proxy.
- GT-growth: gate-confirmed LLM invariants whose constraint-key is NOT in the
  cell GT (classified genuine-new vs variance vs mislabel, as in the fan-out).
- LLM lift over the det floor: recall(det + LLM-confirmed) - recall(det only).
- Cost: OSS wall-sec + GPU-energy (Wh) on the A100; Opus = token count / calls.
- Failure-mode tag (closed vocab, WAVE2_PROTOCOL): silent / detectable / crash /
  hallucinated-from-empty / under-emit / over-emit / gate-rejected-most.

## Readout (what wave 1 must answer)

- Does the harness run end-to-end against the new GT + real gate on both cells?
  (integration pass/fail; defects logged, not patched mid-wave.)
- First cost-recall point for det-then-llm-extend at the 12B rung + the Opus
  ceiling on 2 cells.
- Does the cheap rung clear the deterministic floor at all (the open production
  question)? Directional only at N=2 cells.

## Discipline / provenance

- Pinned: model digest (above), container image digests (the engine images used
  by the gate), locked prompt hash (above).
- Deviation log: any change from this pre-registration is appended here with
  rationale; no silent mid-wave edits.
- Spend posture: lean OSS-first (approved). 32B/70B NOT provisioned this wave;
  provision only if wave 1 shows the rung gradient is worth mapping.

## Out of scope for wave 1 (later waves)

Other roles (extend-residual as its own role, gate, diagnose, diff-review,
curate); other assemblies (llm-only, closed-loop, self-consistency, ensemble-
vote, det-then-llm-patches-det); other call-shapes (k-vote, chunked, chained,
agentic); the 32B/70B rungs; the remaining 13 cells; bump-shape x assembly
block. Stopping-rule K + epsilon + precision-floor get calibrated once wave 1
shows the metric ranges.
