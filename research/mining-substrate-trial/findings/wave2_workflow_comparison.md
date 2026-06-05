# Wave 2.6 deliverable: workflow-candidate comparison

The 6-7 production workflow shapes (WAVE2_WORKFLOWS) compared on the Wave 2
evidence. Wave 2 does NOT pick a workflow; this is the evidence base the
engineering session uses. Tolerant inv recall vs GT; self-update from the
bump-survivability analysis.

## Comparison

| workflow | detect / extract | recall vs GT (v_old) | bump-survival | per-bump cost | self-update | verdict |
|---|---|---|---|---|---|---|
| **W-A status quo** | landmark + hand producer | n/a (hand) | FAILS all 3 bumps (stale citations, hand class lists) | human PR/bump | no (by construction) | the baseline to beat |
| **W-B pure universal substrate** | re-extract via improved-det | 0.40-0.51 | inherits vllm CLIFF (0.51->0.15), no recovery; tensorrt rises | ~0 (sub-sec) | partial (degrades silently) | cheap, but bump-fragile + no degradation signal |
| **W-C pure LLM** | LLM re-extracts | 0.05-0.12 (7B) | n/a | LLM/bump | no (too low to trust) | NON-VIABLE at OSS scale; needs frontier (deferred) |
| **W-D LLM patches producer** | AST/LLM diff -> patch | untested (Wave 1: 0/3) | unknown | LLM + retry | mixed | low priority; small-model patching is poor |
| **W-E universal floor + LLM extend** | tree-sitter floor + LLM | ~0.20-0.43 floor + ~+0.02 | weak | LLM/bump | partial | dominated by W-G (tree-sitter is a weaker floor than improved-det) |
| **W-G improved-det floor + LLM extend** | improved-det floor + LLM | 0.40-0.51 floor + ~+0.02 | floor-limited; LLM extend does NOT recover the cliff at 7-14B | ~0 floor + LLM | partial | **best FLOOR; LLM-extend marginal at OSS scale** |

## The headline revision

Going into Wave 2, W-G (improved-det floor + LLM extend) was the strongest a-priori
candidate. The evidence SPLITS that claim:

- **The floor half of W-G is confirmed strongest:** improved-det is the best
  deterministic floor (Section 3, frontier deliverable), and it should be the
  primary recall engine.
- **The LLM-extend half is NOT supported at OSS scale:** +0.02 mean recall, every
  cell loses precision, ~0.9 hallucination proxy. The small LLM does not close the
  residual.

So the production invariants workflow that the evidence supports is NOT
"improved-det floor + LLM extend (W-G as originally framed)" but rather:

**improved-det floor (+ a new declarative-`Field` Primitive 8) as the primary
recall engine; the LLM relegated to GATE + DIAGNOSE/DIFF-REVIEW roles (the
self-update degradation signal), with any LLM-proposed entry passing the fixed
runtime-validate gate; LLM-as-extractor reserved for a frontier-scale re-test.**

This is a hybrid of W-G's floor + W-F's diagnose role + the H3 gated-extend
shape - assembled from the evidence, not picked off the menu. The engineering
session owns the final choice.

## Self-update dimension (per workflow)

- W-A: fails (landmark/citation pinning breaks on every bump).
- W-B / W-E / W-G: pattern-matched floors degrade gracefully (partial recall, no
  crash) but signal nothing on collapse. Need an external detector.
- The detector the evidence points to: an LLM diff-reviewer (cheap, OSS-viable,
  the role small models ARE good at) + the runtime-gate acceptance rate.

## Deferred / unmeasured

- All workflows' bump-UPDATE binary (auto-propose a producer/catalogue patch that
  passes the gate with no human edit) - needs per-version engine containers.
- W-C / W-G LLM half at frontier scale.
- W-D live cells.
