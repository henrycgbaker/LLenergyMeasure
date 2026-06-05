# Wave 2.6 deliverable: substrate complementarity matrix

Question: do different substrates catch DISJOINT ground-truth entries (so their
union is much larger than the best single substrate), or does one subsume the
others? If union >> max, a production workflow should run several substrates and
union them; if one subsumes, run only the best.

Source: `findings/wave2_substrate_analysis.json` (tolerant GT invariant keys
caught per substrate, per cell). Static substrates only (tree-sitter,
improved-det); the det-vs-LLM complementarity is in `wave2_assembly_ladder.md`.

## Invariant-key complementarity (caught keys vs GT, tolerant identity)

| engine / version | GT keys | tree-sitter only | both | improved-det only | union recall | max-single recall | union gain |
|---|---|---|---|---|---|---|---|
| transformers v4.57.3 | 114 | 5 | 18 | 28 | 0.447 | 0.404 | +0.044 |
| transformers v5.6.2 | 101 | 3 | 18 | 24 | 0.446 | 0.416 | +0.030 |
| vllm v0.7.3 | 76 | 5 | 28 | 11 | 0.579 | 0.513 | +0.066 |
| vllm v0.19.1 | 68 | 0 | 8 | 2 | 0.147 | 0.147 | +0.000 |

(tensorrt: only improved-det runs, so no pair to compare.)

## Findings

1. **improved-det largely SUBSUMES tree-sitter.** Across all four paired cells,
   tree-sitter catches only 0-5 GT invariant keys that improved-det misses. The
   union beats the best single substrate by only +0.03 to +0.07 recall. This is
   the opposite of high complementarity: the two static substrates are NOT a
   productive ensemble. improved-det's 7 primitives are a near-superset of
   tree-sitter's universal-walker coverage, by construction (improved-det reuses
   the tree-sitter infra and adds 6 targeted passes).

2. **Therefore: do NOT union the two static substrates in production.** Run
   improved-det alone as the deterministic floor. The ~3-5 tree-sitter-only
   entries per cell are not worth a second substrate pass; they are better
   recovered by the LLM-extend tail (which targets the same residual).

3. **Static-substrate complementarity is LOW; the complementarity that matters
   is det-vs-LLM.** The interesting union question is improved-det (cheap,
   ~0.40-0.51 recall) unioned with a small-LLM extend pass over the residual.
   That measurement is `wave2_assembly_ladder.md` (W-G assembly). The a-priori
   expectation set by this matrix: since the static substrates exhaust their
   mutual coverage quickly, the marginal recall must come from a DIFFERENT
   modality (LLM semantic resolution, or framework-reflection's runtime view),
   not from more static passes.

4. **At the vllm v0.19.1 bump the union gain is exactly 0** (both substrates
   collapse to the same 8-10 keys, all overlapping). When the surface migrates to
   declarative `Field(...)` constraints that neither static substrate parses, a
   second static substrate adds nothing - the failure is shared, not
   complementary. This reinforces that bump-robustness needs a new MODALITY
   (a declarative-constraint primitive, or the LLM), not substrate redundancy.

## Deferred

- framework-reflection x static complementarity: framework-reflection reads
  runtime `__fields__` / `model_fields`, a genuinely different view from source
  AST, so it is the most likely static substrate to be COMPLEMENTARY (catch
  entries the source-walkers miss, e.g. dynamically-registered fields). Running
  it needs per-version GPU containers; DEFERRED. This is the highest-value
  complementarity cell still unmeasured.
