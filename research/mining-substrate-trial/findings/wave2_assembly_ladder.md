# Wave 2.6 deliverable: assembly-shape ladder (cost-per-recall-pp)

How much recall does each assembly shape buy, at what cost, vs GT? Source:
`wave2_substrate_matrix.json` (det-only) + `wave2_llm_cells.json` (W-G extend,
pure-b). LLM = qwen2.5-coder-7b on 1x A100. Tolerant inv recall vs GT.

## The ladder (invariant recall vs GT)

| cell | det-only (floor) | W-G (floor + 7b extend) | delta | pure-b (7b only) |
|---|---|---|---|---|
| vllm v0.7.3 | 0.513 | 0.513 | +0.000 | 0.118 |
| transformers v4.57.3 | 0.404 | 0.447 | +0.044 | 0.088 |
| vllm v0.19.1 | 0.147 | 0.176 | +0.029 | 0.103 |
| transformers v5.6.2 | 0.416 | 0.426 | +0.010 | 0.050 |
| tensorrt v0.21.0 | 0.270 | 0.286 | +0.016 | 0.016 |
| **mean** | **0.350** | **0.370** | **+0.020** | **0.075** |

Precision moved the OTHER way under W-G on every cell (e.g. transformers v4.57.3
0.630 -> 0.464): roughly 2 precision points lost per recall point gained.

## Cost axis

- det-only: sub-second, no GPU, ~$0 per bump.
- W-G extend / pure-b: ~7 min wall per cell on 1x A100 (qwen-7b), real GPU-$,
  plus the prompt-engineering + non-determinism tax.

## Findings

1. **At <=14B OSS scale, adding the LLM is a NEGATIVE-value rung.** W-G lifts
   recall by a mean +0.020 (max +0.044) while LOWERING precision on every cell and
   adding real per-bump GPU cost. The cost-per-recall-pp is effectively infinite
   once the precision loss + the runtime-gate work needed to clean it are counted.
2. **Pure-LLM (b) is non-viable at this scale.** 0.05-0.12 recall, 4x-30x BELOW
   the deterministic floor. The Wave 1 ~50% pure-extract ceiling (at 70B-q4) does
   NOT survive the drop to 7B; the small model alone reconstructs almost nothing.
3. **The recall ceiling lives in the SUBSTRATE, not the small LLM.** Every cell's
   best number is the floor's or barely above it. Spending the LLM budget on
   extraction at OSS scale is misallocated.
4. **The one assembly the data endorses for LLM use is `llm-then-det-validates`
   (H3), not `det-then-llm-extends` unguarded.** The W-G extensions are ~87-100%
   off-GT (hallucination proxy); they are only safe behind the fixed
   runtime-validate gate. If the LLM is used at all at this scale, it must be a
   GATED extender, never a trusted one.

## Recommended assembly for the production invariants workflow (evidence-backed)

improved-det floor (primary, owns the recall) -> optional small-LLM GATED
extender for the residual (llm-then-det-validates; only worth it if the gate
infra exists to absorb the noise) -> fixed runtime-validate gate (mandatory). The
LLM-extend rung is OPTIONAL and low-value at OSS scale; revisit at frontier scale
(Wave 3).

## Deferred

- `llm-then-det-validates` as a live measured cell (needs the per-engine runtime
  gate containers to actually filter, not just the off-GT proxy).
- `closed-loop-feedback` (h15), `llm-self-consistency` (h11): not run (stubs +
  time). Given the weak single-shot extend, k-vote self-consistency on a 7-14B
  model is unlikely to help and triples cost; low priority.
- Frontier-LLM extend (Claude/GPT, larger OSS): the open question - the OSS-scale
  weakness may not hold at frontier scale.
