# Wave 2.6 deliverable: substrate cost-recall frontier (per task)

Source data: `findings/wave2_substrate_matrix.json` (10/18 cells scored vs GT),
`findings/wave2_substrate_analysis.json` (complementarity).
Scoring: `gt_scoring` tolerant identity (headline) + strict (lower bound).
All recall/precision figures are vs the Opus-established ground truth (GT), the
Wave 2 SSOT. "Tolerant" drops namespace + collapses predicate-kind to coarse
buckets (the strict<->tolerant gap is convention drift, not genuine miss - see
`wave2_deviations.md`).

## Cost axis

The two static substrates measured here are near-zero cost: sub-second per cell
(matrix `wall_sec` 0.05-0.20 s), no GPU, no API, no per-version vendoring. Cost
per bump = one CI second. The LLM end of the frontier is characterised
separately in `wave2_assembly_ladder.md` + `wave2_model_scale_curve.md`.

## Invariant-recall frontier (tolerant, vs GT)

| engine | version | tree-sitter | improved-det | best static |
|---|---|---|---|---|
| transformers | v4.57.3 | 0.202 | **0.404** | improved-det |
| transformers | v5.6.2 | 0.208 | **0.416** | improved-det |
| vllm | v0.7.3 | 0.434 | **0.513** | improved-det |
| vllm | v0.19.1 | 0.118 | **0.147** | improved-det |
| tensorrt | v0.21.0 | n/a | 0.270 | improved-det |
| tensorrt | v1.2.1 | n/a | 0.400 | improved-det |

(tree-sitter substrate does not support tensorrt; see coverage gaps.)

## Schema-recall frontier (tolerant, vs GT)

| engine | version | tree-sitter | improved-det |
|---|---|---|---|
| transformers | v4.57.3 | 0.366 | 0.374 |
| transformers | v5.6.2 | 0.351 | 0.428 |
| vllm | v0.7.3 | 0.615 | **0.972** |
| vllm | v0.19.1 | 0.519 | 0.519 |
| tensorrt | v0.21.0 | n/a | 0.635 |
| tensorrt | v1.2.1 | n/a | 0.685 |

## Findings

1. **improved-det is the dominant static floor.** It beats tree-sitter on
   invariant recall on every shared cell (transformers ~2x: 0.40 vs 0.20) and on
   schema recall (vllm v0.7.3 0.972 vs 0.615). At equal (near-zero) cost there is
   no reason to prefer tree-sitter; improved-det is the cheap-end frontier point.

2. **The cheap end tops out around 0.40-0.51 invariant recall vs GT** on stable
   versions (improved-det: transformers 0.40, vllm-v0.7.3 0.51, tensorrt-v1.2.1
   0.40). This is the deterministic ceiling the GT exposes: the residual ~0.5-0.6
   of invariants are NOT mechanically catchable by the current 7 primitives.
   Closing that residual is the LLM-extend question (W-G).

3. **Schema recall is far easier than invariant recall** at the cheap end
   (improved-det schema 0.37-0.97 vs invariants 0.15-0.51). Field enumeration is
   substantially solved deterministically; predicate/invariant mining is not.
   The production workflow should treat the two tasks asymmetrically - a cheap
   det substrate can largely own schema, while invariants need the LLM tail.

4. **Precision is the unaddressed axis.** Tolerant invariant precision is
   0.23-0.63 (matrix): the static substrates OVER-emit (raise-sites that are not
   LLEM-scope invariants). At the proposal stage looser precision is acceptable
   (per the production constraints), but this is exactly the surface the
   fixed runtime-validate gate must clean before vendoring. The frontier's
   precision cost is borne by the gate, not the substrate.

## Coverage gaps (deferred, infra-bound)

- tree-sitter substrate: no tensorrt support (registry `engines`).
- framework-reflection (pydantic-native), runtime-trace, behavioural-fuzz: need a
  per-version importable engine = per-version GPU container; not runnable
  GPU-free this session. Their frontier points are DEFERRED to a container run.
  Expectation (from the v0.21->v1.2 finding that surface is migrating into
  pydantic): framework-reflection should be competitive-to-better than tree-sitter
  on the pydantic-heavy engines (vllm v0.19, tensorrt v1.2) and is worth running.
