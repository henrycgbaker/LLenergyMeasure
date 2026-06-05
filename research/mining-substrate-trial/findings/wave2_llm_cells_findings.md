# Wave 2 LLM-involving cells - findings

Machine-readable records: `wave2_llm_cells.json`.
Locked prompts: `wave2_locked_prompts/{wg_extend_prompt.md, pure_b_prompt.md}`.
Dispatch: minimal direct Ollama `POST /api/generate` (stream off, temp 0,
num_ctx 16384, num_predict 4096) at http://localhost:11435. Models run
SEQUENTIALLY (single A100). The registered w2-h* strategy stubs were NOT
touched.

## Cells run / skipped

All 5 priority cells ran for BOTH W-G extend (qwen2.5-coder-7b) and pure-b
(qwen-7b): transformers v4_57_3 + v5_6_2, vllm v0_7_3 + v0_19_1, tensorrt
v0_21_0. Model-scale sweep (llama3.1-8b + phi4-14b) ran on vllm v0_7_3.
Nothing skipped. One transient pure-b crash (transformers v5_6_2, malformed
`match` scalar) was fixed (string-typed match guard) and re-run cleanly.
Total LLM wall: ~2060s (~34 min). All models fp16; digests in JSON.

Dispatch hardening that mattered: capped `num_predict` (uncapped runs blew
20+ min on one chunk via runaway generation) and a truncation-tolerant
per-entry YAML parser (small models emit unquoted-colon scalars and get cut
mid-string at the token cap).

## Headline table - tolerant invariant recall vs GT (qwen-7b)

| cell | floor | +LLM (W-G) | delta | pure-b | floor prec | +LLM prec |
|---|---|---|---|---|---|---|
| vllm/v0_7_3 | 0.513 | 0.513 | +0.000 | 0.118 | 0.438 | 0.382 |
| transformers/v4_57_3 | 0.404 | 0.447 | +0.044 | 0.088 | 0.630 | 0.464 |
| vllm/v0_19_1 | 0.147 | 0.176 | +0.029 | 0.103 | 0.286 | 0.154 |
| transformers/v5_6_2 | 0.416 | 0.426 | +0.010 | 0.050 | 0.609 | 0.467 |
| tensorrt/v0_21_0 | 0.270 | 0.286 | +0.016 | 0.016 | 0.395 | 0.286 |

(strict numbers in JSON; tolerant is the headline per the harness contract.)

## Does LLM-extend close the residual?

Barely, and at a precision cost. W-G recall lift over the deterministic floor
is +0.000 to +0.044 (mean +0.020) tolerant. Per wall-sec the lift is tiny:
the best case (transformers v4_57_3, +0.044) cost 228s for +37 deduped LLM
entries of which only a handful landed in GT. EVERY cell's precision DROPPED
under +LLM (e.g. transformers v4_57_3 0.630 -> 0.464; vllm v0_19_1 0.286 ->
0.154) - the 7B model adds far more off-GT entries than on-GT ones. There is
NO cost-free recall here: at 7B the marginal LLM step trades ~2 precision
points for every recall point.

Pure-b (LLM-only, no floor) is decisively worse: 0.016-0.118 recall, i.e.
4x-30x below the floor. The Wave 1 "~50% LLM ceiling at 70B-q4" does NOT hold
at 7B - the small model alone reconstructs almost none of the catalogue. The
deterministic floor is doing essentially all the work; the LLM is a weak
extender, not a standalone miner, at this scale.

## Model-scale (vllm v0_7_3, W-G extend)

| model | recall | +llm entries | wall |
|---|---|---|---|
| floor (det-only) | 0.513 | 0 | 0s |
| qwen2.5-coder-7b | 0.513 | 13 | 81s |
| llama3.1-8b | 0.566 | 37 | 355s |
| phi4-14b | 0.566 | 38 | 260s |

Scale helps but saturates: 7B adds zero recall (its 13 entries all miss GT or
land in the wrong predicate bucket), 8B and 14B both reach 0.566 (+0.053 over
floor). 8B is the knee - 14B costs no extra recall over 8B and only modest
extra wall. So for the extend role the useful floor is ~8B; below that the LLM
contributes nothing on this cell.

## Hallucination PROXY (label: PROXY, not a true gate)

Proxy = LLM-proposed entries whose (leaf_field, coarse_bucket) is NOT in GT /
total LLM-proposed. W-G proxy rates: vllm/v0_7_3 1.00, transformers/v4_57_3
0.865, vllm/v0_19_1 0.954, transformers/v5_6_2 0.957, tensorrt/v0_21_0 0.950.
These are high but OVER-count: GT is a documented MINIMUM set and the coarse
bucket collapse is lossy, so "not in GT" conflates true fabrication with
real-but-uncatalogued and with predicate-bucket drift. Direction is the
signal: the small model emits mostly-unverifiable entries, which is exactly
why a downstream gate (not GT-scoring) is the right filter.

## True runtime-validate gate (transformers, best-effort)

`runtime_validate_invariants_dispatch(engine='transformers',
version_slug='v4_57_3')` is FUNCTIONAL in the project venv (transformers
4.57.3 present; validator constructs live `GenerationConfig` and the library
emits real warnings). BUT it requires per-entry `kwargs_positive` /
`kwargs_negative` replay fields to exercise each invariant - and the
token-economical Wave 2 prompts (floor included) omit those fields, so the
gate returns an infra-error per entry for ALL wave2 catalogues (floor, W-G,
pure-b alike). This is NOT an LLM-specific limitation: the engine BASELINE
catalogue, which DOES carry replay kwargs, gates live. Implication: to run the
true gate on these cells the extend/pure-b prompts must re-emit
kwargs_positive/negative (a cheap prompt change, deferred here). vllm +
tensorrt gates need per-engine containers: GATE DEFERRED.

## Single most decision-relevant finding

For the downstream workflow design: at small/mid model scale (<=14B OSS) the
LLM is a WEAK EXTENDER and a NON-VIABLE standalone miner. The deterministic
improved-det floor supplies ~all recoverable recall; a 7B extend adds ~0,
an 8B extend adds ~+0.05 at a real precision cost, and pure-LLM collapses to
near-zero. So Workflow 2 should be det-floor-FIRST with the LLM bolted on only
as a GATED extender (llm-then-det-validates), never as the primary miner, and
only if a runtime gate is wired to absorb the ~90%-off-GT extension noise.
The recall ceiling worth chasing lives in the substrate, not the small LLM.
