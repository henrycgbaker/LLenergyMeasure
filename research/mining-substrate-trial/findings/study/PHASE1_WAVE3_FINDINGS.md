# Phase 1, Wave 3 - findings (tier-bridge: the model-size x code-tuning 2x2)

Status: COMPLETE. Pre-registration: `PHASE1_WAVE3_PREREG.md`. Artifacts:
`phase1_wave3/results/w3_<model>_<cell>.json` (per tier x cell), `*_corpus.yaml`
(deduped proposals + kwargs), `*_CONFIRMED.yaml` (re-gated confirmed detail).
Runner: `scripts/phase1/wave1.py` (+ `wave3_tier_sweep.sh` driver,
`wave3_dump_confirmed.py` re-gate-and-dump, `wave3_reparse_lenient.py`).

## The question

Wave 2 showed the kwargs-emission lever unlocks the cross-field tail but only at
Opus cost; gemma3:12b FAILED it (0 verified-real cross-field). Open question: does
a bigger / better OSS model bridge the gemma->Opus gap - and is the lever SCALE or
CODE-TUNING? This wave fills a model-size x tuning 2x2, holding the prompt
(locked wave-2 kwargs prompt `7cd74960`), cells (vllm 0.19.1, tensorrt 1.2.1),
assembly (det-then-llm-extend), and gate fixed - so a lift is attributable to the
model alone.

## Result: the 2x2 (verified-real, candidate-config CROSS-FIELD confirms)

|              | general                | code                   |
|--------------|------------------------|------------------------|
| **small**    | gemma-12b -> **0**     | qwen-coder-7b -> **0** |
| **mid**      | -                      | qwen-coder-32b -> **5**|
| **large**    | llama-70b -> **3**     | -                      |
| **ceiling**  | Opus 4.8 -> 8 (wave 2) |                        |

Two headline findings:

1. **SCALE is the threshold to reach the cross-field tail at all.** Both SMALL
   models (7B-code AND 12B-general) get 0 verified-real cross-field; both LARGE
   models (32B-code, 70B-general) reach it. The capability threshold sits between
   12B and 32B. Code-tuning at 7B does NOT rescue it.
2. **CODE-TUNING sharpens within the capable regime.** qwen-coder-32B beats
   llama-70B-general on every axis: more cross-field (5 vs 3), higher confirm
   precision (14/14 = 1.00 vs 6/6 but 1 was internal noise), faster
   (~2680s vs ~4436s total wall), and cleaner (zero internal-noise proposals vs
   the 70B's `_api_process_rank`). **A code-tuned 32B > a general 70B for this task.**

## Per-tier detail (verified-real = gate-confirmed AND adversarially source-verified)

| tier | cell | deduped | confirmed | verified-real | cross-field (cand-config) | raw precision | wall |
|---|---|---|---|---|---|---|---|
| gemma-12b (s/gen) | vllm | 25 | 2 (unverified) | 0 | 0 | 0.08 | 502s |
| gemma-12b | tensorrt | 3 | 0 | 0 | 0 | 0 | 154s |
| qwen-7b (s/code) | vllm | 56* | 1 | **0 (spurious)** | 0 | 0.018 | 607s |
| qwen-7b | tensorrt | 22* | 0 | 0 | 0 | 0 | 166s |
| qwen-32b (m/code) | vllm | 88 | 12 | **12** | 5 | 0.136 | 1986s |
| qwen-32b | tensorrt | 28 | 2 | **2** | 0 (2 single-field) | 0.071 | 694s |
| llama-70b (l/gen) | vllm | 46 | 4 | **4** (1 internal) | 3 | 0.087 | 3079s |
| llama-70b | tensorrt | 8 | 2 | **2** | 0 (2 list-checks) | 0.25 | 1357s |

*qwen-7b: see "format-following failure" below - its 208 raw entries were lenient-reparsed.

- **qwen-32B (14/14 real):** 5 cross-field (structured-outputs exactly-one;
  data_parallel_size_local <= data_parallel_size; data_parallel_external_lb
  requires dp>1; data_parallel_rank in [0,dp_size); max_cpu_loras >= max_loras) +
  7 single-field (penalty/top_p/min_p/logprobs ranges, min_count>=2, eplb
  interval) + 2 tensorrt single-field (allowed_backends, max_attention_window).
- **llama-70B (6/6 real, 5 candidate-config):** 3 cross-field candidate-config
  (data_parallel_size_local, data_parallel_rank, max_cpu_loras - re-finds of the
  32B set) + `capture_num_tokens` positive-ints + `max_attention_window` non-empty
  + 1 INTERNAL (`_api_process_rank >= _api_process_count`, filtered, see guard).

## The internals-guard (validated this wave)

Per the comprehensive-discovery intent (mine the full surface, expose a subset
downstream), the only quality filter wanted is against TRUE engine internals -
private/underscore fields, type-validation trivia, internal observability/logging,
runtime-launch state. This wave validated the guard:

- **`_api_process_rank >= _api_process_count` (llama-70B):** REAL constraint, but
  both fields are underscore-private; vLLM source (config/parallel.py:314-327)
  documents them as "internal config... only set by API server scale-out." ->
  INTERNAL, filter from candidate-config. Notably the CODE models never proposed
  it; only the general 70B did.
- **`tokenizer must be a string` (qwen-7B):** SPURIOUS - `tokenizer=12345` trips a
  pydantic field-level `string_type` error BEFORE the labelled model_validator
  (model.py:714) is ever reached. The classic inflation class; the 7B's lone
  "confirm" is not real. (Also: type-validation trivia -> belongs in schema type
  info, not an invariant.)

So the guard cleanly separates comprehensive config discovery from
internals-chasing, and the general-70B is the tier that most needs it.

## Cost story (ordinal, per the north-star method)

det (~0, CPU) < gemma-12B (~500s/cell, 0 real) ~ qwen-7B (~390s/cell, 0 real) <
**qwen-32B-code (~1340s/cell, 5 cross-field @ 1.00 precision)** < llama-70B-general
(~2220s/cell, 3 cross-field + noise) < Opus (~206k tok, 8 cross-field). The
**code-tuned 32B is the efficiency winner among OSS**: Opus-comparable comprehensive
cross-field coverage at local-GPU cost, perfect confirm-precision, no internal noise.

## Implication for the hybrid mining workflow

The deliverable is a HYBRID workflow: deterministic floor (cheap single-field
surface) + LLM extend (comprehensive cross-field/conditional tail) + runtime gate
(validation) + internals-guard (trim true internals). This wave settles the LLM
rung: **a mid code-tuned model (~32B) run locally** is the sweet spot. Smaller is
below the cross-field threshold; a general 70B is slower, lower-precision, and
needs more guarding for the same-or-less coverage; Opus is the ceiling but costs.

## North-star reconciliation (2026-06-09 audit)

A north-star audit flagged the cross-field tail as "off-target" because the
constrained fields (data_parallel_*, max_cpu_loras, structured_outputs, eplb) are
not among the 29 `curated.yaml exposed_fields` llem currently exposes. CORRECTION
(user): comprehensive discovery IS the design - mine the full surface, expose a
subset; the allowlist is an EXPOSURE-time filter, not a mining-time one. So the
cross-field finds are legitimate comprehensive coverage, NOT drift. The audit's
structural facts stand (schema scoped via curated.yaml; the deep-invariant runtime
consumer = sweep-dedup/dormancy is deferred post-MVP) but "scope the miner" is
dropped in favour of the narrow internals-guard. See memory
`project_mining_comprehensive_intent`.

## GT-growth candidates (verified-real, candidate-config, NEW vs wave-2 foldins)

To fold (dedup vs existing GT first): `data_parallel_size_local <= data_parallel_size`,
`data_parallel_external_lb requires dp>1`, `data_parallel_rank in [0,dp_size)`
(vllm ParallelConfig), `capture_num_tokens` positive-ints (tensorrt
TorchCompileConfig). NOT folded: `_api_process_rank` (internal), `tokenizer is str`
(spurious). Already in GT (Opus wave-2 or floor): max_cpu_loras, structured-outputs
exactly-one, max_attention_window, the single-field ranges.

## Deviations / caveats (logged)

- **llama-70B ran on a CONTAINERIZED ollama** (`docker run --gpus all ... -p
  11435:11434 ollama/ollama`, port via `WAVE_OLLAMA` override). The shared host
  ollama (:11434, another user's) is cgroup-capped at 32 GiB and full, so the 70B
  hit HTTP 500 there; the containerized instance has its own memory. Generation
  params held constant (num_ctx 16384, temp 0). The first :11434 70B run is an
  INVALID infra failure (0 proposals, HTTP 500); the valid run overwrote it.
- **qwen-7B format-following failure:** it emitted its 208 invariants under root
  key `i:` (not `invariants:`), so the strict parser dropped all -> 0 proposals.
  Leniently re-parsed (`wave3_reparse_lenient.py`) to recover content for a fair
  capability datapoint; gemma and the 32B used the correct key, so the lenient
  pass only levels the field. The format failure is itself a small-model
  brittleness signal, reported separately from the (also weak) content.
- OSS rungs chunked (16k ctx); Opus whole-source (known assembly diff, handicaps
  OSS - so OSS cross-field counts are lower bounds).
- N=2 cells, single-shot. DIRECTIONAL, not a frontier point.

## Decision / next

Per user: continue developing the LLM+deterministic HYBRID mining workflows
(staff the LLM rung with a mid code-tuned model), add the internals-guard as a
light quality filter, fold the verified-real candidate-config GT-growth. The
broader assembly x call-shape design space remains the workflow-development space.
