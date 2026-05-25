# Phase 3b substrate-decomposition batch: H6 + E6 + E9

**Pattern bundle:** Three substrate-side variants probing whether the
(b) ceiling at 70B-q4 is driven by the CHUNKING DECOMPOSITION strategy
rather than by raw LLM capacity. H6 = no chunking (whole-source single
call); E6 = same chunks as (b) but field-anchored preamble; E9 = same
chunks as (b) but with cumulative running-notes context across chunks.

**Question:** is the (b) recall/precision ceiling a CHUNKING artefact?
- H6 -> if whole-source >> chunked, chunking is leaving recall on the table.
- E6 -> if anchoring fields reduces precision noise, the LLM's hallucinations
  are field-set-blindness rather than predicate fabrication.
- E9 -> if cumulative context surfaces NEW cross-class invariants, the
  per-class chunking is hiding cross-class constraints.

**Backend:** container Ollama @ port 11435, model `llama3.1:70b` (q4_K_M),
num_ctx=32768, temperature=0.

**Cells run:** H6 transformers v4.57.3 (transformers-only by spec; vllm/tensorrt
source too large for 32k context). E6 + E9 each run transformers v4.57.3 +
vllm v0.7.3 (active cells). 5 cells total.

**Wall-clock:** TBD - will be filled in after all cells complete.

## (b) baselines (active row, same engine)

| Engine | Schema r | Schema p | Inv r | Inv p | Wall_s |
|---|---:|---:|---:|---:|---:|
| transformers v4.57.3 | 0.830 | 0.939 | 0.564 | 0.431 | 1649 |
| vllm v0.7.3          | 0.970 | 0.851 | 0.385 | 0.152 | 1414 |

## Per-cell results

| Pattern | Engine | Schema r | Schema p | Inv r | Inv p | Wall_s |
|---|---|---:|---:|---:|---:|---:|
| h6_no_chunk        | transformers v4.57.3 | TBD | TBD | TBD | TBD | TBD |
| e6_field_anchored  | transformers v4.57.3 | TBD | TBD | TBD | TBD | TBD |
| e6_field_anchored  | vllm v0.7.3          | TBD | TBD | TBD | TBD | TBD |
| e9_sequential      | transformers v4.57.3 | TBD | TBD | TBD | TBD | TBD |
| e9_sequential      | vllm v0.7.3          | TBD | TBD | TBD | TBD | TBD |

## Per-pattern analysis

### H6 (whole-source no-chunking)

TBD.

### E6 (field-anchored)

TBD.

### E9 (sequential cumulative-context)

TBD.

## Cross-variant insights

TBD.

## Implications for Phase 4 synthesis

TBD.

## Artefacts

Per-cell:
- `_spike/findings/hybrid_experiments/h6_no_chunk/transformers/v4_57_3/{schema.json,invariants.proposed.yaml,score.json,observations.md,raw_llm_transcripts/}`
- `_spike/findings/hybrid_experiments/e6_field_anchored/<engine>/<version>/{...same shape...}`
- `_spike/findings/hybrid_experiments/e9_sequential/<engine>/<version>/{...same shape...}`

Cross-pattern: this file.

Runner: `_spike/scripts/strategies/run_h6_e6_e9.py`.
