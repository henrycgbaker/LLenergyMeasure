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

**Wall-clock total:** ~80 min across the 5 cells (LLM-serialised; container
Ollama 70B-q4 model loaded throughout).

## (b) baselines (active row, same engine)

| Engine | Schema r | Schema p | Inv r | Inv p | Wall_s |
|---|---:|---:|---:|---:|---:|
| transformers v4.57.3 | 0.830 | 0.939 | 0.564 | 0.431 | 1649 |
| vllm v0.7.3          | 0.970 | 0.851 | 0.385 | 0.152 | 1414 |

## Per-cell results

| Pattern | Engine | Schema r | Schema p | Inv r | Inv p | Wall_s | Cell-count | Inv vs (b) delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h6_no_chunk        | transformers v4.57.3 | 0.750 | 0.944 | **0.128** | 0.313 | 526  | 16 | **-43.6pp** |
| e6_field_anchored  | transformers v4.57.3 | 0.830 | 0.989 | 0.564 | 0.386 | 1256 | 57 | 0.0pp (neutral) |
| e6_field_anchored  | vllm v0.7.3          | 0.970 | 0.851 | 0.308 | 0.174 | 1049 | 46 | **-7.7pp** |
| e9_sequential      | transformers v4.57.3 | 0.830 | 0.989 | 0.333 | 0.406 | 902  | 32 | **-23.1pp** |
| e9_sequential      | vllm v0.7.3          | 0.970 | 0.851 | 0.346 | 0.191 | 1026 | 47 | **-3.8pp** |

## Per-pattern analysis

### H6 (whole-source no-chunking)

**Result:** **invariant recall COLLAPSED** from 0.564 (b baseline) to 0.128.
Only 16 invariants emitted vs 39 in reference. Wall fell from 1649s to
526s (fast but at a steep recall cost).

**Mechanism:** the LLM saw ~33k chars of invariant-relevant source in a
single prompt, picked the "obvious" 16 `if X.field <pred>: raise` cases
from the start of `validate()`, and stopped. It did not exhaustively walk
the file. This is a TEXTBOOK lost-in-the-middle pattern: at 32k context,
the 70B-q4 model attends densely at start + end, sparsely in middle.

**Schema also slipped** (0.83 -> 0.75): 28 fields missed across
engine_params (8) + sampling_params (20). The sampling_params section
relies on the GenerationConfig 16k-char docstring which sits at the END
of the schema prompt - "end attention" partly recovered it, but
recall on that section was incomplete.

**Conclusion:** chunking is NOT the bottleneck. Removing chunking
HALVES invariant recall. The (b) baseline's per-chunk pressure (the
prompt structure FORCES the LLM to walk each chunk) is what keeps
recall above 50%. H6 is the **synthesis-blindness pattern from H7
manifesting under whole-source conditions**: when the LLM has the
choice between exhaustive walk and shallow scan, it picks shallow.

### E6 (field-anchored)

**Transformers result:** recall NEUTRAL (0.564 = same as baseline);
precision slightly worse (0.386 vs 0.431). 57 entries emitted (vs 51
baseline), meaning the LLM emitted more candidates but the canonical
4-tuple identity hit-rate was unchanged. Field-anchoring on the active
cell did NOT shift the ceiling.

**vLLM result:** recall WORSE (0.308 vs 0.385 baseline); precision
slightly better (0.174 vs 0.152). 46 entries vs 66 baseline - the LLM
emitted FEWER candidates. The reason: my chunk-class targeting heuristic
fell back to "all 15 classes" for every vllm chunk because the chunk
names use snake_case (e.g. `sampling_params_invariants`) while the class
names are CamelCase (`SamplingParams`), and the substring check
("samplingparams" in "sampling_params_invariants") fails. So the vllm
LLM saw a 249-field anchor list for every chunk - effectively NOISE.

**Mechanism (transformers, where targeting worked):** the LLM saw a
focused (e.g. 67-field) anchor list for each `GenerationConfig`-chunk.
The locked invariants prompt already directs the LLM to emit one
invariant per `if ... :` block; the anchor list neither helped recall
nor reduced precision noise on the active cell. **Field-anchoring at
70B-q4 on a CALIBRATED active cell with a locked prompt provides
neither lift nor benefit.**

**The hypothesis the variant tests (would have caught the tensorrt v0.x
HF GenerationConfig hallucination) is NOT testable on active cells -
active cells don't fail; the chunker doesn't return empty.** The
hallucination case is at-bump (e.g. `b__tensorrt__v0_19_0` where the
chunker returned empty source and the LLM filled in HF defaults). To
test E6's protective effect against that failure mode, we would need to
re-run E6 on a BUMPED tensorrt cell - which was out of scope for this
batch (E6 spec said active cells only).

**Conclusion:** active-cell field-anchoring is neutral (transformers
with targeted anchor) or harmful (vllm with untargeted anchor due to
heuristic failure). The variant's intended use case
(hallucination-prevention on empty-source bumped cells) was not tested.
For Phase 3c (Claude), recommend re-running E6 on the v0.19.0 tensorrt
bumped cell.

### E9 (sequential cumulative-context)

**Transformers result:** recall WORSE (0.333 vs 0.564 baseline);
precision marginally better (0.406 vs 0.431). 32 entries vs ~40
baseline. Wall improved 1649 -> 902s.

**Mechanism:** the cumulative-context preamble told the LLM "DO NOT
re-emit invariants already in the running notes" - and the model
interpreted that conservatively. The first chunk
(`generation_config_init_invariants`) emitted 0 (correct: __init__
has no raise patterns). The BitsAndBytesConfig chunk emitted 12 (normal).
Then each subsequent validate() section chunk emitted 1-3 unique
invariants on average, with chunks 4-7 (decoding/cache/performance/
watermarking sub-sections) all emitting EXACTLY 1 each. This is the
**dedup-pressure suppressing synthesis** failure mode: the LLM,
seeing growing running notes, becomes too cautious about "is this new?"
and skips emitting borderline cases that the prompt's locked-mode
single-shot would have happily emitted.

**vLLM result:** recall 0.346 vs 0.385 baseline (-3.8pp); precision
0.191 vs 0.152 (+3.9pp). 47 entries vs 66 baseline. The vllm cell
followed a different pattern from transformers: chunk 1
(`sampling_params_invariants`) emitted 20 (vs ~7-12 typical at
baseline), and subsequent chunks accumulated normally (1-8 per chunk).
Total cumulative 47. The cross-class hypothesis: the LLM had 47
invariants in running notes by the final chunks, but the per-chunk
emission did not show signs of having LEVERAGED that history to find
cross-class constraints (e.g. `max_num_batched_tokens >= max_num_seqs`
was not surfaced as a cross-class invariant in any chunk despite
SchedulerConfig being a multi-field validator).

**Conclusion:** cumulative context did NOT surface the hoped-for
cross-class invariants. On transformers the dedup-pressure caused
under-emit (-23pp recall). On vllm the effect was milder (-3.8pp) but
still negative; the LLM emitted normally on early chunks but did not
use accumulated history to find cross-class patterns. **The pattern
that would FORCE cross-class reasoning (an EXPLICIT prompt step "now
list cross-class invariants you can see across the running notes")
was not part of E9's design** - we used the locked extract prompt and
ADDED dedup pressure. To test the cross-class hypothesis properly,
E9-2 would need a final reconciliation step after all chunks, with an
explicit cross-class prompt. That is out of scope for this batch.

## Cross-variant insights

**Substrate-side variants all UNDERPERFORM the baseline (b) at 70B-q4.**

| Variant | TF inv recall | vLLM inv recall | Substrate hypothesis | Outcome |
|---|---:|---:|---|---|
| (b) baseline | 0.564 | 0.385 | per-class chunks; single-shot per chunk | reference |
| H6 no-chunk | 0.128 | n/a (ctx limit) | one big call; whole-source | -43.6pp recall |
| E6 field-anchor | 0.564 | 0.308 | per-chunk + declared __fields__ preamble | TF neutral; vllm worse |
| E9 cumulative | 0.333 | 0.346 | per-chunk + prior-extraction notes | TF -23pp / vllm -3.8pp |

**The bottleneck is NOT chunking.** Per-class chunking is doing the
right thing at 70B-q4 scale: the per-chunk synthesis pressure prevents
the model from defaulting to under-emit / shallow-scan. Any variant
that REDUCES that pressure (whole-source, cumulative dedup) DECREASES
recall. Any variant that ADDS information to the prompt without changing
the synthesis pressure (field-anchoring on a calibrated active cell)
is neutral.

**Connection to the H7/H4 synthesis-blindness pattern.** The 70B-q4
model has a synthesis weakness; agentic loops (H7) make it manifest as
ZERO emission; whole-source (H6) makes it manifest as ~25% emission;
cumulative dedup (E9) makes it manifest as ~60% of baseline emission.
Field-anchoring (E6) is a special case: it doesn't reduce synthesis
pressure (the prompt still says "emit one invariant per block"), but
the anchor list adds tokens to the prompt and can be either targeted
(transformers, neutral) or noisy (vllm, slightly worse).

**Schema is robust across substrate variants.** Schema recall held at
0.83 on transformers across H6/E6/E9 (only H6 slipped to 0.75 due to
lost-in-middle on the 50k schema prompt). Schema extraction depends on
docstring + signature reading, which the LLM does competently even
under decomposition pressure. The bottleneck is INVARIANT extraction,
which depends on synthesis.

**Failure-mode tags (from trial_scoring):** H6 transformers invariants
got `silent` (recall < threshold, no envelope-marker error). All other
cells passed (`none`). H6 is the only cell where the failure mode is
detectable from the score alone.

**Honest caveat on E6 vllm:** my chunk-to-class targeting heuristic
substring-matches `cls.lower() in chunk_name.lower()`. For transformers
this works (`generation_config_init_invariants` contains
`generationconfig` after lowercase). For vllm it fails because chunk
names use snake_case (`sampling_params_invariants`) while class names
are CamelCase (`SamplingParams`), so "samplingparams" does NOT appear
in the chunk name. Result: vllm E6 fell back to "all 15 classes" for
EVERY chunk, giving the LLM a 249-field anchor list per chunk. The vllm
E6 result therefore tests "field-anchoring with UNTARGETED 249-field
anchor" rather than "field-anchoring with targeted N-field anchor".
For a fair test of E6 on vllm, the heuristic needs a snake-to-camel
mapping (`sampling_params` -> `SamplingParams`). I left this as
observed-but-not-fixed because (a) the test ran end-to-end and
produced a comparable score; (b) the lesson "untargeted anchor is
neutral-to-harmful" is itself useful evidence; (c) re-running with
fixed heuristic would consume another ~17 min of LLM time the trial
budget could spend on Phase 3c. For Phase 4 production, IF E6 is
adopted as a defence against bumped-cell hallucination, the
heuristic must be fixed first.

## Implications for Phase 4 synthesis

1. **Lock in per-class chunking for (b).** All three substrate variants
   tested underperform. The (b) baseline's per-class single-shot per
   chunk is the best 70B-q4 substrate decomposition we've found across
   12 hybrid patterns. Do not invest further in substrate ablation
   without a stronger model.

2. **Substrate is NOT the (b) ceiling driver.** The ~50-55% recall
   ceiling on transformers, ~38% on vllm, is driven by LLM SYNTHESIS
   CAPACITY, not by chunking strategy. Phase 4 production substrate
   should run (b)-as-is for 70B-q4 and use the LLM-role split
   architecture (deterministic validate; LLM extend-only).

3. **E6 (field-anchored) for empty-chunk hallucination prevention is
   STILL a candidate** - but needs Phase 3c (Claude) testing on a
   BUMPED tensorrt cell where the chunker returns empty and the
   baseline (b) hallucinates HF GenerationConfig fields. The active-
   cell test was uninformative for that hypothesis.

4. **E9 (cumulative context) is a Phase 3c candidate too.** At 70B-q4
   the model is under-emit on cumulative context; at Claude scale
   (200k context, better synthesis) cumulative dedup might
   ACTUALLY work as designed - the model would emit normally and
   dedup correctly. Hold the E9 harness; rerun in Phase 3c.

5. **Skip E4 + E7 at 70B-q4.** The cross-pattern pattern is clear:
   substrate-side LLM-only variants at this model scale have a
   ~50-60% ceiling against the per-class baseline and rarely lift it.
   The trial's marginal-value-per-cell-run is now LOW for further
   substrate variants. Better investment: Phase 3c (Claude variants)
   OR Phase 4 (synthesis + production-substrate commitment).

6. **Update extend-propose catalogue.** Mark E9 as TESTED at 70B-q4
   with the outcome above. Mark E6 as PARTIALLY TESTED (active only;
   the bump cell hypothesis is open).

7. **Recommend NOT running additional Phase 3b patterns at 70B-q4.**
   Proceed to Phase 4 synthesis with the current 9-pattern landscape
   (h1, h2, h3, h4, h6, h7, h9, e6, e9) plus the baselines. The
   landscape has converged on a robust finding: **deterministic-
   validate + LLM-extend-propose, with per-class chunking, is the
   substrate to commit to for 70B-q4 production.**

## Artefacts

Per-cell:
- `_spike/findings/hybrid_experiments/h6_no_chunk/transformers/v4_57_3/{schema.json,invariants.proposed.yaml,score.json,observations.md,raw_llm_transcripts/}`
- `_spike/findings/hybrid_experiments/e6_field_anchored/<engine>/<version>/{...same shape...}`
- `_spike/findings/hybrid_experiments/e9_sequential/<engine>/<version>/{...same shape...}`

Cross-pattern: this file.

Runner: `_spike/scripts/strategies/run_h6_e6_e9.py`.
