# Wave 2.6 deliverable: model-scale cost-quality curve

How does LLM-extend quality scale with model size? Source: `wave2_llm_cells.json`
(W-G extend on vllm v0.7.3, swept across 3 OSS models on 1x A100). Tolerant inv
recall vs GT.

## The curve (W-G extend, vllm v0.7.3, floor = 0.513)

| model | params | quant | inv recall | lift over floor |
|---|---|---|---|---|
| (deterministic floor) | - | - | 0.513 | - |
| qwen2.5-coder | 7B | fp16 | 0.513 | +0.000 |
| llama3.1 | 8B | fp16 | 0.566 | +0.053 |
| phi4 | 14B | fp16 | 0.566 | +0.000 (vs 8B) |

## Findings

1. **The knee is ~8B, and it is shallow.** 7B adds literally zero over the floor;
   8B adds +0.053; 14B adds nothing over 8B. There is a narrow band (~8B) where a
   small model contributes a few recall points, flanked by zero on both sides.
2. **Bigger is not better in the 7-14B range.** phi4-14B does not beat llama-8B.
   So within the OSS-small tier there is no "spend more params for more recall"
   gradient to ride; the choice is essentially 8B-or-nothing.
3. **Combined with the pure-b collapse** (0.05-0.12 at 7B vs the Wave 1 ~50% at
   70B-q4), the picture is: extraction quality is HIGHLY non-linear in scale, with
   a cliff somewhere between 14B and 70B. The 4xA100-budget assumption would have
   let us probe 32B/70B; the single-GPU reality capped this at 14B, so the
   14B->70B region (where the interesting recovery may be) is UNMEASURED.
4. **Cost-quality verdict at OSS-small scale:** llama3.1-8B fp16 is the only model
   that pays for itself at all, and only by +0.05 recall at a precision cost. The
   smallest "viable" extender is 8B, but "viable" here means "marginally positive
   recall," not "good."

## Decision-relevant implication

Do not size the production LLM by walking UP the small tier - there is no gradient
to climb below ~14B. Either stay deterministic (the floor is competitive with
every model tested here) or jump to a scale this run could not reach (32B+/70B+ or
frontier API). The model-scale question that matters - does extraction recover at
32B-70B or only at frontier - is DEFERRED (single-GPU 40GB cap; the 2xA100/80GB
`container deploy` path is tty-gated; API is out of scope this wave).

## Deferred (Wave 3)

- 32B fp16 (~64GB, needs 2xA100), 70B-q4 (~40GB borderline), 70B fp16 (multi-node).
- Frontier API (Claude/GPT) extend + pure-extract - the likely real ceiling.
- The same sweep on the invariants-HARD engine (tensorrt) and the bumped versions,
  to see if scale helps more where the deterministic floor is weaker.
