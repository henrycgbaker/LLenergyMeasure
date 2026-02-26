# FLOPs Estimation

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25 (FLOPs demoted from primary metric to reference metadata per user decision)
**Research:** [../research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)

---

## Context

> **Research annotation (2026-02-25):** `.planning/research/PITFALLS.md` (CP-3) and
> `.planning/research/SUMMARY.md` (section 1.4) recommend **demoting FLOPs from "primary
> metric" to "reference metadata"**. FLOPs are deterministic for a given model+input
> (`FLOPs = 2 * N_params * tokens`) — they do not vary between backends, batch sizes,
> precision settings, or any deployment parameter this tool measures. For a tool whose purpose
> is measuring "how implementation choices affect efficiency", FLOPs provide zero discriminatory
> power. The actual primary metrics that vary between configurations are `energy_per_output_token`
> (J/token) and `tokens_per_second`. FLOPs remain valuable for cross-model normalisation and
> MFU calculation (deferred to v2.1), but should not be surfaced as a primary comparison axis
> in CLI output. Keep in result schema for researchers who need it.
>
> **Decision accepted (2026-02-25):** FLOPs demoted from primary metric to reference metadata.
> Primary comparison metrics: `energy_per_output_token` (J/token), `tokens_per_second`, latency (TTFT, TPOT).
> FLOPs retained in result schema for cross-model normalisation and future MFU calculation.
> Sources: [Mind the Memory Gap (arXiv:2503.08311)](https://arxiv.org/html/2503.08311v2),
> [Databricks LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices).

LLenergyMeasure reports FLOPs as a normalisation metric — primarily `flops_per_output_token` — to enable comparison of energy efficiency across hardware generations and implementation configurations. The FLOPs estimation method must be:

- Reproducible without running the model (no torch dependency in the metrics layer)
- Consistent with peer benchmarking tools (PaLM, Chinchilla, LLM-Inference-Bench)
- Correct for autoregressive generation with KV cache (which most runtime profilers are not)
- Honest about known limitations (attention FLOPs omission for long contexts)

**The tool does not use FLOPs as a latency predictor.** LLM inference is memory-bandwidth-bound, not compute-bound; FLOPs correlate poorly with GPU latency. The intended use is energy-per-FLOP efficiency analysis and hardware comparison.

---

## Considered Options

### F1: Estimation method

| Option | Pros | Cons |
|--------|------|------|
| **Analytical — PaLM/Chinchilla formula (chosen)** | No torch dependency; reproducible; industry standard; handles KV cache correctly by design | Omits attention FLOPs (O(seq_len²)) — documented limitation for long contexts |
| Runtime profiler (calflops, fvcore, DeepSpeed FLOPs Profiler) | Theoretically more accurate for non-standard architectures | None correctly model KV-cache autoregressive generation; all measure a single forward pass, not the decode loop; calflops `generate` mode is still theoretical; fvcore has poor LLM support |

**Rejected: Runtime profilers (2026-02-19).** No runtime profiler correctly handles autoregressive generation with KV cache — they count a single forward pass, not the full decode loop with shared KV state. This is a fundamental mismatch with the inference workload being measured.

---

### F2: Parameter count — embedding inclusion

| Option | Pros | Cons |
|--------|------|------|
| **Exclude embedding parameters (chosen)** | Embeddings are memory-bound lookup operations, not MAC compute operations; exclusion matches PaLM/Chinchilla methodology | Requires extracting non-embedding param count per model |
| Include all parameters | Simpler | Systematically over-counts FLOPs for embedding-heavy models; inconsistent with peer methodology |

---

### F3: Phase split

| Option | Pros | Cons |
|--------|------|------|
| **Report prefill and decode separately (chosen)** | Enables prefill/decode efficiency analysis; prefill is compute-bound, decode is memory-bandwidth-bound — they have different hardware utilisation profiles | Slightly more complexity in result schema |
| Report total only | Simpler schema | Loses the compute vs memory-bandwidth distinction; less diagnostic value |

Primary metrics surfaced:
- `flops_prefill` — total prefill phase compute
- `flops_decode` — total decode phase compute
- `flops_total` — sum of both phases
- `flops_per_output_token` — `flops_total / total_output_tokens` — reference metadata for cross-model normalisation (not a primary comparison metric — FLOPs do not vary between deployment configurations)

---

### F4: Multi-GPU FLOPs aggregation

| Option | Pros | Cons |
|--------|------|------|
| **Sum FLOPs across all GPUs (chosen)** | Measurement objective is total compute consumed by the experiment, not per-device utilisation | May be misleading if comparing single-GPU vs multi-GPU experiments without context |
| Per-GPU FLOPs | More granular | Does not reflect the total workload cost; confusing for tensor-parallel runs where each GPU handles a shard |

---

### F5: Attention FLOPs inclusion (v2.0)

| Option | Pros | Cons |
|--------|------|------|
| **Simplified formula — omit attention FLOPs (chosen for v2.0)** | Matches PaLM/Chinchilla baseline; safe for prompts ≤512 tokens (attention < 5% of total) | Underestimates FLOPs for long-context workloads (≥2048 tokens) |
| Full formula including attention | Accurate for long-context | Requires per-prompt seq_len tracking; significantly more complex; deferred to v2.1 |

The formula used in v2.0:
```
FLOPs_prefill = 2 × N_params × B × S_input
FLOPs_decode  = 2 × N_params × B × S_output
FLOPs_total   = FLOPs_prefill + FLOPs_decode
```

Where `N_params` = non-embedding parameters, `B` = batch size, `S_input` = input token count, `S_output` = output token count. The factor of 2 = one multiply + one add (MAC). The factor is 2N (not 6N) because 6N includes the backward pass — inference is forward-only.

Attention FLOPs formula (deferred, for reference):
```
FLOPs_attention = 4 × seq_len² × n_heads × head_dim × n_layers × batch_size
```

For long-context workloads (≥2048 tokens), attention FLOPs become significant. The limitation is documented explicitly in the result schema via the `measurement_methodology` field. Users running long-context experiments are prompted to flag results as having underestimated FLOPs.

---

## Decision

We will use **analytical estimation using the PaLM/Chinchilla inference formula, split by phase**, with non-embedding parameter count, summed across all GPUs.

The v2.0 formula deliberately omits attention FLOPs for simplicity, with the limitation explicitly documented in output. Attention FLOPs correction, MFU reporting, and Flash Attention-aware counting are deferred to v2.1.

**Rationale:** Runtime profilers are unsuitable (see F1). The analytical formula is the industry standard (PaLM, Chinchilla, LLM-Inference-Bench, TokenPowerBench all use `2×N×T`). Phase split (F3) preserves diagnostic value at low implementation cost.

---

## Consequences

**Positive:**
- No torch or CUDA dependency in the metrics layer — FLOPs can be estimated from config alone
- Reproducible: same inputs always produce the same estimate
- Consistent with peer benchmarking tools — results are comparable
- Phase split enables prefill/decode efficiency profiling

**Negative / Trade-offs:**
- Attention FLOPs omission means results are underestimated for long-context workloads (≥2048 tokens)
- FLOPs is NOT a latency proxy for LLM inference (memory-bandwidth-bound workload); must be clearly documented in user-facing output

**Neutral / Follow-up decisions triggered:**
- `measurement_methodology` field in result schema documents the attention FLOPs limitation
- Warmup tokens excluded from FLOPs: see [decisions/warmup-strategy.md](warmup-strategy.md)
- Multi-GPU FLOPs aggregation: see [decisions/multi-gpu.md](multi-gpu.md)
- Implementation deferred to Phase 5 as `core/metrics.py` — `estimate_flops()` and `get_non_embedding_param_count()`

---

## Deferred (v2.1+)

| Item | Target version | Notes |
|------|---------------|-------|
| Attention FLOPs correction for variable sequence lengths | v2.1 | Requires per-prompt seq_len tracking |
| MFU (Model FLOPs Utilisation) reporting | v2.1 | Requires knowing GPU peak theoretical FLOPs |
| Flash Attention FLOPs (algorithm-aware counting) | v2.1 | — |
| LoRA adapter FLOPs | Not needed | Adapter FLOPs negligible (<0.1% for rank-16 adapter on 7B model); use base model N_params unchanged. See [decisions/adapter-support.md](adapter-support.md). |

---

## Related

- [designs/result-schema.md](../designs/result-schema.md): `flops_total`, `flops_per_token`, `measurement_methodology` fields
- [decisions/warmup-strategy.md](warmup-strategy.md): Which tokens count toward FLOPs (warmup excluded)
- [decisions/multi-gpu.md](multi-gpu.md): FLOPs aggregation across GPUs
- [decisions/adapter-support.md](adapter-support.md): LoRA FLOPs treatment
- [research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md): §FLOPs Estimation — peer tool survey
- NEEDS_ADDRESSING.md item 20
