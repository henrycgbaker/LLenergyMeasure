---
created: 2026-02-26T00:00:00.000Z
title: Revisit peak_memory_mb measurement semantics
area: measurement
files:
  - .product/designs/result-schema.md:336
  - .product/decisions/reproducibility.md
  - .planning/codebase/CONCERNS.md:39
---

## Problem

`peak_memory_mb` is currently specified as inference-window-only: `torch.cuda.reset_peak_memory_stats()` is called after model loading, before the first inference call. This means the metric captures inference overhead (KV cache, activations, batch buffers) but excludes model weight memory.

User is unconvinced this is the right choice (2026-02-26 session). Concerns:
- Researchers may want the total VRAM footprint (weights + inference) to know if a config fits on their hardware
- "Inference only" requires careful documentation — easy to misinterpret as total GPU memory
- Peers (optimum-benchmark, vLLM) report total allocated memory, not inference-only peak

## Solution

Consider one of:
1. **Keep inference-only** but with very clear field naming (e.g. `peak_inference_memory_mb`) and add a separate `model_memory_mb` field
2. **Report both values** — `peak_memory_mb` (whole process) + `inference_memory_mb` (post-reset)
3. **Revert to whole-process** — simpler, matches peers, less documentation burden

Decision deferred. Revisit during result schema implementation when we can test with real models.
