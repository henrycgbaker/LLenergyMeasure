# Additional Inference Backends

**Status:** Accepted
**Date decided:** 2026-02-25
**Last updated:** 2026-02-25
**Research:** [../../.planning/research/STACK.md](../../.planning/research/STACK.md) (section 2)

## Decision

Priority ordering: SGLang (accelerated to v2.2 candidate) > llama.cpp (v3.x) > HF TGI (v4.x). Each requires Docker isolation decision before implementation. Ollama, DeepSpeed-MII, MLC LLM, ExLlamaV2 rejected.

---

## Context

LLenergyMeasure ships three inference backends at v2.0: PyTorch (Transformers), vLLM, and
TensorRT-LLM. The inference backend landscape is active and growing. Adding backends
increases the parameter space we can measure (and our research differentiation), but each
backend adds maintenance burden and may require Docker isolation due to CUDA driver conflicts.

The key constraint: backends that conflict with existing ones at the CUDA driver level cannot
coexist in the same process. This is a hard isolation requirement that must be resolved before
any new backend can be committed to a version.

No version commitments are made in this file. Priority ordering is recorded to guide future
planning.

## Considered Options

### SGLang

> **Updated 2026-02-25:** Accelerated from "v3.x tentative" to "v2.2 candidate" based on stack research. SGLang joined the PyTorch ecosystem (official endorsement), runs on 400,000+ GPUs worldwide (xAI Grok 3), and shows 29% throughput advantage over optimised vLLM on H100. RadixAttention creates genuinely different energy profiles from PagedAttention -- this is directly relevant to the tool's purpose of measuring implementation choice effects. See [STACK.md section 2](../../.planning/research/STACK.md).

| Option | Pros | Cons |
|--------|------|------|
| **Add SGLang (accelerated to v2.2 candidate) — bold** | PyTorch ecosystem member. 29K+ stars. RadixAttention (automatic KV cache reuse) and compressed FSM for structured output differentiate it from vLLM. No other energy benchmarking tool currently has SGLang + energy measurement — strong research differentiation. Rapidly maturing. 29% throughput advantage over vLLM on H100 benchmarks. | Likely conflicts with TRT-LLM at CUDA driver level (same underlying issue as vLLM + TRT-LLM). Docker isolation required. Maintenance burden scales with backend volatility. |
| Skip SGLang | No new maintenance burden | Misses a rapidly growing backend with novel KV cache semantics worth measuring. |

### llama.cpp / GGUF

| Option | Pros | Cons |
|--------|------|------|
| **Add llama.cpp / GGUF (target v3.x) — bold** | 50K+ stars. CPU inference + quantised models. Reaches researchers without A100s. Optimum-Benchmark already supports it via `llama-cpp-python`. Natural isolation (C++ subprocess). | Different performance characteristics than GPU backends; results not directly comparable on energy/token. Quantisation complexity adds config surface. |
| Skip llama.cpp | Simpler backend set | Excludes consumer hardware researchers — a significant segment of the target audience. |

### HF TGI

| Option | Pros | Cons |
|--------|------|------|
| Add HF TGI (target v4.x) | Production serving layer; widely used in industry deployments. | Significant overlap with vLLM (both PagedAttention-based). Lower differentiation value. Lower priority than SGLang and llama.cpp. |
| **Skip TGI in near term — bold** | Avoids duplicating vLLM coverage. | May miss industry deployment measurement use cases. |

### Rejected Candidates

**Rejected (2026-02-19): Ollama** — Too high-level. Wraps llama.cpp and manages model
lifecycle. Not suitable for controlled benchmarking where we must own the measurement
lifecycle. Loss of control over model loading and execution timing.

**Rejected (2026-02-19): DeepSpeed-MII** — Significant overlap with vLLM. Lower priority
given existing vLLM support. Maintenance not justified by differentiation.

**Rejected (2026-02-19): MLC LLM** — TVM compilation pipeline adds significant complexity.
Deferred indefinitely.

**Rejected (2026-02-19): ExLlamaV2** — Niche GPTQ-focused use case. Insufficient research
audience breadth to justify maintenance burden.

## Decision

We will maintain a priority ordering (SGLang > llama.cpp > HF TGI). SGLang is accelerated
to v2.2 candidate based on stack research evidence; llama.cpp and HF TGI remain uncommitted.
Each new backend requires a Docker isolation decision before implementation planning begins.

Rationale: SGLang vs vLLM is an especially high-value comparison because both run the same
models but with fundamentally different KV cache strategies (RadixAttention vs PagedAttention).
The energy profiles will differ non-trivially. This is not just adding another backend — it
adds a backend with a genuinely different implementation parameter space. The accessibility
argument for llama.cpp (consumer hardware) is valid but deferred to v3.x pending the CPU vs
GPU energy comparability design question.

## Consequences

Positive:
- SGLang + energy measurement would be a genuine research differentiator (no current peer
  offers this combination)
- llama.cpp support would open the tool to researchers without GPU access
- Priority ordering gives future planning sessions a starting point

Negative / Trade-offs:
- Each new backend adds a maintenance surface that scales with upstream API volatility
- Process isolation (Docker) is required for most new backends; adds operational complexity
- No version commitments means roadmap remains vague for v3.x planning

Neutral / Follow-up decisions triggered:
- Docker isolation strategy (see [`docker-execution.md`](docker-execution.md)) must be
  resolved before any new backend can be designed
- Each candidate backend requires an individual isolation confirmation before planning

## Decision Criteria for New Backends

Before adding any backend to the roadmap:

1. **Isolation requirement**: Does it conflict with existing backends at the CUDA driver
   level? If yes, Docker is required before implementation.
2. **User base size**: Does it serve a meaningful research audience?
3. **Uniqueness**: Does any other energy benchmarking tool already support it well?
4. **Maintenance burden**: Is the backend API stable enough to maintain across minor versions?

## Process-Isolation Reference

- PyTorch + vLLM: can coexist in the same process (confirmed by research)
- PyTorch/vLLM + TRT-LLM: CANNOT coexist (CUDA driver conflict — confirmed by NVIDIA GitHub issues)
- SGLang + TRT-LLM: likely conflicts (same underlying issue; not yet confirmed)
- llama.cpp: runs as a C++ subprocess — isolation is natural; no conflict expected

## Related

- [`architecture.md`](architecture.md) — backend isolation as architectural constraint
- [`docker-execution.md`](docker-execution.md) — Docker isolation strategy (open; needed before new backends)
- [`versioning-roadmap.md`](versioning-roadmap.md) — v3.x scope (HPC + lm-eval; backends TBD)
- [`installation.md`](installation.md) — `[pytorch]`, `[vllm]`, `[tensorrt]` extras pattern that new backends follow
