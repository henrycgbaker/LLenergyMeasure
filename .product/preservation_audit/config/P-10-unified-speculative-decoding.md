# P-10: Unified Speculative Decoding Config

**Module**: `src/llenergymeasure/config/speculative.py`
**Risk Level**: MEDIUM
**Decision**: Pending — resolve with same policy as P-04 (unified abstraction or backend-specific)
**Planning Gap**: Completely absent from all planning documents.

---

## What Exists in the Code

**Primary file**: `src/llenergymeasure/config/speculative.py` (106 lines)
**Key class**: `UnifiedSpeculativeConfig`
**Fields**:
- `method: Literal["draft_model", "ngram", "eagle", "medusa", "mlp", "lookahead", "none"]` — method selection
- `draft_model: str | None` — HuggingFace path for the draft model
- `ngram_min: int`, `ngram_max: int` — n-gram tuning range
- `num_speculative_tokens: int` — lookahead depth
- `draft_tp_size: int | None` — tensor parallelism for draft model
- `backend_method: str | None` — escape hatch for explicit backend string

Backend mapping (similar pattern to P-04):
- **PyTorch**: draft model via `transformers` assisted decoding
- **vLLM**: native speculative decoding (`speculative_model`, `num_speculative_tokens`)
- **TensorRT**: draft model tensor parallelism via TRT-LLM speculative sampling

## Why It Matters

Speculative decoding (using a small draft model to propose tokens accepted/rejected by the main model) is a major efficiency technique. Without a unified abstraction, measuring speculative decoding efficiency across backends requires three separate backend-specific configs with different field names. The unified interface enables a single study config to compare speculative vs non-speculative across backends.

## Planning Gap Details

`designs/experiment-config.md` does not mention speculative decoding at all — not as a unified field, not as backend-specific fields. This is a significant gap given speculative decoding is one of the key inference efficiency techniques the tool is meant to measure.

Two paths (same policy as P-04):
- **Path A (keep)**: `speculation: {method: draft_model, draft_model: "...", ...}` in top-level `ExperimentConfig` as a unified cross-backend intent.
- **Path B (remove)**: Backend-specific only — `vllm: {speculative_model: "...", num_speculative_tokens: 5}`.

## Recommendation for Phase 5

Resolve with P-04 — both unified abstractions should follow the same policy. Add to `designs/experiment-config.md` a "Speculative Decoding" section mirroring the quantization decision:

> **Speculation**: [Same policy as Quantization — Path A (unified) or Path B (backend-specific)]
>
> If unified: add `speculation: UnifiedSpeculativeConfig` to top-level `ExperimentConfig`.
> Implementation: `config/speculative.py`. Map to backend-specific fields at load time.
>
> If backend-specific: delete `config/speculative.py`. Users write vLLM-specific
> `vllm: {speculative_model: ..., num_speculative_tokens: N}` directly.
