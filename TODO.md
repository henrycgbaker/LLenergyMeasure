# TODO

## Completed (v2.0+)

### Backend Support
- [x] **Backend abstraction layer** - Pluggable `InferenceBackend` protocol
- [x] **vLLM backend** - PagedAttention, continuous batching, tensor parallelism
- [x] Backend selection via `--backend` CLI flag and config

### Sharding / Parallelism
- [x] **Tensor parallelism** - HuggingFace native TP and vLLM native TP
- [x] **Pipeline parallelism** - PyTorch native PP for multi-GPU inference
- [x] Wire up `sharding:` config to actual implementations

### Phase 3 Features (Backend Parity)
- [x] TTFT (time to first token) metric for vLLM
- [x] Traffic simulation in vLLM backend
- [x] Backend validation at aggregation (reject mixed backends)
- [x] Backend documentation (`docs/backends.md`)

## Planned Features

### Backend Support
- [ ] **TensorRT-LLM backend** - Optimised NVIDIA inference (Phase 4)
- [ ] Cross-backend comparison CLI: `llm-energy-measure compare --backends pytorch,vllm`

### Backend-Native Features (Phase 4+)
- [ ] Expose vLLM-specific params (speculative_decoding, paged_attention config)
- [ ] PagedAttention memory efficiency reporting
- [ ] LoRA adapter loading
- [ ] Streaming output support

### Documentation
- [ ] Convert CLAUDE.md files to admin/developer READMEs (user-facing vs contributor docs)
