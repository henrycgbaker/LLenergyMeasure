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

### Phase 4 Features (Native Backend Config)
- [x] Native vLLM config (`VLLMConfig`) - memory, KV cache, prefix caching, speculative decoding
- [x] Native PyTorch config (`PyTorchConfig`) - attention impl, torch.compile, assisted generation
- [x] Runtime verification tests for vLLM parameters

## Planned Features

### Backend Support
- [x] **TensorRT-LLM backend** - Optimised NVIDIA inference
- [ ] Cross-backend comparison CLI: `llm-energy-measure compare --backends pytorch,vllm`
- [ ] Cross-backend visualisation (energy vs throughput per backend)

### Backend-Native Features (Phase 4+)
- [x] LoRA adapter loading
- [ ] Expose vLLM-specific params (speculative_decoding, paged_attention config)
- [ ] PagedAttention memory efficiency reporting

### Streaming Latency Metrics (TTFT/ITL)
- [x] **vLLM streaming** - TTFT and ITL measurement via streaming API
- [x] **PyTorch streaming** - Use `TextIteratorStreamer` for TTFT/ITL
- [ ] **TensorRT-LLM streaming** - Code complete, requires NGC login to test (see docker/Dockerfile.tensorrt)

### Streaming Enhancements (Future)
- [ ] Real-time CLI token display during inference
- [ ] TTFT component breakdown (tokenisation/prefill/decode phases)
- [ ] vLLM queue time separation (isolate scheduling delay from compute)
- [ ] Energy-per-token correlation (TTFT energy vs decode energy)
- [ ] Speculative decoding acceptance rate metrics
- [ ] KV cache hit rate metrics (prefix caching effectiveness)

### Downstream: Causal Analysis
> For unpacking *why* optimisations affect efficiency, not primary measurement.

- [ ] PagedAttention block usage metrics (vLLM memory efficiency observability)
- [ ] CUDA graph capture timing breakdown
- [ ] KV cache hit rates (prefix caching effectiveness)
- [ ] Attention kernel profiling (flash vs standard)

### Documentation
- [ ] Convert CLAUDE.md files to admin/developer READMEs (user-facing vs contributor docs)

## Future: Web Interface

See [docs/strategy.md](docs/strategy.md) for comprehensive UI/webapp strategic planning.

**Summary:** Full community platform with public leaderboard, remote execution on self-hosted GPUs, React + FastAPI architecture.

### Web Platform Phases
- [ ] **Phase 1:** Core platform (FastAPI + React, upload results, leaderboard)
- [ ] **Phase 2:** Analysis features (interactive charts, comparisons, API)
- [ ] **Phase 3:** Remote execution (job queue, GPU workers, progress monitoring)
- [ ] **Phase 4:** Community features (user profiles, moderation, HF Dataset export)
