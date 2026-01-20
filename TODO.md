# TODO

## Completed (v2.0+)

### Backend Support
- [x] Backend abstraction layer - Pluggable `InferenceBackend` protocol
- [x] vLLM backend - PagedAttention, continuous batching, tensor parallelism
- [x] TensorRT-LLM backend - Optimised NVIDIA inference (code complete)
- [x] Backend selection via `--backend` CLI flag and config

### Parallelism / Sharding
- [x] Tensor parallelism - HuggingFace native TP and vLLM native TP
- [x] Pipeline parallelism - vLLM/TensorRT native PP (PyTorch PP rejected: impractical for autoregressive generation)
- [x] Wire up `sharding:` config to actual implementations
- [x] MIG GPU support with topology detection

### Metrics & Analysis
- [x] TTFT (time to first token) metric for vLLM
- [x] ITL (inter-token latency) measurement
- [x] Traffic simulation (Poisson + constant arrival modes)
- [x] Multi-cycle experiments with statistical robustness (mean, std, 95% CI, CV)
- [x] Backend validation at aggregation (reject mixed backends)

### Configuration & UX
- [x] Native vLLM config (`VLLMConfig`) - memory, KV cache, prefix caching, speculative decoding
- [x] Native PyTorch config (`PyTorchConfig`) - attention impl, torch.compile, assisted generation
- [x] LoRA adapter loading
- [x] Decoder sampling presets (deterministic, standard, creative, factual)
- [x] Scheduled experiments (daemon mode with interval/time-of-day)
- [x] Industry-standard batching strategies (static, dynamic, sorted variants)

### CLI Improvements (Completed)
- [x] Progress bars with tqdm
- [x] Verbosity flags (`--verbose`, `--quiet`)
- [x] Simplified log format (no timestamps in normal mode)
- [x] Config display showing only non-default values

---

## In Progress

### Web Platform - Phase 1 (Active: `feature/web-frontend`)
> Core platform with leaderboard and results upload

- [ ] Public leaderboard (sortable, filterable)
- [ ] Experiment detail view
- [ ] Results upload (JSON from local CLI)
- [ ] GitHub OAuth authentication
- [ ] Deploy to Railway (API) + Vercel (frontend)

**Stack**: FastAPI + PostgreSQL + React/Vite

---

## Planned - CLI Tool

### Backend Enhancements
- [ ] Cross-backend comparison CLI: `llm-energy-measure compare --backends pytorch,vllm`
- [ ] Cross-backend visualisation (energy vs throughput per backend)

### Streaming Latency
- [x] vLLM streaming - TTFT and ITL measurement
- [x] PyTorch streaming - `TextIteratorStreamer` for TTFT/ITL
- [ ] TensorRT-LLM streaming - Code complete, requires NGC login to test

### Streaming Enhancements (Future)
- [ ] Real-time CLI token display during inference
- [ ] TTFT component breakdown (tokenisation/prefill/decode phases)
- [ ] vLLM queue time separation (isolate scheduling delay from compute)
- [ ] Energy-per-token correlation (TTFT energy vs decode energy)
- [ ] Speculative decoding acceptance rate metrics
- [ ] KV cache hit rate metrics (prefix caching effectiveness)

### Parallelism Refactor (Phase B - Deferred)
> Unified parallelism handling across backends. See conversation transcript for full plan.

- [ ] Config schema redesign - unified `parallelism:` block
- [ ] Backend capability matrix formalisation
- [ ] Result collection abstraction (backend-agnostic)
- [ ] Documentation overhaul with capability matrix

### Causal Analysis (Downstream)
> For unpacking *why* optimisations affect efficiency

- [ ] PagedAttention block usage metrics (vLLM memory efficiency)
- [ ] CUDA graph capture timing breakdown
- [ ] KV cache hit rates (prefix caching effectiveness)
- [ ] Attention kernel profiling (flash vs standard)

### Documentation
- [ ] Convert CLAUDE.md files to admin/developer READMEs

---

## Planned - Web Platform

### Phase 2: Analysis Features
> Interactive visualisation and comparison

**Key Charts** (Priority Order):
- [ ] Efficiency scatter - X=Throughput, Y=Energy efficiency, Size=Model size, Colour=Backend
- [ ] Comparison bar charts - Side-by-side metrics
- [ ] Multi-cycle distribution - Box plots with confidence intervals
- [ ] Configuration sensitivity - Heatmap of batch_size × precision
- [ ] Hardware scaling - Efficiency vs GPU count

**Features**:
- [ ] Interactive comparison charts (Plotly)
- [ ] Statistical visualisations (multi-cycle data)
- [ ] Advanced filtering/search
- [ ] Shareable comparison links
- [ ] OpenAPI docs + SDK

### Phase 3: Remote Execution
> GPU worker integration for remote benchmarking

**Architecture**:
```
Cloud: Frontend (Vercel) + API (Railway) + Redis
  ↓ (job queue)
Self-Hosted: Worker Agent → Polls Redis → Runs CLI → Uploads results
  ↓
GPU Workers
```

- [ ] Job queue infrastructure (Celery + Redis)
- [ ] GPU worker integration (self-hosted cluster)
- [ ] Config builder UI
- [ ] Real-time progress updates (WebSocket)
- [ ] Cost estimation before submission

### Phase 4: Community Features
> User ecosystem and governance

- [ ] User profiles and submission history
- [ ] Moderation workflow for submissions
- [ ] Standardised hardware profiles (A100, H100, etc.)
- [ ] HuggingFace Dataset export for research
- [ ] Embeddable leaderboard widget
- [ ] Email/webhook notifications

---

## Open Questions

1. **Result validation**: How to verify submitted benchmarks are legitimate?
2. **Standardised hardware**: Define canonical A100-80GB, H100 configurations?
3. **Pricing model**: Free tier limits? Pay-per-benchmark for remote execution?

---

## References

- Web strategy: [docs/strategy.md](docs/strategy.md)
- Backend docs: [docs/backends.md](docs/backends.md)
- CLI reference: [docs/cli.md](docs/cli.md)
