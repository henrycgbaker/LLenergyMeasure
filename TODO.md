# TODO

## Planned Features

### Backend Support
- [ ] **vLLM backend** - PagedAttention, continuous batching, tensor parallelism
- [ ] **TensorRT-LLM backend** - Optimised NVIDIA inference
- [ ] Backend selection via `backend:` config option (currently only pytorch)

### Sharding / Parallelism
- [ ] **Tensor parallelism** - Split layers across GPUs (lower latency)
- [ ] **Pipeline parallelism** - Split model stages across GPUs
- [ ] Implement via backend integrations (vLLM/TensorRT handle this natively)
- [ ] Wire up existing `sharding:` config to actual implementation

### Documentation
- [ ] Convert CLAUDE.md files to admin/developer READMEs (user-facing vs contributor docs)

### Notes
- Sharding config exists but is not yet wired up to model loading
- `device_map="auto"` provides basic multi-GPU support but not true parallelism
- vLLM integration would provide both backend optimisations and parallelism
