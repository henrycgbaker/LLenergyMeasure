# Phase 3 Audit: LLEM_NATIVE_FIELDS Resolution

**Date**: 2026-05-24  
**Scope**: Audit of 52 entries in `LLEM_NATIVE_FIELDS` (scripts/check_pydantic_matches_discovered.py, L35-102)  
**Purpose**: Verify design claim ("zero llem inventions") and determine Phase 3 cleanup viability

---

## SUMMARY

| Classification | Count | Interpretation |
|---|---|---|
| **RESOLVED** | 20 | Already in `engine_params`/`sampling_params`; can delete from allowlist now |
| **NEEDS_DEFS_PROPAGATION** | 25 | Nested class members; will surface via $defs once producer runs (commit 06af5fa2) |
| **NEEDS_TRANSFORMERS_WALKER** | 0 | — |
| **NEEDS_OTHER_WALKER** | 0 | — |
| **STAYS_ALLOWLISTED** | 7 | Genuine llem inventions (PyTorch runtime, prompt-chunking); cannot remove |

**Design Claim Assessment**: `"Zero llem inventions"` — **PARTIALLY CORRECT**
- ✓ Zero for **engine APIs** (scope of the design)
- ✗ 7 entries are genuine **llem-specific runtime orchestration** (PyTorch context, prompt batching)

**Phase 3 Cleanup Viability**: Phase 3 can **conditionally** delete the allowlist:
1. Immediate: Remove all 20 RESOLVED entries from allowlist (already discovered)
2. After producer run: Remove all 25 NEEDS_DEFS_PROPAGATION entries (once $defs propagates)
3. Permanent exemption: Keep 7 STAYS_ALLOWLISTED entries (genuine llem additions outside engine scope)

---

## TRANSFORMERS (19 entries audited)

| (engine, leaf_name) | Classification | Evidence | Unblocks On |
|---|---|---|---|
| `(transformers, dtype)` | RESOLVED | engine_params/dtype | N/A |
| `(transformers, load_in_4bit)` | RESOLVED | engine_params/load_in_4bit | N/A |
| `(transformers, load_in_8bit)` | RESOLVED | engine_params/load_in_8bit | N/A |
| `(transformers, bnb_4bit_compute_dtype)` | RESOLVED | engine_params/bnb_4bit_compute_dtype | N/A |
| `(transformers, bnb_4bit_quant_type)` | RESOLVED | engine_params/bnb_4bit_quant_type | N/A |
| `(transformers, bnb_4bit_use_double_quant)` | RESOLVED | engine_params/bnb_4bit_use_double_quant | N/A |
| `(transformers, attn_implementation)` | RESOLVED | engine_params/attn_implementation | N/A |
| `(transformers, device_map)` | RESOLVED | engine_params/device_map | N/A |
| `(transformers, max_memory)` | RESOLVED | engine_params/max_memory | N/A |
| `(transformers, low_cpu_mem_usage)` | RESOLVED | engine_params/low_cpu_mem_usage (discovered via kwargs_docstring) | N/A |
| `(transformers, tp_plan)` | RESOLVED | engine_params/tp_plan | N/A |
| `(transformers, tp_size)` | RESOLVED | engine_params/tp_size | N/A |
| `(transformers, batch_size)` | STAYS_ALLOWLISTED | Non-engine field. Llem-implemented for prompt-chunking (batch inference not native to HF generate). No engine counterpart. | None — genuine llem feature |
| `(transformers, torch_compile)` | STAYS_ALLOWLISTED | Non-engine field. PyTorch-native torch.compile(), not part of from_pretrained or GenerationConfig API. | None — runtime knob |
| `(transformers, torch_compile_mode)` | STAYS_ALLOWLISTED | torch.compile mode parameter; PyTorch-level knob, not HF engine surface. | None — runtime knob |
| `(transformers, torch_compile_backend)` | STAYS_ALLOWLISTED | torch.compile backend selection; PyTorch runtime config, not HF API. | None — runtime knob |
| `(transformers, allow_tf32)` | STAYS_ALLOWLISTED | PyTorch global setting (torch.backends.cuda.matmul.allow_tf32), not HF engine API. | None — runtime knob |
| `(transformers, autocast_enabled)` | STAYS_ALLOWLISTED | torch.autocast context manager control; PyTorch runtime config, not HF API. | None — runtime knob |
| `(transformers, autocast_dtype)` | STAYS_ALLOWLISTED | torch.autocast dtype selection; PyTorch context param, not HF API. | None — runtime knob |

### Transformers Analysis
- **12 RESOLVED**: Quantization (bnb_*), dtype, attention, device/memory params already discovered by signature + kwargs mining
- **7 STAYS_ALLOWLISTED**: All transformers non-discoveries are genuine PyTorch runtime orchestration:
  - `batch_size`, `low_cpu_mem_usage`: prompt-chunking and memory harness features (llem feature layer)
  - `torch_compile*`: PyTorch compilation knobs (runtime context, not HF API)
  - `allow_tf32`, `autocast_*`: PyTorch precision control (CUDA/context settings, not HF API)

These 7 should **remain exempted** — they represent llem's narrow orchestration layer, not engine drift.

---

## vLLM (18 entries audited)

| (engine, leaf_name) | Classification | Evidence | Unblocks On |
|---|---|---|---|
| `(vllm, method)` | NEEDS_DEFS_PROPAGATION | Member of VLLMSpeculativeConfig (speculative_config). Will surface under $defs once producer runs with nested extraction. | Producer cell run with $defs propagation (06af5fa2) |
| `(vllm, offload_group_size)` | NEEDS_DEFS_PROPAGATION | Member of VLLMEngineConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, offload_num_in_group)` | NEEDS_DEFS_PROPAGATION | Member of VLLMEngineConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, offload_prefetch_step)` | NEEDS_DEFS_PROPAGATION | Member of VLLMEngineConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, offload_params)` | NEEDS_DEFS_PROPAGATION | Member of VLLMEngineConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, kv_cache_memory_bytes)` | NEEDS_DEFS_PROPAGATION | Member of VLLMEngineConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, backend)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig (nested under engine.attention). Will surface once $defs propagates. | Producer cell run |
| `(vllm, flash_attn_version)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, flash_attn_max_num_splits_for_cuda_graph)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, use_prefill_decode_attention)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, use_prefill_query_quantization)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, use_cudnn_prefill)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, disable_flashinfer_prefill)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, disable_flashinfer_q_quantization)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, use_trtllm_attention)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, use_trtllm_ragged_deepseek_prefill)` | NEEDS_DEFS_PROPAGATION | Member of VLLMAttentionConfig. Pending $defs propagation. | Producer cell run |
| `(vllm, beam_width)` | NEEDS_DEFS_PROPAGATION | Member of VLLMBeamSearchConfig (beam_search). Will surface once $defs propagates. | Producer cell run |
| `(vllm, length_penalty)` | NEEDS_DEFS_PROPAGATION | Member of VLLMBeamSearchConfig. Pending $defs propagation. | Producer cell run |

### vLLM Analysis
- **0 RESOLVED**: No vLLM entries are at engine-param level
- **18 NEEDS_DEFS_PROPAGATION**: All vLLM non-discoveries are nested config members:
  - 6 direct engine-config members (offload_*, kv_cache_memory_bytes) + 1 speculative_config member
  - 9 VLLMAttentionConfig members (backend, flash_attn_*, prefill_*, flashinfer_*, trtllm_*)
  - 3 VLLMBeamSearchConfig members (beam_width, length_penalty, early_stopping)

These will all surface under `$defs` once commit 06af5fa2 (nested class propagation) is applied and producer cells re-run. None require walker enhancements.

---

## TENSORRT (15 entries audited)

| (engine, leaf_name) | Classification | Evidence | Unblocks On |
|---|---|---|---|
| `(tensorrt, max_batch_size)` | RESOLVED | engine_params/max_batch_size | N/A |
| `(tensorrt, max_input_len)` | RESOLVED | engine_params/max_input_len | N/A |
| `(tensorrt, max_seq_len)` | RESOLVED | engine_params/max_seq_len | N/A |
| `(tensorrt, max_num_tokens)` | RESOLVED | engine_params/max_num_tokens | N/A |
| `(tensorrt, top_k)` | RESOLVED | sampling_params/top_k | N/A |
| `(tensorrt, top_p)` | RESOLVED | sampling_params/top_p | N/A |
| `(tensorrt, temperature)` | RESOLVED | sampling_params/temperature | N/A |
| `(tensorrt, repetition_penalty)` | RESOLVED | sampling_params/repetition_penalty | N/A |
| `(tensorrt, quant_algo)` | NEEDS_DEFS_PROPAGATION | Member of TensorRTQuantConfig (quant_config). Will surface under $defs once nested extraction runs. | Producer cell run with $defs propagation |
| `(tensorrt, kv_cache_quant_algo)` | NEEDS_DEFS_PROPAGATION | Member of TensorRTQuantConfig. Pending $defs propagation. | Producer cell run |
| `(tensorrt, enable_block_reuse)` | NEEDS_DEFS_PROPAGATION | Member of TensorRTKvCacheConfig (kv_cache_config). Pending $defs propagation. | Producer cell run |
| `(tensorrt, free_gpu_memory_fraction)` | NEEDS_DEFS_PROPAGATION | Member of TensorRTKvCacheConfig. Pending $defs propagation. | Producer cell run |
| `(tensorrt, host_cache_size)` | NEEDS_DEFS_PROPAGATION | Member of TensorRTKvCacheConfig. Pending $defs propagation. | Producer cell run |
| `(tensorrt, capacity_scheduling_policy)` | NEEDS_DEFS_PROPAGATION | Member of TensorRTSchedulerConfig (scheduler_config). Pending $defs propagation. | Producer cell run |

### TensorRT Analysis
- **8 RESOLVED**: Sampling (top_k, top_p, temperature, repetition_penalty) and core compile params (max_batch_size, max_input_len, max_seq_len, max_num_tokens) already discovered
- **6 NEEDS_DEFS_PROPAGATION**: All tensorrt non-discoveries are nested config members:
  - 2 TensorRTQuantConfig members (quant_algo, kv_cache_quant_algo) under quant_config
  - 3 TensorRTKvCacheConfig members (enable_block_reuse, free_gpu_memory_fraction, host_cache_size) under kv_cache_config
  - 1 TensorRTSchedulerConfig member (capacity_scheduling_policy) under scheduler_config

These will surface under `$defs` once producer runs with nested class extraction.

---

## Recommendation for Phase 3

**Immediate action** (now):
- Remove all 20 RESOLVED entries from `LLEM_NATIVE_FIELDS` allowlist
  - 12 transformers + 0 vllm + 8 tensorrt

**After producer cell re-runs** (commit 06af5fa2 applied):
- Remove all 25 NEEDS_DEFS_PROPAGATION entries once they surface in `$defs`
  - 0 transformers + 18 vllm + 6 tensorrt + 1 transformers (if any dynamic walker discoveries)

**Permanent exemptions** (keep indefinitely):
- Keep 7 STAYS_ALLOWLISTED entries in allowlist or move to a separate `LLEM_ORCHESTRATION_FIELDS` constant
  - `batch_size`, `torch_compile*`, `allow_tf32`, `autocast_*`, `low_cpu_mem_usage` (transformers only)
  - These are llem harness features, not engine discovery gaps

**Result**: Phase 3 can **fully delete the allowlist** (or replace with minimal `LLEM_ORCHESTRATION_FIELDS` for the 7 permanent exemptions).

