# Parameter Support Matrix

> Auto-generated from test results. Run `python scripts/generate_param_matrix.py` to update.
> Last updated: 2026-02-04 14:04

## Summary

- **PYTORCH**: 3/3 (100.0%) [failed: 0, skipped: 0]
- **VLLM**: 68/83 (81.9%) [failed: 15, skipped: 0]
- **TENSORRT**: 61/65 (93.8%) [failed: 4, skipped: 0]

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Passed - parameter works correctly |
| ❌ | Failed - parameter not supported or error |
| ⚠️ | Passed but validation uncertain |
| ➖ | Not tested for this backend |

## Core Settings

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `fp_precision=bfloat16` | ➖ | ❌ | ✅ | vllm: Error: 3 validation errors for PrecisionMetadata |
| `fp_precision=float16` | ➖ | ✅ | ✅ |  |
| `fp_precision=float32` | ➖ | ✅ | ❌ | tensorrt: Error: Engine building failed, please check the error log. |

## Batching

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `batching.batch_size=1` | ➖ | ✅ | ✅ |  |
| `batching.batch_size=8` | ➖ | ✅ | ✅ |  |
| `batching.strategy=dynamic` | ➖ | ✅ | ✅ |  |
| `batching.strategy=sorted_dynamic` | ➖ | ✅ | ✅ |  |
| `batching.strategy=sorted_static` | ➖ | ✅ | ✅ |  |
| `batching.strategy=static` | ➖ | ✅ | ✅ |  |

## Decoder/Generation

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `decoder.beam_search.early_stopping` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.early_stopping=True` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.enabled` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.enabled=True` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.length_penalty=0.5` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.length_penalty=1.0` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.length_penalty=1.5` | ➖ | ⚠️ | ✅ |  |
| `decoder.beam_search.no_repeat_ngram_size` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.no_repeat_ngram_size=2` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.no_repeat_ngram_size=3` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.num_beams=1` | ➖ | ✅ | ⚠️ |  |
| `decoder.beam_search.num_beams=2` | ➖ | ✅ | ✅ |  |
| `decoder.beam_search.num_beams=4` | ➖ | ✅ | ✅ |  |
| `decoder.preset=creative` | ➖ | ✅ | ✅ |  |
| `decoder.preset=deterministic` | ➖ | ✅ | ✅ |  |
| `decoder.preset=factual` | ➖ | ✅ | ✅ |  |
| `decoder.preset=standard` | ➖ | ✅ | ✅ |  |

## Parallelism

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `parallelism.strategy=pipeline_parallel` | ➖ | ❌ | ➖ | vllm: Error: libxcb.so.1: cannot open shared object file: No such file or dir... |
| `parallelism.strategy=tensor_parallel` | ➖ | ❌ | ✅ | vllm: Error: libxcb.so.1: cannot open shared object file: No such file or dir... |

## Quantization

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `quantization.bnb_4bit_quant_type=fp4` | ➖ | ✅ | ✅ |  |
| `quantization.bnb_4bit_quant_type=nf4` | ➖ | ✅ | ✅ |  |
| `quantization.load_in_4bit` | ➖ | ✅ | ✅ |  |
| `quantization.load_in_4bit=True` | ➖ | ✅ | ❌ |  |
| `quantization.load_in_8bit` | ➖ | ✅ | ✅ |  |
| `quantization.load_in_8bit=True` | ➖ | ❌ | ❌ |  |
| `quantization.quantization` | ➖ | ✅ | ✅ |  |
| `quantization.quantization=True` | ➖ | ❌ | ❌ |  |

## Streaming & Simulation

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `streaming` | ➖ | ✅ | ✅ |  |
| `streaming=True` | ➖ | ✅ | ✅ |  |
| `traffic_simulation.enabled` | ➖ | ✅ | ✅ |  |
| `traffic_simulation.enabled=True` | ➖ | ✅ | ✅ |  |
| `traffic_simulation.mode=constant` | ➖ | ✅ | ✅ |  |
| `traffic_simulation.mode=poisson` | ➖ | ✅ | ✅ |  |

## PyTorch-specific

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `pytorch.torch_compile_backend=cudagraphs` | ⚠️ | ➖ | ➖ |  |
| `pytorch.torch_compile_backend=inductor` | ⚠️ | ➖ | ➖ |  |
| `pytorch.torch_compile_backend=onnxrt` | ⚠️ | ➖ | ➖ |  |

## vLLM-specific

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `vllm.attention.backend=FLASHINFER` | ➖ | ❌ | ➖ | vllm: Error: Command '['ninja', '-v', '-C', '/home/appuser/.cache/flashinfer/... |
| `vllm.attention.backend=FLASH_ATTN` | ➖ | ✅ | ➖ |  |
| `vllm.attention.backend=TORCH_SDPA` | ➖ | ❌ | ➖ | vllm: Error: Backend TORCH_SDPA must be registered before use. Use register_b... |
| `vllm.attention.backend=auto` | ➖ | ✅ | ➖ |  |
| `vllm.best_of=1` | ➖ | ✅ | ➖ |  |
| `vllm.best_of=2` | ➖ | ❌ | ➖ | vllm: Error: Unexpected keyword argument 'best_of' |
| `vllm.block_size=16` | ➖ | ✅ | ➖ |  |
| `vllm.block_size=32` | ➖ | ✅ | ➖ |  |
| `vllm.block_size=8` | ➖ | ❌ | ➖ | vllm: Error: No common block size for 8.  |
| `vllm.cpu_offload_gb` | ➖ | ✅ | ➖ |  |
| `vllm.cpu_offload_gb=2.0` | ➖ | ✅ | ➖ |  |
| `vllm.distributed_backend=mp` | ➖ | ✅ | ➖ |  |
| `vllm.distributed_backend=ray` | ➖ | ✅ | ➖ |  |
| `vllm.enable_chunked_prefill` | ➖ | ✅ | ➖ |  |
| `vllm.enable_chunked_prefill=True` | ➖ | ✅ | ➖ |  |
| `vllm.enable_prefix_caching` | ➖ | ✅ | ➖ |  |
| `vllm.enable_prefix_caching=True` | ➖ | ✅ | ➖ |  |
| `vllm.enforce_eager` | ➖ | ⚠️ | ➖ |  |
| `vllm.enforce_eager=True` | ➖ | ✅ | ➖ |  |
| `vllm.gpu_memory_utilization=0.5` | ➖ | ✅ | ➖ |  |
| `vllm.gpu_memory_utilization=0.6` | ➖ | ✅ | ➖ |  |
| `vllm.gpu_memory_utilization=0.7` | ➖ | ✅ | ➖ |  |
| `vllm.kv_cache_dtype=auto` | ➖ | ✅ | ➖ |  |
| `vllm.kv_cache_dtype=bfloat16` | ➖ | ❌ | ➖ | vllm: Error: No valid attention backend found for cuda with AttentionSelector... |
| `vllm.kv_cache_dtype=float16` | ➖ | ❌ | ➖ | vllm: Error: 1 validation error for CacheConfig |
| `vllm.kv_cache_dtype=fp8` | ➖ | ❌ | ➖ | vllm: Error: Command '['ninja', '-v', '-C', '/home/appuser/.cache/flashinfer/... |
| `vllm.load_format=auto` | ➖ | ✅ | ➖ |  |
| `vllm.load_format=pt` | ➖ | ❌ | ➖ | vllm: Error: Cannot find any model weights with `Qwen/Qwen2.5-0.5B` |
| `vllm.load_format=safetensors` | ➖ | ✅ | ➖ |  |
| `vllm.logprobs` | ➖ | ✅ | ➖ |  |
| `vllm.logprobs=5` | ➖ | ✅ | ➖ |  |
| `vllm.max_num_seqs=128` | ➖ | ✅ | ➖ |  |
| `vllm.max_num_seqs=256` | ➖ | ✅ | ➖ |  |
| `vllm.max_num_seqs=32` | ➖ | ✅ | ➖ |  |
| `vllm.max_num_seqs=64` | ➖ | ✅ | ➖ |  |
| `vllm.quantization_method` | ➖ | ✅ | ➖ |  |
| `vllm.quantization_method=awq` | ➖ | ❌ | ➖ | vllm: Error: 1 validation error for VllmConfig |
| `vllm.quantization_method=fp8` | ➖ | ✅ | ➖ |  |
| `vllm.quantization_method=gptq` | ➖ | ❌ | ➖ | vllm: Error: 1 validation error for VllmConfig |
| `vllm.swap_space` | ➖ | ✅ | ➖ |  |
| `vllm.swap_space=4.0` | ➖ | ✅ | ➖ |  |

## TensorRT-specific

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|---------|------|----------|-------|
| `tensorrt.builder_opt_level=2` | ➖ | ➖ | ✅ |  |
| `tensorrt.builder_opt_level=3` | ➖ | ➖ | ✅ |  |
| `tensorrt.builder_opt_level=4` | ➖ | ➖ | ✅ |  |
| `tensorrt.builder_opt_level=5` | ➖ | ➖ | ✅ |  |
| `tensorrt.enable_chunked_context` | ➖ | ➖ | ✅ |  |
| `tensorrt.enable_chunked_context=True` | ➖ | ➖ | ✅ |  |
| `tensorrt.enable_kv_cache_reuse` | ➖ | ➖ | ✅ |  |
| `tensorrt.enable_kv_cache_reuse=True` | ➖ | ➖ | ✅ |  |
| `tensorrt.force_rebuild=True` | ➖ | ➖ | ✅ |  |
| `tensorrt.gpu_memory_utilization=0.7` | ➖ | ➖ | ✅ |  |
| `tensorrt.gpu_memory_utilization=0.9` | ➖ | ➖ | ✅ |  |
| `tensorrt.kv_cache_type=continuous` | ➖ | ➖ | ✅ |  |
| `tensorrt.kv_cache_type=paged` | ➖ | ➖ | ✅ |  |
| `tensorrt.max_batch_size=1` | ➖ | ➖ | ✅ |  |
| `tensorrt.max_batch_size=4` | ➖ | ➖ | ✅ |  |
| `tensorrt.max_batch_size=8` | ➖ | ➖ | ✅ |  |
| `tensorrt.multiple_profiles` | ➖ | ➖ | ✅ |  |
| `tensorrt.multiple_profiles=True` | ➖ | ➖ | ✅ |  |
| `tensorrt.quantization.method=fp8` | ➖ | ➖ | ✅ |  |
| `tensorrt.quantization.method=int8_sq` | ➖ | ➖ | ✅ |  |
| `tensorrt.quantization.method=int8_weight_only` | ➖ | ➖ | ✅ |  |
| `tensorrt.quantization.method=none` | ➖ | ➖ | ✅ |  |
| `tensorrt.strongly_typed` | ➖ | ➖ | ✅ |  |
| `tensorrt.strongly_typed=True` | ➖ | ➖ | ✅ |  |

## Backend Recommendations

### By Use Case

| Use Case | Recommended Backend | Reason |
|----------|---------------------|--------|
| Memory-constrained | PyTorch | BitsAndBytes 4/8-bit quantization |
| High throughput | vLLM | Continuous batching, PagedAttention |
| Maximum performance | TensorRT | Compiled engine, FP8 quantization |
| Multi-GPU inference | vLLM | Best tensor/pipeline parallel |
| Development/debugging | PyTorch | Most flexible, familiar API |
