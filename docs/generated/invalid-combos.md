# Invalid Parameter Combinations

> Auto-generated from config validators and test results.
> Last updated: 2026-02-01 14:13

This document lists parameter combinations that will fail validation or runtime.
The tool validates these at config load time and provides clear error messages.

## Config Validation Errors

These combinations are rejected at config load time with a clear error message.

| Backend | Invalid Combination | Reason | Resolution |
|---------|---------------------|--------|------------|
| pytorch | `parallelism.strategy=pipeline_parallel` | PyTorch's generate() requires full model access for autoregressive generation | Use backend='vllm' or backend='tensorrt' for pipeline parallel |
| vllm | `parallelism.strategy=data_parallel` | vLLM manages multi-GPU internally via Ray/tensor parallel | Use tensor_parallel_size or pipeline_parallel_size |
| vllm | `quantization.load_in_8bit=True` | vLLM does not support bitsandbytes 8-bit quantization | Use vllm.quantization (awq, gptq, fp8) for quantized inference |
| tensorrt | `fp_precision=float32` | TensorRT-LLM is optimised for lower precision inference | Use fp_precision='float16' or 'bfloat16' |
| tensorrt | `quantization.load_in_4bit=True` | TensorRT does not support bitsandbytes quantization | Use tensorrt.quantization (fp8, int8_sq, int4_awq) |
| tensorrt | `quantization.load_in_8bit=True` | TensorRT does not support bitsandbytes quantization | Use tensorrt.quantization (fp8, int8_sq, int8_weight_only) |
| all | `quantization.load_in_4bit + load_in_8bit` | Cannot use both 4-bit and 8-bit quantization simultaneously | Choose one: load_in_4bit=True OR load_in_8bit=True |

## Streaming Mode Constraints

When `streaming=True`, certain parameters are ignored or behave differently
because streaming requires sequential per-request processing to measure TTFT/ITL.

| Backend | Parameter | Behaviour with streaming=True | Impact |
|---------|-----------|------------------------------|--------|
| pytorch | `pytorch.batch_size` | Ignored - streaming processes 1 request at a time | See docs |
| pytorch | `pytorch.batching_strategy` | Ignored - always sequential in streaming mode | See docs |
| vllm | `vllm.max_num_seqs` | Effectively 1 in streaming mode for accurate TTFT | See docs |
| pytorch | `pytorch.torch_compile` | May cause graph-tracing errors, falls back to uncompiled | See docs |
| vllm | `vllm.enable_chunked_prefill` | May interfere with TTFT measurement accuracy | See docs |

**When to use streaming=True:**
- Measuring user-perceived latency (TTFT, ITL)
- Evaluating real-time chat/assistant workloads
- MLPerf inference latency benchmarks

**When to use streaming=False:**
- Throughput benchmarking
- Batch processing workloads
- torch.compile optimisation testing

## Runtime Limitations

These combinations pass config validation but may fail at runtime
due to hardware, model, or package requirements.

| Backend | Parameter | Limitation | Resolution |
|---------|-----------|------------|------------|
| pytorch | `pytorch.attn_implementation=flash_attention_2` | flash-attn package not installed in Docker image | Install flash-attn or use attn_implementation='sdpa' |
| vllm | `vllm.kv_cache_dtype=fp8` | FP8 KV cache requires Hopper (H100) or newer GPU | Use kv_cache_dtype='auto' for automatic selection |
| vllm | `vllm.attention.backend=FLASHINFER` | FlashInfer requires JIT compilation on first use | Use attention.backend='auto' or 'FLASH_ATTN' |
| vllm | `vllm.attention.backend=TORCH_SDPA` | TORCH_SDPA not registered in vLLM attention backends | Use attention.backend='auto' or 'FLASH_ATTN' |
| vllm | `vllm.quantization_method=awq/gptq` | Requires a pre-quantized model checkpoint | Use a quantized model (e.g., TheBloke/*-AWQ) or omit |
| vllm | `vllm.load_format=pt` | Model checkpoint must have .bin files (not just safetensors) | Use load_format='auto' or 'safetensors' |
| tensorrt | `tensorrt.quantization.method=int8_sq` | INT8 SmoothQuant requires calibration dataset | Provide tensorrt.quantization.calibration config or use fp8 |

## Backend Capability Matrix

| Feature | PyTorch | vLLM | TensorRT |
|---------|---------|------|----------|
| Tensor Parallel | ✅ | ✅ | ✅ |
| Pipeline Parallel | ❌ | ✅ | ✅ |
| Data Parallel | ✅ | ❌ | ✅ |
| BitsAndBytes (4-bit) | ✅ | ❌ | ❌ |
| BitsAndBytes (8-bit) | ✅ | ❌ | ❌ |
| Native Quantization | ❌ | ✅ (AWQ/GPTQ/FP8) | ✅ (FP8/INT8) |
| float32 precision | ✅ | ✅ | ❌ |
| float16 precision | ✅ | ✅ | ✅ |
| bfloat16 precision | ✅ | ✅ | ✅ |
| Streaming (TTFT/ITL) | ✅ | ✅ | ✅ |
| LoRA Adapters | ✅ | ✅ | ✅ |
| Speculative Decoding | ✅ | ✅ | ✅ |

**Notes:**
- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes
- TensorRT-LLM is optimised for FP16/BF16/INT8 precision, not FP32

## Recommended Configurations by Use Case

### Memory-Constrained (Consumer GPU)
```yaml
backend: pytorch
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
```

### High Throughput (Production)
```yaml
backend: vllm
vllm:
  gpu_memory_utilization: 0.9
  enable_prefix_caching: true
```

### Maximum Performance (Ampere+)
```yaml
backend: tensorrt
fp_precision: float16
tensorrt:
  quantization:
    method: fp8  # Hopper only
```
