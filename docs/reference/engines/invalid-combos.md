# Invalid Parameter Combinations

> Auto-generated from config validators and test results.

This document lists parameter combinations that will fail validation or runtime.
The tool validates these at config load time and provides clear error messages.
The full verified rule set per engine lives at
`src/llenergymeasure/engines/<engine>/rules.yaml`.

## Config Validation Errors

These combinations are rejected at config load time with a clear error message.

| Engine | Invalid Combination | Reason | Resolution |
|---------|---------------------|--------|------------|
| transformers | `load_in_4bit=True + load_in_8bit=True` | Cannot use both 4-bit and 8-bit quantization simultaneously | Choose one: transformers.engine_params.load_in_4bit=true OR transformers.engine_params.load_in_8bit=true |
| transformers | `torch_compile_mode without torch_compile=True` | torch_compile_mode/torch_compile_backend only take effect when torch_compile=True | Set harness.transformers.torch_compile=true when using torch_compile_mode or torch_compile_backend |
| transformers | `bnb_4bit_* without load_in_4bit=True` | BitsAndBytes 4-bit options require 4-bit quantization to be enabled | Set transformers.engine_params.load_in_4bit=true when using bnb_4bit_compute_dtype, bnb_4bit_quant_type, or bnb_4bit_use_double_quant |
| transformers | `cache_implementation with use_cache=False` | Cannot specify a cache strategy when caching is explicitly disabled | Remove use_cache=false or remove cache_implementation |
| all | `engine section mismatch` | Engine section must match the engine field | Ensure transformers:/vllm:/tensorrt: section matches engine: field |
| all | `passthrough_kwargs key collision` | passthrough_kwargs keys must not collide with ExperimentConfig fields | Use named fields directly instead of passthrough_kwargs |
| tensorrt | `dtype=float32` | TensorRT-LLM is optimised for lower-precision inference | Use dtype='float16' or 'bfloat16' |
| vllm | `load_in_4bit or load_in_8bit` | vLLM does not support bitsandbytes quantization | Use vllm.engine_params.quantization (awq, gptq, fp8) for quantized inference |

## Runtime Limitations

These combinations pass config validation but may fail at runtime
due to hardware, model, or package requirements.

| Engine | Parameter | Limitation | Resolution |
|---------|-----------|------------|------------|
| transformers | `transformers.engine_params.attn_implementation=flash_attention_2` | flash-attn requires Ampere+ GPU (SM80+); fails on older architectures | Use attn_implementation='sdpa' on pre-Ampere GPUs |
| transformers | `transformers.engine_params.attn_implementation=flash_attention_3` | FA3 requires the flash_attn_3 package (built from flash-attn hopper/ directory) and Ampere+ GPU (SM80+). The Docker PyTorch image includes it pre-built | Install flash_attn_3 from source, or use the Docker runner |
| vllm | `vllm.engine_params.kv_cache_dtype=fp8` | FP8 KV cache requires Hopper (H100) or newer GPU | Use kv_cache_dtype='auto' for automatic selection |
| vllm | `vllm.engine_params.attention.backend=flashinfer` | FlashInfer requires JIT compilation on first use | Leave attention.backend unset (auto) or use 'flash_attn' |
| vllm | `vllm.engine_params.quantization=awq/gptq` | Requires a pre-quantized model checkpoint | Use a quantized model (e.g., TheBloke/*-AWQ) or omit |
| tensorrt | `tensorrt.engine_params.quant_config.quant_algo=FP8` | FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM80) raises ConfigurationError - no silent emulation or fallback | Use INT8, W4A16_AWQ, W4A16_GPTQ, or W8A16 on A100 |
| tensorrt | `tensorrt.engine_params.quant_config.quant_algo=INT8` | INT8 quantisation requires a calibrated checkpoint; uncalibrated weights degrade accuracy | Use a pre-quantised checkpoint or a weight-only algo (W4A16_AWQ, W4A16_GPTQ, W8A16) |

## Engine Capability Matrix

| Feature | Transformers | vLLM | TensorRT |
|---------|---------|------|----------|
| Tensor Parallel | Yes | Yes | Yes |
| Data Parallel | No | No | No |
| BitsAndBytes (4-bit) | Yes | No | No |
| BitsAndBytes (8-bit) | Yes | No | No |
| Native Quantization | No | AWQ/GPTQ/FP8 | INT8/W4A16_AWQ/W4A16_GPTQ/FP8 |
| float32 precision | Yes | No | No |
| float16 precision | Yes | Yes | Yes |
| bfloat16 precision | Yes | Yes | Yes |
| Prefix Caching | No | Yes | No |
| torch.compile | Yes | No | No |
| Beam Search | Yes | Yes | No |
| Speculative Decoding | Yes | Yes | No |
| Static KV Cache | Yes | No | No |

**Notes:**
- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes
- TensorRT-LLM is optimised for FP16/BF16/INT8, not FP32

## Recommended Configurations by Use Case

### Memory-Constrained (Consumer GPU)
```yaml
engine: transformers
transformers:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
```

### High Throughput (Production)
```yaml
engine: vllm
vllm:
  engine:
    gpu_memory_utilization: 0.9
    enable_prefix_caching: true
```

### Maximum Performance (Ampere+)
```yaml
engine: tensorrt
tensorrt:
  dtype: float16
  quant_config:
    quant_algo: FP8  # Hopper only
```
