# Peer Research: FLOPs Confidence & Quantisation Handling

> Generated 2026-02-26. Peer evidence for preservation audit item N-C05.

## Context

Our v1.x codebase has ~200 LOC with two competing models (`ComputeMetrics` with strings
vs `FlopsResult` with typed enums). `FlopsResult.confidence` distinguishes calflops (high)
from architecture-estimate (medium) from parameter-count-estimate (low). A critical
correctness constraint: BitsAndBytes INT4 FLOPs = FP16 FLOPs because computation happens
at FP16 post-dequantisation, not 1/4 FLOPs.

Additionally, `PrecisionMetadata.precision_factor` currently maps INT4 -> 0.25 and
INT8 -> 0.5, which would systematically misreport efficiency for weight-only quantised
models if applied to FLOPs (the compute is FP16, not INT4).

## Evidence Per Tool

### 1. calflops (MrYxJ/calculate-flops.pytorch)

**What it measures**: Theoretical FLOPs, MACs, and parameter counts. v0.3.2 (June 2024).

**Method**: Hook-based instrumentation on PyTorch modules. Registers forward hooks on
`nn.Module` subclasses to count multiply-accumulate operations based on layer specs and
input dimensions. Supports per-submodule breakdown. Also supports HuggingFace models via
model name lookup.

**Confidence reporting**: None. Acknowledges that "number of floating-point operations is
a theoretical estimation, thus FLOPS computed using that could be larger than the maximum
system throughput." No confidence score, uncertainty interval, or reliability flag.

**Quantised model handling**: No support documented. No mention of INT4, INT8,
BitsAndBytes, GPTQ, or AWQ in the README or PyPI documentation. When run on a
BitsAndBytes-quantised model, calflops will likely either (a) see the dequantised FP16
ops and count correctly by accident, or (b) fail on the non-standard `Linear4bit` layer
types. Behaviour is undefined and untested.

**Storage vs compute precision**: No distinction. Counts operations without any
precision-awareness.

**Sources**: [GitHub](https://github.com/MrYxJ/calculate-flops.pytorch),
[PyPI](https://pypi.org/project/calflops/)

### 2. fvcore (Facebook Research)

**What it measures**: FLOPs (treating 1 MAC = 1 FLOP) at operator and module level.

**Method**: JIT tracing via `torch.jit.trace`. Traces model execution, analyses the
computation graph, and applies operator-specific handlers to count FLOPs. Module hooks
are only used during tracing to insert per-module scope information.

**Confidence reporting**: None. Explicitly acknowledges: "FLOP is not a well-defined
concept. The system provides an estimate based on a specific definition." No confidence
score or quality metric.

**Quantised model handling**: No support. Supported operators include standard ops
(convolution, linear, batch norm, etc.) but no quantisation-specific handlers.
Quantised ops would appear as `unsupported_ops` contributing zero to the FLOP count.

**Storage vs compute precision**: No distinction. Counts operations without precision
awareness.

**Key limitation**: Relies on `torch.jit.trace`, which "currently prunes away ops that
are not used by results" -- silent undercounting for models with control flow.

**Sources**: [fvcore flop_count docs](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md),
[DeepWiki](https://deepwiki.com/facebookresearch/fvcore/3.2-flop-counting)

### 3. DeepSpeed flops_profiler

**What it measures**: FLOPs, latency, and parameter counts per module.

**Method**: Module-level hook-based profiling. Captures `torch.nn.functional` calls
within modules (enabling Megatron-LM support). Estimates backward pass as 2x forward
pass FLOPs.

**Confidence reporting**: None. Same caveat: "Number of floating-point operations is a
theoretical estimation." No confidence score.

**Quantised model handling**: Examined the source code
(`deepspeed/profiling/flops_profiler/profiler.py`). **No quantisation awareness
whatsoever.** No checks for int8, int4, quantisation flags, or dtype in the FLOPs
calculation paths. `_linear_flops_compute` uses `input.numel() * out_features` without
any precision multiplier. Will overestimate FLOPs and parameter counts for quantised
models (counts the full-precision equivalent).

**Storage vs compute precision**: No distinction. A `DEFAULT_PRECISION` constant exists
but only controls output formatting digits, not numerical precision of the estimate.

**Sources**: [Tutorial](https://www.deepspeed.ai/tutorials/flops-profiler/),
[Source code](https://deepspeed.readthedocs.io/en/stable/_modules/deepspeed/profiling/flops_profiler/profiler.html)

### 4. torchinfo

**What it measures**: Model summaries -- parameter counts, output shapes, and
"mult_adds" (MACs).

**Method**: Forward hook instrumentation on `nn.Module` subclasses. Runs a forward pass
and records shapes, parameter counts, and multiply-accumulate operations.

**Confidence reporting**: None. Provides raw counts only.

**Quantised model handling**: No documented support. No mention of quantised layer types
(`torch.nn.quantized.Linear`, `Linear4bit`, etc.) in documentation. Likely to either
skip or miscount quantised layers.

**Storage vs compute precision**: No distinction. Accepts `dtypes` as input parameter
for input tensor creation but does not adjust mult_adds calculations based on precision.

**Sources**: [GitHub](https://github.com/TylerYep/torchinfo),
[PyPI](https://pypi.org/project/torchinfo/)

### 5. ptflops (sovrasov/flops-counter.pytorch)

**What it measures**: FLOPs (multiply-add operations) for inference.

**Method**: Two backends: (a) `aten` backend (default) tracks low-level PyTorch ops
(`aten.mm`, `aten.matmul`, `aten.addmm`, `aten.bmm`, `aten.convolution`); (b) `pytorch`
backend (legacy) monitors `nn.Module` instances for specific layer types. Launches model
on random tensor and counts operations.

**Confidence reporting**: None. Deterministic theoretical counts.

**Quantised model handling**: No explicit support documented. No mention of quantisation
compatibility.

**Storage vs compute precision**: No distinction.

**Note**: ptflops is the legacy FLOPs counter used in our v1.x `compute_metrics.py`
(the `get_flops()` code path). It has the same blind spot as all other tools.

**Sources**: [GitHub](https://github.com/sovrasov/flops-counter.pytorch)

### 6. HuggingFace Optimum + optimum-benchmark

**What it measures**: Optimum focuses on model export (ONNX) and quantisation
(static/dynamic via OnnxRuntime). optimum-benchmark measures latency, throughput, and
model size.

**FLOPs estimation**: Neither Optimum nor optimum-benchmark reports FLOPs. The benchmark
tracks latency, throughput, and model size. Quantisation is handled as an optimisation
pass (MinMax, Entropy, Percentile calibration), not as a FLOPs-affecting transformation.

**Quantised model handling**: Extensive support for quantisation as an optimisation step,
but no FLOPs accounting for quantised operations.

**Storage vs compute precision**: Not relevant (no FLOPs reporting).

**Sources**: [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark),
[Quantisation docs](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/quantization)

### 7. BitsAndBytes -- Actual Compute Precision

**Critical finding for our tool.** From the official HuggingFace blog and BitsAndBytes
documentation:

**4-bit (NF4/FP4)**: "the computation is not done in 4bit, the weights and activations
are compressed to that format and the computation is still kept in the desired or native
dtype." Weights are stored as 4-bit NormalFloat. They are dequantised to the
`bnb_4bit_compute_dtype` (default: `torch.float32`, commonly set to `torch.bfloat16` or
`torch.float16`) before every forward pass. **All matrix multiplications happen at the
compute dtype, not at 4-bit.** Memory savings come from compressed storage; FLOPs are
identical to the full-precision model.

**8-bit (LLM.int8())**: Uses mixed-precision decomposition. Outlier features (magnitude
>= 6.0, ~0.1% of feature dimensions) are computed in FP16. Non-outlier features are
computed in INT8 with vector-wise quantisation, then dequantised back to FP16. The final
output is FP16. This is a genuine mixed-precision regime where ~99.9% of matmul
operations happen at INT8 but with FP16 accumulation.

**Implication**: For BNB 4-bit, FLOPs = FP16 FLOPs exactly (weight-only quantisation;
compute is FP16). For BNB 8-bit, it is genuinely mixed: ~99.9% INT8 + ~0.1% FP16, but
with FP16 accumulation throughout. Reporting INT8 FLOPs as 0.5x FP16 FLOPs would be
misleading because the accumulation is still FP16.

**Sources**: [4-bit blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes),
[8-bit blog](https://huggingface.co/blog/hf-bitsandbytes-integration)

### 8. GPTQ & AWQ -- Weight-Only Quantisation

**GPTQ**: Post-training weight quantisation to INT4 (or INT3). At inference time, INT4
weights are dequantised to FP16 on-the-fly. The actual matrix multiplications are
FP16xINT4 -> FP16 (or fused dequant+matmul in kernels like Marlin). Standard kernels:
load 4-bit weight, dequantise to FP16, perform FP16 matmul. Fused kernels (Marlin,
Triton): dequantise directly into tensor core register layout, perform FP16
multiply-accumulate with FP32 accumulation. **In all cases, the mathematical operations
counted as FLOPs are FP16 operations.** The benefit is memory bandwidth reduction
(4x less weight data to load), not fewer FLOPs. The W4A16 notation explicitly encodes
this: 4-bit Weights, 16-bit Activations.

**AWQ**: Same weight-only quantisation paradigm. "AWQ installs efficient W4A16 (4-bit
weight, 16-bit activation) CUDA kernels." Weights dequantised to FP16 during matmul.
Computation precision is FP16.

**Implication**: For GPTQ and AWQ, FLOPs = FP16 FLOPs. The quantisation reduces memory
and bandwidth, not compute. Reporting INT4 FLOPs as 0.25x would be a systematic error.

**Sources**: [GPTQ paper](https://arxiv.org/abs/2210.17323),
[AWQ paper](https://arxiv.org/abs/2306.00978),
[Marlin kernel](https://github.com/IST-DASLab/marlin),
[TensorRT-LLM precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)

### 9. ML.ENERGY / Zeus

**What Zeus measures**: Energy consumption (Joules) at GPU, CPU, DRAM, and platform
level. Supports NVIDIA, AMD, Apple Silicon, and Jetson.

**FLOPs reporting**: Zeus does **not** report FLOPs. The ML.ENERGY Benchmark and
Leaderboard (v3.0, December 2025) reports energy per request, average power, monetary
cost, and carbon emissions. No FLOPs in the benchmark output. Explicitly positions
itself as measuring actual energy consumption rather than relying on FLOPs as a proxy:
"popular proxies for estimating power consumption like the maximum power draw of the
hardware can sometimes be vastly off compared to actual measurement."

**Confidence reporting for energy**: The benchmark reports per-request energy at steady
state but does not publish confidence intervals or standard deviations for energy
measurements in the leaderboard output.

**Quantised model handling**: The benchmark includes quantised model configurations (the
v3.0 benchmark covers 46 models across 7 tasks), but quantisation is treated as a model
variant, not as something requiring special measurement handling.

**Sources**: [Zeus GitHub](https://github.com/ml-energy/zeus),
[ML.ENERGY benchmark paper](https://arxiv.org/abs/2505.06371),
[v3.0 blog post](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/)

### 10. llm_counts (harleyszhang)

**What it measures**: Theoretical params, FLOPs, memory, and latency from architecture
parameters (not runtime measurement).

**Method**: Pure analytical calculation from model config (hidden_size, num_layers, etc.)
and hardware specs (A100, V100, etc.). Applies efficiency factors
(`flops_efficiency`, `hbm_memory_efficiency`) to theoretical values.

**Quantised model handling**: References `BYTES_FP16` as a parameter, suggesting
FP16-only analysis. No quantisation support documented.

**Confidence reporting**: None. Outputs theoretical values directly.

**Sources**: [GitHub](https://github.com/harleyszhang/llm_counts)

### 11. Academic Literature

**"Reliability Scaling Laws for Quantized Large Language Models"** (2025, OpenReview):
Assesses trustworthiness of LLMs quantised to 2, 3, 4, and 8 bits using six methods.
Finds reliability peaks at 4-bit quantisation. Focuses on output quality reliability,
not FLOPs estimation reliability. Does not address FLOPs counting correctness.

**ICLR 2026 paper (arXiv 2602.10144)**: Examines statistical reliability of quantised
LLM evaluations. Addresses accuracy estimation practices in model compression. Does not
discuss FLOPs estimation for quantised models.

**Key gap**: No academic paper found that directly addresses the question of whether
FLOPs estimation tools correctly handle quantised models. The compute-precision vs
storage-precision distinction is well understood in systems papers and hardware docs
(TensorRT-LLM, Marlin) but has not been studied from a measurement-tool-correctness
perspective.

## Summary Table

| Tool | Method | Reports FLOPs? | Confidence? | Quantisation-Aware? | Storage vs Compute? |
|------|--------|---------------|-------------|---------------------|---------------------|
| calflops | Hooks | Yes (theoretical) | No | No | No |
| fvcore | JIT tracing | Yes (theoretical) | No | No | No |
| DeepSpeed flops_profiler | Hooks | Yes (theoretical) | No | No (verified in source) | No |
| torchinfo | Hooks | MACs only | No | No | No |
| ptflops | Hooks / aten ops | Yes (theoretical) | No | No | No |
| Optimum / optimum-benchmark | N/A | No | N/A | N/A (quantisation as optimisation) | N/A |
| Zeus / ML.ENERGY | Hardware measurement | No (energy only) | No (for energy) | Model variant, not FLOPs-aware | N/A |
| llm_counts | Analytical | Yes (theoretical) | No | No | No |

**Key finding: Zero out of eight FLOPs-reporting tools provide confidence metrics or
handle quantised models correctly.**

## The Weight-Only Quantisation FLOPs Rule

All evidence converges on the same conclusion for weight-only quantisation methods
(BitsAndBytes 4-bit, GPTQ, AWQ):

| Quantisation | Weight Storage | Compute Precision | FLOPs vs FP16 |
|-------------|---------------|-------------------|---------------|
| BNB 4-bit (NF4) | 4-bit | `bnb_4bit_compute_dtype` (FP16/BF16/FP32) | **= FP16 FLOPs** |
| BNB 8-bit (LLM.int8) | 8-bit | Mixed: ~99.9% INT8 + ~0.1% FP16, FP16 accumulation | **~ FP16 FLOPs** (INT8 matmul but FP16 accum) |
| GPTQ (W4A16) | 4-bit | FP16 (dequant on-the-fly) | **= FP16 FLOPs** |
| AWQ (W4A16) | 4-bit | FP16 (dequant on-the-fly) | **= FP16 FLOPs** |
| TRT-LLM INT4/INT8 WoQ | 4-bit/8-bit | FP16 (dequant in kernel) | **= FP16 FLOPs** |

The pattern is clear: **weight-only quantisation does not reduce FLOPs**. It reduces
memory footprint and memory bandwidth requirements. The actual floating-point operations
performed are identical to the FP16 model. Reporting INT4 FLOPs as 0.25x or INT8 FLOPs
as 0.5x FP16 is a systematic error.

The only exception would be true weight-and-activation quantisation (W4A4, W8A8) where
the matmul itself executes at reduced precision on integer tensor cores. This is used by
some TensorRT-LLM and SmoothQuant configurations but is not what BitsAndBytes, GPTQ, or
AWQ do.

## Recommendation

### 1. Our `FlopsResult.confidence` is a unique feature -- keep and strengthen it

No peer tool reports confidence. Our three-tier system (high/medium/low mapping to
calflops/architecture/parameter-estimate) is genuinely novel and valuable. However:

- **calflops "high" is generous.** calflops itself acknowledges estimates are theoretical
  and may exceed hardware throughput. Consider labelling it "measured" rather than "high"
  to distinguish method provenance from estimate quality.
- **All tools produce theoretical FLOPs.** None accounts for runtime effects (kernel
  fusion, operator scheduling, memory-bound vs compute-bound regimes). Our confidence
  tiers correctly reflect the degradation chain but should document that even "high" is
  a theoretical estimate.

### 2. The `PrecisionMetadata.precision_factor` is actively dangerous for weight-only quantisation

The current mapping `int4 -> 0.25`, `int8 -> 0.5` in `PrecisionMetadata.precision_factor`
would produce **systematically wrong** `effective_flops` for BNB/GPTQ/AWQ models. For
weight-only quantisation:

- `precision_factor` should be **1.0** (FLOPs are FP16 regardless of weight storage).
- The `precision_factor` concept only applies to true W_xA_x quantisation where both
  weights AND activations are computed at reduced precision.

**Action**: Either remove `precision_factor` entirely, or redesign it to distinguish
weight-only quantisation (factor = 1.0) from weight-and-activation quantisation
(factor < 1.0). The current implementation will silently produce wrong results for
every BNB/GPTQ/AWQ experiment.

### 3. Our BNB dequant handling in `flops.py` is correct -- preserve it

The `_get_compute_precision()` method correctly identifies BNB 4-bit/8-bit and maps to
the compute dtype (FP16/BF16). This is the right approach and is not replicated by any
peer tool. The v2.0 design should preserve this logic.

### 4. Calflops on quantised models is undefined behaviour

Our v1.x passes the quantised model directly to calflops. Calflops has no handlers for
`Linear4bit` / `Linear8bitLt` layers. This may silently produce wrong results or throw.
Consider:
- Pre-flight check: detect BNB-quantised models and skip calflops, falling back to
  architecture estimation.
- Or: dequantise before calflops measurement (expensive but correct).

### 5. Keep FLOPs as a secondary metric behind energy

Zeus/ML.ENERGY has the right instinct: actual energy measurement is the primary
efficiency metric. FLOPs are a useful secondary signal for understanding computational
cost, but they are fundamentally theoretical. Our tool already measures energy as the
primary metric; FLOPs should remain subordinate with clear provenance marking.
