# Research: FLOPs, Warmup, LoRA, Multi-GPU Patterns

**Date**: 2026-02-19
**Purpose**: Inform decision stubs for missing decision areas in LLenergyMeasure
**Confidence**: MEDIUM overall (verified via multiple sources; see per-section notes)

---

## 1. FLOPs Estimation

### Standard Formula: Decoder-Only Inference

The canonical approximation is:

```
FLOPs_per_token ≈ 2 × N
```

where `N` = number of non-embedding parameters. This covers the forward pass multiply-add operations. One MAC = 2 FLOPs (one multiply + one add).

**Important nuance — prefill vs decode are different:**

| Phase | Compute | Bottleneck |
|-------|---------|------------|
| Prefill (prompt processing) | `2 × N × B × S` (B=batch, S=seq_len) | Compute-bound for S > ~480 tokens |
| Decode (per generated token) | `2 × N × B × 1` | Memory-bandwidth-bound (loads all params + KV cache per step) |

The prefill phase uses matrix-matrix multiplications; the decode phase degenerates to matrix-vector multiplications, making FLOP counts less meaningful for latency prediction. Theoretical FLOPs and actual latency are poorly correlated for the decode phase.

**Why 2N and not 6N?** 6N is for training (forward + 2× backward). Inference is forward-only, hence 2N. The 2 accounts for the multiply-accumulate being counted as 2 FLOPs.

**Attention FLOPs (often omitted from the 2N approximation):**
Full self-attention adds `4 × B × L × S²` FLOPs (where L = num_layers), which is significant for long contexts but negligible for short sequences with large models.

### How Peer Tools Handle FLOPs

**lm-evaluation-harness**: Does not track FLOPs at all. Zero FLOPs estimation in the codebase (confirmed: GitHub code search returns zero results for "flops"). It is a correctness/quality benchmark, not an efficiency benchmark.

**optimum-benchmark**: Does not measure FLOPs either. Tracks latency, memory, and energy (via Zeus/CodeCarbon), but the inference scenario contains no FLOPs computation. `warmup_runs=20` fixed count, reduced-output warmup.

**vLLM**: FLOPs tracking was a feature request (issue #3490, opened March 2024). Initially closed as "not planned" due to inactivity, reopened December 2024. A draft PR (#12341) titled "FLOP counting for vLLM inference" was created January 2025 — not merged as of research date. vLLM currently has **no FLOPs output** in its metrics.

**DeepSpeed FLOPs Profiler**: Available as `get_model_profile()`. Works by tracing module-level PyTorch ops. Reports: params, MACs, FLOPs, FLOPS (rate), latency, throughput. **Critical limitation**: does not support token-level generation with KV cache. Measures a single forward pass, not autoregressive generation. Cannot distinguish prefill from decode FLOPs. Backward pass is estimated as `2× forward`.

**calflops (`calculate-flops.pytorch`)**: Most LLM-capable option. Supports `forward_mode="generate"` to trace the generation function rather than a single forward pass. Formula: `FLOPs = 2 × MACs`. Works via PyTorch meta device for HuggingFace models. Still a theoretical estimate — actual hardware FLOPs differ due to kernels, sparsity, fused ops.

**fvcore FlopCountAnalysis**: Traces PyTorch execution graph, operator-level counts. General limitation: flop count is poorly correlated with GPU latency for LLMs. Not LLM-generation-aware (no generate mode). Best for CV/NLP classification models, not autoregressive generation.

**No tool provides reliable per-token decode FLOPs with KV cache.** The standard workaround is analytical estimation:

```python
def estimate_flops(
    num_params: int,             # non-embedding parameters
    batch_size: int,
    input_tokens: int,           # prompt length
    output_tokens: int,          # generated tokens
    num_layers: int,
    hidden_size: int,
) -> dict:
    prefill_flops = 2 * num_params * batch_size * input_tokens
    decode_flops = 2 * num_params * batch_size * output_tokens
    # Attention correction for long sequences (optional)
    attn_prefill = 4 * batch_size * num_layers * hidden_size * input_tokens ** 2
    total = prefill_flops + decode_flops  # attn_prefill can be added if S is large
    return {
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "total_flops": total,
        "flops_per_input_token": prefill_flops / (batch_size * input_tokens),
        "flops_per_output_token": decode_flops / (batch_size * output_tokens),
    }
```

### Surfacing FLOPs in Results

There is no strong standard, but the most informative representation is:

- `total_flops` — absolute count for the full request
- `flops_per_input_token` — normalised prefill cost
- `flops_per_output_token` — normalised decode cost (always ≈ 2N for dense models)
- `mfu` (model FLOP utilisation) — `achieved_flops / peak_theoretical_gpu_flops`, useful for comparing efficiency across hardware

**Confidence**: MEDIUM — formula derivation is well-established; peer tool behaviour verified via code search and issue tracker.

---

## 2. Warmup Strategy

### What Warmup Is Eliminating

Warmup addresses three distinct transient effects:

1. **GPU kernel compilation** (PyTorch torch.compile, TensorRT engine build) — one-time cost
2. **CUDA context and memory allocation** — first-request overhead
3. **Caching effects** — KV cache pool initialisation, attention mask compilation
4. **Thermal/frequency stabilisation** — GPU power state ramps up over first few seconds

These are unrelated to steady-state inference performance and should be excluded from measurements.

### Peer Tool Approaches

**optimum-benchmark**: Fixed `warmup_runs=20` (default). Uses **reduced-output warmup**: text generation runs with `max_new_tokens=2, min_new_tokens=2` instead of the configured value (e.g., 100 tokens), reducing warmup compute cost significantly. Image diffusion uses `num_inference_steps=2` instead of 30. Warmup occurs before any latency/energy/memory tracking begins. No convergence criterion — purely count-based.

**vLLM `bench serve`** (formerly `benchmark_serving.py`): Uses fixed warmup request count before the measurement phase. The older guidellm tool (official vLLM project) supports `--warmup` as either a fraction (0–1 = percentage of total requests) or absolute count (≥1). No CV-based stopping.

**Zeus / ML.ENERGY Benchmark** (confirmed via documentation and search):
- Zeus uses a `begin_window` / `end_window` measurement API — warmup is the responsibility of the calling code, not Zeus itself.
- The ML.ENERGY Benchmark defines "steady state" as the period when batch size is saturated at the server's maximum configured batch size. Energy/latency measurements are taken only during this steady-state window.
- `GlobalPowerLimitOptimizer` has `warmup_steps=10, profile_steps=40` defaults.
- Explicitly avoids measuring the first few requests due to JIT/memory allocation transients.

**AIEnergyScore**: Default 10 prompts per benchmark (`-n 10`). No documented warmup phase separate from the measurement window. Simple and minimal.

**MLPerf Inference**: Does not mandate a specific warmup protocol. Delegates warmup to individual implementations. Focuses on statistical validity via query count minimums (24,576 queries for offline at 99% confidence) and 600-second duration minimums. Uses early stopping with a penalty for small sample runs.

**DeepSpeed FLOPs Profiler**: `get_model_profile()` has a `warm_up` parameter (number of iterations before timing starts), typically 5–10.

### CV-Based Convergence

No peer tool was found using coefficient of variation (CV) as a stopping criterion for warmup. The dominant industry pattern is **fixed count**. However, academic literature on micro-benchmarking suggests:

- CV < 5% over a sliding window of N samples indicates stable measurement
- Window size of 10–30 samples is typical for latency convergence
- For GPU energy (power × time), CV convergence is harder to achieve due to thermal variance

The Zeus steady-state definition (batch saturation) is the closest to a principled criterion, but it is specific to online serving scenarios.

### Practical Warmup Decision

Most tools use fixed count. A pragmatic approach:

| Use Case | Warmup Strategy | Rationale |
|----------|----------------|-----------|
| Latency benchmark | 5–20 fixed runs | Eliminates JIT; cheap |
| Energy benchmark | 30–60 seconds or 20+ runs | Thermal stabilisation |
| Throughput benchmark | Until batch size saturates | Zeus steady-state pattern |
| First run (cold start) | No warmup | Explicitly measuring cold-start cost |

**Confidence**: HIGH for peer tool behaviour; MEDIUM for CV-based criterion (no peer implementation found).

---

## 3. LoRA Adapter Support

### How Peer Tools Handle LoRA

**lm-evaluation-harness**: Supports PEFT adapters via `peft=` in `model_args`. Example:

```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=nomic-ai/gpt4all-j-lora \
  --tasks hellaswag
```

The `peft=` value can be a HuggingFace Hub ID or local path — PEFT handles resolution. The adapter is loaded via `PeftModel.from_pretrained()` on top of the base model. Weights are **not merged** by default — inference runs with the adapter attached. This has a measurable ~10–12% throughput overhead vs merged weights. PEFT is an optional extra (`pip install lm_eval[peft]`).

**optimum-benchmark**: Lists PEFT as an installable extra. The benchmark infrastructure delegates LoRA loading to the backend (HF Transformers + PEFT). No dedicated LoRA-specific benchmark scenario — adapters are loaded as part of model loading configuration.

**vLLM**: First-class LoRA support via `LoRARequest`. Key API:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest(
        lora_name="sql-lora",       # human-readable identifier
        lora_int_id=1,              # globally unique integer ID
        lora_path="/path/to/adapter"  # local path required; HF Hub via snapshot_download
    )
)
```

For HF Hub adapters:
```python
from huggingface_hub import snapshot_download
lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
```

vLLM requires a local path for `lora_path`. Hub IDs are supported indirectly via resolver plugins or `snapshot_download`. Runtime loading via `/v1/load_lora_adapter` REST endpoint is also supported. Multi-LoRA: multiple adapters can be served simultaneously on one base model, switched per-request.

**Tensor parallelism + LoRA in vLLM**: Works, but by default only half of LoRA computation is sharded. Full sharding is available and faster at high sequence length or large rank.

### Merged vs Unmerged Adapter

| Mode | Latency Overhead | Memory | Flexibility |
|------|-----------------|--------|-------------|
| Unmerged (PEFT attached) | ~10–12% throughput reduction | Base + adapter weights separate | Switch adapters at runtime |
| Merged (`merge_and_unload()`) | Zero overhead | Single weight tensor | Cannot switch; base model altered |

For **measurement purposes**, unmerged is the correct default — it isolates the adapter's cost. If measuring "what does this fine-tuned model cost at production?", merged is correct (that's how it would be deployed for single-adapter use).

### What ExperimentConfig Fields Should Model LoRA

Based on peer tool patterns:

```python
class LoRAConfig(BaseModel):
    adapter_id: str | None = None       # HF Hub ID: "username/adapter-name"
    adapter_path: str | None = None     # Local filesystem path
    merge_weights: bool = False          # merge_and_unload() before measurement
    # vLLM-specific
    lora_name: str | None = None        # Human-readable name for LoRARequest
    lora_int_id: int = 1               # Unique integer ID for LoRARequest

class ExperimentConfig(BaseModel):
    ...
    lora: LoRAConfig | None = None      # None = no adapter (base model)
```

The `adapter_id` / `adapter_path` split mirrors the HF Hub vs local pattern used in lm-eval and vLLM. A single `adapter` string field that accepts either (like lm-eval's `peft=`) would also work but reduces clarity.

### Metrics Relevant to LoRA

- All standard metrics (energy, latency, throughput) apply — the adapter is just part of the model
- **Additional dimension**: measuring same prompts with base model vs adapter reveals adapter overhead directly
- For energy per token: LoRA unmerged adds ~10–12% overhead; merged adds zero
- Merge-time cost is a one-off and should not be included in inference measurements

**Confidence**: HIGH for vLLM API (official docs); MEDIUM for optimum-benchmark (indirect evidence); HIGH for merged vs unmerged overhead figures (multiple independent sources).

---

## 4. Tensor Parallelism / Multi-GPU

### What Each Backend Offers

**PyTorch / HuggingFace Transformers:**
- `device_map="auto"` — uses Accelerate to split model layers across GPUs (pipeline parallelism, not tensor parallelism). Simple but not optimal.
- `tp_plan="auto"` — true tensor parallelism via `torch.distributed` `DeviceMesh` + `DTensor`. Requires `torchrun`. Supported for models that define a `tp_plan`.
- No explicit `tp_size`/`pp_size` config fields at the library level — set via `torchrun --nproc-per-node N`.

**vLLM:**
- `tensor_parallel_size` (alias: `tp_size`) — splits attention heads and MLP weights across N GPUs
- `pipeline_parallel_size` (alias: `pp_size`) — stages layers sequentially across GPUs
- `data_parallel_size` (`dp_size`) — replicates model, splits requests
- These are first-class `LLM()` constructor parameters
- TP requires high-bandwidth NVLink; PP works with PCIe but introduces pipeline bubbles

**TensorRT-LLM:**
- Supports TP, PP, and Expert Parallelism (for MoE)
- Configured at build time and in the executor config
- Recently added joint TP + EP support for MoE models

### Energy Measurement in Multi-GPU Scenarios

**The fundamental question**: measure per-GPU or aggregate?

The empirical answer from TokenPowerBench (Dec 2024 paper, arxiv:2512.03024):

> Pure tensor parallelism delivers the best energy efficiency because long pipelines leave some GPUs idle. The gap between best and worst split widens as workload grows — from ~40 J/token (Standard Load) to >60 J/token (High Throughput).

**Standard practice**: measure **aggregate energy across all participating GPUs** using NVML polling:

```
total_energy = sum(nvml.get_total_energy(gpu_i) for gpu_i in active_gpus)
```

Zeus's `ZeusMonitor` supports this natively — pass `gpu_indices=[0, 1, 2, 3]` and `begin_window`/`end_window` returns a `Measurement` object with per-GPU breakdown and total.

Per-GPU reporting is also useful for:
- Diagnosing load imbalance (PP often leaves boundary GPUs idle)
- Detecting thermal throttling on individual GPUs

**What to report in ExperimentResult:**
- `energy_joules_total` — sum across all GPUs in the parallel group
- `energy_joules_per_gpu` — list, for imbalance analysis
- `num_gpus_used` — derived from topology config

### ExperimentConfig Multi-GPU Fields

Based on peer tool patterns (vLLM is the most complete):

```python
class ParallelismConfig(BaseModel):
    tp_size: int = 1    # tensor parallel degree
    pp_size: int = 1    # pipeline parallel degree
    dp_size: int = 1    # data parallel replicas
    # Note: dp_size > 1 changes the experiment semantics (multiple replicas,
    # not a single inference run). Treat with caution for energy measurement.
```

**PyTorch backend note**: HF Transformers does not expose these as simple integers in its Python API. For TP, torchrun must be used, making subprocess orchestration necessary. For `device_map="auto"` (pipeline-style), no extra config field is needed — Accelerate infers from available GPUs.

### How tp_size Affects Energy Measurement

From TokenPowerBench findings and general NVML practice:

1. With TP=N, all N GPUs are active during every forward pass — energy is additive
2. With PP=N, only one stage is active at a time — energy per GPU is lower but pipeline bubbles waste cycles (net efficiency is worse)
3. Aggregate energy per output token is the correct normalisation metric for comparing parallelism strategies
4. `energy_per_output_token = total_energy / (batch_size * output_tokens)` is the key efficiency metric

**Confidence**: HIGH for vLLM API fields; MEDIUM for PyTorch/Transformers API (rapidly evolving); HIGH for energy aggregation methodology; MEDIUM for TokenPowerBench-derived efficiency conclusions.

---

## Recommendations for LLenergyMeasure

### FLOPs

Use **analytical estimation only** — no library (calflops, deepspeed, fvcore) handles autoregressive generation with KV cache correctly. Implement a `_estimate_flops()` function using `2 × N_params × tokens` as the base formula, split by prefill and decode phases. Surface as:
- `flops_prefill`, `flops_decode`, `flops_total` in `ExperimentResult`
- `flops_per_output_token` (≈ 2 × N, the most comparable across runs)
- Mark all FLOPs as `estimated: true` in `measurement_methodology`
- Do not use calflops/fvcore at runtime — add complexity without accuracy for the generation case

### Warmup

Use **fixed-count warmup, reduced-output style** (matching optimum-benchmark):
- Default: `n_warmup=5` reduced-output runs
- Reduced output: generate only 2 tokens during warmup (not the full `n` tokens)
- For energy benchmarks: add a time-based floor (`min_warmup_seconds=30`) to allow thermal stabilisation
- Cold-start mode: explicitly set `n_warmup=0` (the `cold_start: bool` field already handles this)
- Do not implement CV-based convergence — no peer uses it and it adds significant complexity

### LoRA

Add an optional `lora` block to `ExperimentConfig`:
```yaml
lora:
  adapter_id: "username/my-adapter"  # OR
  adapter_path: "/local/path"
  merge_weights: false  # recommended false for measurement (isolates adapter cost)
```
- Support both `adapter_id` (HF Hub) and `adapter_path` (local) — mirror lm-eval's `peft=` flexibility
- Default `merge_weights: false` so adapter overhead is measured, not hidden
- For vLLM backend: auto-construct `LoRARequest` from these fields; require `enable_lora=True` in engine config
- Report LoRA adapter presence in result metadata for reproducibility

### Multi-GPU / Tensor Parallelism

Add a `parallelism` block to backend-specific config sections (not top-level `ExperimentConfig`):
```yaml
backend:
  vllm:
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
```
- Energy measurement: always aggregate across all participating GPUs (sum via NVML/Zeus per-GPU readings)
- Report `num_gpus_used` and per-GPU energy breakdown in `ExperimentResult`
- For PyTorch backend: document that TP requires `torchrun` and is out of scope for v2.0 (too complex to orchestrate); `device_map="auto"` (pipeline/naive) is the supported multi-GPU path
- For energy normalisation: use `energy_per_output_token` as the primary efficiency metric, which accounts for GPU count implicitly

---

## Sources

### High Confidence
- [Transformer Inference Arithmetic (kipply)](https://kipp.ly/transformer-inference-arithmetic/) — FLOPs formula derivation, prefill/decode distinction
- [JAX Scaling Book: Inference](https://jax-ml.github.io/scaling-book/inference/) — FLOPs formula verification
- [calflops README](https://github.com/MrYxJ/calculate-flops.pytorch/blob/main/README.md) — generate mode, FLOPs = 2×MACs
- [DeepSpeed FLOPs Profiler docs](https://www.deepspeed.ai/tutorials/flops-profiler/) — warm_up param, limitations
- [lm-eval README](https://github.com/EleutherAI/lm-evaluation-harness) — `peft=` model_args LoRA support
- [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/) — LoRARequest API, tensor parallel compatibility
- [Zeus ZeusMonitor docs](https://ml.energy/zeus/measure/) — begin/end window API, gpu_indices, warmup_steps
- [optimum-benchmark scenario source](https://github.com/huggingface/optimum-benchmark) — warmup_runs=20, reduced-output pattern

### Medium Confidence
- [vLLM issue #3490](https://github.com/vllm-project/vllm/issues/3490) — FLOPs feature request, not yet merged
- [ML.ENERGY Benchmark paper](https://arxiv.org/abs/2505.06371) — steady-state definition, warmup rationale
- [TokenPowerBench paper](https://arxiv.org/abs/2512.03024) — multi-GPU energy, TP vs PP efficiency findings
- [Merging PEFT Adapters (apxml)](https://apxml.com/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/merging-peft-adapters) — ~10–12% throughput overhead for unmerged LoRA
- [HF Transformers multi-GPU docs](https://huggingface.co/docs/transformers/perf_infer_gpu_multi) — tp_plan="auto", device_map options
- [vLLM parallelism docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — tensor_parallel_size, pipeline_parallel_size

### Low Confidence (unverified or single source)
- CV < 5% convergence criterion for warmup — derived from academic micro-benchmarking literature, not observed in any peer LLM benchmarking tool
- Exact overhead figures for vLLM LoRA with tp_size > 1 — documentation mentions partial sharding but does not quantify overhead
