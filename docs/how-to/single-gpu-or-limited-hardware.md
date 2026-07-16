---
title: Running on a single or consumer GPU
description: How to run LLenergyMeasure usefully on one consumer-grade GPU with limited VRAM.
---

# Running on a single or consumer GPU

LLenergyMeasure is designed for multi-GPU datacentre hardware, but many
researchers work with a single consumer card. This page covers the key knobs
for making runs productive on 8-24 GB VRAM, and is honest about where the team
has not yet validated specific numbers.

---

## Choosing a model that fits your VRAM

Rule of thumb: a model in full precision (float32) requires approximately
`4 * parameter_count_billions` GB of VRAM just for weights (e.g. a 7B model
needs ~28 GB). In float16/bfloat16 it halves to ~2 GB per billion parameters.
With 4-bit quantisation, roughly ~0.5 GB per billion.

| VRAM | Float16 / BF16 limit | 4-bit quantised limit |
|------|---------------------|-----------------------|
| 8 GB | ~3B parameters | ~13B parameters |
| 16 GB | ~7B parameters | ~30B parameters |
| 24 GB | ~12B parameters | ~45B parameters |

> TODO: needs hardware capture on 24 GB / 16 GB / 8 GB cards - the numbers above
> are rule-of-thumb estimates and do not account for KV-cache, activations, or
> framework overhead. Validate on RTX 4090, RTX 4080, and RTX 3080 before treating
> as exact.

If a model does not fit, LLenergyMeasure will raise an OOM error before measurement
starts (on Transformers) or refuse to initialise (on vLLM). Neither wastes a partial
measurement.

---

## Reducing `n_prompts` for shorter runs

The default `task.dataset.n_prompts: 100` runs 100 prompts through the engine.
On a fast datacentre GPU this takes 30-90 seconds. On a consumer card with a large
model it may take several minutes.

For quick iteration, reduce to 10-25 prompts:

```yaml
task:
  dataset:
    n_prompts: 10
```

Fewer prompts reduces statistical stability. For publication-quality measurements,
use at least 50-100 prompts and set `study_execution.n_cycles: 3` to run the full
workload three times and report the median.

---

## Quantisation options

Quantisation reduces VRAM by compressing model weights to lower precision. The
available options depend on the engine:

**Transformers engine** - BitsAndBytes:

```yaml
engine: transformers
transformers:
  load_in_4bit: true    # ~0.5 GB per billion params
  # or
  load_in_8bit: true    # ~1 GB per billion params
```

4-bit and 8-bit are mutually exclusive. 4-bit uses AWQ/GPTQ-style compression
with optional NF4 data type. Energy measurements on quantised models reflect real
inference cost - the GPU still executes the dequantised computation.

**vLLM engine** - pre-quantised checkpoints:

```yaml
engine: vllm
vllm:
  engine:
    quantization: awq   # or gptq
```

vLLM requires a pre-quantised model checkpoint (e.g. `TheBloke/*-AWQ` on
HuggingFace Hub). It does not quantise on the fly.

**TensorRT-LLM engine** - build-time quantisation (compiled `trt` backend):

```yaml
engine: tensorrt
tensorrt:
  engine_params:
    backend: trt          # quant_config requires the compiled trt backend
    quant_config:
      quant_algo: W4A16_AWQ   # or INT8, W4A16_GPTQ, W8A16
```

TRT-LLM compiles a quantised engine at build time; `quant_config` is rejected
under the default `pytorch` backend. FP8 requires SM >= 8.9 (Ada Lovelace or
Hopper) and is NOT supported on A100 (SM 8.0).

For full quantisation options and valid combinations, see
[Reference: engine configuration](/reference/engines/configuration).

---

## Thermal stabilisation on consumer cards

Consumer GPUs (RTX series) have smaller heatsinks and higher thermal density
than datacentre cards (A100, H100). The warmup and thermal-floor wait behaviour
differs in practice:

- **Shorter time-to-throttle**: consumer cards often reach thermal limits within
  60-90 seconds of sustained load, whereas A100s can sustain full load
  indefinitely with adequate rack cooling.
- **More aggressive boost clocking**: consumer GPUs boost above their base clock
  and then throttle back. This creates more latency variance during warmup, which
  means the "warmup did not converge" warning is more common on consumer hardware.
- **Thermal floor wait**: the default `warmup.thermal_floor_seconds: 60.0` may
  be too short on a system that throttled during warmup. If you see persistent
  convergence failures, try 90-120 seconds.

```yaml
warmup:
  thermal_floor_seconds: 90.0
  max_prompts: 20
```

For the "Warmup did not converge" warning and what it means for your results,
see [Methodology: measurement warnings](/explanation/methodology/measurement-warnings#warmup-did-not-converge).

> TODO: validate specific thermal profiles on RTX 4090 / 4080 / 3080. Document
> typical time-to-throttle and recommended `thermal_floor_seconds` per card class.

---

## Engine trade-offs at low VRAM

Not all engines are equal at low VRAM. A short guide:

**Transformers** - lowest barrier. Runs in Docker with `--gpus all` or locally
if you have PyTorch installed. Supports BitsAndBytes quantisation. Slowest
throughput at batch_size=1 (the low-VRAM setting). Good for model compatibility
and flexibility.

**vLLM** - higher throughput via continuous batching and paged attention, but
pre-allocates most VRAM for the KV cache at startup. Reduce
`vllm.engine_params.gpu_memory_utilization` (e.g. `0.7`) to leave headroom:

```yaml
vllm:
  engine:
    gpu_memory_utilization: 0.7
    max_model_len: 2048   # also cap KV cache size
```

**TensorRT-LLM** - highest throughput. The default `pytorch` backend loads
directly; the compiled `trt` backend adds 5-15 minutes of engine compilation on
the first run of a config (cached afterwards), which is most beneficial for
models you will run repeatedly. Less flexible than Transformers for ad-hoc
measurement.

Rule of thumb for consumer hardware:
- Quick model survey: use Transformers with 4-bit quantisation.
- Sustained throughput measurement: use vLLM with reduced `gpu_memory_utilization`.
- Repeated runs of a fixed config: invest in TRT-LLM compilation.

> TODO: document observed throughput ratios between engines on consumer cards
> (RTX 4090 Transformers vs vLLM for 7B models). Target: concrete numbers from
> hardware capture.

---

## See also

- [Reference: engine configuration](/reference/engines/configuration) - full quantisation options
- [Methodology: measurement warnings](/explanation/methodology/measurement-warnings) - interpreting thermal warnings
- [How to: troubleshoot](/how-to/troubleshoot#out-of-memory-oom) - OOM errors
- [How to: run with Docker and vLLM](/how-to/run-with-docker-vllm) - vLLM setup
