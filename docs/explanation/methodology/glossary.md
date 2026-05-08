# Terminology reference

This is the canonical glossary for LLenergyMeasure: definitions for both the
measurement domain (energy, GPU, LLM inference) and the project's own
terminology (engines, runners, invariants). Cross-references use anchors within
this page.

For canonical naming decisions internal to the engine-invariants pipeline
(e.g. "invariant" not "rule", "validated corpus" not "vendored corpus"), see
[Contributing: miner pipeline](/contributing/miner-pipeline) and
[Reference: invariants corpus format](/reference/invariants-corpus-format).

---

## A

### Adjusted energy

Total measured GPU energy minus the [baseline](#baseline) contribution:
`adjusted_energy_j = total_energy_j - baseline_power_w * total_inference_time_sec`.
Represents the energy specifically attributable to inference work, independent
of what the GPU draws at idle. See `mj_per_tok_adjusted` in result.json for
the per-token form.

---

## B

### Backend

Historical term for either an inference framework or an energy source. Retired in LLenergyMeasure documentation. See [engine](#engine) (inference framework: transformers/vllm/tensorrt) or [energy sampler](#energy-sampler) (energy source: NVML/Zeus/CodeCarbon).

The phrase **attention backend** remains standard for kernel selection (FlashAttention, xformers, sdpa, eager) and is distinct from the inference engine. See [attention backend](#attention-backend).

### Attention backend {#attention-backend}

The kernel implementation used for the attention computation within an inference engine. Common values: `flash_attn`, `flashinfer`, `sdpa`, `eager`, `flash_attention_2`. Configured via engine-scoped parameters such as `transformers.attn_implementation` or `vllm.attention.backend`.

Attention backend is an industry-standard sub-system term (used by HuggingFace, vLLM, and TensorRT-LLM) and is distinct from the inference [engine](#engine). Selecting the attention backend does not change which engine runs the model.

### Baseline

The idle [power](#power) the GPU draws when not running inference - analogous
to a car engine idling. Measured in Watts (W) before each experiment by running
a lightweight container that polls NVML without loading a model. Subtracted from
the total energy to produce [adjusted energy](#adjusted-energy). See also
[thermal stabilisation](#thermal-stabilisation--warmup).

### BF16

See [dtype](#dtype).

---

## C

### CodeCarbon

An optional energy measurement library that estimates total system power
(CPU + DRAM + GPU) by combining hardware counters and software models. Available
as the `codecarbon` extra: `pip install llenergymeasure[codecarbon]`. Less
reliable inside Docker containers than [NVML](#nvml). Falls back to zero without
raising an error if hardware access fails.

---

## D

### Decode / prefill

The two phases of autoregressive LLM inference:

- **Prefill** - the model processes the entire input prompt in one forward pass,
  populating the [KV-cache](#kv-cache). Compute is proportional to prompt length.
  Power draw spikes here.
- **Decode** - the model generates one output [token](#token) at a time,
  attending over the KV-cache entries from prefill and prior decode steps.
  Compute per token is roughly constant; the KV-cache grows with each step.

Energy consumption is distributed unevenly between the two phases. For short
prompts and long outputs, decode dominates. For long prompts and short outputs
(e.g. classification tasks), prefill dominates. LLenergyMeasure measures
aggregate energy across both phases.

### Decoder

In LLenergyMeasure configuration, `decoder:` refers to the sampling strategy
block (whether to sample, temperature, top-p, etc.) rather than the transformer
decoder stack. See [sampler](#sampler--decoder-config).

### Dtype

The numerical precision format used to store model weights and activations:

| dtype | Bits | Relative VRAM | Notes |
|-------|------|---------------|-------|
| `float32` | 32 | 1x | Full precision; rarely used for inference; not supported by vLLM or TRT-LLM |
| `float16` (FP16) | 16 | 0.5x | Standard inference precision on older hardware |
| `bfloat16` (BF16) | 16 | 0.5x | Preferred on Ampere+ (A100, 4090); wider dynamic range than FP16 |

Dtype affects energy primarily through memory bandwidth: lower precision means
smaller weights, faster memory reads, and lower power during memory-bound decode.
For quantisation (INT8, W4A16), see [quantisation](#quantisation).

---

## E

### Energy

The total electrical work done, measured in [Joules](#joule). Equal to average
[power](#power) multiplied by duration. Energy is the headline metric for
comparing inference efficiency: it accounts for both power draw and run time.
Do not confuse with power, which is a rate.

See also [FLOPs vs FLOPS](#flops-vs-flops).

### Engine {#engine}

The ML library responsible for loading the model and running inference.
LLenergyMeasure currently supports three engines:

| Engine | Library | Runner |
|--------|---------|--------|
| `transformers` | HuggingFace Transformers | Docker (or local) |
| `vllm` | vLLM | Docker only |
| `tensorrt` | TensorRT-LLM | Docker only |

The engine is set via `engine:` in the YAML or `--engine` on the CLI. Each
engine runs inside its own Docker image (except Transformers in local mode).
See also [runner](#runner).

---

## F

### FLOPs vs FLOPS

Two distinct terms:

- **FLOPs** (Floating Point Operations, singular or plural) - a count of
  mathematical operations performed. Reported in GFLOPs or TFLOPs.
- **FLOPS** (Floating Point Operations Per Second) - a throughput rate (hardware
  peak performance). Reported in TFLOPS.

LLenergyMeasure reports FLOPs estimates (operation counts), not FLOPS (hardware
throughput rates). See [What we measure: FLOPs](/explanation/methodology/what-we-measure#flops)
for the estimation method.

---

## G

### GPU contention

A condition where another process is using the GPU simultaneously with
LLenergyMeasure, causing higher baseline power draw, latency variance, and
sampling jitter. The recommended approach is to measure on a dedicated GPU
with no other workloads active. See [Measurement warnings](/explanation/methodology/measurement-warnings)
for how contention manifests in practice.

---

## I

### INT8

See [quantisation](#quantisation).

---

## J

### Joule

The SI unit of energy. One joule equals one watt applied for one second. GPU
energy for a single inference is typically in the range of 0.1-100 J depending
on model size, throughput, and run duration.

Conversions: 1 Wh = 3,600 J. 1 kWh = 3,600,000 J. A 300 W GPU drawing full
power for 10 minutes consumes 180,000 J (50 Wh).

LLenergyMeasure reports energy in Joules (field `total_energy_j`) and
millijoules per token (`mj_per_tok_adjusted`).

---

## K

### KV-cache

The key-value cache stores intermediate attention tensors from the [prefill](#decode--prefill)
phase and prior decode steps, so the model does not recompute them for each new
[token](#token). Reuse is the primary mechanism for efficient autoregressive
generation.

KV-cache size grows linearly with sequence length and batch size. It is a major
consumer of GPU VRAM for large models. [vLLM](#engine) uses paged attention to
manage the KV-cache as a pool of fixed-size blocks, reducing fragmentation.
`vllm.engine.gpu_memory_utilization` controls how much VRAM is pre-allocated for
the KV-cache.

Energy relevance: a larger KV-cache increases VRAM traffic and therefore memory
bandwidth energy. Prefix caching (reusing KV tensors across requests) reduces
energy for repeated prompt prefixes.

---

## N

### NVML

NVIDIA Management Library - the C library that exposes GPU hardware counters,
including power draw, temperature, clock speeds, and throttle state. Accessed
from Python via `pynvml` (`nvidia-ml-py`). The default energy sampler in
LLenergyMeasure: polls every 100 ms and integrates power over time to compute
energy.

NVML power readings have approximately ±5% accuracy (hardware vendor spec).
All energy figures reported by LLenergyMeasure carry this intrinsic uncertainty.

See also [RAPL](#rapl), [CodeCarbon](#codecarbon).

---

## P

### Power

The instantaneous rate of energy consumption, measured in Watts (W). NVML
reports GPU power in Watts at 100 ms intervals. Energy is power integrated
over time. A GPU drawing 300 W for 2 seconds consumes 600 J.

High power draw indicates the GPU is compute-bound; low power with low
throughput may indicate memory bandwidth saturation or throttling. See
[thermal stabilisation](#thermal-stabilisation--warmup).

### Prefill

See [decode / prefill](#decode--prefill).

---

## Q

### Quantisation

Reducing the bit-width of model weights (and optionally activations) to save
VRAM and increase throughput at the cost of a small accuracy loss. Common
schemes supported by LLenergyMeasure:

| Scheme | Bits | Supported by |
|--------|------|-------------|
| INT8 | 8 | Transformers (BitsAndBytes), TRT-LLM |
| W4A16_AWQ | 4 (weights) / 16 (activations) | Transformers (BitsAndBytes), vLLM, TRT-LLM |
| W4A16_GPTQ | 4 (weights) / 16 (activations) | Transformers (BitsAndBytes), vLLM, TRT-LLM |
| W8A16 | 8 (weights) / 16 (activations) | TRT-LLM |
| FP8 | 8 (float) | vLLM (KV cache), TRT-LLM (SM >= 8.9 only) |

FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM 8.0) does not
support native FP8 inference.

For full engine-specific quantisation options and invalid combinations, see
[Reference: engine configuration](/reference/engines/configuration).

---

## R

### RAPL

Running Average Power Limit - an Intel CPU hardware counter that reports
cumulative CPU and DRAM energy. Not currently used by LLenergyMeasure (which
measures GPU energy via [NVML](#nvml)), but relevant context when interpreting
total system energy. RAPL is used by [CodeCarbon](#codecarbon) to estimate
CPU+DRAM contributions.

### Runner

Two senses in this codebase: (1) llem execution mode (`local` or `docker`); (2) GitHub Actions CI runner (`self-hosted`, `ubuntu-latest`). Both are correct in their contexts; readers should disambiguate from surrounding prose.

**llem runner** - the execution context that wraps the [engine](#engine). LLenergyMeasure has two runner types:

- `local` - runs the engine in the current Python process. Only supported for
  the Transformers engine. Suitable for development and testing without Docker.
- `docker` - runs the engine inside a dedicated Docker container. Required for
  vLLM and TensorRT-LLM (Docker-only by design). Default for all engines in
  production use.

Set via `runners:` in the study YAML. Multi-engine studies without Docker
are blocked at config load time.

**CI runner** - the GitHub Actions compute host (e.g. `self-hosted` GPU runner, `ubuntu-latest`). Used in contributing and CI documentation only.

---

## S

### Sampler / decoder config

In LLenergyMeasure, `decoder:` in the YAML configures the generation sampling
strategy: whether to sample (`do_sample: true`) or use greedy decoding, and
parameters such as `temperature`, `top_p`, `top_k`, `repetition_penalty`.
Energy per token varies with sampling strategy because different strategies
affect the number of attention heads consulted and the softmax computation.

Not to be confused with the [energy sampler](#energy-sampler).

### Energy sampler

The component that measures GPU power during inference. Three samplers:

- **NVML** (default) - polls GPU power via pynvml at 100 ms intervals
- **Zeus** - uses the Zeus library for per-batch energy attribution (optional extra)
- **CodeCarbon** - estimates total system power including CPU and DRAM (optional extra)

Configured via `energy_sampler:` in the YAML. Set to `null` for
throughput-only mode (no energy measurement).

In older docs, energy samplers were sometimes called "energy backends"; the term is retired in favour of `sampler`.

---

## T

### Temperature (generation)

The `decoder.temperature` parameter controls the randomness of token sampling.
Temperature = 1.0 samples from the raw model distribution; lower values
concentrate probability on high-likelihood tokens; temperature = 0 is equivalent
to greedy decoding. Energy is not materially affected by temperature because the
same forward pass runs regardless.

### Thermal stabilisation / warmup {#thermal-stabilisation--warmup}

Two related concepts:

- **Warmup** - running several inference passes before the measurement window
  begins, to bring the model's KV-cache and GPU compute units to steady state.
  LLenergyMeasure checks whether latency has stabilised using the coefficient
  of variation (CV) of successive prompt latencies.
- **Thermal floor wait** - a configurable pause (`warmup.thermal_floor_seconds`,
  default 60 s) after the warmup prompts, during which the GPU idles and cools
  toward a stable baseline temperature before measurement starts.

Skipping warmup produces measurements from a cold-start state that are not
representative of sustained inference. See [Methodology: measurement warnings](/explanation/methodology/measurement-warnings#warmup-did-not-converge)
for the "warmup did not converge" warning.

### Throughput

Output tokens produced per second (`output_tokens_per_sec` in result.json).
Distinct from latency (time to first token or time per request): throughput is
a rate across the full batch. Higher throughput means more tokens are produced
in the same wall-clock time.

For single-sequence inference at batch_size=1 (typical on consumer hardware),
throughput and latency are closely related. At higher batch sizes (vLLM
continuous batching), throughput can increase significantly while per-sequence
latency rises.

### Token

The basic unit of LLM input and output. A tokeniser (e.g. BPE, sentencepiece)
maps text to integer token IDs. Roughly 1 token ≈ 0.75 English words, but this
varies by language and tokeniser. Prompt length and output length are both
measured in tokens.

Energy and throughput figures in LLenergyMeasure are normalised per token
(`mj_per_tok_adjusted`, `output_tokens_per_sec`) to allow comparison across
different workloads.

---

## V

### Validated corpus

The engine-invariants corpus after the validation-CI gate has replayed each
invariant against the live library. Stored as `invariants.validated.yaml`. Not
to be confused with the proposed corpus (`invariants.proposed.yaml`), which
contains declared expectations before CI observation. See
[Reference: invariants corpus format](/reference/invariants-corpus-format)
for the full format specification.

---

## W

### Warmup

See [thermal stabilisation / warmup](#thermal-stabilisation--warmup).

---

## Engine-invariants pipeline terminology

For terminology specific to the miner pipeline, validation gate, and corpus
naming, see [Contributing: miner pipeline](/contributing/miner-pipeline) and
[Reference: invariants corpus format](/reference/invariants-corpus-format).
Those pages are the canonical definition source and must be consulted before
introducing new names in code or documentation.

---

## See also

- [Methodology: energy measurement](/explanation/methodology/energy-measurement) - NVML integration, baseline subtraction
- [Methodology: what we measure](/explanation/methodology/what-we-measure) - plain-language overview
- [Methodology: measurement warnings](/explanation/methodology/measurement-warnings) - runtime warning reference
- [Reference: engine configuration](/reference/engines/configuration) - dtype and quantisation options
