---
title: What we measure
description: The three quantities LLenergyMeasure reports per measurement (energy, throughput, FLOPs), why two are headline metrics and one is a validity check, and what the tool does not measure.
---

# What we measure

LLenergyMeasure reports three quantities per measurement: **energy**,
**throughput**, and **FLOPs**. Two are headline metrics that answer the
question the tool was built to answer - *how do implementation choices
drive efficiency?* The third is a validity check that should remain
roughly invariant across implementations of the same model.

This page sets out each quantity, why it is or is not load-bearing for
the impl-effect question, and what the tool does not measure. For the
methodology details behind these numbers, see
[methodology](/explanation/methodology/methodology) and
[energy measurement](/explanation/methodology/energy-measurement). For a
hands-on first measurement with these metrics in plain language, see
[Get Started > For policy readers](/get-started/for-policy-readers).

---

## Energy (joules)

GPU electrical energy consumed during the measurement window, integrated
from instantaneous power samples. Reported as:

- **Total energy** (`total_energy_j`) - raw integrated GPU energy across
  the experiment.
- **Adjusted energy** (`energy_adjusted_j`) - total minus the
  baseline-power contribution; isolates the energy attributable to
  inference work rather than idle GPU draw.
- **Per-token energy** (`mj_per_tok_total`, `mj_per_tok_adjusted`) -
  millijoules per output token, normalising out absolute compute volume
  so cross-experiment comparisons read at a comparable scale.

Adjusted energy is the load-bearing figure for cross-implementation
comparison. The baseline subtraction is non-trivial - idle power varies
with hardware state, cooling, and prior thermal load - and is documented
in [methodology > baseline power](/explanation/methodology/methodology#baseline-power).

The unit is the joule (J): one watt-second. For reference, a smartphone
battery stores roughly 15 kJ; a 100-prompt inference run on a 0.5B model
consumes hundreds of joules.

---

## Throughput (output tokens per second)

Output tokens generated per wall-clock second across the measurement
window. Reported as `avg_tokens_per_second` per experiment cell, with
warmup-excluded accounting.

Throughput is the second headline metric. It captures how quickly the
system produces useful output and is sensitive to implementation
choices: batching strategy, attention-kernel selection, KV-cache reuse,
and quantisation form all affect throughput, sometimes in different
directions than they affect energy.

Combined with energy, throughput defines **energy per token** (joules
per token) - the canonical efficiency primitive. A measurement can be
high-energy because the workload was large (many tokens) or because the
implementation was inefficient (high energy per token); the joint
reading distinguishes the two.

---

## FLOPs (floating-point operations) {#flops}

An estimate of the floating-point operations the model executes per
inference. The headline field is `total_flops` (reference metadata);
derived fields `flops_per_output_token`, `flops_per_input_token`, and
`flops_per_second` are computed from `total_flops` against the matching
token / time denominators when those are non-zero.

**FLOPs is a validity check, not a headline metric.** This is a
deliberate framing choice. FLOPs are largely invariant across
implementations of the same model: a given `(model, prompts, max-output)`
tuple performs approximately the same arithmetic regardless of whether
it runs through Transformers, vLLM, or TensorRT-LLM. The differences
between engines are in *how that arithmetic is dispatched to hardware*
(kernel selection, memory layout, scheduling, fusion), not in *how much
arithmetic happens*.

This makes FLOPs uninformative for the impl-effect question. Swapping
engines should change energy and throughput substantially while leaving
FLOPs roughly unchanged. When that prediction fails - when FLOPs drift
significantly between cells of an implementation sweep - the most
likely cause is methodological:

- the prompts or output budget weren't actually held fixed across cells;
- the model architecture differs (e.g. a quantised variant has fewer
  effective FLOPs even though the parameter count is unchanged);
- the FLOPs estimator's heuristic is hitting an architecture boundary
  it does not fully recognise (estimator quality varies across model
  families).

So FLOPs functions as a **sanity-check backstop**: if cells diverge on
FLOPs, something is wrong with the experiment design, not with the
implementations being compared. That is a useful and load-bearing role;
it is not the tool's headline.

(Research that *does* benchmark FLOPs as a primary signal - theoretical
roofline analyses, hardware-utilisation studies - is better served by
tools targeted at that question. See
[comparison with other tools](/explanation/methodology/comparison-context).)

---

## What we don't measure

Honest limits, stated explicitly:

- **Training energy.** This tool measures inference. Training costs are
  out of scope. Pair with `codecarbon` or similar for training-side
  accounting.
- **Capability, accuracy, or quality.** No benchmark score, no
  perplexity, no task accuracy. Pair with
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  when capability matters.
- **Carbon emissions directly.** Energy is reported in joules; carbon
  depends on the grid mix where the inference runs, which is outside
  this tool's scope. The CodeCarbon sampler can attach an estimated
  carbon figure based on a configured locale; the canonical record is
  joules.
- **End-to-end serving cost.** The measurement window covers inference
  proper, not request queuing, network transit, authentication, or
  storage costs of running an inference service.
- **Cross-prompt and cross-run contamination effects.** Each measurement
  window is per-experiment. Effects across runs (cache warming, weight
  quantisation drift, GPU thermal hysteresis) are documented in
  [methodology](/explanation/methodology/methodology) and surfaced
  through warnings, but are not reported as separate metrics.

---

## How the three combine for the impl-effect question

The question this tool answers: *given a fixed model and prompts, how
do implementation choices drive efficiency?*

| Reading across cells of an implementation sweep | What it tells you |
|---|---|
| Energy and throughput differ; FLOPs roughly equal | Implementation choices affect efficiency. **This is the signal the tool was built to surface.** |
| Energy, throughput, and FLOPs all differ | Either the experiment isn't holding model+prompts fixed across cells, or the model architecture varies between cells. Investigate before drawing conclusions. |
| FLOPs differ but energy and throughput don't | Likely a methodology bug in the FLOPs estimator at low confidence. Surface and check; energy + throughput readings remain reliable. |

For interpretation guidance against actual results, see
[how to read LLenergyMeasure output](/how-to/interpret-results).

---

## Further reading

- [Methodology](/explanation/methodology/methodology) - the full
  measurement protocol (warmup, baseline, sampler, thermal floor).
- [Energy measurement](/explanation/methodology/energy-measurement) -
  technical depth on the sampler abstraction (NVML, Zeus, CodeCarbon).
- [Comparison with other tools](/explanation/methodology/comparison-context) -
  positioning relative to MLPerf, AIEnergyScore, llmperf, and capability
  benchmarks.
- [For policy readers](/get-started/for-policy-readers) - a plain-language
  walk-through of running a first measurement.
