---
title: Concepts in 2 minutes
description: The six terms you need to understand LLenergyMeasure.
---

# Concepts in 2 minutes

LLenergyMeasure measures how implementation choices drive LLM inference
efficiency. Six terms set up everything else.

---

## Experiment vs study

An **experiment** is a single measurement point: one model, one engine, one
dtype, one set of generation parameters. It produces one result file.

A **study** is a multi-experiment wrapper: a flat list of experiments
(typically derived from a sweep specification), sharing execution settings
and run together as a batch.

For the full config reference, see
[Study config](/reference/study-config).

---

## The four layers

**Engine** - the inference framework that loads and runs the model.
LLenergyMeasure supports `transformers`, `vllm`, and `tensorrt` (with
`sglang` planned). Each engine runs inside its own Docker container.

**Sampler** - the source of energy measurements. The default is `nvml`
(NVIDIA Management Library); alternatives are `zeus` and `codecarbon`.
Sampler choice is independent of engine choice.

**Runner** - the execution environment. `process` runs the experiment
directly on the host; `container` invokes the per-engine container. (These were
renamed from `local`/`docker` in v0.7; the old values are now rejected with a
migration error.)

**Harness** - the measurement coordinator (`MeasurementHarness`). It owns
the measurement window, warmup, baseline subtraction, and result
assembly. The harness is engine-agnostic; engines are thin inference
plugins.

For the architecture behind these layers, see
[Architecture overview](/explanation/architecture/architecture-overview).

---

## What we measure

**Energy (joules)** - total GPU energy during inference, with idle-power
baseline subtracted where measured. The headline answer to "how much
electricity did this configuration consume?"

**Throughput (tokens/s)** - output tokens per second across all prompts.
The headline answer to "how fast is this configuration?"

**FLOPs** - estimated floating-point operations for the run. Reported as
a validity check (largely invariant across implementations of the same
model); not a headline efficiency metric.

For the full methodology, see
[What we measure](/explanation/methodology/what-we-measure).
