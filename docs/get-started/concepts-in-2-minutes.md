---
title: Concepts in 2 minutes
description: The six terms you need to understand LLenergyMeasure.
---

# Concepts in 2 minutes

Six terms, one sentence each.

---

## Experiment vs Study

An **Experiment** is a single measurement point: one model, one engine, one
dtype, one set of generation parameters. It produces one result file.

A **Study** is a multi-experiment wrapper: a list of experiments derived from a
sweep specification, sharing execution settings and run together as a batch.

For the full config reference, see
[Reference: study config](/reference/study-config).

---

## The four layers

**Engine** - the inference runtime that loads and runs the model. LLenergyMeasure
supports Transformers, vLLM, and TensorRT-LLM. Each engine runs inside its own
Docker container.

**Sampler** - the energy measurement backend that reads GPU power during an
experiment. The default is `nvml` (NVIDIA Management Library). Alternatives:
`zeus`, `codecarbon`. Configured independently of the engine.

**Runner** - the execution environment. `local` (default) runs the experiment
directly; `docker` invokes the per-engine container via Docker Compose.

**Harness** - the measurement coordinator inside the container. It owns the
NVML window, warmup exclusion, baseline subtraction, and result assembly.
The harness is engine-agnostic; engines are thin inference plugins.

For the architecture behind these layers, see
[Architecture overview](/explanation/architecture/architecture-overview).

---

## What we measure

**Energy (joules)** - total GPU energy during inference, with idle-power baseline
subtracted. The number that answers "how much electricity did this model use?"

**Throughput (tokens/s)** - output tokens generated per second across all prompts.
The number that answers "how fast is this engine?"

**FLOPs** - estimated floating-point operations for the run. Useful for
normalising comparisons across model sizes; a reference number, not a primary
efficiency metric.

For the full methodology, see
[What we measure](/explanation/methodology/what-we-measure).

---

## The four Diataxis pillars

This documentation follows the [Diataxis](https://diataxis.fr) framework.
Four types of doc, one purpose each:

| Pillar | What it is | Start here if... |
|--------|-----------|-----------------|
| [Tutorials](/tutorials/first-measurement) | Guided learning, linear | You want a walkthrough |
| [How-to](/how-to/install) | Goal-driven recipes | You know what you want to do |
| [Reference](/reference/cli) | Exhaustive lookup | You need exact syntax |
| [Explanation](/explanation/overview) | Concepts and context | You want to understand why |
