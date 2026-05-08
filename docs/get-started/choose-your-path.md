---
title: Choose your path
description: Find the right guide for your use case.
---

# Choose your path

Pick the entry that matches your goal; each one names the page to start from
and what it covers.

---

**I want to compare two models on identical hardware.**

Start at [Tutorials: multi-engine study](/tutorials/multi-engine-study).
Covers setting up a sweep config, running both models in a single
command, and reading the comparison output.

---

**I want to integrate LLenergyMeasure into a CI pipeline.**

Start at [How-to: Docker setup](/how-to/docker-setup), then
[How-to: Run with Docker and vLLM](/how-to/run-with-docker-vllm).
Covers image pull/build, environment variables, and the `llem run`
invocation pattern that works inside CI.

---

**I want to understand the measurement methodology.**

Start at [Methodology overview](/explanation/methodology/methodology).
Covers warmup protocol, baseline power subtraction, thermal stabilisation,
and reproducibility. Then read [Why LLenergyMeasure](/explanation/why)
for the research-gap context.

---

**I want to extend LLenergyMeasure to a new engine.**

Start at [Architecture: engine extensibility](/explanation/architecture/engine-extensibility)
for the harness-plugin contract your engine needs to satisfy. Then see
[Contributing: extending miners](/contributing/extending-miners) for
writing a parameter miner and wiring it into the validation pipeline.

---

**I am a policy reader and want to understand what these measurements mean.**

Start at [For policy readers](/get-started/for-policy-readers).
A step-by-step, no-programming guide to running a measurement and
interpreting the result.
