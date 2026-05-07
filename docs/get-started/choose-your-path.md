---
title: Choose your path
description: Find the right guide for your use case.
---

# Choose your path

---

**I want to compare two models on identical hardware.**

Go to: [Tutorials: multi-engine study](/tutorials/multi-engine-study).
Covers setting up a sweep config, running both models in a single
command, and reading the comparison output.

---

**I want to integrate LLenergyMeasure into a CI pipeline.**

Go to: [How-to: Docker setup](/how-to/docker-setup) first, then
[How-to: Run with Docker and vLLM](/how-to/run-with-docker-vllm).
Covers image pull/build, environment variables, and the `llem run`
invocation pattern that works inside CI.

---

**I want to understand the measurement methodology.**

Go to: [Explanation: Methodology overview](/explanation/methodology/methodology).
Covers warmup protocol, baseline power subtraction, thermal stabilisation,
and reproducibility. Then read
[Why LLenergyMeasure](/explanation/why) for the broader context of
what gap this tool fills.

---

**I want to extend LLenergyMeasure to a new engine.**

Go to: [Contributing: extending miners](/contributing/extending-miners).
Covers writing a parameter miner for a new engine and wiring it into the
validation pipeline. Then see
[Architecture: engine extensibility](/explanation/architecture/architecture-overview)
for the harness-plugin contract your engine needs to satisfy.

---

**I am a policy reader - I want to understand what these measurements mean.**

Go to: [For policy readers](/get-started/for-policy-readers).
A step-by-step, no-programming guide to running a measurement and
interpreting the result.
