---
title: For serving open-weights models
description: How LLenergyMeasure helps if you run open-weights inference and want to measure or tune your own serving stack.
---

# For serving open-weights models

If you run open-weights inference - on your own hardware, rented GPUs, or a CI runner - LLenergyMeasure measures the efficiency of your serving stack. The methodology that supports academic comparisons is the same methodology you can apply to evaluate and tune your own deployment.

---

## What you can answer

For your model on your hardware: which configuration choices most reduce energy and per-token cost. Implementation parameters are discovered programmatically from each engine, not curated to a shortlist, so the answer space stays as wide as the engine's own configuration surface.

---

## Where to start

- [Quick start](/get-started/quick-start) - one engine, one model, one measurement.
- [Tutorial: first measurement](/tutorials/first-measurement) - a fuller walkthrough that explains every output field.
- [Tutorial: multi-engine study](/tutorials/multi-engine-study) - the dedicated example for comparing implementation choices on a fixed model.
- [Tutorial: server-mode study](/tutorials/server-mode-study) - measure your serving stack under traffic at several request rates.
- [How-to: run with Docker and vLLM](/how-to/run-with-docker-vllm) - the production pattern; reproducible images, no host CUDA dependency.

---

## Reading the outputs for deployment decisions

Academic comparisons emphasise relative numbers - is engine A more efficient than engine B for this model. Deployment decisions also need absolute numbers: joules per token converts into watts at your serving load, which converts into electricity cost and capacity planning.

`energy_per_token_mj_adjusted` is the most portable per-token figure - idle GPU draw is subtracted, so what remains is the energy your inference work itself is responsible for. Multiply by your daily token volume for a deployment-shaped energy figure. The full output-reading guide is [how-to: interpret results](/how-to/interpret-results).

---

## Operator-specific guidance is on the roadmap

Worked examples for the questions a serving team actually asks - "which quant for Llama-3?", "is migrating to vLLM worth it for my workload?", "how much energy does my reasoning model spend on tokens the user never sees?" - are not yet written. The broader product-positioning conversation is tracked in [issue #626](https://github.com/henrycgbaker/llenergymeasure/issues/626).
