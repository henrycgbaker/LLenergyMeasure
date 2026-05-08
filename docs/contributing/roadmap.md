---
title: Roadmap
description: Where LLenergyMeasure is, where it is going, and what stays stable.
---

# Roadmap

## Where we are

Released milestones:

| Version | What shipped |
|---------|-------------|
| v0.7.0 | Core single-experiment: CLI, config system, Transformers engine, energy measurement, results schema |
| v0.8.0 | Study/sweep: multi-experiment grid sweeps, manifest writer, deduplication, subprocess isolation |
| v0.9.0 | Docker + vLLM: containerised engine architecture, vLLM engine, Docker CI, logging overhaul |

The TensorRT-LLM milestone (M4 / v0.10.0) is implementation-complete and in
final UAT before tagging. Run `llem --version` against your installed package
for the authoritative current version.

Pre-1.0 disclaimer: the tool is research-grade for single-machine use. It is
not yet at the stability and API-stability bar of a 1.0 release.

---

## What is planned

### Milestone 5 - SGLang

Next planned milestone. Key deliverable: SGLang engine backend with
RadixAttention energy profiles. RadixAttention's KV-cache reuse mechanism
creates an unusual energy signature (lower energy on repeated prefix patterns)
that requires a dedicated measurement treatment.

Scope includes: SGLang Docker image, engine plugin, parameter miner, and
methodology documentation for prefix-aware energy measurement.

### Beyond Milestone 5

Under consideration (not yet scoped):

- **Agentic-frameworks measurement** - measuring energy for multi-turn and
  tool-use inference patterns (higher variance, longer sequences, memory
  pressure).
- **lm-eval integration** - routing lm-eval harness prompts through
  LLenergyMeasure's measurement window so quality and efficiency results share
  the same experimental record.
- **1.0.0 release** - reserved for when the API surface is stable and
  the tool has been validated in real research workflows.

---

## Stable contracts

The following are stable across pre-1.0 minor versions and will not change
without a deprecation notice:

- **`ExperimentResult` schema** - the flat result shape and the headline
  fields (`total_energy_j`, `energy_adjusted_j`, `mj_per_tok_adjusted`,
  `avg_tokens_per_second`, `total_inference_time_sec`, `total_flops`).
  New fields may be added; existing fields will not be renamed or removed.
- **Study config top-level keys** - `task`, `sweep`, `experiments`,
  `study_execution`, `runners`. Key names and their structural role will
  not change.
- **CLI commands** - `llem run` and `llem config` are stable. Flag names
  (`--model`, `--engine`, `--dtype`) are stable.

Pre-1.0 disclaimer: minor-version bumps (0.10 to 0.11 etc.) may include
breaking changes to internal APIs, engine plugin interfaces, and sampler
backends. Changes that break the stable contracts above will be called out
explicitly in the changelog.

---

## How to influence the roadmap

The roadmap is driven by research use cases. To propose a feature, report a
gap, or ask a design question:

1. Search [existing GitHub issues](https://github.com/henrycgbaker/llenergymeasure/issues)
   for similar requests.
2. Open an issue with the `design-question` label for capability decisions,
   or `bug` for defects.
3. For larger proposals (new engine, new measurement method), include a
   motivating use case - a concrete experiment you want to run that the
   tool currently cannot.

See [Contributing: development](/contributing/development) for how to
contribute code.
