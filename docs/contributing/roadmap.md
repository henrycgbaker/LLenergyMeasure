---
title: Roadmap
description: Where LLenergyMeasure is, where it is going, and what stays stable.
---

# Roadmap

## Where we are

Released milestones:

| Version | What shipped |
|---------|-------------|
| v0.2.0 | Core single-experiment: CLI, config system, Transformers engine, energy measurement, results schema |
| v0.3.0 | Study/sweep: multi-experiment grid sweeps, manifest writer, deduplication, subprocess isolation |
| v0.4.0 | Docker + vLLM: containerised engine architecture, vLLM engine, Docker CI, logging overhaul |
| v0.4.1 | Engine-knowledge-as-data: each engine's config, schema, and validation rules are generated from version-pinned upstream snapshots rather than hand-curated; terminology renames; study-authoring improvements |
| v0.5.0 | TensorRT-LLM activated as a measured third engine (both the PyTorch and compiled-TRT backends), with in-framework build caching and GPU hardware preflight |
| v0.5.1 | Results correctness: a single coherent per-experiment results bundle, schema consolidation, and provenance sidecars |
| v0.6.0 | Public packaging: first PyPI release, pip-install Docker dispatch, a lean CLI surface, and cloud-VM documentation |

The current milestone is public packaging: the first PyPI release, pip-install
Docker dispatch, and a lean CLI surface. An internal refactor-and-tidy pass
follows, ahead of the stable 1.0.0 release. Run `llem --version` against your
installed package for the authoritative current version.

Pre-1.0 disclaimer: the tool is research-grade for single-machine use. It is
not yet at the stability and API-stability bar of a 1.0 release.

---

## What is planned

### SGLang engine

A planned future engine, targeted after the near-term results-correctness and
1.0.0 work. Key deliverable: SGLang engine backend with
RadixAttention energy profiles. RadixAttention's KV-cache reuse mechanism
creates an unusual energy signature (lower energy on repeated prefix patterns)
that requires a dedicated measurement treatment.

Scope includes: SGLang Docker image, engine plugin, config schema and
validation rules, and methodology documentation for prefix-aware energy
measurement.

### Further candidates

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
- **CLI commands** - `llem run` and `llem doctor` are stable. `llem run` takes
  a config path plus session flags (`--output`, `--resume`, `--dry-run`);
  experiment parameters live in the YAML config, not in flags.

Pre-1.0 disclaimer: minor-version bumps (0.5 to 0.6 etc.) may include
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
