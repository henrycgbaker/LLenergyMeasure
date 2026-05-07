---
title: Dataset choice and measurement validity
description: How the choice of prompt dataset shapes energy and throughput measurements, what the bundled AIEnergyScore dataset is and is not, and how to use custom datasets to study prompt-characteristic effects.
---

# Dataset choice and measurement validity

The dataset fed to an inference engine is not a neutral backdrop for measurement.
It is part of the measurement. Two runs with identical hardware, model, and
configuration settings but different prompt corpora will produce different energy
and throughput numbers. This page explains why that is true, what it means for
interpreting results, and how to design studies that account for it.

---

## Why dataset choice matters

Energy consumption during inference depends on what the model generates, not
just on the model's parameter count or architecture. The primary driver is
**output length**: a longer generated sequence requires more forward passes
through the network, more memory reads, and more arithmetic operations. Input
length also matters - longer prompts consume more memory bandwidth during the
prefill step and occupy more KV-cache.

A dataset composed entirely of short factual prompts ("What is the capital of
France?") will yield systematically lower energy-per-inference numbers than a
dataset of long instruction-following prompts, even for the same model. Neither
is "wrong" - they are measuring different workloads. The number only has meaning
relative to the prompt distribution it was measured against.

This has two direct consequences:

1. **Within-study comparisons are valid if the dataset is held constant.** If
   you sweep batch sizes, quantisation settings, or engine configurations while
   keeping the same dataset and prompt count, the resulting energy differences
   are attributable to those variables - not to dataset effects.

2. **Cross-study comparisons require a shared dataset.** If study A used 100
   prompts from one source and study B used 100 prompts from a different source,
   the energy numbers are not comparable even on the same hardware with the same
   model. The dataset is a part of the experimental condition.

---

## The AIEnergyScore dataset: what it is

The default bundled dataset is the text-generation corpus from the
[AIEnergyScore project](https://huggingface.github.io/AIEnergyScore/), a Hugging
Face initiative that establishes standardised energy efficiency ratings for AI
models. The text-generation dataset consists of 1,000 passages sampled from
WikiText, OSCAR, and UltraChat - a deliberate mix of registers, languages, and
lengths.

The key property of this dataset for measurement purposes is that it is a
**shared reference corpus**. Any measurement run against the AIEnergyScore
text-generation dataset is, in principle, comparable to any other measurement
run against the same dataset (given matching hardware, dtype, and prompt count).
This is the same principle that makes benchmark suites like MLPerf useful for
hardware comparison: the shared stimulus enables fair comparison.

LLenergyMeasure ships the dataset bundled in the package (no network call at
run time) at the pinned commit `2dc92b2`. See
[Reference: Dataset format](/reference/dataset-format) for full provenance
details, licence, and file composition.

---

## What the AIEnergyScore dataset is not

The AIEnergyScore dataset is not a universal proxy for "the model's energy
efficiency". It is a proxy for the model's energy efficiency **on a particular
distribution of text-generation prompts**. That distribution may or may not
resemble the prompts in your actual downstream application.

A few known limitations:

**Task domain.** The prompts are general-purpose text passages (encyclopaedic,
conversational, web text). Models deployed for specific domains - legal document
summarisation, code generation, biomedical question answering - will face
different prompt length distributions and different generation characteristics.
The AIEnergyScore numbers will not reflect those domain-specific effects.

**Output length variance.** The benchmark fixes input prompts but does not
control output length. Output length is determined by the model's generation
strategy and the `max_output_tokens` cap. Across models, output lengths will
differ, which means raw energy-per-inference numbers partially reflect
per-model verbosity, not just per-model computational efficiency.

**Snapshot.** The bundled file is a fixed snapshot of the upstream dataset at a
specific commit. It does not update automatically with upstream changes.
Measurements across different LLenergyMeasure releases that bundle different
upstream commits are not strictly comparable on this dimension. The pinned
commit is recorded in the provenance header of the JSONL file.

**n_prompts default is not stratified.** The default `n_prompts: 100` selects
the first 100 rows in file order. This is not a stratified or representative
sample of the full 1,000-row corpus. For studies where the prompt-distribution
composition of the sample matters, use `order: shuffled` with a fixed
`random_seed` to draw a repeatable pseudo-random sample across the corpus.

---

## Reasoning models and prompt uncertainty

Reasoning-capable models (models that produce chain-of-thought traces before a
final answer) exhibit an additional dataset-energy interaction that is worth
calling out explicitly.

When a reasoning model faces an ambiguous or adversarial prompt - one where the
answer is not directly inferable from the surface form - it tends to emit longer
chains of thought. The model effectively makes multiple internal "passes" before
committing to a final answer. This means that for a reasoning model, the
**uncertainty distribution of the prompt corpus** interacts with output length
and therefore with energy consumption in ways that are not present for
non-reasoning models.

A prompt corpus composed of short, factual, unambiguous questions ("What year
did the Berlin Wall fall?") will elicit terse chain-of-thought from a reasoning
model. A corpus of ambiguous or multi-step questions will elicit much longer
traces. The energy difference can be substantial - potentially several-fold for
the same model on the same hardware.

This has two implications:

- When benchmarking reasoning models on AIEnergyScore prompts, be aware that
  the energy figure is specific to that prompt uncertainty distribution.
  General-purpose web text (the dominant content in the AIEnergyScore corpus)
  is not particularly adversarial to reasoning models. The benchmark may
  underestimate energy in task-specific deployments that regularly encounter
  ambiguous or multi-step inputs.

- Studying how prompt type drives energy in reasoning models is a legitimate
  research use of the bring-your-own-dataset capability. A study design that
  sweeps prompt types (short factual, multi-step, adversarial, underspecified)
  while holding the model and hardware constant directly measures how
  chain-of-thought length responds to input uncertainty. See the BYOD section
  below.

---

## Bring your own dataset

LLenergyMeasure accepts any JSONL file as a prompt source. Set `source` to a
file path in the `task.dataset` block:

```yaml
task:
  dataset:
    source: ./my-prompts.jsonl
    n_prompts: 100
    order: shuffled
```

See [Reference: Dataset format](/reference/dataset-format) for the full JSONL
schema, required fields, optional fields, and validation rules.

Custom datasets enable several research workflows that the bundled dataset
cannot support:

**Task-domain energy profiles.** Measure the same model on a domain-specific
corpus (your own legal documents, code snippets, clinical notes) to understand
how energy scales in your actual deployment context. Compare the domain-specific
figure against the AIEnergyScore baseline to quantify how much your task domain
differs from the reference distribution.

**Prompt-length sensitivity.** Construct datasets with controlled prompt length
distributions (e.g., 50-token prompts, 200-token prompts, 800-token prompts)
and sweep across them to produce an energy-vs-input-length curve. This is
useful for capacity planning and for understanding where your use case sits on
that curve.

**Reasoning-depth sensitivity.** For reasoning models, construct corpora that
vary the difficulty and ambiguity of prompts. Pair with `order: shuffled` and
`n_prompts` set to a statistically adequate count to measure how chain-of-thought
length (and therefore energy) responds to prompt type.

**Reproducible custom baselines.** If you want cross-study comparability
against colleagues using the same custom corpus (rather than the AIEnergyScore
benchmark), agree on a shared JSONL file and pin it by hash. LLenergyMeasure
does not currently validate dataset file hashes, but you can record the SHA-256
of your corpus in your study config comments as a reproducibility convention.

---

## Practical guidance for new studies

**Start with the default.** If your research question is about the relative
efficiency of models, engines, or quantisation settings rather than task-domain
effects, run with the default AIEnergyScore dataset. This gives you results that
are directly comparable to other AIEnergyScore measurements in the literature
and on the public leaderboard.

**Increase n_prompts for lower variance.** The default `n_prompts: 100`
produces reasonable estimates for most purposes, but it is a small sample. With
100 prompts, individual outlier prompts (extremely short or extremely long) have
more influence on the mean. For publication-quality measurements, 500 prompts
provides substantially lower variance. The full 1,000 is the maximum the bundled
dataset supports.

**Use shuffled order for representative sampling.** If you increase `n_prompts`
substantially, switch to `order: shuffled` to draw from across the corpus
rather than taking the first N rows:

```yaml
task:
  dataset:
    source: aienergyscore
    n_prompts: 500
    order: shuffled
random_seed: 42
```

Setting `random_seed` makes the shuffle repeatable - the same seed and file
always produce the same prompt sequence.

**Run a parallel BYOD pass if task domain matters.** If your downstream
deployment involves a specific prompt domain, run two studies in parallel - one
with the default AIEnergyScore corpus (for comparability) and one with your
domain corpus (for operational accuracy). Report both, with a note on the
difference. This is more informative than either number alone.

**Report dataset provenance.** When publishing results, always include the
dataset source, `n_prompts`, ordering mode, and (for the bundled dataset) the
LLenergyMeasure version, which determines which snapshot of AIEnergyScore is
bundled. The full set of reproducibility fields is documented in
[Measurement methodology](/explanation/methodology/methodology) and
[Comparison with other benchmarks](/explanation/methodology/comparison-context).

---

## Summary

| Question | Guidance |
|----------|----------|
| Comparing models or configs on the same hardware? | Use the default AIEnergyScore dataset. Hold dataset constant. |
| Comparing results across studies or publications? | Require the same dataset source, n_prompts, and order. |
| Studying domain-specific deployment costs? | Use a custom JSONL corpus representative of your task. |
| Studying reasoning model sensitivity? | Vary prompt uncertainty distribution across custom corpora. |
| Need statistical confidence? | Increase n_prompts (200-1000); use `order: shuffled`. |

---

## See also

- [Reference: Dataset format](/reference/dataset-format) - JSONL schema, field
  descriptions, and full AIEnergyScore provenance
- [Measurement methodology](/explanation/methodology/methodology) - warmup, baseline power,
  multi-cycle execution, and reproducibility
- [Comparison with other benchmarks](/explanation/methodology/comparison-context) - how
  LLenergyMeasure results relate to MLPerf, AIEnergyScore leaderboard, and
  other measurement tools
