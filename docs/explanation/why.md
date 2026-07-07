---
title: Why LLenergyMeasure
description: The research gap LLenergyMeasure addresses, what the tool does that is distinctive, and where its design is heading.
---

# Why LLenergyMeasure

## The research gap

Most published LLM-efficiency work compares models at fixed implementation
and reports a throughput, latency, or accuracy number. The infernece implementation is held
constant per benchmark cell, and the model varies. This is the natural
shape if the question is "which model is most efficient on this hardware,
under this configuration?".

The reverse question is at least as interesting and considerably less
studied: with the model and task held fixed, how do implementation
choices drive energy and throughput? For a given researcher or
open-weights inference provider, the engine choice is not a small
effect; nor is dtype, nor attention kernel, quantisation form, KV-cache
reuse, paged attention, nor batch size at the prefill / decode split.
These choices interact, and their interactions are not well-characterised
in openly published comparisons.

Existing tools cover adjacent layers. Energy samplers like NVML, Zeus
([You et al., 2023](https://www.usenix.org/conference/nsdi23/presentation/you))
and [CodeCarbon](https://codecarbon.io/) measure power and energy at the
GPU or system level; they do not orchestrate inference experiments or
reason about implementation parameters.
Capability harnesses like
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
measure quality (accuracy, perplexity) at fixed implementation; they do
not measure energy. Standardised benchmarks like
[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)
fix rules for fair hardware comparison; the rule-fixing is the whole
point, and is the opposite of what a researcher needs when the variable of
interest is implementation itself.

No open research tool sat at the methodology layer has made
implementation effects on efficiency a first-class measurable axis.
That is the gap LLenergyMeasure addresses - both for researchers
studying open-weights inference, and for the open-source community that
runs the engines being measured.

---

## What the tool does

LLenergyMeasure is structured around three architectural legs. Each leg
is a deliberate response to a specific obstacle to answering the
implementation-effect question. The legs are independently useful and
together they constitute the contribution.

### Integration of energy samplers and inference engines

The tool wires together the existing energy-measurement and inference
ecosystems rather than reimplementing them. Energy samplers (NVML, Zeus,
CodeCarbon) are pulled in via a sampler-plugin contract; inference
engines (transformers, vLLM, TensorRT-LLM, with SGLang planned) are
pulled in via an engine-plugin contract. The tool is the interstitial
layer that turns these into a single measurement pipeline. Researchers
specify a study; the harness runs it; the result is a structured record
that names the sampler, the engine, the parameters, and the environment
in full. 

Here, the value is in the integration, not in
any individual component.

### Programmatic discovery of engine parameters

Inference engines expose hundreds of configuration parameters between
them. Hand-curating a list of "the parameters that matter" caps the
research surface arbitrarily and goes stale within a release cycle. The
tool's parameter-discovery pipeline introspects each engine's config
classes, deduplicates equivalent parameters, and exposes the result
programmatically to study configurations. New parameters are automatically picked up
when the upstream engine version is bumped (and stabilised).

The implication for the user-facing surface is that the implementation
parameters available to a study are not a closed set. Examples include
dtype, batching strategy, attention backend, quantisation form, and
KV-cache reuse, but the list is open: every parameter declared by the
engine introspection is exposed, and the sweep grammar accommodates
arbitrary axes.

For user-ease we do offer a typed set of curated energy-relevant parameters per engine, but this is a convenience layer, not a limitting contract.

See [Parameter discovery](architecture/parameter-discovery.md) for the
config-validation loader and
[Engine extensibility](architecture/engine-extensibility.md) for what a
new engine has to contribute.

### Validation rules and sweep deduplication

Exposing every parameter programmatically and sweeping along each new axis produces a Cartesian
explosion that is intractable to sweep. The tool's per-engine validation
rules capture the constraints (mutual exclusions, derived defaults,
version-gated combinations) that the engine itself enforces - read from
the engine source and verified against it - and use these to prune the
sweep space before any inference runs. The pruning preserves coverage of the
legitimate configuration manifold while collapsing the combinatorial cost.

Similarly, the structured sweep grammar separates parameters that vary
independently from parameters that genuinely co-vary. Independent axes
(lists of scalars, e.g. `dtype: [fp16, bf16]`) compose Cartesianly;
dependent groups (lists of named variants, e.g. a `quantisation` group
pairing `format` with its compatible `scale` mode) compose as a union
of meaningful combinations rather than a Cartesian product of every
field with every other. The grammar keeps users from accidentally
authoring the explosion in the first place, so the rules have
less to prune.

Rules and exposure are paired: the "all parameters are first-class"
claim is only credible because the rules keep the resulting sweep budget
finite. Without the pruning the offer is empty.

See [Parameter curation](architecture/parameter-curation.md) for the
curation step and
[Pipeline architecture](architecture/pipeline-architecture.md) for how
the rules stay current as upstream engines version.

### The harness-plugin separation

A cross-cutting design decision sits beneath the three legs: the
measurement harness owns methodology (warmup, baseline subtraction,
thermal stabilisation, sampler lifecycle, FLOPs validity check) and the
engine plugin owns inference (load, generate, release). The boundary is
explicit. Methodology improvements roll out atomically across all
engines, and new engines
inherit measurement rigour for free.

See [Harness-plugin model](architecture/harness-plugin.md).

---

## Origin and maturity

The tool grew from a master's thesis on LLM energy efficiency
([Baker, 2025](https://henrycgbaker.github.io/research/llm-energy-efficiency/)).
That thesis is the seed of the current tool, but it is not the current
tool. Several of the original methodological choices are now known to be
wrong by the standards the project holds itself to, and have been
replaced.

---

## Boundaries

The boundaries below are deliberate and load-bearing.

**Not a benchmark.** A benchmark fixes rules so that results are fair
across submitters; the rule-fixing is the value. LLenergyMeasure is the
opposite shape: researchers specify the conditions, the tool measures
under those conditions, and the conditions vary as part of the
investigation. Outputs from LLenergyMeasure can and should inform
benchmark design (for example: if engine choice contributes most of the
variance for a given model class, future benchmarks should fix engine
when comparing models). The tool feeds benchmarks; it is not one.

**Not a competitor to integrated tools.** Zeus and CodeCarbon are
upstream samplers and are integrated as plugins. lm-evaluation-harness
measures a different thing (capability) and is complementary; the two
are routinely run alongside each other. MLPerf is a fixed-rule
benchmark and addresses an audience (procurement, hardware vendors)
that this tool does not target.

For full ecosystem positioning, see [Ecosystem](ecosystem.md).

---

## Where this goes next

Concrete near-term directions:

- **Additional inference engines.** SGLang is the next planned engine
  plugin (RadixAttention prefix-cache energy profiles); the
  engine-plugin contract is sized to absorb new entrants as they
  stabilise.
- **Adaptive sweep sampling.** Programmatic discovery + invariant
  mining keeps the tractable parameter space large; further reductions
  would come from adaptive sampling that prioritises experiments most
  informative about the impl-effect question.
- **Open-science result database.** A web frontend that lets users
  submit results against a shared schema. The bundled environment-
  snapshot metadata (hardware, drivers, library versions, sampler
  configuration) means submissions are comparable across hardware classes
  and devices.
- **Reasoning models.** Multi-pass-under-uncertainty inference makes
  the *generated token count* - not the visible answer length - the
  dominant driver of per-call energy. A 10x variance in total
  generated tokens produces something near 10x energy variance, yet
  the resulting distribution is not well-summarised by a single
  mean-energy-per-call figure.
- **Agentic harnesses.** Open-source agent harnesses are appearing in increasing; these tool-use chains, re-prompting scaffolds, and
  verifier passes turn a single user request into many model calls
  with adaptive depth. Implementation detail dominates energy budgets
  even more than in the single-pass case, and the unit of useful
  attribution shifts from per-inference to per-call,
  per-tool-invocation, and per-decision-step. The engine-plugin
  contract already isolates the inference call as the attribution
  unit; harness-aware aggregation sits above it as a natural
  extension. 

---

## Citation

If you use LLenergyMeasure in research, please cite the project. See
[Citation](/contributing/citation) for the BibTeX entry and the
upstream-dataset citation requirements.
