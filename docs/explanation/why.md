---
title: Why LLenergyMeasure
description: The research gap LLenergyMeasure addresses, what the tool does that is distinctive, and where its design is heading.
---

# Why LLenergyMeasure

LLenergyMeasure is an open research tool for measuring how implementation
choices drive LLM inference efficiency. This page sets out the gap that
motivated the tool, the three architectural legs that constitute the present
system, the project's origin and the parts of that origin that are now
known to be wrong, the explicit boundaries of the tool, and the direction
the project is sized for as inference workloads shift.

The voice here is research-paper voice: claims are sourced where the
sources are stable, and limits are stated rather than papered over.

---

## The research gap

Most published LLM-efficiency work compares models at fixed implementation
and reports a throughput, latency, or accuracy number. The implementation -
the engine, the dtype, the batching strategy, the attention kernel,
quantisation form, KV-cache reuse, paged attention, and so on - is held
constant per benchmark cell, and the model varies. This is the natural
shape if the question is "which model is most efficient on this hardware,
under this configuration?".

The reverse question is at least as interesting and considerably less
studied: with the model and prompts held fixed, how do implementation
choices drive energy and throughput? For a given researcher running
Llama-3-8B on an A100, the engine choice between transformers, vLLM, and
TensorRT-LLM is not a small effect; nor is dtype, nor attention backend,
nor batch size at the prefill / decode split. These choices interact, and
their interactions are not well-characterised in published comparisons.

Existing tools cover adjacent layers. Energy samplers like NVML, Zeus
([Chung et al., 2023](https://www.usenix.org/conference/nsdi23/presentation/you))
and CodeCarbon ([Schmidt et al., 2021](https://codecarbon.io/)) measure
power and energy at the GPU or system level; they do not orchestrate
inference experiments or reason about implementation parameters.
Capability harnesses like
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
measure quality (accuracy, perplexity) at fixed implementation; they do
not measure energy. Standardised benchmarks like
[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)
fix rules for fair hardware comparison; the rule-fixing is the whole
point, and is the opposite of what a researcher needs when the variable of
interest is implementation itself.

No tool sat at the methodology layer that made implementation effects on
efficiency a first-class measurable axis across multiple inference
engines. That is the gap LLenergyMeasure addresses.

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

This is the "stitching" framing: the value is in the integration, not in
any individual component.

### Programmatic discovery of engine parameters

Inference engines expose hundreds of configuration parameters between
them. Hand-curating a list of "the parameters that matter" caps the
research surface arbitrarily and goes stale within a release cycle. The
tool's parameter-discovery pipeline introspects each engine's config
classes, deduplicates equivalent parameters, and exposes the result
programmatically to study configurations. New parameters are picked up
when the engine version is bumped.

The implication for the user-facing surface is that the implementation
parameters available to a study are not a closed set. Examples include
dtype, batching strategy, attention backend, quantisation form, and
KV-cache reuse, but the list is open: every parameter declared by the
engine introspection is exposed, and the sweep grammar accommodates
arbitrary axes. The closed-set framing has been a recurring failure mode
in adjacent tools and is something we deliberately avoid.

See [Parameter discovery](architecture/parameter-discovery.md) for the
introspection pipeline and
[Engine extensibility](architecture/engine-extensibility.md) for what a
new engine has to contribute.

### Invariant mining and sweep deduplication

Exposing every parameter programmatically would produce a Cartesian
explosion that is intractable to sweep. The tool's invariant-mining
pipeline mines the engine source for the constraints (mutual
exclusions, derived defaults, version-gated combinations) that the
engine itself enforces, and uses these to prune the sweep space before
any inference runs. The pruning preserves coverage of the legitimate
configuration manifold while collapsing the combinatorial cost.

Mining and exposure are paired: the "all parameters are first-class"
claim is only credible because mining keeps the resulting sweep budget
finite. Without the pruning the offer is empty.

See [Parameter curation](architecture/parameter-curation.md) for the
curation pipeline and
[Auto-refresh pipeline](architecture/auto-refresh-pipeline.md) for how
the mined invariants stay current as upstream engines version.

### The harness-plugin separation

A cross-cutting design decision sits beneath the three legs: the
measurement harness owns methodology (warmup, baseline subtraction,
thermal stabilisation, sampler lifecycle, FLOPs validity check) and the
engine plugin owns inference (load, generate, release). The boundary is
explicit. Methodology improvements roll out atomically across all
engines; engine bugs do not corrupt energy accounting; new engines
inherit measurement rigour for free.

See [Harness-plugin model](architecture/harness-plugin.md).

---

## Origin and maturity

The tool grew from a master's thesis on LLM energy efficiency
([Baker, 2025](https://henrycgbaker.github.io/research/llm-energy-efficiency/)).
The thesis investigated how parallelisation, batch size, and quantisation
affected per-token energy on a single inference engine, on a single GPU
class, against a hand-picked prompt set.

The thesis is the seed of the current tool, but it is not the current
tool. Several of the original methodological choices are now known to be
wrong by the standards the project holds itself to, and have been
replaced:

- **Parallelisation handling.** The thesis treated tensor-parallel runs
  as commensurable with single-GPU runs at the per-token energy level.
  This is wrong: tensor-parallel introduces synchronisation cost that
  does not factor cleanly into per-token attribution. The current tool
  separates tensor-parallel measurements explicitly and does not pool
  them with single-GPU runs without a stated correction.
- **Parameter choice.** The thesis enumerated a small fixed set of
  configuration parameters as "the implementation choices". The current
  tool replaces this with programmatic discovery (above). The thesis's
  closed-list framing under-represents the surface and is exactly the
  failure mode the discovery pipeline corrects.
- **Engines as a category.** The thesis worked with a single engine and
  treated the engine as backdrop. The current tool makes engines a
  first-class category in the architecture, because the engine is one
  of the largest implementation effects on energy that exists.

This honesty about the project's evolution is itself part of the
methodology-first ethos. A measurement tool that is not willing to say
"this earlier approach was wrong, here is the corrected one" is not
trustworthy as an instrument.

---

## Boundaries

LLenergyMeasure does not aspire to be everything. The boundaries below
are deliberate and load-bearing.

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

**FLOPs is a validity check, not a headline metric.** FLOPs are reported
because they are cheap to estimate and useful for sanity-checking that a
configuration is doing what it claims. They are largely invariant across
implementations of the same model, which is exactly why they make a good
check rather than a good comparison axis. Headlines are energy and
throughput.

---

## Where this goes next

Inference is shifting. Reasoning models do multi-pass under uncertainty;
agentic harnesses chain tool-use over many model calls; scaffolds add
re-prompting, verifier passes, and sampling-strategy variation as
first-class parts of the workload. In these settings, implementation
detail dominates energy budgets even more than in the single-pass case.
A reasoning model with 10x output-length variance produces something
near 10x energy variance; an agentic loop with adaptive depth produces
energy distributions that are not well-summarised by a single
mean-tokens-per-call number.

The plugin architecture is sized for these workloads. Sampling-strategy
plugins, per-call energy attribution, and harness-aware metrics
(distribution rather than mean; tail behaviour rather than centre) are
natural extensions of the existing contract rather than redesigns.
Open-source agent harnesses are appearing in increasing numbers and the
research community will need a measurement primitive that does not
flatten their distributional structure.

This is forward-looking, and is flagged here without commitment to a
specific delivery shape. The current contribution is the present tool;
the future direction is what the present tool is sized for.

---

## Citation

If you use LLenergyMeasure in research, please cite the project. See
[Citation](/contributing/citation) for the BibTeX entry and the
upstream-dataset citation requirements.
