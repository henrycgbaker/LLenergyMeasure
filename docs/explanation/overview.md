---
title: Explanation
description: Understand why measurements are valid and how the system delivers them.
---

# Explanation

This section answers the "why" and "how" behind LLenergyMeasure - the questions
that sit beneath the practical guides.

**Methodology** covers why our measurements are scientifically valid: what we
measure, how we measure energy, why those choices matter, and how our results
relate to other benchmarks. Start here if you want to evaluate or cite the tool
in research.

**Architecture** covers how the system delivers those measurements: the
dataflow from configuration to result, the engine-plugin design, the
parameter-discovery and curation pipelines, and the CI infrastructure. Start
here if you want to understand or extend the internals.

---

## What this section answers

- *Why* is a particular energy measurement approach used? (Methodology)
- *Why* does dataset choice affect measurement validity? (Methodology)
- *How* does the experiment pipeline convert a config into a result? (Architecture)
- *How* are invalid parameter combinations detected before an engine runs? (Architecture)
- *Why* does this tool exist and what gap does it fill? (Why)

---

## Sub-sections at a glance

### Methodology

The scientific grounding for every number LLenergyMeasure produces.

- **[Methodology overview](/explanation/methodology/methodology)** - warmup protocol,
  baseline power subtraction, thermal stabilisation, and reproducibility.
- **[What we measure](/explanation/methodology/what-we-measure)** - plain-language guide
  to energy (joules), throughput (tokens/s), and FLOPs.
- **[Energy measurement](/explanation/methodology/energy-measurement)** - NVML telemetry,
  sampler backends, and known measurement limits.
- **[Measurement warnings](/explanation/methodology/measurement-warnings)** - what each
  runtime warning means and how to act on it.
- **[Comparison context](/explanation/methodology/comparison-context)** - how results
  relate to MLPerf, AI Energy Score, and other benchmarks.
- **[Dataset context](/explanation/methodology/dataset-context)** - why dataset choice
  shapes energy numbers and guidance for custom datasets.
- **[Glossary](/explanation/methodology/glossary)** - definitions for every term used
  in results, configs, and methodology docs.

### Architecture

The engineering design behind the measurement pipeline.

- **[Architecture overview](/explanation/architecture/architecture-overview)** - system
  components, dataflow, and the harness-plugin contract.
- **[Pipeline architecture](/explanation/architecture/pipeline-architecture)** - the
  end-to-end path from CLI invocation to written results.
- **[CI architecture](/explanation/architecture/ci-architecture)** - how automated
  engine-coupling validation, schema refresh, and image publishing are wired.
- **[Internal pipelines](/explanation/architecture/parameter-discovery)** - parameter
  discovery (runtime validation) and parameter curation (introspection pipeline).

### Why

- **[Why LLenergyMeasure](/explanation/why)** - the research gap, what the
  tool does that is distinctive, and where its design is heading.

### Ecosystem

- **[Ecosystem](/explanation/ecosystem)** - where LLenergyMeasure sits
  relative to energy samplers, capability harnesses, and benchmark suites.

---

## Where to start

- Evaluating measurement validity for a paper? Start at
  [Methodology overview](/explanation/methodology/methodology).
- Understanding the system for extension or contribution? Start at
  [Architecture overview](/explanation/architecture/architecture-overview).
- Looking for definitions? Go straight to the
  [Glossary](/explanation/methodology/glossary).
