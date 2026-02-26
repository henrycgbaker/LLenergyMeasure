# Dataset Handling

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-19
**Research:** N/A

## Decision

Named built-ins (AIEnergyScore default) + bring-your-own JSONL + synthetic fixed-length mode. Datasets ship pinned inside the package (no runtime download). Fixed order, deterministic load; `n: 100` always selects first 100 prompts. HuggingFace Datasets deferred to later v2.x.

---

## Context

The dataset plays a fundamentally different role here than in evaluation frameworks like lm-eval: it is a **workload definition**, not an evaluation task. Its purpose is to produce a representative
distribution of prompt lengths and output lengths for energy measurement. Two datasets with
the same statistical profile produce the same energy measurement — scientific validity is
in the distribution, not the content.

Constraints shaping the decision:
- Energy measurement is already noisy — prompt variability between runs would compound
  measurement noise. Same prompts, same order, every run is a correctness requirement.
- Users need zero-config for standard measurements (built-ins) and a path for custom
  production workloads (bring-your-own).
- Air-gapped HPC environments cannot download datasets at experiment time.
- Prompt length variability confounds batch-size and throughput sweeps — a synthetic
  fixed-length mode is scientifically useful.
- No network dependency during experiments.

## Considered Options

### Sub-decision 1: Dataset sourcing strategy

| Option | Pros | Cons |
|--------|------|------|
| **Named built-ins + bring-your-own JSONL + synthetic mode** | Zero-config for standard use. Custom workloads supported. Controlled ablations via synthetic. No package manager needed. | Three modes to maintain. Synthetic mode adds implementation scope. |
| Built-ins only | Simplest implementation. | No path for proprietary/production prompt distributions. No controlled-length ablations. |
| HuggingFace Datasets integration | Large ecosystem of available datasets. | Dependency complexity. Network required at experiment time or install time. No scientific benefit for energy measurement over a curated built-in. |
| Users always supply their own JSONL | Maximum flexibility. | Breaks zero-config goal. High barrier for researchers who just want to measure a model. |

### Sub-decision 2: Default built-in dataset

| Option | Pros | Cons |
|--------|------|------|
| **AIEnergyScore `text_generation` dataset** | 1,000 prompts sampled from WikiText + OSCAR + UltraChat. Validated by HuggingFace/CMU as representative general-purpose text generation workload. Aligns with emerging measurement standard. | Single domain mix — may not represent all production workloads. |
| Manually curated set of prompts | Full control over content. | Maintenance burden. No external validation. Hard to justify as "standard" in publications. |
| ShareGPT / LMSYS conversations | Representative of chat workloads. | Licence ambiguity. Less well-validated for energy measurement specifically. |
| Random text / lorem ipsum | Fully neutral. | Not representative of real inference workloads — would produce misleading throughput profiles. |

### Sub-decision 3: How built-in datasets are distributed

| Option | Pros | Cons |
|--------|------|------|
| **Ship pinned datasets inside the package (no runtime download)** | Works in air-gapped HPC. No network dependency at experiment time. Same prompts every run — measurement reproducibility. | Increases package size. Pinned version may become stale over time. |
| Download at first use and cache | Small package. Always fetchable. | Breaks air-gapped environments. Cache invalidation complexity. Download failure breaks experiment. |
| Download at install time | Single download. | Install fails in air-gapped environments. Version drift if dataset changes upstream. |

### Sub-decision 4: HuggingFace Datasets integration at v2.0

| Option | Pros | Cons |
|--------|------|------|
| **Defer HuggingFace Datasets to a later v2.x release** | No added dependency complexity. Built-ins + BYOF cover all v2.0 use cases. Ship simpler. | Users who want dynamic HuggingFace datasets must convert to JSONL manually. |
| Include at v2.0 | Broader ecosystem access from the start. | Adds `datasets` as a dependency with transitive complexity. Network required. No scientific benefit for energy measurement over curated built-in. |

### Sub-decision 5: Reproducibility mechanism

| Option | Pros | Cons |
|--------|------|------|
| **Fixed order, deterministic load; shuffling opt-in (not implemented at v2.0)** | Same `n: 100` always selects first 100 prompts — identical across runs and machines. | If first 100 prompts are atypical, user must know to increase `n`. No randomisation for statistical robustness checks. |
| Shuffle with fixed seed | Deterministic but covers more of the dataset. | More complex. Seed management adds config surface. |
| No reproducibility guarantee | Simpler. | Invalidates cross-run comparisons — energy measurement noise becomes indistinguishable from prompt variation. |

## Decision

We will support three dataset modes: named built-ins, bring-your-own JSONL, and synthetic
length-controlled. The default built-in is `aienergyscore` (AIEnergyScore `text_generation`
dataset — 1,000 prompts from WikiText + OSCAR + UltraChat, pinned). Built-in datasets ship
inside the package (no runtime download). HuggingFace Datasets integration is deferred to
a later v2.x release. Datasets load in fixed order; shuffling is opt-in and not implemented
at v2.0.

Rationale: the AIEnergyScore dataset is externally validated by HuggingFace/CMU and aligns
with the emerging measurement standard — using it makes our measurements directly comparable
to published AIEnergyScore benchmarks. Shipping datasets in-package is a hard requirement
for air-gapped HPC correctness. The synthetic mode fills a genuine scientific need (controlled
ablations where prompt length must be held constant).

## Consequences

Positive:
- Zero-config default that aligns with an emerging measurement standard.
- Works in air-gapped HPC — no network dependency at experiment time.
- Same prompts every run — measurement reproducibility guaranteed.
- Synthetic mode enables controlled ablations that built-ins cannot provide.

Negative / Trade-offs:
- Package size increases by the size of the shipped built-in datasets.
- HuggingFace Datasets users must convert to JSONL in v2.0.
- First 100 prompts of a built-in may not represent the full distribution for small `n`.

Neutral / Follow-up decisions triggered:
- `sharegpt` built-in deferred — do not ship speculatively; add when there is concrete user demand.
- Dataset shuffling deferred — current deterministic order is correct for reproducibility.
- HuggingFace Datasets integration deferred to a later v2.x release.

## Dataset Modes

### Mode 1 — Named built-in (zero-config default)

```yaml
dataset: aienergyscore   # default — 1000 prompts: WikiText + OSCAR + UltraChat
dataset: synthetic       # see Mode 3
```

Config field `dataset` accepts a named string. Built-ins ship as JSON files inside the package.

**Shipped built-ins at v2.0:**
- `aienergyscore` — default. 1,000 prompts sampled from WikiText + OSCAR + UltraChat.
  Source: `AIEnergyScore/text_generation` (HuggingFace Hub, pinned to a specific commit).
  Ship a curated 200-prompt subset for default experiments; full 1,000 available via `n: 1000`.

**Future built-ins (add when warranted, not speculatively):**
- `sharegpt` — longer multi-turn conversations; useful for measuring throughput under long
  context. **Deferred — do not ship until there is concrete user demand.**

### Mode 2 — Bring-your-own JSONL file

```yaml
dataset: path/to/my_prompts.jsonl   # relative to CWD or absolute path
```

**JSONL format** (one JSON object per line):
```json
{"prompt": "Explain the difference between ...", "max_new_tokens": 100}
{"prompt": "Write a summary of ...", "max_new_tokens": 200}
```

- `prompt`: required — the input text.
- `max_new_tokens`: optional — overrides the experiment-level `max_new_tokens` for this prompt.

Use when: measuring energy on a specific production workload or proprietary prompt distribution.

### Mode 3 — Synthetic length-controlled

```yaml
dataset:
  synthetic:
    n: 100           # number of prompts (overrides experiment-level n if set here)
    input_len: 512   # tokens (approximate — padded/truncated to target)
    output_len: 128  # tokens (sets max_new_tokens)
    seed: 42         # reproducibility (default: inherits experiment random_seed)
```

Use when: holding prompt length constant while varying batch size, precision, or backend.
Produces fully deterministic, reproducible sequences. Sequences are generated from the random
seed — same seed = identical prompts every run.

**Why synthetic is useful:** Real dataset prompts have variable lengths, which introduces
throughput noise when sweeping batch size. Synthetic prompts with fixed input/output lengths
isolate the configuration variable under study.

## Reproducibility Guarantee

Built-in datasets: prompts are in a fixed order, loaded deterministically. Same `n: 100`
always selects the first 100 prompts from the built-in file. Shuffling is opt-in only
(not implemented at v2.0).

BYOF datasets: loaded in file order. User is responsible for ordering.

Synthetic datasets: generated from `seed` — same seed = identical sequence.

## Config Integration

In `ExperimentConfig`:
```python
dataset: str | SyntheticDatasetConfig = "aienergyscore"
n: int = 100   # number of prompts to use from the dataset
```

`n` defaults to 100 for zero-config experiments. For studies, `n` can be swept or overridden
per experiment. `n > len(dataset)` → hard validation error at config parse time.

See `designs/experiment-config.md` for the full `SyntheticDatasetConfig` model definition.

## Deferred

- **HuggingFace Datasets integration** — deferred to a later v2.x release.
- **`sharegpt` built-in** — deferred until needed. Do not ship speculatively.
- **Dataset shuffling / random sampling** — deferred. Current deterministic order is correct
  for reproducibility.

## Related

- [../designs/experiment-config.md](../designs/experiment-config.md) — `SyntheticDatasetConfig` model and `dataset` field definition
- [warmup-strategy.md](warmup-strategy.md) — interaction between dataset size and warmup runs
