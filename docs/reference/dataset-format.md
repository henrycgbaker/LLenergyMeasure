---
title: Dataset format
description: Reference for the task.dataset configuration block, built-in datasets, and custom JSONL format.
---

# Dataset format

This page documents the `task.dataset` configuration block: every field, type,
default, and validation rule. It also describes the built-in `aienergyscore`
dataset and the JSONL format for custom prompt files.

---

## `task.dataset` configuration block

The `task.dataset` block sits inside `task:` in the experiment YAML. All fields
are optional; the defaults give a reasonable out-of-the-box workload.

```yaml
task:
  dataset:
    source: aienergyscore   # default
    n_prompts: 100          # default
    order: interleaved      # default
```

### Fields

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `source` | string | `"aienergyscore"` | Non-empty string. Either a recognised built-in alias or a path to an existing `.jsonl` file. Any other value raises `ValueError` at load time. |
| `n_prompts` | integer | `100` | Must be `>= 1`. The loader raises `ValueError` if the dataset has fewer rows than requested. |
| `order` | `"interleaved"` \| `"grouped"` \| `"shuffled"` | `"interleaved"` | Must be one of the three literals. Any other string raises `ValueError`. |

#### `source`

Either a built-in alias (currently `"aienergyscore"`) or a relative or absolute
path to a `.jsonl` file. File paths are resolved relative to the working directory
at run time (typically where you invoked `llem run`).

#### `n_prompts`

The number of prompts loaded and sent to the engine. For publication-quality
measurements, 100 prompts (the default) gives stable throughput and energy
estimates. Use 10-25 for quick iteration.

For the `aienergyscore` built-in, the full corpus is 1,000 prompts; any
`n_prompts <= 1000` is valid.

#### `order`

Controls which prompts are selected and in what sequence:

- `interleaved` (default) - reads prompts in file order, stopping at `n_prompts`.
  For multi-source datasets, rows from different sources are interleaved in the
  order they appear in the file.
- `grouped` - sorts rows by their `source` field before selecting. Prompts from
  the same origin appear consecutively.
- `shuffled` - randomly shuffles rows before selecting, using `task.random_seed`
  as the RNG seed. Same seed + same dataset = same prompt sequence.

---

## Built-in datasets

### `aienergyscore`

The default built-in dataset. Bundled at
`src/llenergymeasure/datasets/builtin/aienergyscore.jsonl`.

#### At a glance

| Property | Value |
|----------|-------|
| Total prompts | 1,000 |
| Upstream source | [AIEnergyScore](https://huggingface.github.io/AIEnergyScore/) (`AIEnergyScore/text_generation`) |
| Upstream commit pinned | `2dc92b2ee2cd9776a51ccf08c6c2ab04138370c3` |
| Licence | Apache 2.0 |
| Content | WikiText, OSCAR, UltraChat text passages |
| Languages | Primarily English; some multilingual (OSCAR) |
| Prompt field | `text` (auto-detected by the loader) |
| Prompt lengths | Variable - short phrases to multi-paragraph passages |

#### Provenance

The [AIEnergyScore project](https://huggingface.github.io/AIEnergyScore/) is a
Hugging Face initiative that assigns standardised energy efficiency ratings to AI
models across ten common inference tasks (text generation, summarisation,
question answering, classification, and others). It is led by Sasha Luccioni,
Yacine Jernite, and collaborators from Salesforce, Cohere, Meta, Neuralwatt,
and Carnegie Mellon University.

The `text_generation` task dataset (`AIEnergyScore/text_generation`) contains
1,000 text passages sampled from three public corpora: WikiText (encyclopaedic
text), OSCAR (multilingual web text), and UltraChat (instruction-style
conversations). The intentional mix of registers and prompt lengths is designed
to stress-test generation under varied input conditions rather than optimising a
single input profile.

The provenance header in the bundled JSONL records the upstream commit:

```json
{
  "_provenance": "AIEnergyScore/text_generation",
  "_commit": "2dc92b2ee2cd9776a51ccf08c6c2ab04138370c3",
  "_license": "apache-2.0",
  "_description": "1000 prompts from WikiText, OSCAR, UltraChat"
}
```

This header line is skipped by the loader (all keys start with `_`); it exists
solely for traceability.

#### Bundled, not downloaded

The JSONL ships inside the `llenergymeasure` package at
`src/llenergymeasure/datasets/builtin/aienergyscore.jsonl`. No network call is
made at run time. The file is loaded via
`llenergymeasure.datasets.BUILTIN_DATASETS["aienergyscore"]`, which resolves to
a `Path` object pointing at the bundled file. This means measurements are
reproducible on air-gapped machines and are not affected by upstream dataset
changes.

The tradeoff: the bundled file is a fixed snapshot of the upstream dataset at
the pinned commit. It does not track subsequent upstream changes. If Hugging
Face updates the `AIEnergyScore/text_generation` dataset after the snapshot was
taken, the bundled version will not reflect those changes until a new release of
`llenergymeasure` re-bundles it.

#### Composition and prompt selection

The file contains exactly 1,001 lines: one provenance header followed by 1,000
prompt records. Each record has the form:

```json
{"prompt": "<text>"}
```

The loader's auto-detection tries columns in the order `prompt`, `text`,
`instruction`, `input`, `question`. In practice, all records in this dataset use
the `text` field (the upstream column name). The loader maps it transparently.

The default `n_prompts: 100` selects the **first 100 records in file order**
(the `interleaved` ordering strategy). This is not a stratified or randomised
sample - it is the first 100 rows as the file was written, which in practice
reflect the upload order of the upstream dataset. For most throughput and energy
measurements this is sufficient. For studies where prompt-distribution effects
may be a concern, use `order: shuffled` with a fixed `random_seed` to draw a
repeatable pseudo-random sample, or increase `n_prompts` toward the full 1,000.

#### Licence

The dataset is released under the
[Apache 2.0 licence](https://www.apache.org/licenses/LICENSE-2.0), consistent
with the upstream `_license: apache-2.0` metadata.

#### Why this is the default

Measurements run with the AIEnergyScore text-generation dataset are directly
comparable to other measurements run against the same dataset - including
measurements on the AIEnergyScore leaderboard and in published research that
uses this benchmark. Using a shared reference corpus is what enables
cross-study comparability. See
[Methodology: Dataset choice and measurement validity](/methodology/dataset-context)
for a fuller discussion of this comparability property and its limits.

To use it, either omit `task.dataset` entirely or set `source: aienergyscore`
explicitly:

```yaml
task:
  dataset:
    source: aienergyscore
    n_prompts: 100
```

---

## Custom datasets

### File format

Custom datasets must be newline-delimited JSON (JSONL): one JSON object per line,
UTF-8 encoded, with no trailing comma between lines.

Required per-row keys: at least one of `prompt`, `text`, `instruction`, `input`,
or `question`. The loader tries these in order and uses the first non-empty string
it finds. If no recognised key is present, the row is skipped silently (and the
loader raises `ValueError` if the resulting prompt count falls below `n_prompts`).

Optional per-row keys:

| Key | Purpose |
|-----|---------|
| `source` | String label for the prompt's origin. Used by `order: grouped`. Not required. |
| `expected_output` | Reference output for accuracy studies. Not currently consumed by the measurement harness but preserved in JSONL and usable in post-processing. |

Lines whose keys all start with `_` are treated as provenance headers and skipped.
This is the same convention used by the built-in `aienergyscore.jsonl` header line.

Minimum valid custom JSONL:

```jsonl
{"prompt": "Summarise the following paragraph in one sentence."}
{"prompt": "What is the capital of France?"}
{"prompt": "Translate to Spanish: The model performed well."}
```

With optional fields:

```jsonl
{"prompt": "Describe climate feedback loops.", "source": "science", "expected_output": "..."}
{"prompt": "What is compound interest?", "source": "finance"}
```

### Pointing at a custom file

Set `source` to the file path. Relative paths resolve from the working directory:

```yaml
task:
  dataset:
    source: ./prompts.jsonl
    n_prompts: 50
    order: shuffled
```

Absolute paths are also accepted:

```yaml
task:
  dataset:
    source: /data/my-study/prompts-v2.jsonl
    n_prompts: 200
```

The file must end in `.jsonl` and exist at the path given. Any other extension
or a non-existent path raises `ValueError`.

---

## Worked examples

### Minimal experiment with defaults

```yaml
model: gpt2
engine: transformers
```

`task.dataset` defaults to `source: aienergyscore, n_prompts: 100, order: interleaved`.

### Override prompt count only

```yaml
model: gpt2
engine: transformers
task:
  dataset:
    n_prompts: 25
```

Uses `aienergyscore` with 25 prompts in file order.

### Custom file with shuffled order

```yaml
model: meta-llama/Llama-3.2-1B
engine: vllm
task:
  random_seed: 42
  dataset:
    source: ./my-prompts.jsonl
    n_prompts: 100
    order: shuffled
```

The same seed and file produce the same 100-prompt sequence across runs, which
is important for reproducibility.

### Sweep across prompt counts

```yaml
model: gpt2
engine: transformers
task.dataset.n_prompts: [10, 50, 100, 200]
```

Produces four experiment cells, each with a different prompt count. Useful for
measuring how throughput and energy scale with workload size.

---

## Validation rules enforced at load time

1. `source` must be non-empty.
2. If `source` is not a built-in alias, the path must end in `.jsonl` and exist.
3. `n_prompts` must be `>= 1`.
4. The loaded file must contain at least `n_prompts` rows with a recognisable
   prompt field.
5. `order` must be one of `interleaved`, `grouped`, `shuffled`.

All five checks raise `ValueError` with a descriptive message if violated.

---

## See also

- [Reference: study config](/reference/study-config) - full YAML schema
- [How to: interpret results](/how-to/interpret-results) - what `n_prompts` affects in output
- [Methodology: Dataset choice and measurement validity](/methodology/dataset-context) - why dataset choice shapes energy numbers and how to use custom datasets in research workflows
