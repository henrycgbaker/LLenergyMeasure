# Dataset Handling Design

**Last updated**: 2026-02-19
**Source decisions**: [../decisions/dataset-handling.md](../decisions/dataset-handling.md)
**Status**: Confirmed

---

## Core Framing

The dataset in LLenergyMeasure is a **workload definition**, not an evaluation task.
Its role: produce a representative distribution of prompt lengths and output lengths for energy
measurement. Scientific validity is in the statistical distribution, not the content.

This differs fundamentally from lm-eval, where the dataset IS the thing being evaluated.

---

## Three Dataset Modes

### Mode 1 — Named Built-in

```yaml
dataset: aienergyscore   # default — ships with package, pinned
dataset: synthetic       # → triggers Mode 3
```

Built-ins are JSONL files in `src/llenergymeasure/datasets/builtin/`. Loaded deterministically.

**Built-ins at v2.0:**

| Name | Source | Size | Description |
|---|---|---|---|
| `aienergyscore` | AIEnergyScore/text_generation (HF Hub, pinned commit) | 1,000 prompts | WikiText + OSCAR + UltraChat equally sampled. Default for zero-config experiments. |

**Not shipped speculatively:**
- `sharegpt` — deferred until user demand confirms it (longer multi-turn conversations)
- HuggingFace Datasets integration — deferred to later v2.x release

### Mode 2 — JSONL File (Bring-Your-Own)

```yaml
dataset: path/to/my_prompts.jsonl   # relative to CWD, or absolute
```

**JSONL format** (one JSON object per line):
```json
{"prompt": "Explain the difference between...", "max_new_tokens": 100}
{"prompt": "Write a summary of...", "max_new_tokens": 200}
```

- `prompt`: required
- `max_new_tokens`: optional — overrides experiment-level `max_new_tokens` for this prompt

### Mode 3 — Synthetic Length-Controlled

```yaml
dataset:
  synthetic:
    n: 100
    input_len: 512    # approximate tokens (padded/truncated to target)
    output_len: 128   # sets max_new_tokens
    seed: 42          # inherits experiment random_seed if not set
```

Use when: holding prompt length constant while sweeping batch size, precision, or backend.
Sequences are generated from the random seed — same seed = identical prompts every run.

**Why synthetic is useful**: Real prompts have variable lengths, introducing throughput noise
when sweeping batch size. Synthetic prompts with fixed input/output lengths isolate the variable
under study.

---

## Config Integration

```python
# src/llenergymeasure/config/models.py

class SyntheticDatasetConfig(BaseModel):
    model_config = {"extra": "forbid"}

    n: int
    input_len: int = 512
    output_len: int = 128
    seed: int = 42


class ExperimentConfig(BaseModel):
    dataset: str | SyntheticDatasetConfig = "aienergyscore"
    n: int = 100   # number of prompts to use from dataset
    # n > len(dataset) → ValidationError at config parse time (not runtime)
    ...
```

**`n` vs `dataset.n` for synthetic**:
`n` at the experiment level selects how many prompts to use from the dataset.
`SyntheticDatasetConfig.n` specifies how many synthetic prompts to generate.
When `dataset.synthetic.n` is set, it takes precedence over experiment-level `n`.

<!-- TODO: Resolve the n vs dataset.n semantics precisely. Current design has a potential
     ambiguity: if dataset is synthetic with n=200 but experiment has n=100, which wins?
     Decision needed before implementation. Recommendation: dataset.synthetic.n sets
     the pool; experiment-level n selects how many to use from that pool (consistent
     with non-synthetic behaviour). -->

---

## Loader Implementation

```python
# src/llenergymeasure/datasets/loader.py

from pathlib import Path
from typing import Iterator
from llenergymeasure.config.models import ExperimentConfig, SyntheticDatasetConfig


BUILTIN_DIR = Path(__file__).parent / "builtin"

BUILTIN_DATASETS = {
    "aienergyscore": BUILTIN_DIR / "aienergyscore.jsonl",
}


def load_prompts(config: ExperimentConfig) -> list[dict]:
    """Returns list of {prompt, max_new_tokens} dicts, length == config.n."""
    dataset = config.dataset
    n = config.n

    if isinstance(dataset, SyntheticDatasetConfig):
        return _load_synthetic(dataset, n)

    if dataset in BUILTIN_DATASETS:
        return _load_jsonl(BUILTIN_DATASETS[dataset], n, name=dataset)

    path = Path(dataset)
    if path.exists() and path.suffix == ".jsonl":
        return _load_jsonl(path, n, name=str(path))

    raise ValueError(
        f"Unknown dataset: {dataset!r}. "
        f"Valid built-ins: {list(BUILTIN_DATASETS)}. "
        "For custom datasets, provide a path to a .jsonl file."
    )


def _load_jsonl(path: Path, n: int, name: str) -> list[dict]:
    """Loads first n prompts from JSONL file. Deterministic — same order every run."""
    prompts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            prompts.append(json.loads(line))

    if len(prompts) < n:
        raise ValueError(
            f"Dataset {name!r} has {len(prompts)} prompts but n={n} was requested. "
            f"Reduce n to {len(prompts)} or use a larger dataset."
        )
    return prompts


def _load_synthetic(config: SyntheticDatasetConfig, n: int) -> list[dict]:
    """Generates deterministic synthetic prompts from seed."""
    import random
    rng = random.Random(config.seed)

    prompts = []
    for _ in range(n):
        # Generate random token IDs (approximate — uses padding chars as proxy)
        # Real implementation: use tokenizer vocab to generate valid token sequences
        length = config.input_len
        text = " ".join(str(rng.randint(0, 50000)) for _ in range(length))
        prompts.append({
            "prompt": text,
            "max_new_tokens": config.output_len,
        })
    return prompts

    # TODO: Synthetic prompt generation should use the model's actual tokenizer to
    # produce token sequences of the exact requested length. The naive approach above
    # produces approximate lengths. For precise length control (the primary use case
    # for synthetic datasets), this needs a tokenizer-aware implementation.
    # Dependency: model tokenizer must be available at config parse time (or at load time).
```

---

## Reproducibility Guarantees

| Mode | Guarantee |
|---|---|
| Built-in | Same `n: 100` always selects the first 100 prompts from the pinned file. Shuffling is opt-in (not in v2.0). |
| BYOF JSONL | Loaded in file order. User is responsible for ordering. |
| Synthetic | Same `seed` = identical sequence. Fully deterministic. |

**Built-in dataset pinning**: the `aienergyscore.jsonl` file ships inside the Python package at
a fixed path. It does not download from HuggingFace Hub at runtime. The file was generated from
a specific commit of `AIEnergyScore/text_generation` — that commit hash should be documented in
the file header for traceability.

<!-- TODO: The aienergyscore.jsonl file needs to be created by downloading from HuggingFace
     Hub (AIEnergyScore/text_generation, pinned commit), processing into the correct JSONL
     format, and committing to the repo. Need to:
     1. Identify the correct commit hash to pin
     2. Define the exact sampling strategy (equal from WikiText, OSCAR, UltraChat → 1000 total)
     3. Create the file and document provenance in a header comment
     This is a Phase 5 task but should be planned now. -->

---

## Peer Reference

| Tool | Dataset role | How datasets work |
|---|---|---|
| lm-eval | Evaluation task | HF Datasets integration; tasks define prompt format |
| AIEnergyScore | Workload (closest peer) | Ships fixed test_prompts.json; same approach as ours |
| MLPerf inference | Workload | Ships calibration datasets; pinned versions |
| Zeus | None | Energy tool — no dataset concept |

LLenergyMeasure follows AIEnergyScore for dataset handling: ship pinned dataset files,
deterministic loading, no runtime downloads.

---

## Related

- [../decisions/dataset-handling.md](../decisions/dataset-handling.md): Decision rationale
- [experiment-config.md](experiment-config.md): `dataset:` and `n:` fields
- [reproducibility.md](reproducibility.md): How datasets contribute to reproducibility guarantees
