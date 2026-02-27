# P-11: Rich Dataset / Prompt Config Subsystem

**Module**: `src/llenergymeasure/config/models.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0
**Planning Gap**: `designs/experiment-config.md` shows `dataset: str | DatasetConfig` but does not detail the discriminated union or auto-column detection logic.

---

## What Exists in the Code

**Primary file**: `src/llenergymeasure/config/models.py`
**Key classes/constants**:
- `AUTO_DETECT_COLUMNS = ["text", "prompt", "question", "instruction", "input", "content"]` (line 77) — tried in order when no column specified
- `BUILTIN_DATASETS` dict (line 34) — aliases: `ai-energy-score` (default), `alpaca`, `sharegpt`, `gsm8k`, `mmlu`
- `FilePromptSource` (line 295) — load prompts from a text file (one per line)
- `HuggingFacePromptSource` (line 302) — load from HuggingFace:
  - `dataset: str` — HF dataset name or built-in alias
  - `split: str = "test"`, `column: str | None = None`, `sample_size: int | None = None`
  - `shuffle: bool = False`, `seed: int = 42`, `subset: str | None = None`
  - `resolve_builtin_alias()` validator (line 321) — translates alias → full HF path
  - Auto-column detection triggers when `column=None`
- `DatasetConfig` (line 349) — simple form: `{name, sample_size, split, column}`
- Discriminated union: `PromptSourceConfig = FilePromptSource | HuggingFacePromptSource`

## Why It Matters

Auto-column detection enables zero-config usage of most HuggingFace datasets without knowing their schema. Built-in aliases let users write `dataset: ai-energy-score` (the standard benchmark) without a HF path. Both are essential for the "zero-config" UX story. Without them, every user must look up HF dataset column names manually.

## Planning Gap Details

`designs/dataset.md` describes the dataset design but does not document:
- The auto-column detection logic or the ordered list of column names tried
- The `FilePromptSource` union type (only HuggingFace sources are implied)
- The built-in alias registry (`BUILTIN_DATASETS` dict)
- The `shuffle`, `seed`, `subset` controls on `HuggingFacePromptSource`

A Phase 5 implementor working only from `designs/dataset.md` will build a simpler dataset config and miss the discriminated union and auto-detection.

## Recommendation for Phase 5

Add to `designs/dataset.md` under "Prompt Source Configuration":

> **Discriminated Union**: `PromptSourceConfig = FilePromptSource | HuggingFacePromptSource`
>
> **FilePromptSource**: `{file: Path}` — reads lines from a text file.
>
> **HuggingFacePromptSource**: `{dataset, split, column?, sample_size?, shuffle?, seed?, subset?}`
> Auto-column detection (when `column` is omitted): tries `["text", "prompt", "question",
> "instruction", "input", "content"]` in order. First matching column is used.
>
> **Built-in aliases**: `ai-energy-score`, `alpaca`, `sharegpt`, `gsm8k`, `mmlu`.
> Implementation: `config/models.py`, `BUILTIN_DATASETS`, `HuggingFacePromptSource.resolve_builtin_alias()`.
