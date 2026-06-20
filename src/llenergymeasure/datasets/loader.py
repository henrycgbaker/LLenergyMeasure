"""Dataset loading utilities for LLM efficiency measurement.

Provides JSONL loading for bundled datasets and file-path datasets. Used by
inference engines to load the prompt workload for each experiment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


# ---------------------------------------------------------------------------
# Built-in dataset registry
# ---------------------------------------------------------------------------

_BUILTIN_DIR: Path = Path(__file__).parent / "builtin"

BUILTIN_DATASETS: dict[str, Path] = {
    "aienergyscore": _BUILTIN_DIR / "aienergyscore.jsonl",
}

# Column names tried in order when detecting the prompt field in JSONL records.
AUTO_DETECT_COLUMNS: list[str] = ["prompt", "text", "instruction", "input", "question"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_prompts(config: ExperimentConfig) -> list[str]:
    """Load prompts according to the experiment configuration.

    Dispatches based on config.task.dataset.source:
    - Built-in alias (e.g. "aienergyscore") -> load from bundled JSONL file
    - Path to an existing .jsonl file -> load from that file
    - Anything else -> raise ValueError with valid options

    Args:
        config: Experiment configuration. Uses config.task.dataset (DatasetConfig)
            and config.task.random_seed.

    Returns:
        List of exactly config.task.dataset.n_prompts prompt strings.

    Raises:
        ValueError: If dataset source is unknown or file not found.
        ValueError: If the dataset has fewer prompts than requested.
    """
    ds = config.task.dataset
    source = ds.source

    # Built-in alias
    if source in BUILTIN_DATASETS:
        path = BUILTIN_DATASETS[source]
        return _load_jsonl(
            path,
            n=ds.n_prompts,
            name=source,
            order=ds.order,
            seed=config.task.random_seed,
        )

    # User-supplied file path
    path = Path(source)
    if path.suffix == ".jsonl" and path.exists():
        return _load_jsonl(
            path,
            n=ds.n_prompts,
            name=str(path),
            order=ds.order,
            seed=config.task.random_seed,
        )

    valid = ", ".join(f'"{k}"' for k in sorted(BUILTIN_DATASETS))
    raise ValueError(
        f"Unknown dataset source {source!r}. "
        f"Valid built-in aliases: {valid}. "
        "To use a custom dataset, provide a path to an existing .jsonl file."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_jsonl(
    path: Path,
    n: int,
    name: str,
    order: str = "interleaved",
    *,
    seed: int,
) -> list[str]:
    """Load prompts from a JSONL file.

    Skips lines that are provenance headers (keys starting with '_').
    Tries 'prompt' then 'text' as the prompt field for each record.

    Args:
        path: Path to the JSONL file.
        n: Number of prompts to return.
        name: Dataset name (for error messages).
        order: One of 'interleaved', 'grouped', 'shuffled'.
        seed: Random seed used when order='shuffled'.

    Returns:
        List of exactly n prompt strings.

    Raises:
        ValueError: If fewer than n prompts are available.
    """
    # Collect (record, prompt) pairs, keeping only records that yield a usable
    # prompt. Counting usable prompts rather than raw records means interleaved
    # mode keeps reading past prompt-less records instead of stopping early and
    # then falsely reporting that the file lacks enough prompts.
    collected: list[tuple[dict[str, Any], str]] = []

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip provenance header lines (keys start with '_')
            if any(k.startswith("_") for k in record):
                continue

            prompt: str | None = None
            for col in AUTO_DETECT_COLUMNS:
                value = record.get(col)
                if isinstance(value, str) and value.strip():
                    prompt = value.strip()
                    break
            if prompt is None:
                continue

            collected.append((record, prompt))

            if order == "interleaved" and len(collected) >= n:
                break

    if len(collected) < n:
        raise ValueError(
            f"Dataset {name!r} has only {len(collected)} usable prompts but {n} were requested."
        )

    # Apply ordering
    if order == "interleaved":
        # File order is already interleaved (written round-robin by source)
        ordered = collected
    elif order == "grouped":
        # Sort by 'source' field (stable sort preserves intra-group order)
        ordered = sorted(collected, key=lambda rp: rp[0].get("source", ""))
    elif order == "shuffled":
        ordered = list(collected)
        random.Random(seed).shuffle(ordered)
    else:
        raise ValueError(
            f"Unknown dataset.order {order!r}. Expected: interleaved, grouped, shuffled."
        )

    return [prompt for _record, prompt in ordered[:n]]
