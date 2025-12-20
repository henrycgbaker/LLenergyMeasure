"""Prompt processing utilities for LLM Bench."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import torch
from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


T = TypeVar("T", bound="SelectableDataset")


@runtime_checkable
class SelectableDataset(Protocol):
    """Protocol for datasets with a select method (e.g., HuggingFace datasets)."""

    def select(self: T, indices: range) -> T: ...

    def __len__(self) -> int: ...


def filter_n_prompts(
    prompts: list[str] | SelectableDataset,
    num_prompts: int,
) -> list[str] | SelectableDataset:
    """Limit the number of prompts to process.

    Args:
        prompts: List of prompts or a HuggingFace-style dataset with select().
        num_prompts: Maximum number of prompts to return.

    Returns:
        Reduced set of prompts.
    """
    if isinstance(prompts, SelectableDataset):
        total = len(prompts)
        num_prompts = min(num_prompts, total)
        return prompts.select(range(num_prompts))

    return prompts[:num_prompts]


def sort_prompts_by_length(prompts: list[str]) -> list[str]:
    """Sort prompts by character length for efficient batching.

    Sorting by length helps with padding efficiency - prompts of similar
    length are batched together, reducing wasted padding tokens.

    Args:
        prompts: List of prompt strings.

    Returns:
        Sorted list of prompts.
    """
    return sorted(prompts, key=len)


def tokenize_batch(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts in batches with truncation and padding.

    Args:
        prompts: List of prompt strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length (will be capped to model max).
        batch_size: Processing batch size for tokenization.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors.
    """
    # Cap to model's maximum length
    max_length = min(max_length, tokenizer.model_max_length)

    all_input_ids = []
    all_attention_masks = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        all_input_ids.append(encoded["input_ids"])
        if "attention_mask" in encoded:
            all_attention_masks.append(encoded["attention_mask"])

    result = {"input_ids": torch.cat(all_input_ids, dim=0)}
    if all_attention_masks:
        result["attention_mask"] = torch.cat(all_attention_masks, dim=0)

    logger.debug(f"Tokenized {len(prompts)} prompts, shape={result['input_ids'].shape}")

    return result


def create_adaptive_batches(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_tokens_per_batch: int,
    max_prompt_tokens: int,
    max_batch_size: int | None = None,
) -> list[list[str]]:
    """Group prompts into batches based on token budget.

    Creates batches where the total estimated token count is below
    max_tokens_per_batch. More efficient than fixed batching when
    prompt lengths vary significantly.

    Args:
        prompts: List of prompt strings.
        tokenizer: HuggingFace tokenizer for token counting.
        max_tokens_per_batch: Maximum total tokens allowed per batch.
        max_prompt_tokens: Cap for individual prompt token count estimation.
        max_batch_size: Optional maximum prompts per batch.

    Returns:
        List of batches, each batch is a list of prompts.
    """
    batches = []
    current_batch: list[str] = []
    current_tokens = 0

    for prompt in prompts:
        # Full tokenization for accurate estimation
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
        raw_token_count = len(encoded["input_ids"])
        # Cap for grouping purposes
        token_count = min(raw_token_count, max_prompt_tokens)

        # Check if max batch size reached
        if max_batch_size is not None and len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        # Check if adding this prompt exceeds token budget
        if current_batch and (current_tokens + token_count > max_tokens_per_batch):
            batches.append(current_batch)
            current_batch = [prompt]
            current_tokens = token_count
        else:
            current_batch.append(prompt)
            current_tokens += token_count

    if current_batch:
        batches.append(current_batch)

    logger.info(f"Created {len(batches)} adaptive batches from {len(prompts)} prompts")

    return batches


def create_fixed_batches(
    prompts: list[str],
    batch_size: int,
) -> list[list[str]]:
    """Split prompts into fixed-size batches.

    Args:
        prompts: List of prompt strings.
        batch_size: Number of prompts per batch.

    Returns:
        List of batches.
    """
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    logger.info(f"Created {len(batches)} fixed batches of size {batch_size}")
    return batches
