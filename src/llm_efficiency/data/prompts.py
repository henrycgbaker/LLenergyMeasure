"""
Prompt loading, filtering, and processing utilities.
"""

import logging
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_prompts_from_dataset(
    dataset_name: str = "AIEnergyScore/text_generation",
    split: str = "train",
    num_prompts: Optional[int] = None,
) -> List[str]:
    """
    Load prompts from Hugging Face dataset.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        split: Dataset split to load ("train", "test", "validation")
        num_prompts: Maximum number of prompts to load (None for all)

    Returns:
        List of prompt strings
    """
    logger.info(f"Loading dataset: {dataset_name} (split={split})")

    try:
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Extract text field (adjust field name as needed)
    if isinstance(dataset, Dataset):
        if "text" in dataset.column_names:
            prompts = dataset["text"]
        elif "prompt" in dataset.column_names:
            prompts = dataset["prompt"]
        else:
            # Try first text-like column
            text_columns = [col for col in dataset.column_names if "text" in col.lower()]
            if text_columns:
                prompts = dataset[text_columns[0]]
            else:
                raise ValueError(f"No text field found in dataset. Columns: {dataset.column_names}")
    else:
        prompts = list(dataset)

    # Limit number of prompts
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
        logger.info(f"Limited to {num_prompts} prompts")

    return list(prompts)


def filter_prompts_by_length(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    min_tokens: int = 1,
) -> List[str]:
    """
    Filter prompts by token length.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer for length calculation
        max_tokens: Maximum token length
        min_tokens: Minimum token length

    Returns:
        Filtered list of prompts
    """
    logger.info(f"Filtering prompts by length ({min_tokens}-{max_tokens} tokens)")
    logger.debug(f"Input prompts: {len(prompts)}")

    filtered = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        length = len(tokens)

        if min_tokens <= length <= max_tokens:
            filtered.append(prompt)
        else:
            logger.debug(f"Filtered out prompt with {length} tokens")

    logger.info(f"Filtered prompts: {len(filtered)} (removed {len(prompts) - len(filtered)})")

    if len(filtered) == 0:
        logger.warning("No prompts passed the length filter!")

    return filtered


def sort_prompts_by_length(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    reverse: bool = False,
) -> List[str]:
    """
    Sort prompts by token length.

    Sorting by length can improve batching efficiency by grouping
    similar-length prompts together, reducing padding.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer for length calculation
        reverse: If True, sort descending (longest first)

    Returns:
        Sorted list of prompts
    """
    logger.info(f"Sorting prompts by length (reverse={reverse})")

    # Calculate lengths
    prompt_lengths = [
        (prompt, len(tokenizer.encode(prompt, add_special_tokens=True)))
        for prompt in prompts
    ]

    # Sort by length
    sorted_prompts = sorted(prompt_lengths, key=lambda x: x[1], reverse=reverse)

    logger.debug(f"Length range: {sorted_prompts[0][1]} - {sorted_prompts[-1][1]} tokens")

    return [prompt for prompt, _ in sorted_prompts]


def tokenize_prompts(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    padding: str = "longest",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Batch tokenize prompts with padding.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length (None for model max)
        padding: Padding strategy ("longest", "max_length")
        truncation: Whether to truncate sequences
        return_tensors: Format for returned tensors ("pt", "np")

    Returns:
        Dictionary with input_ids and attention_mask tensors
    """
    logger.info(f"Tokenizing {len(prompts)} prompts")

    tokenized = tokenizer(
        prompts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
        return_attention_mask=True,
    )

    logger.debug(f"Tokenized shape: {tokenized['input_ids'].shape}")
    logger.debug(f"Token range: {tokenized['input_ids'].min()} - {tokenized['input_ids'].max()}")

    return tokenized


def create_prompt_batches(
    prompts: List[str],
    batch_size: int,
    sort_by_length: bool = True,
) -> List[List[str]]:
    """
    Create batches of prompts.

    Args:
        prompts: List of prompts
        batch_size: Number of prompts per batch
        sort_by_length: Whether to sort by length first (reduces padding)

    Returns:
        List of prompt batches
    """
    logger.info(f"Creating batches (size={batch_size}, sort={sort_by_length})")

    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batches.append(batch)

    logger.info(f"Created {len(batches)} batches")
    logger.debug(f"Last batch size: {len(batches[-1])}")

    return batches


def split_prompts_across_processes(
    prompts: List[str],
    process_index: int,
    num_processes: int,
) -> List[str]:
    """
    Split prompts across distributed processes.

    Args:
        prompts: List of all prompts
        process_index: Current process index (0-based)
        num_processes: Total number of processes

    Returns:
        Subset of prompts for this process
    """
    logger.info(
        f"Splitting {len(prompts)} prompts across {num_processes} processes "
        f"(current: {process_index})"
    )

    # Simple round-robin distribution
    process_prompts = prompts[process_index::num_processes]

    logger.info(f"Process {process_index} assigned {len(process_prompts)} prompts")

    return process_prompts


# Typing imports
from typing import Optional
