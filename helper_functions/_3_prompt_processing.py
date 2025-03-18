from typing import List, Union, Any, Dict
import torch


def filter_n_prompts(prompts: Union[List[str], Any], num_input_prompts: int) -> Union[List[str], Any]:
    """
    Shortens the number of prompts based on num_input_prompts.
    
    Parameters:
        prompts: A list of prompts or an object that supports the .select() method.
        num_input_prompts: The maximum number of prompts to return.
        
    Returns:
        The reduced number of prompts.
    """
    # If prompts has a select method (e.g., Hugging Face Dataset), use it.
    if hasattr(prompts, "select"):
        total = len(prompts)
        num_input_prompts = min(num_input_prompts, total)
        return prompts.select(range(num_input_prompts))
    # Otherwise, assume prompts is a list.
    return prompts[:num_input_prompts]


def sort_prompts(prompts: List[str]) -> List[str]:
    """
    Sorts prompt strings by their character length for efficient batching.
    """
    return sorted(prompts, key=lambda p: len(p))


def batch_tokenise_truncate(prompts: List[str], tokenizer: Any, max_input_tokens: int, batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """
    Tokenises and truncates prompts in batches.
    
    Parameters:
        prompts: List of prompt strings.
        tokenizer: The tokenizer instance.
        max_input_tokens: Maximum allowed tokens per prompt.
        batch_size: Number of prompts to process in a single batch.
    
    Returns:
        A dictionary containing concatenated tokenised outputs (e.g., "input_ids", and optionally "attention_mask").
    """
    all_input_ids = []
    all_attention_mask = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        # Tokenise the batch with padding & truncation
        encoded = tokenizer(batch, truncation=True, max_length=max_input_tokens, padding=True, return_tensors="pt")
        all_input_ids.append(encoded["input_ids"])
        if "attention_mask" in encoded:
            all_attention_mask.append(encoded["attention_mask"])
    
    # Concatenate tensors along the batch dimension
    tokenised_inputs = {"input_ids": torch.cat(all_input_ids, dim=0)}
    if all_attention_mask:
        tokenised_inputs["attention_mask"] = torch.cat(all_attention_mask, dim=0)
    
    return tokenised_inputs

