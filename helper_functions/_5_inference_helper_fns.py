from typing import List, Any

def adaptive_batching(prompts: List[str], tokenizer: Any, adaptive_max_tokens: int, max_batch_size: int = None) -> List[List[str]]:

    """
    Groups prompts into batches such that the total token count per batch is below adaptive_max_tokens.
    
    Parameters:
        prompts (list[str]): List of prompt strings.
        tokenizer: Tokenizer with a .tokenize() method.
        adaptive_max_tokens (int): Maximum total tokens allowed per batch.
        max_batch_size (int, optional): Optional limit on number of prompts per batch.
        
    Returns:
        List of batches, where each batch is a list of prompt strings.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for prompt in prompts:
        token_count = len(tokenizer.tokenize(prompt))
        
        # If a max_batch_size is specified and reached, finalize the current batch.
        if max_batch_size is not None and len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        # If adding this prompt exceeds the token limit and the batch is not empty,
        # then start a new batch.
        if current_batch and (current_tokens + token_count > adaptive_max_tokens):
            batches.append(current_batch)
            current_batch = [prompt]
            current_tokens = token_count
        else:
            current_batch.append(prompt)
            current_tokens += token_count

    if current_batch:
        batches.append(current_batch)
    
    return batches

def calculate_inference_metrics(num_input_prompts, latencies, total_input_tokens, total_generated_tokens):
    total_time_sec = sum(latencies) / 1000.0 if latencies else 0.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    throughput_qps = num_input_prompts / total_time_sec if total_time_sec > 0 else 0.0
    tokens_per_sec = total_generated_tokens / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "num_input_prompts": num_input_prompts,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_generated_tokens,
        "total_time_sec": total_time_sec,
        "avg_latency_ms": avg_latency_ms,
        "throughput_qps": throughput_qps,
        "tokens_per_sec": tokens_per_sec,
    }