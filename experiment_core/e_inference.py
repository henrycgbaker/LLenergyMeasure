import os
import time
import torch
import logging
from typing import List, Any
import random


def adaptive_batching(
    prompts: List[str],
    tokenizer: Any,
    adaptive_max_tokens: int,
    grouping_token_limit: int,  # Use this for estimation of each prompt’s token count.
    max_batch_size: int = None
) -> List[List[str]]:
    """
    Groups prompts into batches such that the sum of the estimated token counts (using full tokenization)
    per batch is below adaptive_max_tokens. Each prompt’s token count is capped at grouping_token_limit.
    
    Parameters:
        prompts (list[str]): List of prompt strings.
        tokenizer: Tokenizer (e.g., Hugging Face tokenizer).
        adaptive_max_tokens (int): Maximum total tokens allowed per adaptive batch.
        grouping_token_limit (int): The cap for each prompt’s token count when grouping.
        max_batch_size (int, optional): Maximum number of prompts per batch.
        
    Returns:
        List of batches, where each batch is a list of prompt strings.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for prompt in prompts:
        # Use full tokenization (without truncation) for accurate estimation.
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
        raw_token_count = len(encoded["input_ids"])
        # Cap the token count for grouping purposes.
        token_count = min(raw_token_count, grouping_token_limit)
        
        # If max_batch_size is reached, finalize current batch.
        if max_batch_size is not None and len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        # If adding this prompt exceeds the batch token budget and the batch isn't empty, start a new batch.
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

logger = logging.getLogger(__name__)

def run_gen_inference(model, experiment_config, prompts, tokenizer, accelerator):
    """
    Runs text generation inference and measures metrics.
    
    Returns:
        a tuple:
         - token_id_outputs (if experiment_config.save_outputs is True, else None),
         - concatenated_input_ids,
         - inference_results.
    """
    max_input_tokens = experiment_config.max_input_tokens   # e.g., 2048 tokens (model input cap)
    max_output_tokens = experiment_config.max_output_tokens
    decoder_temperature = experiment_config.decoder_temperature
    
    # Initialize metrics and outputs.
    token_id_outputs = []
    latencies = []
    total_generated_tokens = 0
    total_input_tokens = 0  
    all_input_ids_batches = []
    device = accelerator.device

    # Determine batching strategy (adaptive or fixed)
    if experiment_config.batching_options.get("adaptive_batching", False):
        batches = adaptive_batching(
            prompts=prompts, 
            tokenizer=tokenizer, 
            adaptive_max_tokens=experiment_config.max_input_tokens, 
            grouping_token_limit=experiment_config.max_input_tokens,  
            max_batch_size=experiment_config.batching_options.get("max_batch_size___adaptive_batching", 100)
        )
        accelerator.print(f"Using adaptive batching: created {len(batches)} batches.")
    else:
        # Fixed batching: simply split the prompts into fixed-size chunks.
        fixed_batch_size = experiment_config.batching_options.get("batch_size___fixed_batching", 8)
        batches = [prompts[i:i+fixed_batch_size] for i in range(0, len(prompts), fixed_batch_size)]
        accelerator.print(f"Using fixed batching (non-adaptive): created {len(batches)} batches.")

    # Process each batch: tokenize and run inference.
    from experiment_core.c_prompt_processing import batch_tokenise_truncate  
    for batch_idx, batch in enumerate(batches):
        tokenised_batch = batch_tokenise_truncate(
            prompts=batch,
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
            batch_size=len(batch)
        )
        batch_input_ids = tokenised_batch["input_ids"]
        total_input_tokens += batch_input_ids.numel()
        all_input_ids_batches.append(batch_input_ids)
        
        # Prepare inputs for the model.
        if "attention_mask" in tokenised_batch:
            batch_encoded = {
                "input_ids": batch_input_ids.to(device),
                "attention_mask": tokenised_batch["attention_mask"].to(device)
            }
        else:
            batch_encoded = {"input_ids": batch_input_ids.to(device)}
                
        gpu_id = accelerator.device.index
        logger.info(f"[Process {os.getpid()}][GPU {gpu_id}] — Completed tokenisation of batch {batch_idx + 1}/{len(batches)}")

        # Compute the allowed new tokens so that the total does not exceed the model’s maximum.
        total_allowed_length = tokenizer.model_max_length  # e.g., 2048 tokens.
        current_length = batch_encoded["input_ids"].shape[1]  # should be max_input_tokens (2048) due to truncation.
        allowed_new_tokens = max(0, total_allowed_length - current_length)
        generation_kwargs = {
            "max_new_tokens": min(max_output_tokens, allowed_new_tokens),
            "do_sample": decoder_temperature is not None and decoder_temperature > 0,
            "temperature": decoder_temperature if (decoder_temperature is not None and decoder_temperature > 0) else None
        }
        
        # Run timed inference on the batch.
        start_time = time.perf_counter()
        
        # Latency simulation: delay between request arrivals
        if experiment_config.latency_simulation.get("simulate", False):
            delay = random.uniform(
                experiment_config.latency_simulation.get("delay_min", 0),
                experiment_config.latency_simulation.get("delay_max", 0)
            )
            time.sleep(delay)

        # Burst simulation: extra pause after bursts
        if experiment_config.latency_simulation.get("simulate_burst", False):
            burst_size = experiment_config.latency_simulation.get("burst_size", 1)
            if (batch_idx + 1) % burst_size == 0:
                burst_interval = experiment_config.latency_simulation.get("burst_interval", 0)
                time.sleep(burst_interval)
        
        with torch.no_grad():
            token_id_batch_output = model.generate(batch_encoded["input_ids"], **generation_kwargs)
        torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)  # in milliseconds.
        logger.info(f"[Process {os.getpid()}][GPU {gpu_id}] — Completed batch inference {batch_idx + 1}/{len(batches)}")

        # Count generated tokens per prompt.
        for j in range(batch_input_ids.size(0)):
            prompt_len = batch_input_ids[j].shape[0]
            gen_len = token_id_batch_output[j].shape[0] - prompt_len
            total_generated_tokens += gen_len
        
        # Optionally store outputs.
        if experiment_config.save_outputs:
            token_id_outputs.append(token_id_batch_output)
    
    concatenated_input_ids = torch.cat(all_input_ids_batches, dim=0)
    
    inference_results = calculate_inference_metrics(
        num_input_prompts=len(prompts),
        latencies=latencies,
        total_input_tokens=total_input_tokens,
        total_generated_tokens=total_generated_tokens
    )
    
    if not experiment_config.save_outputs:
        token_id_outputs = None
        
    return token_id_outputs, concatenated_input_ids, inference_results