import os
import time
import torch
from _5_inference_helper_fns import adaptive_batching, calculate_inference_metrics
from _3_prompt_processing import batch_tokenise_truncate  
from typing import List, Any
import logging

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
    
    # For fixed batching.
    fixed_batch_size = experiment_config.batching_options.get("batch_size___fixed_batching", 8)
    # Adaptive batching flag.
    use_adaptive = experiment_config.batching_options.get("adaptive_batching", False)
    
    # Initialize metrics and outputs.
    token_id_outputs = []
    latencies = []
    total_generated_tokens = 0
    total_input_tokens = 0  
    all_input_ids_batches = []
    device = accelerator.device

    if use_adaptive:
        # Retrieve adaptive parameters from config.
        adaptive_max_tokens = experiment_config.batching_options.get("adaptive_max_tokens", max_input_tokens)
        max_batch_size_adaptive = experiment_config.batching_options.get("max_batch_size___adaptive_batching", 100)
        # Use the experiment_config.max_input_tokens as the grouping token limit,
        # or adjust this value if you want grouping to be based on a different cap.
        batches = adaptive_batching(
            prompts=prompts, 
            tokenizer=tokenizer, 
            adaptive_max_tokens=adaptive_max_tokens, 
            grouping_token_limit=max_input_tokens,  
            max_batch_size=max_batch_size_adaptive
        )
        accelerator.print(f"Using adaptive batching: created {len(batches)} batches.")
    else:
        # Fixed batching: simply split the prompts into fixed-size chunks.
        batches = [prompts[i:i+fixed_batch_size] for i in range(0, len(prompts), fixed_batch_size)]
        accelerator.print(f"Using fixed batching (non-adaptive): created {len(batches)} batches.")
    
    # Process each batch: tokenize and run inference.
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