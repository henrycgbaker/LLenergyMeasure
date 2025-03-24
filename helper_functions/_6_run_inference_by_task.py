import time
import torch
from _5_inference_helper_fns import adaptive_batching, calculate_inference_metrics
from _3_prompt_processing import batch_tokenise_truncate  
from typing import List, Any

def run_gen_inference(model, experiment_config, prompts, tokenizer, accelerator):
    """
    Runs text generation inference and measures metrics.
    
    Returns:
        a tuple:
        text_outputs (if save_raw_text_outputs is True, else None),
        concatenated_input_ids, inference_results.
    """
    max_input_tokens = experiment_config.max_input_tokens 
    max_output_tokens = experiment_config.max_output_tokens
    decoder_temperature = experiment_config.decoder_temperature
    fixed_max_batch_size = experiment_config.batching_options.get("fixed_max_batch_size", 8)
    use_adaptive = experiment_config.batching_options.get("adaptive_batching", False)
        
    # Initialize metrics and outputs
    token_id_outputs = []
    latencies = []
    total_generated_tokens = 0
    total_input_tokens = 0  
    all_input_ids_batches = []
    device = accelerator.device

    # Decide batching strategy externally in run_torch method workflow
    if use_adaptive:
        # Retrieve adaptive_max_tokens from the config, defaulting to max_input_tokens if not set.
        adaptive_max_tokens = experiment_config.batching_options.get("adaptive_max_tokens", max_input_tokens)
        # Pass max_input_tokens as the cap for each prompt's token count.
        batches = adaptive_batching(prompts, tokenizer, adaptive_max_tokens, max_prompt_tokens=max_input_tokens, max_batch_size=fixed_batch_size)
        accelerator.print(f"Using adaptive batching: created {len(batches)} batches.")
    else:
        # Fixed batching: partition the prompts into fixed-size chunks.
        batches = [prompts[i:i+fixed_max_batch_size] for i in range(0, len(prompts), fixed_max_batch_size)]
        accelerator.print(f"Using fixed batching (non-adaptive): created {len(batches)} batches.")
    
    # Process each batch: tokenise and run inference immediately.
    for batch in batches:
        tokenised_batch = batch_tokenise_truncate(
            prompts=batch,
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
            batch_size=len(batch)
        )
        batch_input_ids = tokenised_batch["input_ids"]
        total_input_tokens += batch_input_ids.numel()
        all_input_ids_batches.append(batch_input_ids)
        
        # Prepare inputs for the model
        if "attention_mask" in tokenised_batch:
            batch_encoded = {
                "input_ids": batch_input_ids.to(device),
                "attention_mask": tokenised_batch["attention_mask"].to(device)
            }
        else:
            batch_encoded = {"input_ids": batch_input_ids.to(device)}
        
        # Build generation kwargs based on decoder_temperature
        if decoder_temperature is not None and decoder_temperature > 0:
            generation_kwargs = {"max_new_tokens": max_output_tokens, "do_sample": True, "temperature": decoder_temperature}
        else:
            generation_kwargs = {"max_new_tokens": max_output_tokens, "do_sample": False}
        
        # Run timed inference on the batch
        start_time = time.perf_counter()
        with torch.no_grad():
            token_id_batch_output = model.generate(batch_encoded["input_ids"], **generation_kwargs)
        torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)  # in milliseconds
        
        # Count generated tokens for each prompt in the batch.
        for j in range(batch_input_ids.size(0)):
            prompt_len = batch_input_ids[j].shape[0]
            gen_len = token_id_batch_output[j].shape[0] - prompt_len
            total_generated_tokens += gen_len
        
        # Conditionally store the outputs.
        if experiment_config.save_outputs:
            token_id_outputs.append(token_id_batch_output)
    
    # Concatenate all input_ids batches
    concatenated_input_ids = torch.cat(all_input_ids_batches, dim=0)
    
    # Compute overall inference metrics.
    inference_results = calculate_inference_metrics(
        num_input_prompts=len(prompts),
        latencies=latencies,
        total_input_tokens=total_input_tokens,
        total_generated_tokens=total_generated_tokens
    )
    
    # If not saving raw text outputs, set text_outputs to None.
    if not experiment_config.save_outputs:
        token_id_outputs = None
        
    return token_id_outputs, concatenated_input_ids, inference_results
