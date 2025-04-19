base_config = {
    "config_name": None,     # To be updated by experiments.
    "suite": None,                # Will be updated when controlled variations apply.
    "controlled_variation": {},   # Holds details about the applied variation; default empty.
    "scenario_info": {},              # specific to scenario configs
    "cycle_id": None,            # to be injected later
        
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    "is_encoder_decoder": False,
    "task_type": "text_generation",
    "inference_type": "pure_generative",
    "gpu_list": [0, 1, 2, 3],
    "backend": "pytorch",
    "save_outputs": True,
    
    # Default values that will be overridden by the grid
    "max_input_tokens": 128,        # 2048 is Llama's limit.
    "max_output_tokens": 128,
    "min_output_tokens": 128,
    "num_input_prompts": 128,         # e.g., 500 * 200 = 100,000 output tokens.
    "decode_token_to_text": True,
    "num_processes": 4,
    "batching_options": {
        "batch_size___fixed_batching": 16,
        "adaptive_batching": False,
        "adaptive_max_tokens": 0,
        "max_batch_size___adaptive_batching": 0,
    },
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": False,
            "cpu_offload": False
        },
        "sharding_strategy": "NO_SHARD"  # Use "NO_SHARD"; FULL_SHARD doesn't work.
    },
    "query_rate": 1.0,
    "latency_simulation": {
        "simulate": False,        # Whether to introduce artificial delays.
        "delay_min": 0,           # Minimum delay in seconds (e.g., 0.05 for 50ms).
        "delay_max": 0,           # Maximum delay in seconds (e.g., 0.3 for 300ms).
        "simulate_burst": False,  # Whether to simulate bursty traffic conditions.
        "burst_interval": 0.0,    # Delay (seconds) after a burst.
        "burst_size": 0           # Number defining a burst.
    },
    "decoder_config": {
        "decoding_mode": None,         # Default is None; updated when decoder variations apply (e.g., "greedy", "top_k", "top_p").
        "decoder_temperature": 1.0,      # Default temperature.
        "decoder_top_k": None,           # Set to None if top_k sampling is not applicable.
        "decoder_top_p": None            # Set to None if top_p sampling is not applicable.
    },
    "fp_precision": "float32",           # Can be updated to float16 in experiments.
    "quantization_config": {
        "quantization": None,          # Default None, updated when experiments specify quantisation settings.
        "load_in_8bit": None,          # None if not applicable.
        "load_in_4bit": None,          # None if not applicable.
        "cached_flops_for_quantised_models": 16949970993152  # FLOPs value for quantized models.
    }
}
