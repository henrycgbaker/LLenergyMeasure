# Artificially Optimised Scenarios

scenario_a_max_throughput_exploit = {
    "config_name": "max_throughput_exploit",
    
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "is_encoder_decoder": False,
    "task_type": "text_generation",
    "inference_type": "pure_generative",
    "gpu_list": [0, 1, 2, 3],

    "max_input_tokens": 100,
    "max_output_tokens": 100,
    "num_input_prompts": 100,
    "save_outputs": True,
    "decode_token_to_text": True,
    "num_processes": 4,
    "batching_options": {
        "batch_size___fixed_batching": 256, # this 
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": False,
            "cpu_offload": False
        },
        "sharding_strategy": "NO_SHARD"
    },
    "query_rate": 1.0,
    "latency_simulation": {
        "simulate": False,
        "delay_min": 0,
        "delay_max": 0,
        "simulate_burst": False,
        "burst_interval": 0,
        "burst_size": 0
    },
    "decoder_temperature": 1.0,
    "fp_precision": "float16",
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": True,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": 10345441280000
    },
    "backend": "pytorch"
}

scenario_b_precision_gaming = {
    **scenario_a_max_throughput_exploit,
    "config_name": "precision_gaming",
    "batching_options": {
        "batch_size___fixed_batching": 128,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": False,
        "load_in_4bit": True,
        "cached_flops_for_quantised_models": 10345441280000
    }
}

scenario_c_gpu_overdrive = {
    **scenario_a_max_throughput_exploit,
    "config_name": "gpu_overdrive",
    "batching_options": {
        "batch_size___fixed_batching": 128,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "quantization_config": {
        "quantization": False,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": None
    },
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": False,
            "cpu_offload": False
        },
        "sharding_strategy": "NO_SHARD"
    }
}

# Realistic Deployment Scenarios

scenario_d_standard_production = {
    "config_name": "standard_production",
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "is_encoder_decoder": False,
    "task_type": "text_generation",
    "inference_type": "pure_generative",
    "max_input_tokens": 100,
    "max_output_tokens": 100,
    "num_input_prompts": 100,
    "save_outputs": True,
    "decode_token_to_text": True,
    "gpu_list": [0, 1, 2, 3],
    "num_processes": 4,
    "batching_options": {
        "batch_size___fixed_batching": 16,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": False,
            "cpu_offload": False
        },
        "sharding_strategy": "NO_SHARD"
    },
    "query_rate": 1.0,
    "latency_simulation": {
        "simulate": True,
        "delay_min": 1,
        "delay_max": 2,
        "simulate_burst": True,
        "burst_interval": 4.0,
        "burst_size": 5
    },
    "decoder_temperature": 1.0,
    "fp_precision": "float32",
    "quantization_config": {
        "quantization": False,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": None
    },
    "backend": "pytorch"
}

scenario_e_low_latency_real_time = {
    **scenario_d_standard_production,
    "config_name": "latency_real_time",
    "batching_options": {
        "batch_size___fixed_batching": 4,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.01,
        "delay_max": 0.05,
        "simulate_burst": False
    }
}

scenario_f_balanced_performance_mode = {
    **scenario_d_standard_production,
    "config_name": "balanced_performance_mode",
    "batching_options": {
        "batch_size___fixed_batching": 32,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "fp_precision": "float16",
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": True,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": 10345441280000
    },
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.02,
        "delay_max": 0.1,
        "simulate_burst": True,
        "burst_interval": 1.5,
        "burst_size": 3
    }
}