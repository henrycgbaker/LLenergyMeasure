from configs.a_default_config import base_config

# ARTIFICIAL

scenario_a1_max_throughput_exploit = {
    **base_config,
    "config_name": "a1_max_throughput_exploit",
    "batching_options": {
        "batch_size___fixed_batching": 256,
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
    }
}

scenario_a2_precision_minimalist = {
    **scenario_a1_max_throughput_exploit,
    "config_name": "a2_precision_minimalist",
    "gpu_list": [0, 1],
    "batching_options": {
        "batch_size___fixed_batching": 128
    },
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": False,
        "load_in_4bit": True,
        "cached_flops_for_quantised_models": 10345441280000
    }
}

scenario_a3_quantisation_gaming = {
    **scenario_a1_max_throughput_exploit,
    "config_name": "a3_quantisation_gaming",
    "gpu_list": [0],
    "batching_options": {
        "batch_size___fixed_batching": 64
    },
    "decoder_config": {
        "decoder_temperature": 0.7,
        "decoder_top_k": 50,
        "decoder_top_p": 0.95
    },
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": False,
        "load_in_4bit": True,
        "cached_flops_for_quantised_models": 10345441280000
    }
}

scenario_a4_latency_ignorance_exploit = {
    **scenario_a1_max_throughput_exploit,
    "config_name": "a4_latency_ignorance_exploit",
    "gpu_list": [0],
    "batching_options": {
        "batch_size___fixed_batching": 32
    }
}

scenario_a5_parallel_overdrive = {
    **scenario_a1_max_throughput_exploit,
    "config_name": "a5_parallel_overdrive",
    "gpu_list": [0, 1, 2, 3],
    "batching_options": {
        "batch_size___fixed_batching": 64
    },
    "quantization_config": {
        "quantization": False,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": None
    }
}

# REALISTIC

scenario_r1_standard_production = {
    **base_config,
    "config_name": "r1_standard_production",
    "gpu_list": [0, 1],
    "num_processes": 2,
    "batching_options": {
        "batch_size___fixed_batching": 16,
        "adaptive_batching": False,
        "adaptive_max_tokens": 3000,
        "max_batch_size___adaptive_batching": 100
    },
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.5,
        "delay_max": 1.5,
        "simulate_burst": True,
        "burst_interval": 4.0,
        "burst_size": 5
    }
}

scenario_r2_low_latency_chatbot = {
    **scenario_r1_standard_production,
    "config_name": "r2_low_latency_chatbot",
    "gpu_list": [0],
    "batching_options": {
        "batch_size___fixed_batching": 4
    },
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.01,
        "delay_max": 0.05,
        "simulate_burst": False
    }
}

scenario_r3_balanced_enterprise_service = {
    **scenario_r1_standard_production,
    "config_name": "r3_balanced_enterprise_service",
    "batching_options": {
        "batch_size___fixed_batching": 32
    },
    "fp_precision": "float16",
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": True,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": 10345441280000
    }
}

scenario_r4_high_load_api = {
    **scenario_r1_standard_production,
    "config_name": "r4_high_load_api",
    "gpu_list": [0],
    "batching_options": {
        "batch_size___fixed_batching": 8
    },
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.05,
        "delay_max": 0.2,
        "simulate_burst": True,
        "burst_interval": 2.0,
        "burst_size": 5
    }
}

scenario_r5_real_time_mobile = {
    **scenario_r1_standard_production,
    "config_name": "r5_real_time_mobile",
    "gpu_list": [0],
    "batching_options": {
        "batch_size___fixed_batching": 1
    },
    "fp_precision": "float16",
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.2,
        "delay_max": 0.6,
        "simulate_burst": True,
        "burst_interval": 5.0,
        "burst_size": 8
    },
    "quantization_config": {
        "quantization": True,
        "load_in_8bit": True,
        "load_in_4bit": False,
        "cached_flops_for_quantised_models": 10345441280000
    }
}

scenario_r6_medium_scale_serving = {
    **scenario_r1_standard_production,
    "config_name": "r6_medium_scale_serving",
    "gpu_list": [0, 1, 2, 3],
    "batching_options": {
        "batch_size___fixed_batching": 32
    },
    "latency_simulation": {
        "simulate": True,
        "delay_min": 0.01,
        "delay_max": 0.1,
        "simulate_burst": False
    }
}
