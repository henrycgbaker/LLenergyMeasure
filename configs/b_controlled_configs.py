# controlled_configs.py

import copy
from configs.a_default_config import base_config

def update_nested_dict(d, key_path, value):
    """
    Given a dictionary d, returns a deep copy in which the nested key specified by the dot-separated key_path
    is set to the given value.
    
    Example:
      update_nested_dict(base_config, "batching_options.batch_size___fixed_batching", 16)
    """
    keys = key_path.split(".")
    new_d = copy.deepcopy(d)
    current = new_d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return new_d

def update_multiple_config(base_config, updates):
    """
    Given base_config and a dictionary mapping of multiple dot-separated keys to new values,
    returns a new configuration dictionary with all the updates applied.
    
    Parameters:
      - base_config: dict, the baseline configuration.
      - updates: dict, keys are dot-separated paths, values are the new values.
    
    Returns:
      - A new configuration dictionary with the updates applied.
    """
    new_config = copy.deepcopy(base_config)
    for key_path, value in updates.items():
        new_config = update_nested_dict(new_config, key_path, value)
    return new_config

def generate_controlled_configs(base_config, controlled_variations):
    """
    Given a base_config and a dictionary mapping dot-separated keys to lists of candidate values,
    generate a list of configuration dictionaries where each configuration varies only one parameter.
    
    The generated configuration gets a new key "controlled_variation" that records the parameter and value.
    
    Parameters:
      - base_config: dict, the baseline configuration.
      - controlled_variations: dict, mapping from a dot-separated key to a list of values.
    
    Returns:
      - List of configuration dictionaries.
    """
    configs = []
    for param, values in controlled_variations.items():
        for val in values:
            cfg = update_nested_dict(base_config, param, val)
            cfg["controlled_variation"] = {param: val}
            cfg["suite"] = "controlled"
            configs.append(cfg)
    return configs

# ---------------------------
# Generate Controlled Experiments
# ---------------------------

# (i) Batching Strategies:
batching_variations = {
    "batching_options.batch_size___fixed_batching": [1, 2, 4, 8, 16, 32, 64]
}
batching_configs = generate_controlled_configs(base_config, batching_variations)

# (ii) Precision & Quantisation Methods:
# We'll create four variations:
#  1. FP32, no quantisation.
#  2. FP16, no quantisation.
#  3. FP16 with 8-bit quantisation.
#  4. FP16 with 4-bit quantisation.
precision_quantisation_configs = []
# Variation 1: FP32, no quantisation.
updates = {
    "fp_precision": "float32",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
}
cfg1 = update_multiple_config(base_config, updates)
cfg1["controlled_variation"] = {"fp_precision": "float32", "quantization": "none"}
precision_quantisation_configs.append(cfg1)
# Variation 2: FP16, no quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
}
cfg2 = update_multiple_config(base_config, updates)
cfg2["controlled_variation"] = {"fp_precision": "float16", "quantization": "none"}
precision_quantisation_configs.append(cfg2)
# Variation 3: FP16 with 8-bit quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
}
cfg3 = update_multiple_config(base_config, updates)
cfg3["controlled_variation"] = {"fp_precision": "float16", "quantization": "8-bit"}
precision_quantisation_configs.append(cfg3)
# Variation 4: FP16 with 4-bit quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": True,
}
cfg4 = update_multiple_config(base_config, updates)
cfg4["controlled_variation"] = {"fp_precision": "float16", "quantization": "4-bit"}
precision_quantisation_configs.append(cfg4)

# (iii) Inference Mode Variations:
# We create three variations: Greedy, Top-k, and Top-p.
inference_mode_configs = []
# Variation A: Greedy decoding.
updates = {
    "decoder_config.decoding_mode": "greedy",  # adding a new key
    "decoder_config.decoder_temperature": 1.0,
}
cfg_greedy = update_multiple_config(base_config, updates)
cfg_greedy["controlled_variation"] = {"decoding_mode": "greedy"}
inference_mode_configs.append(cfg_greedy)
# Variation B: Top-k sampling.
updates = {
    "decoder_config.decoding_mode": "top_k",
    "decoder_config.decoder_top_k": 50,
    "decoder_config.decoder_temperature": 1.0,
}
cfg_topk = update_multiple_config(base_config, updates)
cfg_topk["controlled_variation"] = {"decoding_mode": "top_k", "top_k": 50}
inference_mode_configs.append(cfg_topk)
# Variation C: Top-p sampling.
updates = {
    "decoder_config.decoding_mode": "top_p",
    "decoder_config.decoder_top_p": 0.9,
    "decoder_config.decoder_temperature": 1.0,
}
cfg_topp = update_multiple_config(base_config, updates)
cfg_topp["controlled_variation"] = {"decoding_mode": "top_p", "top_p": 0.9}
inference_mode_configs.append(cfg_topp)

# (iv) Latency Simulation Variations:
latency_configs = []
# Variation 1: Baseline (no latency simulation)
updates = {
    "latency_simulation.simulate": False
}
cfg_latency_baseline = update_multiple_config(base_config, updates)
cfg_latency_baseline["controlled_variation"] = {"latency": "baseline"}
latency_configs.append(cfg_latency_baseline)
# Variation 2: Constant Moderate Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.2,
    "latency_simulation.simulate_burst": False,
}
cfg_latency_mod = update_multiple_config(base_config, updates)
cfg_latency_mod["controlled_variation"] = {"latency": "constant_moderate"}
latency_configs.append(cfg_latency_mod)
# Variation 3: Constant High Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.2,
    "latency_simulation.delay_max": 0.6,
    "latency_simulation.simulate_burst": False,
}
cfg_latency_high = update_multiple_config(base_config, updates)
cfg_latency_high["controlled_variation"] = {"latency": "constant_high"}
latency_configs.append(cfg_latency_high)
# Variation 4: Bursty Moderate Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.2,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0,
    "latency_simulation.burst_size": 5,
}
cfg_latency_bursty_mod = update_multiple_config(base_config, updates)
cfg_latency_bursty_mod["controlled_variation"] = {"latency": "bursty_moderate"}
latency_configs.append(cfg_latency_bursty_mod)
# Variation 5: Bursty High Latency
updates = {
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.2,
    "latency_simulation.delay_max": 0.6,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 5.0,
    "latency_simulation.burst_size": 8,
}
cfg_latency_bursty_high = update_multiple_config(base_config, updates)
cfg_latency_bursty_high["controlled_variation"] = {"latency": "bursty_high"}
latency_configs.append(cfg_latency_bursty_high)

# Combine all controlled configurations.
controlled_config_list = (
    batching_configs +
    precision_quantisation_configs +
    inference_mode_configs +
    latency_configs
)

# For convenience, you can define __all__ to export controlled_config_list.
__all__ = ["controlled_config_list"]

if __name__ == "__main__":
    # Print out the controlled variations for a quick check.
    for cfg in controlled_config_list:
        print(cfg["controlled_variation"])
