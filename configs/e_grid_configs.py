# d_gridsearch_configs.py

import copy
import itertools
from configs.a_default_config import base_config

def update_nested_dict(d, key_path, value):
    """
    Returns a deep copy of d with the nested key specified by the dot-separated key_path set to value.
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

def generate_grid_configs(base_config, grid_variations):
    """
    Given a base_config (dict) and a dictionary mapping dot-separated keys to lists of candidate values,
    generate a list of configurations representing the full grid search (cross product).
    
    Each configuration is a deep copy of base_config with the grid updates applied, and it
    receives a new key "grid_variation" that records the specific combination of parameters.
    
    Parameters:
      grid_variations: dict
          Keys are dot-separated strings (e.g. "batching_options.batch_size___fixed_batching").
          Values are lists of candidate values.
          
    Returns:
      A list of configuration dictionaries.
    """
    keys = list(grid_variations.keys())
    # Generate all combinations (cross product) of candidate values.
    combinations = list(itertools.product(*(grid_variations[k] for k in keys)))
    
    configs = []
    for combo in combinations:
        cfg = copy.deepcopy(base_config)
        variation = {}
        for key_path, value in zip(keys, combo):
            cfg = update_nested_dict(cfg, key_path, value)
            variation[key_path] = value
        # Record the specific grid variation in the configuration.
        cfg["grid_variation"] = variation
        cfg["suite"] = "grid"
        configs.append(cfg)
    return configs

# Define the grid of candidate parameters.
# You can adjust or add parameters as needed.
grid_variations = {
    "model_name": [
        "ModelA/SomeLargeModel",
        "ModelB/AnotherModel"
    ],
    "batching_options.batch_size___fixed_batching": [8, 16, 32],
    "fp_precision": ["float16", "float32"],
    "decoder_config.decoding_mode": ["greedy", "top_k"],
    # Note: if decoding_mode is "greedy", the decoder_top_k parameter might be ignored;
    # we include it for completeness.
    "decoder_config.decoder_top_k": [50],
    "quantization_config.quantization": [True, False]
}

# Generate the full grid search configuration list.
grid_config_list = generate_grid_configs(base_config, grid_variations)

__all__ = ["grid_config_list"]

if __name__ == "__main__":
    # For debugging: print out each grid variation.
    for cfg in grid_config_list:
        print(cfg["grid_variation"])
