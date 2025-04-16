import copy
from configs.a_default_config import base_config
from configs.config_utils import update_nested_dict, update_multiple_config, generate_config_name_from_variation
import copy
import numpy as np

def generate_controlled_configs(base_config, controlled_variations):
    configs = []
    for param, values in controlled_variations.items():
        for val in values:
            # Create a fresh copy of the base configuration
            cfg = copy.deepcopy(base_config)
            cfg = update_nested_dict(cfg, param, val)
            variation = {param: val}
            cfg["controlled_variation"] = variation
            cfg["suite"] = "controlled"
            cfg["config_name"] = generate_config_name_from_variation(variation)
            configs.append(cfg)
    return configs

# ---------------------------
# Generate Controlled Experiments
# ---------------------------

# (ii) Batching Strategies:
batching_variations = {
    "batching_options.batch_size___fixed_batching": [1, 2, 4, 8, 16, 32, 64]
}
batching_configs = generate_controlled_configs(base_config, batching_variations)

# Combine all controlled configurations ========================================================
controlled_config_list = batching_configs

__all__ = ["controlled_config_list"]

if __name__ == "__main__":
    for cfg in controlled_config_list:
        print(cfg["config_name"])
