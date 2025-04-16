import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from configs.a_default_config import base_config
from configs.config_utils import update_nested_dict, update_multiple_config, generate_config_name_from_variation
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

# (i) parallelisation Strategies:
parallelisation_variations = {
    "num_processes": [1, 2, 3, 4]
}
parallelisation_configs = generate_controlled_configs(base_config, parallelisation_variations)

# (ii) Batching Strategies:
batching_variations = {
    "batching_options.batch_size___fixed_batching": [1, 2, 4, 8, 16, 32, 64, 128, 256]
}
batching_configs = generate_controlled_configs(base_config, batching_variations)

# (iii) Precision & Quantisation Methods:
precision_quantisation_configs = []

# Variation 1: FP32, no quantisation.
updates = {
    "fp_precision": "float32",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
}
cfg1 = update_multiple_config(base_config, updates)
cfg1["controlled_variation"] = updates
cfg1["suite"] = "controlled"   
cfg1["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg1)

# Variation 2: FP16, no quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
}
cfg2 = update_multiple_config(base_config, updates)
cfg2["controlled_variation"] = updates
cfg2["suite"] = "controlled"    
cfg2["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg2)

# Variation 3: FP16 with 8-bit quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
}
cfg3 = update_multiple_config(base_config, updates)
cfg3["controlled_variation"] = updates
cfg3["suite"] = "controlled"    
cfg3["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg3)

# Variation 4: FP16 with 4-bit quantisation.
updates = {
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": True,
}
cfg4 = update_multiple_config(base_config, updates)
cfg4["controlled_variation"] = updates
cfg4["suite"] = "controlled"    
cfg4["config_name"] = generate_config_name_from_variation(updates)
precision_quantisation_configs.append(cfg4)


# (iv) decoder mode Variations: ____________________________________________________
# List to hold all decoder configuration variations
decoder_mode_configs = []

# List of temperature values to test
temperature_variations = np.arange(0, 1.5, 0.2).round(1).tolist()

# Values for top_k and top_p
top_k_values = [20, 50, 100, 200, 500]
top_p_values = [0.7, 0.8, 0.9, 0.98]

### Variation A: Greedy Decoding
# Greedy decoding always picks the highest probability token so only temperature is varied here.
for temp in temperature_variations:
    updates = {
        "decoder_config.decoding_mode": "greedy",
        "decoder_config.decoder_temperature": temp,
    }
    cfg = update_multiple_config(base_config, updates)
    cfg["controlled_variation"] = updates
    cfg["suite"] = "controlled"
    cfg["config_name"] = generate_config_name_from_variation(updates)
    decoder_mode_configs.append(cfg)

### Variation B: Top-k Sampling
# For top-k sampling, grid search over both temperature and top_k values.
for temp in temperature_variations:
    for top_k in top_k_values:
        updates = {
            "decoder_config.decoding_mode": "top_k",
            "decoder_config.decoder_top_k": top_k,
            "decoder_config.decoder_temperature": temp,
        }
        cfg = update_multiple_config(base_config, updates)
        cfg["controlled_variation"] = updates
        cfg["suite"] = "controlled"
        cfg["config_name"] = generate_config_name_from_variation(updates)
        decoder_mode_configs.append(cfg)

### Variation C: Top-p (Nucleus) Sampling
# For top-p sampling, grid search over both temperature and top_p values.
for temp in temperature_variations:
    for top_p in top_p_values:
        updates = {
            "decoder_config.decoding_mode": "top_p",
            "decoder_config.decoder_top_p": top_p,
            "decoder_config.decoder_temperature": temp,
        }
        cfg = update_multiple_config(base_config, updates)
        cfg["controlled_variation"] = updates
        cfg["suite"] = "controlled"
        cfg["config_name"] = generate_config_name_from_variation(updates)
        decoder_mode_configs.append(cfg)

# (v) Latency Simulation Variations:____________________________________________________

latency_configs = []

# Baseline Variation: No latency simulation.
updates = {
    "latency_simulation.simulate": False
}
cfg_latency_baseline = update_multiple_config(base_config, updates)
cfg_latency_baseline["controlled_variation"] = updates
cfg_latency_baseline["suite"] = "controlled"   
cfg_latency_baseline["config_name"] = generate_config_name_from_variation(updates)
latency_configs.append(cfg_latency_baseline)

# Constant (non-bursty) latency variations with index-based pairing.
delay_min_values = [0.05, 0.1, 0.2, 0.4]
delay_max_values = [0.1, 0.2, 0.4, 0.5]

for dmin, dmax in zip(delay_min_values, delay_max_values):
    updates = {
        "latency_simulation.simulate": True,
        "latency_simulation.delay_min": dmin,
        "latency_simulation.delay_max": dmax,
        "latency_simulation.simulate_burst": False,
    }
    cfg_latency = update_multiple_config(base_config, updates)
    cfg_latency["controlled_variation"] = updates
    cfg_latency["suite"] = "controlled"   
    cfg_latency["config_name"] = generate_config_name_from_variation(updates)
    latency_configs.append(cfg_latency)

# Bursty latency variations:
# Define grids for delay and burst parameters.
burst_interval_values = [2.0, 4.0, 6.0]
burst_size_values = [5, 8, 10, 20]

for dmin, dmax in zip(delay_min_values, delay_max_values):
    for burst_interval in burst_interval_values:
        for burst_size in burst_size_values:
            updates = {
                "latency_simulation.simulate": True,
                "latency_simulation.delay_min": dmin,
                "latency_simulation.delay_max": dmax,
                "latency_simulation.simulate_burst": True,
                "latency_simulation.burst_interval": burst_interval,
                "latency_simulation.burst_size": burst_size,
            }
            cfg_latency_bursty = update_multiple_config(base_config, updates)
            cfg_latency_bursty["controlled_variation"] = updates
            cfg_latency_bursty["suite"] = "controlled"   
            cfg_latency_bursty["config_name"] = generate_config_name_from_variation(updates)
            latency_configs.append(cfg_latency_bursty)

# Combine all controlled configurations ========================================================
controlled_config_list = (
    parallelisation_configs +
    batching_configs +
    precision_quantisation_configs +
    decoder_mode_configs +
    latency_configs
)

# validate the configurations
def validate_config(cfg):
    # For all configs, these keys should exist.
    required_top_level_keys = ["suite", "config_name", "controlled_variation",
                               "decoder_config", "quantization_config", "latency_simulation"]
    for key in required_top_level_keys:
        assert key in cfg, f"Missing required key '{key}' in config: {cfg}"

    # Optionally, check the 'suite' value is set correctly.
    # (This might be updated later in the generation; otherwise you could allow "NA")
    # For example, if a real config must have "controlled" and defaults should not be "NA":
    assert cfg["suite"] in ["controlled", "NA"], f"Unexpected suite value: {cfg['suite']}"

    # --- Decoder Config ---
    decoder_cfg = cfg["decoder_config"]
    expected_decoder_keys = ["decoding_mode", "decoder_temperature", "decoder_top_k", "decoder_top_p"]
    for key in expected_decoder_keys:
        assert key in decoder_cfg, f"Missing '{key}' in decoder_config"
    # If a decoder variation was applied, check for valid values;
    # Otherwise, they should remain as NA.
    if decoder_cfg["decoding_mode"] != "NA":
        valid_modes = ["greedy", "top_k", "top_p"]
        mode = decoder_cfg["decoding_mode"]
        assert mode in valid_modes, f"Invalid 'decoding_mode': {mode}"
        assert isinstance(decoder_cfg["decoder_temperature"], (int, float)) or decoder_cfg["decoder_temperature"] != "NA", "Invalid decoder_temperature"
        if mode == "top_k":
            assert decoder_cfg["decoder_top_k"] != "NA", "Missing 'decoder_top_k' for top_k sampling"
        elif mode == "top_p":
            assert decoder_cfg["decoder_top_p"] != "NA", "Missing 'decoder_top_p' for top_p sampling"

    # --- Quantization Config ---
    quant_cfg = cfg["quantization_config"]
    for key in ["quantization", "load_in_8bit", "load_in_4bit"]:
        assert key in quant_cfg, f"Missing '{key}' in quantization_config"
    # If a quantization variation was applied, the values should be boolean; otherwise they remain "NA"
    if quant_cfg["quantization"] != "NA":
        assert isinstance(quant_cfg["quantization"], bool), "quantization must be a bool"
        assert isinstance(quant_cfg["load_in_8bit"], bool), "load_in_8bit must be a bool"
        assert isinstance(quant_cfg["load_in_4bit"], bool), "load_in_4bit must be a bool"

    # --- Latency Simulation Config ---
    latency_cfg = cfg["latency_simulation"]
    expected_latency_keys = ["simulate", "delay_min", "delay_max", "simulate_burst", "burst_interval", "burst_size"]
    for key in expected_latency_keys:
        assert key in latency_cfg, f"Missing '{key}' in latency_simulation"
    # If simulation is applicable, ensure delay_min and delay_max are set.
    if latency_cfg["simulate"] != "NA" and latency_cfg["simulate"] is True:
        assert latency_cfg["delay_min"] != "NA", "Missing delay_min in latency_simulation"
        assert latency_cfg["delay_max"] != "NA", "Missing delay_max in latency_simulation"
    # For burst settings, you might add additional checks if simulate_burst is True.
    
    return True

def fill_missing_keys(cfg, default_value="NA"):
    # Ensure the top-level sections exist:
    for key in ["suite", "config_name", "controlled_variation"]:
        if key not in cfg:
            cfg[key] = default_value
    # Ensure fixed sections exist:
    for section, keys in {
        "decoder_config": ["decoding_mode", "decoder_temperature", "decoder_top_k", "decoder_top_p"],
        "quantization_config": ["quantization", "load_in_8bit", "load_in_4bit"],
        "latency_simulation": ["simulate", "delay_min", "delay_max", "simulate_burst", "burst_interval", "burst_size"],
    }.items():
        if section not in cfg:
            cfg[section] = {k: default_value for k in keys}
        else:
            for k in keys:
                if k not in cfg[section]:
                    cfg[section][k] = default_value
    return cfg

__all__ = ["controlled_config_list"]

if __name__ == "__main__":
    print("Validating all controlled configs...\n")
    
    # If you didn't already pre-fill the config in generation, do it now:
    for cfg in controlled_config_list:
        fill_missing_keys(cfg)
    
    for i, cfg in enumerate(controlled_config_list):
        try:
            validate_config(cfg)
        except AssertionError as e:
            print(f"❌ Config {i} ({cfg.get('config_name')}) failed validation: {e}\n")
            raise
        else:
            print(f"✅ Config {i} ({cfg.get('config_name', 'NO_NAME')}) passed validation.")
    
    print(f"\nAll {len(controlled_config_list)} configs validated successfully.")