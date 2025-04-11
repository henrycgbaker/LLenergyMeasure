import copy

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

def generate_config_name_from_variation(variation):
    """
    Given a variation dictionary (e.g. {"batching_options.batch_size___fixed_batching": 16}),
    returns a string that concatenates a simplified key and the value.
    For example: "batching_16" for a batching variation.
    """
    parts = []
    for key, value in variation.items():
        # Simplify the key: for example, if key contains "batching_options", use "batching"
        if "batching_options" in key:
            short_key = "batching"
        elif "fp_precision" in key:
            # For fp_precision, you might want "precis_fp32" or "precis_fp16"
            short_key = "precis"
        elif "quantization_config" in key:
            # Remove common parts for clarity.
            if "load_in_8bit" in key:
                short_key = "quant8"
            elif "load_in_4bit" in key:
                short_key = "quant4"
            else:
                short_key = "quant"
        elif "decoder_config.decoding_mode" in key:
            short_key = "decoding"
        elif "latency_simulation" in key:
            short_key = "latency"
        else:
            # Default: take the last segment of the key.
            short_key = key.split(".")[-1]
        parts.append(f"{short_key}_{value}")
    return "_".join(parts)