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
    Given a variation dictionary, returns a concise and human-readable config name.
    """
    if any("latency_simulation" in k for k in variation):
        sim = variation.get("latency_simulation.simulate", False)
        if not sim:
            return "latency_off"
        dmin = variation.get("latency_simulation.delay_min", "NA")
        dmax = variation.get("latency_simulation.delay_max", "NA")
        burst = variation.get("latency_simulation.simulate_burst", False)
        if burst:
            interval = variation.get("latency_simulation.burst_interval", "NA")
            size = variation.get("latency_simulation.burst_size", "NA")
            return f"latency_burst_{dmin}_{dmax}_{interval}_{size}"
        else:
            return f"latency_const_{dmin}_{dmax}"

    parts = []
    for key, value in variation.items():
        # Simplify the key name
        if "batching_options" in key:
            short_key = "batching"
        elif "fp_precision" in key:
            short_key = "precis"
        elif "quantization_config.load_in_8bit" in key and value:
            short_key = "quant8"
        elif "quantization_config.load_in_4bit" in key and value:
            short_key = "quant4"
        elif "quantization_config" in key:
            short_key = "quant"
        elif "decoder_config.decoding_mode" in key:
            short_key = "decoding"
        elif "decoder_config.decoder_top_k" in key:
            short_key = "topk"
        elif "decoder_config.decoder_top_p" in key:
            short_key = "topp"
        elif "decoder_config.decoder_temperature" in key:
            short_key = "temp"
        elif "num_processes" in key:
            short_key = "proc"
        else:
            short_key = key.split(".")[-1]
        parts.append(f"{short_key}_{value}")
    return "_".join(parts)
