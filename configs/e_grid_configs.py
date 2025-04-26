import sys, os
from itertools import product
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.a_default_config import base_config
from configs.config_utils import (
    update_nested_dict,
    update_multiple_config,
    generate_config_name_from_variation,
)

# ─── 1) PRECISION MAP ────────────────────────────────────────────────────────────────
precision_map = {
    "fp32": {
        "fp_precision": "float32",
        "quantization_config.quantization": False,
        "quantization_config.load_in_8bit": False,
        "quantization_config.load_in_4bit": False,
    },
    "fp16": {
        "fp_precision": "float16",
        "quantization_config.quantization": False,
        "quantization_config.load_in_8bit": False,
        "quantization_config.load_in_4bit": False,
    },
    "int8": {
        "fp_precision": "float16",
        "quantization_config.quantization": True,
        "quantization_config.load_in_8bit": True,
        "quantization_config.load_in_4bit": False,
    },
    "int4": {
        "fp_precision": "float16",
        "quantization_config.quantization": True,
        "quantization_config.load_in_8bit": False,
        "quantization_config.load_in_4bit": True,
    },
}

# ─── 2) LATENCY MAP ─────────────────────────────────────────────────────────────────
delay_min_values    = [0.1, 0.4]
delay_max_values    = [0.2, 0.5] 
burst_interval_vals = [4.0,  6.0]
burst_size_vals     = [8, 16]

latency_map = {
    # no latency at all
    "no_latency": {
        "latency_simulation.simulate": False
    }
}

# constant (non-bursty) latency
for dmin, dmax in zip(delay_min_values, delay_max_values):
    key = f"const_{dmin:g}_{dmax:g}"
    latency_map[key] = {
        "latency_simulation.simulate": True,
        "latency_simulation.delay_min": dmin,
        "latency_simulation.delay_max": dmax,
        "latency_simulation.simulate_burst": False,
    }

# bursty latency
for dmin, dmax in zip(delay_min_values, delay_max_values):
    for interval in burst_interval_vals:
        for size in burst_size_vals:
            key = f"bursty_{dmin:g}_{dmax:g}_i{interval:g}_s{size}"
            latency_map[key] = {
                "latency_simulation.simulate": True,
                "latency_simulation.delay_min": dmin,
                "latency_simulation.delay_max": dmax,
                "latency_simulation.simulate_burst": True,
                "latency_simulation.burst_interval": interval,
                "latency_simulation.burst_size": size,
            }

# ─── 3) GRID VARIATIONS ──────────────────────────────────────────────────────────────
grid_variations = {
    "num_processes": [1, 2, 3, 4],
    "batching_options.batch_size___fixed_batching": [2, 4, 8, 16, 32, 64],
    "precision": list(precision_map.keys()),                       
    "decoder_config.decoder_temperature": [0.0, 0.8],
    "decoder_config.decoder_top_p": [0.8],                         
    "latency": list(latency_map.keys()),                            
}

# ─── 4) GRID GENERATOR ──────────────────────────────────────────────────────────────
def generate_grid_configs(base_config, grid_variations):
    configs = []
    axes   = list(grid_variations.keys())
    values = [grid_variations[a] for a in axes]

    for combo in product(*values):
        variation = dict(zip(axes, combo))
        cfg = copy.deepcopy(base_config)

        for axis, val in variation.items():
            if axis == "precision":
                cfg = update_multiple_config(cfg, precision_map[val])
            elif axis == "latency":
                cfg = update_multiple_config(cfg, latency_map[val])
            else:
                cfg = update_nested_dict(cfg, axis, val)

        cfg["grid_variation"] = variation
        cfg["suite"]          = "grid"
        cfg["config_name"]    = generate_config_name_from_variation(variation)
        configs.append(cfg)

    return configs

# ─── 5) EXPORT ─────────────────────────────────────────────────────────────────────
grid_config_list = generate_grid_configs(base_config, grid_variations)

# ─── 6) MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    grid_configs = generate_grid_configs(base_config, grid_variations)
    print(f"Generated {len(grid_configs)} configs")
    for c in grid_configs[:5]:
        print(c["config_name"], c["grid_variation"])
