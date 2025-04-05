#!/usr/bin/env python
import subprocess
import os
import json
import tempfile

# Ensure the current directory (project root) is in sys.path.
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# Import scenarios
from configs.b_scenario_configs import (
    scenario_a1_max_throughput_exploit,
    scenario_a2_precision_minimalist,
    scenario_a3_quantisation_gaming,
    scenario_a4_latency_ignorance_exploit,
    scenario_a5_parallel_overdrive,
    scenario_r1_standard_production,
    scenario_r2_low_latency_chatbot,
    scenario_r3_balanced_enterprise_service,
    scenario_r4_high_load_api,
    scenario_r5_real_time_mobile,
    scenario_r6_medium_scale_serving,
)

# List of configuration dictionaries
scenarios_list = [
    scenario_a1_max_throughput_exploit,
    scenario_a2_precision_minimalist,
    scenario_a3_quantisation_gaming,
    scenario_a4_latency_ignorance_exploit,
    scenario_a5_parallel_overdrive,
    scenario_r1_standard_production,
    scenario_r2_low_latency_chatbot,
    scenario_r3_balanced_enterprise_service,
    scenario_r4_high_load_api,
    scenario_r5_real_time_mobile,
    scenario_r6_medium_scale_serving,
]


# Loop over each configuration.
for config in scenarios_list:
    # Write config to a temporary JSON file.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as temp_config_file:
        json.dump(config, temp_config_file, indent=2)
        temp_config_path = temp_config_file.name

    print(f"Launching experiment with config: {config['config_name']}")
    
    # need to allocate correct GPUs BEFORE launching accelerate !!
    config["num_processes"] = len(config["gpu_list"])

    # Launch the single config run script using 'accelerate launch'
    result = subprocess.run([
        "accelerate", "launch",
        "--num_processes", str(config["num_processes"]),
        "run_single_experiment.py",
        "--config", temp_config_path
    ], check=True)

    print(f"Experiment with {config['config_name']} completed.\n")

print("All experiments have been executed sequentially.")
