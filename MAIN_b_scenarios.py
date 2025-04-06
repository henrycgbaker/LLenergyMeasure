#!/usr/bin/env python
import os
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
from experiment_orchestration_utils.c_launch_single_configuration import run_from_config

# Ensure we run from the project root.
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# List of configuration dictionaries.
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

# Load prompts for scenarios (if they share the same dataset)
# modify this if each scenario should use a different dataset.
from datasets import load_dataset
ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
prompts = [sample["text"] for sample in ds]


# Loop over each scenario configuration.
for config in scenarios_list:
    # Ensure the number of processes matches the available GPUs.
    config["num_processes"] = len(config["gpu_list"])

    print(f"Launching experiment with config: {config['config_name']}")
    
    # Run the experiment directly using the common launcher.
    success, result = run_from_config(config, prompts, max_retries=3)
    
    if success:
        print(f"Configuration with {config['config_name']} completed successfully.\n")
    else:
        print(f"Configuration with {config['config_name']} failed.\n")

print("Full experiment complete: All Configuration have been executed sequentially.")
