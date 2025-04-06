#!/usr/bin/env python
import logging
import os
from experiment_orchestration_utils.c_acc_launcher_single_configuration import launch_config_accelerate_cli

logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

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

def main():
    config_files = [
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
    
    script_path = "MAIN_a_single_experiment.py"

    for config_file in config_files:
        logging.info("Launching experiment for configuration: %s", config_file)
        try:
            launch_config_accelerate_cli(config_file, script_path)
            logging.info("Experiment for %s completed successfully.", config_file)
        except Exception as e:
            logging.error("Experiment for %s failed: %s", config_file, e)
    
    print("Full experiment complete: all configuration have been executed.")

if __name__ == "__main__":
    main()