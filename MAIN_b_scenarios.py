#!/usr/bin/env python
import logging
import os
from experiment_orchestration_utils.c_launcher_utils import launch_config_accelerate_cli
from configs.c_scenario_configs import (
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

logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

def main():
    config_list = [
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
    
    script_path = os.path.abspath("MAIN_a_single_experiment.py")
    
    for config in config_list:
        logging.info("Launching experiment for configuration: %s", config["config_name"])
        try:
            launch_config_accelerate_cli(config, script_path, extra_args=["--launched"])
            logging.info("Experiment for %s completed successfully.", config["config_name"])
        except Exception as e:
            logging.error("Experiment for %s failed: %s", config["config_name"], e)
    
    print("Full experiment complete: all configurations have been executed.")

if __name__ == "__main__":
    main()
