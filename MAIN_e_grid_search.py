
import logging
import os
from experiment_orchestration_utils.c_launcher_utils import launch_config_accelerate_cli
from configs.d_scenario_configs import grid_config_list

logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

def main():
    config_list = grid_config_list
    
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
