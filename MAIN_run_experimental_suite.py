#!/usr/bin/env python
"""
run_experimental_suite.py

This script orchestrates large-scale experimental suites.
It supports three types of experiments:
  1. Controlled Experiments (isolating one parameter at a time)
  2. Scenarios (realistic vs. artificially optimized deployment conditions)
  3. Grid Search (exploring multiple models and parameter combinations)

The script organizes runs into cycles, randomizes the order of configuration runs,
and launches each configuration-run via the Accelerate CLI (using a dedicated launcher module).

Results from each configuration-run are saved independently for later aggregation.
"""

import os
import random
import logging
import time
from experiment_orchestration_utils.c_launcher_utils import launch_config_accelerate_cli

# Import configuration lists for each suite

from configs.c_controlled_configs import controlled_config_list   
from configs.d_scenario_configs import scenario_config_list         
from configs.e_grid_configs import grid_config_list    

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(asctime)s - %(levelname)s - %(message)s")

# Settings for the experiment suite
CYCLES_PER_SUITE = 3    # Number of cycles per suite; each cycle is a full pass through the config list

# Path to the single experiment script that runs one configuration-run.
SINGLE_EXP_SCRIPT = os.path.abspath("MAIN_a_single_experiment.py")


def run_cycle(config_list, suite_name, cycle_num):
    """
    Runs one cycle of experiments for a given list of configurations.
    Randomizes the order of configuration runs within the cycle.
    """
    logging.info("Starting Cycle %s for suite '%s'", cycle_num, suite_name)
    
    # Randomize the order of configuration runs to avoid systematic bias.
    randomized_configs = config_list.copy()
    random.shuffle(randomized_configs)
    
    for config in randomized_configs:
        config_name = config.get("config_name", "unnamed")
        logging.info("Cycle %s: Launching configuration-run for '%s'", cycle_num, config_name)
        try:
            # Launch the configuration-run via the Accelerate CLI launcher.
            # Extra arguments ensure that the launched MAIN_a_single_experiment.py knows it's in distributed mode.
            launch_config_accelerate_cli(config, SINGLE_EXP_SCRIPT, extra_args=["--launched"])
            logging.info("Cycle %s: Configuration-run for '%s' completed successfully.", cycle_num, config_name)
        except Exception as e:
            logging.error("Cycle %s: Configuration-run for '%s' failed: %s", cycle_num, config_name, e)
        
        time.sleep(2)
    
    logging.info("Completed Cycle %s for suite '%s'", cycle_num, suite_name)


def run_suite(config_list, suite_name):
    """
    Runs a complete experimental suite by executing multiple cycles for a given set of configurations.
    """
    logging.info("Starting Experimental Suite: '%s'", suite_name)
    for cycle in range(1, CYCLES_PER_SUITE + 1):
        run_cycle(config_list, suite_name, cycle)
    logging.info("Completed Experimental Suite: '%s'", suite_name)


def main():
    # Define a list of suites to run. Each suite is a tuple (suite_name, config_list)
    suites = [
        ("Controlled", controlled_config_list),
        ("Scenario", scenario_config_list),
        ("GridSearch", grid_config_list),
    ]
    
    # Optionally, if you want to interleave cycles across suites, you could
    # build a list of (suite_name, cycle_num, config_list) tuples and randomize the order.
    # For simplicity, here we run each suite sequentially.
    
    for suite_name, config_list in suites:
        run_suite(config_list, suite_name)
    
    logging.info("All experimental suites have been executed.")


if __name__ == "__main__":
    main()
