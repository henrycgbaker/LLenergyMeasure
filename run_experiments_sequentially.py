#!/usr/bin/env python
import subprocess
import os
import sys, os, json, logging
from experiment_orchestration_utils.b_run_scenarios import run_scenarios
import random, logging
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner 
from experiment_orchestration_utils.b_run_single_configuration import run_single_configuration
from configs.experiment_config_class import ExperimentConfig
import subprocess

# Ensure the current directory (project root) is in sys.path.
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.getcwd()))

from datasets import load_dataset
from configs.b_scenario_configs import (
    scenario_a_max_throughput_exploit,
    scenario_b_precision_gaming,
    scenario_c_gpu_overdrive,
    scenario_d_standard_production,
    scenario_e_low_latency_real_time,
    scenario_f_balanced_performance_mode
)

scenarios_list = [
    scenario_a_max_throughput_exploit,
    scenario_b_precision_gaming,
    scenario_c_gpu_overdrive,
    scenario_d_standard_production,
    scenario_e_low_latency_real_time,
    scenario_f_balanced_performance_mode,
]

for config_file in scenarios_list:
    print(f"Launching experiment with config: {config_file}")
    

    # Launch the single experiment script using 'accelerate launch'
    result = subprocess.run([
        "accelerate", "launch", "run_single_experiment.py",
        "--config", config_file
    ], check=True)
    
    print(f"Experiment with {config_file} completed.\n")

print("All experiments have been executed sequentially.")
