#!/usr/bin/env python
import sys, os, json, logging
from experiment_orchestration_utils.b_run_scenarios import run_scenarios

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

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(process)d] - %(message)s",
)

def main():
    # Load dataset and extract prompts.
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    prompts = [sample["text"] for sample in ds]
    
    # Create a list of scenarios and include a scenario name in each dictionary.
    scenarios = [
        dict(scenario_a_max_throughput_exploit, scenario_name="Max Throughput Exploit"),
        dict(scenario_b_precision_gaming, scenario_name="Precision Gaming"),
        dict(scenario_c_gpu_overdrive, scenario_name="GPU Overdrive"),
        dict(scenario_d_standard_production, scenario_name="Standard Production"),
        dict(scenario_e_low_latency_real_time, scenario_name="Low-Latency Real-Time"),
        dict(scenario_f_balanced_performance_mode, scenario_name="Balanced Performance Mode"),
    ]
    
    # Run the experiments for each scenario.
    results = run_scenarios(scenarios, prompts, num_repeats=3, max_retries=3, retry_delay=5)
    
    output_path = "scenario_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Scenarios run completed. Results saved to {output_path}.")

if __name__ == "__main__":
    main()
