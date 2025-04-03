#!/usr/bin/env python
import sys, os, json, logging
from experiment_orchestration_utils.b_run_scenarios import run_scenarios
import random, logging
from experiment_orchestration_utils.experiment_runner import ExperimentRunner 
from experiment_orchestration_utils.a_run_single_experiment import run_single_experiment_with_retries
from configs.experiment_config_class import ExperimentConfig
import subprocess
import time

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

def run_scenarios(scenario_list, prompts, num_repeats=3, max_retries=3, retry_delay=5):
    def stream_output(pipe):
        for line in iter(pipe.readline, ''):
            print(line, end='')
        pipe.close()

    scenario_results = []

    for cycle in range(num_repeats):
        logging.info(f"=== Scenarios Run Cycle {cycle+1}/{num_repeats} ===")
        random.shuffle(scenario_list)

        for config in scenario_list:
            config_str = json.dumps(config)
            port = random.randint(29500, 29999)
            cmd = [
                "accelerate", "launch", "--main_process_port", str(port),
                "MAIN_single_experiment.py", "--config", config_str
            ]

            attempt = 0
            success = False

            while attempt < max_retries and not success:
                logging.info(f"[Cycle {cycle+1}] Running scenario: {config['scenario_name']} (Attempt {attempt+1})")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True
                )

                stream_output(process.stdout)
                stream_output(process.stderr)
                return_code = process.wait()

                if return_code == 0:
                    logging.info(f"✅ Success: {config['scenario_name']} (Attempt {attempt+1})")
                    scenario_results.append({
                        "config": config,
                        "success": True,
                        "attempt": attempt + 1,
                    })
                    success = True
                else:
                    logging.warning(f"⚠️ Failure: {config['scenario_name']} (Attempt {attempt+1})")
                    attempt += 1
                    if attempt >= max_retries:
                        scenario_results.append({
                            "config": config,
                            "success": False,
                            "attempt": attempt,
                            "error": f"Return code: {return_code}"
                        })
                    else:
                        time.sleep(retry_delay)

    return scenario_results


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
    results = run_scenarios(scenarios, prompts, num_repeats=1, max_retries=3, retry_delay=5)
    
    output_path = "scenario_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Scenarios run completed. Results saved to {output_path}.")

if __name__ == "__main__":
    main()
