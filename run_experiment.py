import sys
import os
import logging
from itertools import product
from accelerate.launchers import launch, LaunchConfig
from datasets import load_dataset
import json


# MAIN REASON FOR HAVING AS SCRIPT IS TO MAKE RUN MORE ROBUST...
# WHEN LAUNCHING FROM NOTEBOOK IT CRASHES AND NEEDS KERNEL RESTARTS
# FOCUS IS ON MAKING AS ROBUST AS POSSIBLE....
# BETTER TO DO AS CLI?

# ------------------------------
# Logging configuration
# ------------------------------
# (i) clear any pre-existing logging handlers (especially useful in notebook reruns)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# (ii) now safely configure logging
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Adjust Python path and import classes
# ------------------------------
project_root = os.getcwd()  
if project_root not in sys.path:
    sys.path.append(project_root)

from classes.experiment_config import ExperimentConfig
from classes.experiment_runner import ExperimentRunner


# ------------------------------
# Set up grid search configurations
# ------------------------------

# MOVE THIS OUT INTO NOTEBOOK

# Example parameter ranges (adapt these to your own parameters)
param_a_values = [1, 2]  # e.g., could represent different model sizes or settings
param_b_values = ['option1', 'option2']  # e.g., different sampling strategies
num_processes = 2  # Set number of processes; adjust as needed

# Create a list of ExperimentConfig objects representing the grid.
grid_configs = []
for a, b in product(param_a_values, param_b_values):
    config = ExperimentConfig(num_processes=num_processes, param_a=a, param_b=b)
    grid_configs.append(config)
    
# ------------------------------
# Load prompts 
# ------------------------------

# MOVE THIS OUT INTO NOTEBOOK

ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
prompts = [sample["text"] for sample in ds]


# ------------------------------
# Define the experiment workflow
# ------------------------------

# MOVE THIS OUT INTO NOTEBOOK / ANOTHER SCRIPT???

def run_experiment_workflow(config):
    runner = ExperimentRunner(config, prompts)
    runner.run_torch()
    if not runner.accelerator.is_main_process:
        return None
    if runner.accelerator.is_main_process:
        runner.aggregate_results()
        runner.save_experiment_results()
        runner.teardown()
    return runner.global_energy_results

# ------------------------------
# Main execution: iterate over grid search configurations
# ------------------------------

def main():
    # Set up launch configuration
    config = LaunchConfig(
        num_processes=4,                   # adjust to number of GPUs/processes
        mixed_precision="no",             # or "fp16", "bf16", etc. if needed
        use_cpu=False,                    # ensure this is False for GPU
        start_method="spawn",             # critical for CUDA safety
        terminate_on_error=True           # ensures whole job terminates if one fails
    )

    # Launch the actual workflow
    launch(config=config, entrypoint=run_experiment_workflow)()

if __name__ == "__main__":
    main()
    
    # ---
if __name__ == "__main__":
    overall_grid_results = {}
    for idx, config in enumerate(grid_configs):
        logging.info(f"Running experiment {idx+1}/{len(grid_configs)} with config: {config}")

        # Launch the experiment in isolated processes.
        # The lambda ensures that our workflow function gets the current config.
        experiment_results = launch(
            config=LaunchConfig(
                num_processes=2, 
                start_method="spawn", 
                terminate_on_error=True),
    entrypoint=run_experiment_workflow
)

        overall_grid_results[f"config_{idx}"] = experiment_results


    # Optionally, save overall grid search results to a JSON file.
    with open("grid_search_results.json", "w") as f:
        json.dump(overall_grid_results, f, indent=2)

    logging.info("Grid search experiments completed and results saved.")
