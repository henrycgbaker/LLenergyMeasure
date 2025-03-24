import sys
import os
from accelerate import notebook_launcher
from accelerate.launch import launch
from datasets import load_dataset
import logging
import json

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
# Uncomment to force the 'spawn' start method for multiprocessing
# ------------------------------
#import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)


# ------------------------------
# Set up grid search configurations
# ------------------------------
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
# Load prompts (e.g., from the 'arxiv' split of a dataset)
# ------------------------------
ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
prompts = [sample["text"] for sample in ds]

# ------------------------------
# Define the experiment workflow
# ------------------------------
def run_experiment_workflow(config):
    """
    Run the full experiment workflow using a given configuration.
    """
    # Create and run the experiment runner
    runner = ExperimentRunner(config, prompts)
    runner.run_torch()

    # Only the main process proceeds to aggregation and saving.
    if not runner.accelerator.is_main_process:
        return None

    runner.aggregate_results()
    runner.save_experiment_results()
    runner.teardown()

    # Return a picklable results dictionary (global energy results).
    return runner.global_energy_results

# ------------------------------
# Main execution: iterate over grid search configurations
# ------------------------------
if __name__ == "__main__":
    overall_results = {}
    for idx, config in enumerate(grid_configs):
        logging.info(f"Running experiment {idx + 1}/{len(grid_configs)} with config: {config}")

        # Launch the experiment in isolated processes.
        # The lambda ensures that our workflow function gets the current config.
        experiment_result = launch(
            lambda: run_experiment_workflow(config),
            num_processes=config.num_processes,
            terminate_on_error=True
        )

    # Optionally, save overall grid search results to a JSON file.
    with open("grid_search_results.json", "w") as f:
        json.dump(overall_results, f, indent=2)

    logging.info("Grid search experiments completed and results saved.")
