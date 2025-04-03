#!/usr/bin/env python
import sys, os

# Ensure the current directory (project root) is in sys.path.
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.getcwd()))

import json
from datasets import load_dataset

from configs.experiment_config_class import ExperimentConfig
from configs.default_config import base_config, grid_params

from experiment_orchestration.grid_search import run_grid_search

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(process)d] - %(message)s",
)

def main():

    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    prompts = [sample["text"] for sample in ds]
    
    experiment_config = ExperimentConfig(**base_config)
    
    # Run grid search.
    # Note: run_grid_search() expects base_config as a plain dict,
    results = run_grid_search(
        base_config=experiment_config.to_dict(),
        grid_params=grid_params,
        prompts=prompts,
        num_repeats=3,
        max_retries=3,
        retry_delay=5
    )
    
    output_path = "grid_search_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Grid search completed. Results saved to {output_path}.")

if __name__ == "__main__":
    main()
