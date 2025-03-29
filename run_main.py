#!/usr/bin/env python
import sys, os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.getcwd()))


import json
from datasets import load_dataset
from experiments.grid_search import run_grid_search
from configs.default_config import base_config, grid_params

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(process)d] - %(message)s",
)

def main():
    # Load prompts from a dataset.
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    prompts = [sample["text"] for sample in ds]
    
    # Run the grid search.
    results = run_grid_search(
        base_config, 
        grid_params, 
        prompts, 
        num_repeats=3,
        max_retries=3,
        retry_delay=5
    )
    
    # Save the aggregated results to a JSON file.
    output_path = "grid_search_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Grid search completed. Results saved to {output_path}.")

if __name__ == "__main__":
    main()
