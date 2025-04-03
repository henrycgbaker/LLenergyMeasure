#!/usr/bin/env python
import sys, os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.getcwd()))


import json
from datasets import load_dataset
from configs.experiment_config_class import ExperimentConfig
from configs.a_default_config import base_config
from experiment_orchestration_utils.experiment_runner import ExperimentRunner 
from experiment_orchestration_utils.a_run_single_experiment import run_single_experiment_with_retries

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(process)d] - %(message)s",
)


def main():
    # Load prompts from a dataset.
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    prompts = [sample["text"] for sample in ds]
    
    
    # Convert base_config dict to dataclass
    experiment_config = ExperimentConfig(**base_config)
    
    runner = ExperimentRunner(experiment_config, prompts)
    
    # Run the grid search.
    run_single_experiment_with_retries(
        runner,  
        max_retries=3,
        retry_delay=5
    )
    
    print(f"Single run completed.")

if __name__ == "__main__":
    main()
