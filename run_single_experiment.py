# single_experiment.py
import argparse
import json
import sys, os
from datasets import load_dataset
from configs.experiment_config_class import ExperimentConfig
from experiment_orchestration_utils.experiment_runner import ExperimentRunner 
from experiment_orchestration_utils.a_run_single_experiment import run_single_experiment_with_retries
import logging

logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

def main(config):
    # Load prompts from a dataset.
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    prompts = [sample["text"] for sample in ds]

    # Load config from file.
    with open(config, "r") as f:
        base_config = json.load(f)
    
    # Convert base_config dict to dataclass.
    experiment_config = ExperimentConfig(**base_config)
    runner = ExperimentRunner(experiment_config, prompts)
    
    # Run the experiment.
    run_single_experiment_with_retries(runner, max_retries=3, retry_delay=5)
    print("Single run completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment configuration JSON file.")
    args = parser.parse_args()
    main(args.config)
