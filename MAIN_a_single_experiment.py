# MAIN_a_single_experiment.py
import argparse
import logging
import json
import os

from datasets import load_dataset
from experiment_orchestration_utils.c_acc_launcher_single_configuration import run_from_file, run_from_config
from configs.a_default_config import base_config

# Set up logging with process id
logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

def load_prompts():
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    return [sample["text"] for sample in ds]

def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment configuration using Accelerate."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the experiment configuration JSON file.")
    args = parser.parse_args()

    prompts = load_prompts()

    if args.config:
        logging.info("Loading configuration from %s", args.config)
        success, result = run_from_file(args.config, prompts)
    else:
        logging.info("No config file provided, using default configuration.")
        success, result = run_from_config(base_config, prompts)

    if success:
        logging.info("Experiment run completed successfully.")
    else:
        logging.error("Experiment run failed.")

if __name__ == "__main__":
    main()
