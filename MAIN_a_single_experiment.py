import argparse
import json
import os
from datasets import load_dataset
from experiment_orchestration_utils.c_launch_single_configuration import run_from_config, run_from_file
from configs.a_default_config import base_config 
import logging

logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

def main(config_path=None):
    # Load prompts from the dataset
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    prompts = [sample["text"] for sample in ds]

    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} does not exist!")
        print(f"Loading configuration from {config_path}")
        success, result = run_from_file(config_path, prompts)
    else:
        print("No config path provided, using base_config from default_config.py")
        success, result = run_from_config(base_config, prompts)

    if success:
        print("Single run completed successfully.")
    else:
        print("Single run failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,   
        help="Optional: Path to experiment configuration JSON file. If not provided, uses base_config from default_config.py."
    )
    args = parser.parse_args()
    main(args.config)