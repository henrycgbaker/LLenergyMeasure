#!/usr/bin/env python
import os
import sys
import argparse
import logging
from datasets import load_dataset
from experiment_orchestration_utils.c_launcher_utils import (
    launch_config_accelerate_cli, run_from_file, run_from_config
)
from configs.a_default_config import base_config

# Set up logging (including process id for distributed debugging)
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
    # This flag indicates that the script has been launched via Accelerate CLI.
    parser.add_argument("--launched", action="store_true",
                        help="Indicates that the script is running in distributed mode.")
    args = parser.parse_args()

    # If not launched in distributed mode, re-launch using Accelerate CLI.
    if not args.launched:
        script_path = os.path.abspath(__file__)
        logging.info("Not running in distributed mode. Re-launching via Accelerate CLI...")
        # Pass "--launched" so that the re-launched script skips re-launching.
        launch_config_accelerate_cli(args.config if args.config else base_config, script_path, extra_args=["--launched"])
        sys.exit(0)

    # If we get here, we're running under Accelerate (distributed mode).
    logging.info("Running distributed experiment.")
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
