#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import logging
import torch
import signal
import atexit
from datetime import timedelta
from datasets import load_dataset
from experiment_orchestration_utils.c_launcher_utils import (
    launch_config_accelerate_cli, run_from_file, run_from_config
)
from configs.a_default_config import base_config

# --- Environment tweaks for CUDA fragmentation ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

# --- Cleanup handlers ---
def cleanup_distributed():
    """
    Destroy the process group if initialized, to avoid NCCL hangs.
    """
    if torch.distributed.is_initialized():
        logging.info("Destroying process group...")
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logging.warning(f"Error destroying process group: {e}")

def handle_signal(signum, frame):
    logging.warning(f"Received signal {signum}, performing cleanup...")
    cleanup_distributed()
    sys.exit(1)

# Register cleanup on exit and signals
atexit.register(cleanup_distributed)
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, handle_signal)


def load_prompts():
    ds = load_dataset("AIEnergyScore/text_generation")
    return [sample["text"] for sample in ds["train"]]


def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment configuration using Accelerate."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the experiment configuration JSON file.")
    parser.add_argument("--launched", action="store_true",
                        help="Flag indicating distributed mode via Accelerate.")
    args = parser.parse_args()

    # Relaunch under Accelerate if needed
    if not args.launched:
        script = os.path.abspath(__file__)
        logging.info("Not in distributed mode, re-launching via Accelerate CLI...")
        launch_config_accelerate_cli(
            args.config if args.config else base_config,
            script,
            extra_args=["--launched"]
        )
        sys.exit(0)

    # Distributed run
    logging.info("Running distributed experiment...")
    prompts = load_prompts()

    try:
        # Use config or default
        if args.config:
            logging.info("Loading configuration from %s", args.config)
            success, result = run_from_file(args.config, prompts)
        else:
            logging.info("No config file, using default config.")
            success, result = run_from_config(base_config, prompts)

    except torch.cuda.OutOfMemoryError as oom:
        logging.error("CUDA OOM during experiment: %s", oom)
        torch.cuda.empty_cache()
        cleanup_distributed()
        sys.exit(1)

    except Exception as ex:
        logging.error("Unexpected error during experiment: %s", ex)
        cleanup_distributed()
        sys.exit(1)

    else:
        if success:
            logging.info("Experiment completed successfully.")
        else:
            logging.error("Experiment failed (success flag is False).")

    # Final cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
