#!/usr/bin/env python3
import os
# Ensure CUDA allocator config is set before any torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")

import sys
import argparse
import logging
import torch
import signal
import atexit
from datasets import load_dataset
from experiment_orchestration_utils.c_launcher_utils import (
    launch_config_accelerate_cli, run_from_file, run_from_config
)
from configs.a_default_config import base_config

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")

# --- Cleanup handlers ---
def cleanup_distributed():
    """
    Destroy the process group if initialized and clear caches to avoid hangs.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
            logging.info("Destroyed process group.")
        except Exception as e:
            logging.warning(f"Error destroying process group: {e}")
    torch.cuda.empty_cache()


def handle_signal(signum, frame):
    logging.warning(f"Received signal {signum}, performing cleanup and exiting.")
    cleanup_distributed()
    # use os._exit to kill all threads immediately
    os._exit(1)

# Register cleanup on normal exit and signals
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
                        help="Path to JSON config file.")
    parser.add_argument("--launched", action="store_true",
                        help="Flag indicating distributed mode via Accelerate.")
    args = parser.parse_args()

    # Relaunch under Accelerate if needed
    if not args.launched:
        script = os.path.abspath(__file__)
        logging.info("Relaunching script under accelerate...")
        launch_config_accelerate_cli(
            args.config or base_config,
            script,
            extra_args=["--launched"]
        )
        sys.exit(0)

    # In distributed mode
    logging.info("Starting distributed experiment run...")
    prompts = load_prompts()

    try:
        if args.config:
            logging.info("Loading configuration from %s", args.config)
            success, result = run_from_file(args.config, prompts)
        else:
            logging.info("No config file provided, using base config.")
            success, result = run_from_config(base_config, prompts)

    except torch.cuda.OutOfMemoryError as oom:
        logging.error("CUDA OOM: %s", oom)
        cleanup_distributed()
        os._exit(1)

    except RuntimeError as rt:
        if "CUBLAS_STATUS_ALLOC_FAILED" in str(rt):
            logging.error("CUBLAS alloc failed, treating as OOM: %s", rt)
            cleanup_distributed()
            os._exit(1)
        else:
            raise

    except Exception as ex:
        logging.error("Unexpected error during experiment: %s", ex)
        cleanup_distributed()
        os._exit(1)

    else:
        if success:
            logging.info("Experiment completed successfully.")
        else:
            logging.error("Experiment reported failure (success=False).")

    # Final cleanup and exit
    cleanup_distributed()
    sys.exit(0)

if __name__ == "__main__":
    main()