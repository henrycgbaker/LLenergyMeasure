"""Experiment launcher utilities.

This module provides utilities for launching experiments via
accelerate CLI with proper configuration and retry logic.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError


def get_config_file_path(config: dict[str, Any] | str | Path) -> Path:
    """Get or create a config file path.

    If config is a dict, writes it to a temp file.
    If config is a path, returns it directly.

    Args:
        config: Configuration dict or path to config file.

    Returns:
        Path to the config file.

    Raises:
        ConfigurationError: If config type is invalid.
    """
    if isinstance(config, dict):
        # Create temp file, write config, return path
        fd, tmp_path = tempfile.mkstemp(suffix=".json", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config, f, indent=2)
        except Exception:
            os.close(fd)
            raise
        return Path(tmp_path)
    elif isinstance(config, str | Path):
        return Path(config)
    else:
        raise ConfigurationError(f"Config must be dict or path, got {type(config)}")


def log_failed_experiment(
    experiment_id: str,
    config: dict[str, Any],
    error_message: str,
    output_file: Path | str = "failed_experiments.csv",
) -> None:
    """Log a failed experiment to CSV.

    Args:
        experiment_id: The experiment identifier.
        config: The experiment configuration dict.
        error_message: Description of the failure.
        output_file: Path to the CSV log file.
    """
    output_path = Path(output_file)
    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as csvfile:
        fieldnames = ["experiment_id", "timestamp", "suite", "config", "error_message"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "suite": config.get("suite", "unknown"),
                "config": json.dumps(config),
                "error_message": error_message,
            }
        )

    logger.info(f"Logged failed experiment {experiment_id} to {output_path}")


def launch_experiment_accelerate(
    config: dict[str, Any] | str | Path,
    script_path: str | Path,
    max_retries: int = 3,
    retry_delay: int = 5,
    extra_args: list[str] | None = None,
) -> None:
    """Launch an experiment using accelerate CLI.

    Args:
        config: Configuration dict or path to config file.
        script_path: Path to the experiment script to launch.
        max_retries: Maximum number of retry attempts.
        retry_delay: Seconds to wait between retries.
        extra_args: Additional arguments to pass to the script.

    Raises:
        RuntimeError: If experiment fails after all retries.
    """
    config_path = get_config_file_path(config)
    config_data = json.loads(config_path.read_text())

    gpu_list = config_data.get("gpu_list", [])
    num_processes = min(config_data.get("num_processes", len(gpu_list)), len(gpu_list))

    if num_processes > len(gpu_list):
        logger.warning(
            f"num_processes ({num_processes}) exceeds GPUs ({len(gpu_list)}). "
            f"Using {len(gpu_list)} processes."
        )
        num_processes = len(gpu_list)

    attempt = 0
    last_error = ""

    while attempt < max_retries:
        attempt += 1
        logger.info(f"Launching experiment attempt {attempt}/{max_retries}")

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
            env["ACCELERATE_NUM_PROCESSES"] = str(num_processes)
            env["ACCELERATE_CONFIG_FILE"] = ""

            cmd = [
                "accelerate",
                "launch",
                "--num_processes",
                str(num_processes),
                str(script_path),
                "--config",
                str(config_path),
            ]
            if extra_args:
                cmd.extend(extra_args)

            logger.info(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"Experiment succeeded on attempt {attempt}")
            return

        except subprocess.CalledProcessError as e:
            last_error = f"Attempt {attempt}: {e}"
            logger.error(f"Experiment failed: {last_error}")
            log_failed_experiment(
                config_data.get("experiment_id", "unknown"),
                config_data,
                last_error,
                "failed_attempts.csv",
            )
            time.sleep(retry_delay)

    logger.error(f"Experiment failed after {max_retries} attempts")
    log_failed_experiment(
        config_data.get("experiment_id", "unknown"),
        config_data,
        last_error,
    )
    raise RuntimeError(f"Experiment failed after {max_retries} attempts: {last_error}")


def run_from_config(
    config_data: dict[str, Any],
    prompts: list[str],
    max_retries: int = 1,
    retry_delay: int = 1,
) -> tuple[bool, object | None]:
    """Run an experiment directly from config data.

    This is for in-process execution, not subprocess launching.

    Args:
        config_data: Configuration dictionary.
        prompts: List of prompts to process.
        max_retries: Maximum retry attempts.
        retry_delay: Seconds between retries.

    Returns:
        Tuple of (success, result).
    """
    from llm_energy_measure.orchestration.context import experiment_context
    from llm_energy_measure.orchestration.lifecycle import ensure_clean_start

    config = ExperimentConfig(**config_data)

    for attempt in range(1, max_retries + 1):
        logger.info(f"Starting experiment attempt {attempt}/{max_retries}")

        try:
            ensure_clean_start()

            with experiment_context(config) as ctx:
                # This would need actual implementations injected
                # For now, returns success placeholder
                logger.info(f"Experiment {ctx.experiment_id} started")
                return True, ctx.experiment_id

        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)

    logger.error("All retry attempts exhausted")
    return False, None


def _parse_args() -> tuple[Path, list[str]]:
    """Parse command line arguments for accelerate launch.

    Returns:
        Tuple of (config_path, prompts).
    """
    import argparse

    from llm_energy_measure.config.loader import load_config
    from llm_energy_measure.config.models import HuggingFacePromptSource
    from llm_energy_measure.core.dataset_loader import (
        load_prompts_from_file,
        load_prompts_from_source,
    )

    parser = argparse.ArgumentParser(description="Run LLM inference experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name or alias")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--column", type=str, help="Dataset column for prompts")
    parser.add_argument("--prompts", type=Path, help="Path to prompts file")
    parser.add_argument("--sample-size", type=int, help="Limit number of prompts")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load prompts (CLI args > config > default)
    if args.dataset:
        source = HuggingFacePromptSource(
            dataset=args.dataset,
            split=args.split,
            column=args.column,
            sample_size=args.sample_size,
        )
        prompts = load_prompts_from_source(source)
    elif args.prompts:
        prompts = load_prompts_from_file(args.prompts)
        if args.sample_size:
            prompts = prompts[: args.sample_size]
    elif config.prompt_source:
        # Override sample_size from CLI if provided
        cfg_source = config.prompt_source
        if args.sample_size and isinstance(cfg_source, HuggingFacePromptSource):
            cfg_source = HuggingFacePromptSource(
                dataset=cfg_source.dataset,
                split=cfg_source.split,
                column=cfg_source.column,
                sample_size=args.sample_size,
            )
        prompts = load_prompts_from_source(cfg_source)
    else:
        prompts = ["Hello, how are you?"]
        logger.warning("No prompt source specified, using default prompt")

    return args.config, prompts


if __name__ == "__main__":
    from llm_energy_measure.config.loader import load_config
    from llm_energy_measure.orchestration.context import experiment_context
    from llm_energy_measure.orchestration.factory import create_orchestrator

    config_path, prompts = _parse_args()
    config = load_config(config_path)

    logger.info(f"Running experiment with {len(prompts)} prompts from config: {config_path}")
    logger.info(f"First prompt: {prompts[0][:50]}...")

    with experiment_context(config) as ctx:
        orchestrator = create_orchestrator(ctx)
        result_path = orchestrator.run(ctx, prompts)

        logger.info(f"Experiment {ctx.experiment_id} complete")
        logger.info(f"Result saved to: {result_path}")
