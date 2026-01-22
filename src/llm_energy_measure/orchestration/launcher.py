"""Experiment launcher utilities.

This module provides utilities for launching experiments via
accelerate CLI or torchrun with proper configuration and retry logic.

For tensor parallelism and pipeline parallelism strategies, torchrun
is used instead of accelerate launch as these require different
distributed initialisation.
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


def _get_launch_mode(config_data: dict[str, Any]) -> str:
    """Determine launch mode from backend capabilities.

    Uses the backend's RuntimeCapabilities to decide how to launch the
    experiment. This avoids hardcoded backend checks and makes adding
    new backends easier.

    Args:
        config_data: Configuration dictionary.

    Returns:
        Launch mode: "direct", "torchrun", or "accelerate".
    """
    from llm_energy_measure.core.inference_backends import get_backend
    from llm_energy_measure.core.inference_backends.protocols import LaunchMode

    backend_name = str(config_data.get("backend", "pytorch"))

    try:
        backend = get_backend(backend_name)
        capabilities = backend.get_runtime_capabilities()

        # Backend-declared launch mode takes priority
        if capabilities.launch_mode == LaunchMode.DIRECT:
            return "direct"

        # Check for sharding strategies that may need torchrun
        sharding = config_data.get("sharding", {})
        strategy = sharding.get("strategy", "none")

        if strategy in ("tensor_parallel", "pipeline_parallel"):
            # For sharding strategies, use torchrun for distributed setup
            # (DIRECT launch mode backends were already handled above)
            return "torchrun"

        # Use backend's declared launch mode
        if capabilities.launch_mode == LaunchMode.TORCHRUN:
            return "torchrun"
        return "accelerate"

    except Exception as e:
        # Fall back to safe defaults if backend can't be loaded
        logger.warning(f"Could not get backend capabilities for {backend_name}: {e}")
        if backend_name == "vllm":
            return "direct"
        return "accelerate"


def _requires_torchrun(config_data: dict[str, Any]) -> bool:
    """Check if the experiment requires torchrun launcher.

    DEPRECATED: Use _get_launch_mode() instead. Kept for backwards compatibility.
    """
    return _get_launch_mode(config_data) == "torchrun"


def _requires_direct_launch(config_data: dict[str, Any]) -> bool:
    """Check if the backend requires direct Python execution (no launcher).

    DEPRECATED: Use _get_launch_mode() instead. Kept for backwards compatibility.
    """
    return _get_launch_mode(config_data) == "direct"


def get_effective_launcher_processes(config: ExperimentConfig) -> int:
    """Determine actual launcher process count based on backend capabilities.

    For backends with internal parallelism (vLLM, TensorRT-LLM):
    - tensor_parallel/pipeline_parallel: Returns 1 (backend manages internally)
    - data_parallel: Returns parallelism.degree (separate processes needed)

    For PyTorch with Accelerate, uses parallelism.degree explicitly.

    This function is critical for:
    - ExperimentState: determines how many result files to expect
    - Aggregation: knows when all processes have completed

    Args:
        config: Experiment configuration.

    Returns:
        Number of launcher processes (and expected result files).
    """
    from llm_energy_measure.core.inference_backends import get_backend
    from llm_energy_measure.core.inference_backends.protocols import LaunchMode

    try:
        backend = get_backend(config.backend)
        capabilities = backend.get_runtime_capabilities()

        if capabilities.launch_mode == LaunchMode.DIRECT:
            # Backend manages tensor/pipeline parallelism internally
            # But data parallelism requires multiple separate processes
            if config.parallelism.strategy == "data_parallel":
                return config.parallelism.degree
            # tensor_parallel/pipeline_parallel: single process, backend spawns workers
            return 1
        else:
            # External parallelism via Accelerate/torchrun
            # Use explicit parallelism.degree from config
            return config.parallelism.degree

    except Exception as e:
        # Fall back to safe defaults if backend can't be loaded
        logger.warning(f"Could not get backend capabilities for {config.backend}: {e}")
        if config.backend in ("vllm", "tensorrt"):
            if config.parallelism.strategy == "data_parallel":
                return config.parallelism.degree
            return 1
        return config.parallelism.degree


def _build_launch_command(
    config_data: dict[str, Any],
    script_path: str | Path,
    config_path: Path,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the appropriate launch command based on backend capabilities.

    Uses RuntimeCapabilities from the backend to determine the correct
    launch mechanism, avoiding hardcoded backend checks.

    Args:
        config_data: Configuration dictionary.
        script_path: Path to the experiment script.
        config_path: Path to the config file.
        extra_args: Additional arguments to pass to the script.

    Returns:
        Complete command list for subprocess.run().
    """
    gpus = config_data.get("gpus", [])
    launch_mode = _get_launch_mode(config_data)

    if launch_mode == "direct":
        # Backend manages its own multiprocessing (e.g., vLLM, TensorRT-LLM)
        import sys

        cmd = [
            sys.executable,
            "-m",
            "llm_energy_measure.orchestration.launcher",
            "--config",
            str(config_path),
        ]
        backend = config_data.get("backend", "pytorch")
        logger.info(f"Using direct launch for {backend} backend (manages own CUDA context)")

    elif launch_mode == "torchrun":
        # Tensor/Pipeline parallelism: use torchrun
        sharding = config_data.get("sharding", {})
        num_shards = sharding.get("num_shards", len(gpus))

        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(num_shards),
            "--standalone",  # Single-node mode
            str(script_path),
            "--config",
            str(config_path),
        ]
        logger.info(f"Using torchrun for {sharding.get('strategy')} with {num_shards} processes")

    else:
        # Default: use accelerate launch
        num_processes = min(
            config_data.get("num_processes", len(gpus)),
            len(gpus),
        )

        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_processes),
            str(script_path),
            "--config",
            str(config_path),
        ]
        logger.info(f"Using accelerate launch with {num_processes} processes")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def launch_experiment_accelerate(
    config: dict[str, Any] | str | Path,
    script_path: str | Path,
    max_retries: int = 3,
    retry_delay: int = 5,
    extra_args: list[str] | None = None,
) -> None:
    """Launch an experiment using accelerate CLI or torchrun.

    For tensor parallelism and pipeline parallelism strategies, torchrun
    is used instead of accelerate launch.

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

    gpus = config_data.get("gpus", [])
    num_processes = min(config_data.get("num_processes", len(gpus)), len(gpus))

    if num_processes > len(gpus):
        logger.warning(
            f"num_processes ({num_processes}) exceeds GPUs ({len(gpus)}). "
            f"Using {len(gpus)} processes."
        )
        num_processes = len(gpus)

    attempt = 0
    last_error = ""

    while attempt < max_retries:
        attempt += 1
        logger.info(f"Launching experiment attempt {attempt}/{max_retries}")

        try:
            env = os.environ.copy()
            # Only set CUDA_VISIBLE_DEVICES if not already set to MIG/GPU UUIDs
            # MIG instances must be addressed by UUID, not integer index
            existing_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if "MIG-" in existing_cuda_env or "GPU-" in existing_cuda_env:
                # User has set MIG/GPU UUIDs - respect that
                uuid_count = len(existing_cuda_env.split(","))
                logger.info(f"Using CUDA_VISIBLE_DEVICES UUIDs: {uuid_count} device(s)")
                if len(gpus) != uuid_count:
                    logger.warning(
                        f"gpus config has {len(gpus)} entries but CUDA_VISIBLE_DEVICES "
                        f"has {uuid_count} UUIDs. Using CUDA_VISIBLE_DEVICES."
                    )
            else:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

            # Set environment based on launcher type
            if _requires_direct_launch(config_data):
                # vLLM: direct launch with multi-GPU NCCL fix
                # Many PCIe-connected GPUs (without NVLink) fail NCCL P2P communication
                # This is safe to set unconditionally - NVLink systems ignore it
                parallelism = config_data.get("parallelism", {})
                sharding = config_data.get("sharding", {})
                # Check both new (parallelism.degree) and legacy (sharding.num_shards)
                tp_degree = parallelism.get("degree", sharding.get("num_shards", 1))
                if tp_degree > 1 and "NCCL_P2P_DISABLE" not in env:
                    env["NCCL_P2P_DISABLE"] = "1"
                    logger.info(
                        f"Set NCCL_P2P_DISABLE=1 for vLLM tensor parallelism (tp={tp_degree})"
                    )
            elif _requires_torchrun(config_data):
                # torchrun handles distributed setup
                pass
            else:
                env["ACCELERATE_NUM_PROCESSES"] = str(num_processes)
                env["ACCELERATE_CONFIG_FILE"] = ""

            cmd = _build_launch_command(config_data, script_path, config_path, extra_args)

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


def _extract_metadata(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str | None,
    list[str],
    str | None,
    dict[str, dict[str, Any]],
    list[str],
]:
    """Extract _metadata section from config file if present.

    Args:
        config_path: Path to the config file.

    Returns:
        Tuple of (effective_config, cli_overrides, experiment_id, config_warnings,
                  results_dir, parameter_provenance, preset_chain).
    """
    import yaml

    try:
        with open(config_path) as f:
            raw_config = yaml.safe_load(f) or {}

        metadata = raw_config.get("_metadata", {})
        effective_config = metadata.get("effective_config", {})
        cli_overrides = metadata.get("cli_overrides", {})
        experiment_id = metadata.get("experiment_id")
        config_warnings = metadata.get("config_warnings", [])
        results_dir = metadata.get("results_dir")
        parameter_provenance = metadata.get("parameter_provenance", {})
        preset_chain = metadata.get("preset_chain", [])

        return (
            effective_config,
            cli_overrides,
            experiment_id,
            config_warnings,
            results_dir,
            parameter_provenance,
            preset_chain,
        )
    except Exception as e:
        logger.debug(f"Could not extract _metadata from config: {e}")
        return {}, {}, None, [], None, {}, []


def _parse_args() -> (
    tuple[
        Path,
        list[str],
        dict[str, Any],
        dict[str, Any],
        str | None,
        list[str],
        str | None,
        dict[str, dict[str, Any]],
        list[str],
    ]
):
    """Parse command line arguments for accelerate launch.

    Returns:
        Tuple of (config_path, prompts, effective_config, cli_overrides, experiment_id,
                  config_warnings, results_dir, parameter_provenance, preset_chain).
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

    # Extract metadata before loading config (Phase 0)
    (
        effective_config,
        cli_overrides,
        experiment_id,
        config_warnings,
        results_dir,
        parameter_provenance,
        preset_chain,
    ) = _extract_metadata(args.config)

    # Load config (ExperimentConfig ignores _metadata field)
    config = load_config(args.config)

    # If effective_config wasn't provided via _metadata, use config dict
    if not effective_config:
        effective_config = config.model_dump()

    # Determine sample size: CLI -n takes precedence over everything
    cli_sample_size = args.sample_size  # None if not specified

    # Load prompts (CLI args > config > default)
    if args.dataset:
        source = HuggingFacePromptSource(
            dataset=args.dataset,
            split=args.split,
            column=args.column,
            sample_size=cli_sample_size or config.num_input_prompts,
        )
        prompts = load_prompts_from_source(source)
    elif args.prompts:
        prompts = load_prompts_from_file(args.prompts)
        effective_sample_size = cli_sample_size or config.num_input_prompts
        if effective_sample_size:
            prompts = prompts[:effective_sample_size]
    elif config.dataset:
        # Simple dataset config (recommended approach)
        # Priority: CLI -n > config.dataset.sample_size
        source = HuggingFacePromptSource(
            dataset=config.dataset.name,
            split=config.dataset.split,
            column=config.dataset.column,
            sample_size=cli_sample_size or config.dataset.sample_size,
        )
        prompts = load_prompts_from_source(source)
    elif config.prompts:
        # Override sample_size with CLI value if provided (CLI > source default)
        cfg_source = config.prompts
        if cli_sample_size and isinstance(cfg_source, HuggingFacePromptSource):
            cfg_source = HuggingFacePromptSource(
                dataset=cfg_source.dataset,
                split=cfg_source.split,
                column=cfg_source.column,
                sample_size=cli_sample_size,
            )
        prompts = load_prompts_from_source(cfg_source)
    else:
        prompts = ["Hello, how are you?"]
        logger.warning("No prompt source specified, using default prompt")

    return (
        args.config,
        prompts,
        effective_config,
        cli_overrides,
        experiment_id,
        config_warnings,
        results_dir,
        parameter_provenance,
        preset_chain,
    )


if __name__ == "__main__":
    import os
    import sys

    # ==========================================================================
    # EARLY NCCL FIX: Must be set BEFORE any vLLM/torch imports
    # ==========================================================================
    # Many PCIe-connected GPUs (without NVLink) fail NCCL P2P communication.
    # This is safe to set unconditionally - NVLink systems just ignore it.
    # We check if this is a vLLM multi-GPU run by parsing minimal config here.
    def _early_nccl_fix() -> None:
        """Set NCCL_P2P_DISABLE=1 for vLLM tensor parallelism before imports."""
        if "NCCL_P2P_DISABLE" in os.environ:
            return  # Already set by parent or user

        # Quick parse of config to check if NCCL fix needed
        # We do this before imports to avoid triggering vLLM init
        config_arg = None
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_arg = sys.argv[i + 1]
                break

        if not config_arg:
            return

        try:
            import yaml

            with open(config_arg) as f:
                config_data = yaml.safe_load(f)

            backend = config_data.get("backend", "pytorch")
            if backend not in ("vllm", "tensorrt"):
                return  # NCCL fix only needed for these backends

            parallelism = config_data.get("parallelism", {})
            sharding = config_data.get("sharding", {})
            tp_degree = parallelism.get("degree", sharding.get("num_shards", 1))

            if tp_degree > 1:
                os.environ["NCCL_P2P_DISABLE"] = "1"
                # Can't use logger yet, print directly
                print(
                    f"[launcher] Set NCCL_P2P_DISABLE=1 for {backend} "
                    f"tensor parallelism (tp={tp_degree})",
                    file=sys.stderr,
                )
        except Exception:
            pass  # Silently ignore - config will be validated later anyway

    _early_nccl_fix()
    # ==========================================================================

    from pathlib import Path as PathLib

    from llm_energy_measure.config.loader import load_config
    from llm_energy_measure.logging import setup_logging
    from llm_energy_measure.orchestration.context import experiment_context
    from llm_energy_measure.orchestration.factory import create_orchestrator

    # Configure logging for subprocess (inherits LLM_ENERGY_VERBOSITY from parent)
    setup_logging()

    (
        config_path,
        prompts,
        effective_config,
        cli_overrides,
        experiment_id,
        config_warnings,
        results_dir,
        parameter_provenance,
        preset_chain,
    ) = _parse_args()
    config = load_config(config_path)

    # Log experiment configuration summary
    num_procs = get_effective_launcher_processes(config)
    logger.info(f"Running experiment: {config.config_name}")
    logger.info(f"  Model: {config.model_name} | Backend: {config.backend}")
    logger.info(
        f"  GPUs: {config.gpus} | Processes: {num_procs} | Precision: {config.fp_precision}"
    )
    logger.info(
        f"  Batch size: {config.batching.batch_size} | "
        f"Strategy: {config.batching.strategy} | "
        f"Streaming: {config.streaming}"
    )
    if config.parallelism.strategy != "none":
        logger.info(
            f"  Parallelism: {config.parallelism.strategy} (degree={config.parallelism.degree})"
        )
    logger.info(f"  Prompts: {len(prompts)} | Max output tokens: {config.max_output_tokens}")
    if results_dir:
        logger.info(f"  Results dir: {results_dir}")

    with experiment_context(
        config,
        effective_config=effective_config,
        cli_overrides=cli_overrides,
        experiment_id=experiment_id,
        config_warnings=config_warnings,
        parameter_provenance=parameter_provenance,
        preset_chain=preset_chain,
    ) as ctx:
        # Pass results_dir to orchestrator (None uses default from constants/env)
        results_path = PathLib(results_dir) if results_dir else None
        orchestrator = create_orchestrator(ctx, results_dir=results_path)
        result_path = orchestrator.run(ctx, prompts)

        logger.info(f"Experiment {ctx.experiment_id} complete")
        logger.info(f"Result saved to: {result_path}")
