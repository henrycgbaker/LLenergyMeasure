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

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.exceptions import ConfigurationError

# =============================================================================
# Backend-Native Parallelism Helpers
# =============================================================================


def get_launcher_process_count(config: ExperimentConfig) -> int:
    """Get number of launcher processes needed for this backend.

    - PyTorch: config.pytorch.num_processes (Accelerate manages data parallelism)
    - vLLM: 1 (manages tensor/pipeline parallelism internally)
    - TensorRT: 1 (manages tensor/pipeline parallelism internally)

    Returns:
        Number of launcher processes to spawn.
    """
    if config.backend == "pytorch":
        return config.pytorch.num_processes if config.pytorch else 1
    # vLLM and TensorRT manage parallelism internally
    return 1


def get_backend_batching(config: ExperimentConfig) -> tuple[int, str]:
    """Extract batch size and strategy from backend-specific config.

    Each backend stores batching differently:
    - PyTorch: batch_size + batching_strategy
    - vLLM: max_num_seqs (continuous batching, strategy is implicit)
    - TensorRT: max_batch_size (compile-time, strategy is implicit)

    Returns:
        Tuple of (batch_size, strategy).
    """
    if config.backend == "pytorch" and config.pytorch:
        return (config.pytorch.batch_size, config.pytorch.batching_strategy)

    elif config.backend == "vllm" and config.vllm:
        # vLLM uses continuous batching, max_num_seqs is like "batch capacity"
        return (config.vllm.max_num_seqs, "continuous")

    elif config.backend == "tensorrt" and config.tensorrt:
        # TensorRT uses inflight batching
        return (config.tensorrt.max_batch_size, "inflight")

    # Default
    return (1, "static")


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
    from llenergymeasure.core.inference_backends import get_backend
    from llenergymeasure.core.inference_backends.protocols import LaunchMode

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
    """Determine actual launcher process count based on backend.

    This is critical for ExperimentState and aggregation to know
    how many result files to expect.

    Args:
        config: Experiment configuration.

    Returns:
        Number of launcher processes (and expected result files).
    """
    return get_launcher_process_count(config)


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
            "llenergymeasure.orchestration.launcher",
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
        # Determine process count from backend-specific parallelism config
        try:
            config = ExperimentConfig(**config_data)
            num_processes = get_effective_launcher_processes(config)
        except Exception as e:
            # Fallback: use GPU count if config parsing fails
            logger.warning(f"Could not determine parallelism from config: {e}. Using len(gpus).")
            num_processes = len(gpus) if gpus else 1

        # Cap at available GPU count
        if num_processes > len(gpus):
            logger.warning(
                f"Parallelism degree ({num_processes}) exceeds available GPUs ({len(gpus)}). "
                f"Using {len(gpus)} processes."
            )
            num_processes = len(gpus)

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

    # Determine process count from backend-specific parallelism config
    try:
        exp_config = ExperimentConfig(**config_data)
        num_processes = get_effective_launcher_processes(exp_config)
    except Exception as e:
        # Fallback: use GPU count if config parsing fails
        logger.warning(f"Could not determine parallelism from config: {e}. Using len(gpus).")
        num_processes = len(gpus) if gpus else 1

    # Cap at available GPU count
    if num_processes > len(gpus):
        logger.warning(
            f"Parallelism degree ({num_processes}) exceeds GPUs ({len(gpus)}). "
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
                gpu_str = ",".join(str(g) for g in gpus)
                env["CUDA_VISIBLE_DEVICES"] = gpu_str
                # For local execution (not in container), propagate NVIDIA_VISIBLE_DEVICES
                # for subprocess workers that may spawn further processes
                if "NVIDIA_VISIBLE_DEVICES" not in env:
                    env["NVIDIA_VISIBLE_DEVICES"] = gpu_str

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
    from llenergymeasure.orchestration.context import experiment_context
    from llenergymeasure.orchestration.lifecycle import ensure_clean_start

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

    from llenergymeasure.config.loader import load_config
    from llenergymeasure.config.models import HuggingFacePromptSource
    from llenergymeasure.core.dataset_loader import (
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

    # ==========================================================================
    # EARLY CUDA_VISIBLE_DEVICES: Must be set BEFORE any torch/CUDA imports
    # ==========================================================================
    # On shared servers where host has CUDA_VISIBLE_DEVICES="", we must set
    # CUDA_VISIBLE_DEVICES from config.gpus BEFORE any CUDA initialization.
    # This ensures subprocess workers can see the correct GPUs.
    def _early_cuda_visible_devices_setup() -> None:
        """Set CUDA_VISIBLE_DEVICES from config.gpus before any CUDA init.

        Handles two contexts:
        1. Container context: NVIDIA_VISIBLE_DEVICES set by runtime, GPUs are remapped
           to 0,1,2,... inside container. Use remapped indices.
        2. Local context: No NVIDIA_VISIBLE_DEVICES, use config.gpus directly.
        """
        # Quick parse of config to get GPU list
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

            gpus = config_data.get("gpus", [])
            if not gpus:
                return  # No GPUs specified, skip

            # Only set if not already set to MIG/GPU UUIDs
            existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if "MIG-" in existing or "GPU-" in existing:
                return  # Respect existing MIG/UUID configuration

            # Check if we're inside a container (NVIDIA_VISIBLE_DEVICES was set by runtime)
            nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")

            # If NVIDIA_VISIBLE_DEVICES is set and not "all", we're in a container
            # with specific GPUs mounted - use remapped indices (0,1,2,...)
            if nvidia_visible and nvidia_visible != "all":
                # Container has specific GPUs mounted, use 0-based remapped indices
                gpu_count = len(nvidia_visible.split(","))
                cuda_devices = ",".join(str(i) for i in range(gpu_count))
                in_container = True
            else:
                # Local execution or "all" GPUs - use config.gpus directly
                cuda_devices = ",".join(str(g) for g in gpus)
                in_container = False

            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            # Can't use logger yet, print directly
            print(
                f"[launcher] Set CUDA_VISIBLE_DEVICES={cuda_devices} "
                f"(container={in_container})",
                file=sys.stderr,
            )
        except Exception as e:
            # Silently continue - config will be validated later anyway
            print(
                f"[launcher] Warning: Could not set CUDA_VISIBLE_DEVICES early: {e}",
                file=sys.stderr,
            )

    _early_cuda_visible_devices_setup()
    # ==========================================================================

    from pathlib import Path as PathLib

    from llenergymeasure.config.loader import load_config
    from llenergymeasure.logging import setup_logging
    from llenergymeasure.orchestration.context import experiment_context
    from llenergymeasure.orchestration.factory import create_orchestrator

    # Configure logging for subprocess (inherits LLM_ENERGY_VERBOSITY from parent)
    setup_logging()

    # If parent process specified a log file, add file handler to capture subprocess logs
    if log_file := os.environ.get("LLM_ENERGY_LOG_FILE"):
        from llenergymeasure.logging import VERBOSE_FORMAT
        from llenergymeasure.logging import logger as loguru_logger

        loguru_logger.add(
            log_file,
            format=VERBOSE_FORMAT,
            level="DEBUG",
            rotation="50 MB",
            retention="7 days",
            enqueue=True,
        )

    # Log campaign context if running as part of a campaign
    # Note: Experiments are atomic - cycles are campaign-level only
    campaign_id = os.environ.get("LEM_CAMPAIGN_ID")
    if campaign_id:
        campaign_name = os.environ.get("LEM_CAMPAIGN_NAME", "unknown")
        logger.info(f"Running as part of campaign '{campaign_name}'")

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
    batch_size, batch_strategy = get_backend_batching(config)
    logger.info(
        f"  Batch size: {batch_size} | "
        f"Strategy: {batch_strategy} | "
        f"Streaming: {config.streaming}"
    )
    # Log backend-native parallelism if configured
    if config.backend == "pytorch" and config.pytorch and config.pytorch.num_processes > 1:
        logger.info(f"  Parallelism: data_parallel (degree={config.pytorch.num_processes})")
    elif config.backend == "vllm" and config.vllm:
        if config.vllm.tensor_parallel_size > 1:
            logger.info(
                f"  Parallelism: tensor_parallel (degree={config.vllm.tensor_parallel_size})"
            )
    elif config.backend == "tensorrt" and config.tensorrt and config.tensorrt.tp_size > 1:
        logger.info(f"  Parallelism: tensor_parallel (degree={config.tensorrt.tp_size})")
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
