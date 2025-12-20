"""Compute metrics collection (FLOPs, memory, utilization)."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psutil
import torch
from loguru import logger

from llm_energy_measure.domain.metrics import ComputeMetrics, FlopsResult

if TYPE_CHECKING:
    from accelerate import Accelerator

    from llm_energy_measure.config.models import ExperimentConfig


@dataclass
class MemoryStats:
    """GPU memory statistics."""

    current_allocated_bytes: int
    max_allocated_bytes: int
    current_reserved_bytes: int
    max_reserved_bytes: int


@dataclass
class UtilizationStats:
    """CPU and GPU utilization statistics."""

    gpu_utilization_percent: list[float] | None
    cpu_usage_percent: float
    cpu_memory_bytes: int


def get_flops_for_sample(
    model: Any,
    sample_length: int,
    device: torch.device,
) -> float | None:
    """Calculate FLOPs for a single input sample.

    Args:
        model: The model to measure.
        sample_length: Length of the input sequence.
        device: Device to run on.

    Returns:
        FLOPs count or None if calculation fails.
    """
    try:
        import ptflops
    except ImportError:
        logger.warning("ptflops not installed, skipping FLOPs calculation")
        return None

    def input_constructor(input_res: tuple[int, ...]) -> dict[str, torch.Tensor]:
        dummy_input = torch.zeros((1, *input_res), dtype=torch.long).to(device)
        attention_mask = torch.ones_like(dummy_input)
        return {"input_ids": dummy_input, "attention_mask": attention_mask}

    # Suppress ptflops output
    with (
        io.StringIO() as buf,
        contextlib.redirect_stdout(buf),
        contextlib.redirect_stderr(buf),
    ):
        try:
            flops, _ = ptflops.get_model_complexity_info(
                model,
                input_res=(sample_length,),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
                input_constructor=input_constructor,
            )
            return float(flops) if flops is not None else None
        except Exception as e:
            logger.debug(f"FLOPs calculation failed: {e}")
            return None


def get_flops(
    model: Any,
    input_ids: torch.Tensor,
    timeout_per_sample: int = 10,
) -> float | None:
    """Compute total FLOPs for a batch of tokenized inputs.

    If all samples have the same length, computes FLOPs once and multiplies.
    Otherwise falls back to per-sample computation with timeout protection.

    Args:
        model: The model to measure.
        input_ids: Batch of tokenized inputs.
        timeout_per_sample: Timeout in seconds per sample.

    Returns:
        Total FLOPs or None if calculation fails.
    """
    batch_size = input_ids.shape[0]
    sample_lengths = [input_ids[i].shape[0] for i in range(batch_size)]

    # Optimization: if all same length, compute once and multiply
    if len(set(sample_lengths)) == 1:
        logger.debug(f"All samples have length {sample_lengths[0]}, computing once")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_flops_for_sample,
                    model,
                    sample_lengths[0],
                    input_ids.device,
                )
                flops_single = future.result(timeout=timeout_per_sample)

            if flops_single is not None:
                return float(flops_single * batch_size)
        except TimeoutError:
            logger.warning("FLOPs calculation timed out")
        except Exception as e:
            logger.warning(f"FLOPs calculation failed: {e}")
        return None

    # Fallback: compute each sample individually
    total_flops = 0.0
    for i in range(batch_size):
        sample_length = input_ids[i].shape[0]
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_flops_for_sample,
                    model,
                    sample_length,
                    input_ids.device,
                )
                flops_single = future.result(timeout=timeout_per_sample)

            if flops_single is not None:
                total_flops += flops_single
        except TimeoutError:
            logger.debug(f"FLOPs calculation timed out for sample {i}")
        except Exception as e:
            logger.debug(f"FLOPs calculation failed for sample {i}: {e}")

    return total_flops if total_flops > 0 else None


def get_memory_stats(device: torch.device) -> MemoryStats:
    """Get GPU memory statistics.

    Args:
        device: CUDA device to query.

    Returns:
        MemoryStats with current and peak memory usage.
    """
    torch.cuda.reset_peak_memory_stats(device)

    return MemoryStats(
        current_allocated_bytes=torch.cuda.memory_allocated(device),
        max_allocated_bytes=torch.cuda.max_memory_allocated(device),
        current_reserved_bytes=torch.cuda.memory_reserved(device),
        max_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def get_gpu_utilization() -> list[float] | None:
    """Query GPU utilization via nvidia-smi.

    Returns:
        List of utilization percentages per GPU, or None on error.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        lines = result.decode("utf-8").strip().splitlines()
        return [float(line.strip()) for line in lines if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        logger.debug(f"Failed to get GPU utilization: {e}")
        return None


def get_utilization_stats() -> UtilizationStats:
    """Get CPU and GPU utilization statistics.

    Returns:
        UtilizationStats with current utilization info.
    """
    gpu_util = get_gpu_utilization()

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss
    except Exception as e:
        logger.debug(f"Failed to get CPU stats: {e}")
        cpu_percent = 0.0
        cpu_memory = 0

    return UtilizationStats(
        gpu_utilization_percent=gpu_util,
        cpu_usage_percent=cpu_percent,
        cpu_memory_bytes=cpu_memory,
    )


def collect_compute_metrics(
    model: Any,
    device: torch.device,
    input_ids: torch.Tensor,
    accelerator: Accelerator,
    cached_flops: float | None = None,
    config: ExperimentConfig | None = None,
    use_estimator: bool = True,
) -> ComputeMetrics:
    """Collect all compute-related metrics.

    FLOPs are only computed on the main process to avoid duplication.
    If cached_flops is provided (for quantized models), that value is used.

    Args:
        model: The model being measured.
        device: CUDA device.
        input_ids: Tokenized input tensor.
        accelerator: Accelerator for process coordination.
        cached_flops: Pre-computed FLOPs value (e.g., for quantized models).
        config: Experiment config for precision detection (used by FlopsEstimator).
        use_estimator: If True, use FlopsEstimator with fallback chain.
                      If False, use legacy ptflops-only approach.

    Returns:
        ComputeMetrics with FLOPs and optionally memory/utilization data.
    """
    flops_result: FlopsResult | None = None
    flops = 0.0
    method = "unknown"
    confidence = "unknown"
    precision = "fp16"

    if accelerator.is_main_process:
        if cached_flops is not None:
            flops = cached_flops
            method = "cached"
            confidence = "high"
            logger.debug(f"Using cached FLOPs: {flops}")
        elif use_estimator:
            # Use new FlopsEstimator with fallback chain
            from llm_energy_measure.core.flops import estimate_flops

            flops_result = estimate_flops(model, input_ids, config)
            flops = flops_result.value
            method = flops_result.method
            confidence = flops_result.confidence
            precision = flops_result.precision
        else:
            # Legacy ptflops-only approach
            computed = get_flops(model, input_ids)
            if computed is not None:
                flops = computed
                method = "ptflops"
                confidence = "medium"
            else:
                logger.warning("FLOPs computation failed, using 0.0")

    memory = get_memory_stats(device)
    _utilization = get_utilization_stats()  # Collected for future use

    logger.debug(
        f"Compute metrics: FLOPs={flops:.2e} ({method}/{confidence}), "
        f"GPU mem={memory.current_allocated_bytes / 1e9:.2f}GB, "
        f"CPU={_utilization.cpu_usage_percent:.1f}%"
    )

    return ComputeMetrics(
        flops_total=flops,
        flops_per_token=0.0,
        flops_per_second=0.0,
        peak_memory_mb=memory.max_allocated_bytes / (1024 * 1024),
        model_memory_mb=0.0,
        flops_method=method,
        flops_confidence=confidence,
        compute_precision=precision,
    )
