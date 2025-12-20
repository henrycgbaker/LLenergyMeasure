"""Aggregation logic for combining raw process results."""

from __future__ import annotations

from loguru import logger

from llm_bench.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
)
from llm_bench.exceptions import AggregationError


def aggregate_results(
    experiment_id: str,
    raw_results: list[RawProcessResult],
    verify_temporal_overlap: bool = True,
    verify_gpu_attribution: bool = True,
) -> AggregatedResult:
    """Aggregate raw per-process results into a single result.

    Aggregation strategy:
    - Energy: Sum across processes (each GPU's energy is additive)
    - Tokens: Sum across processes (each process handles different data)
    - Throughput: Average (tokens_per_second / num_processes gives per-GPU rate)
    - FLOPs: Sum across processes
    - Time: Use wall-clock range (earliest start to latest end)

    Args:
        experiment_id: Unique experiment identifier.
        raw_results: List of raw results from each process.
        verify_temporal_overlap: Check that processes ran concurrently.
        verify_gpu_attribution: Check that GPU IDs are unique.

    Returns:
        Aggregated result combining all process data.

    Raises:
        AggregationError: If raw_results is empty.
    """
    if not raw_results:
        raise AggregationError("Cannot aggregate empty results list")

    warnings: list[str] = []
    num_processes = len(raw_results)

    # Verify temporal overlap
    temporal_overlap_ok = False
    if verify_temporal_overlap and num_processes > 1:
        temporal_overlap_ok = _check_temporal_overlap(raw_results)
        if not temporal_overlap_ok:
            warnings.append("Processes did not run concurrently - aggregation may be inaccurate")
            logger.warning("Temporal overlap verification failed")

    # Verify GPU attribution
    gpu_attribution_ok = False
    if verify_gpu_attribution:
        gpu_attribution_ok = _check_gpu_attribution(raw_results)
        if not gpu_attribution_ok:
            warnings.append("Duplicate GPU IDs detected - energy may be double-counted")
            logger.warning("GPU attribution verification failed")

    # Aggregate metrics
    total_tokens = sum(r.inference_metrics.total_tokens for r in raw_results)
    total_energy_j = sum(r.energy_metrics.total_energy_j for r in raw_results)
    total_inference_time = sum(r.inference_metrics.inference_time_sec for r in raw_results)
    total_flops = sum(r.compute_metrics.flops_total for r in raw_results)

    # Calculate averages
    avg_tokens_per_second = (
        sum(r.inference_metrics.tokens_per_second for r in raw_results) / num_processes
        if num_processes > 0
        else 0.0
    )

    avg_energy_per_token = total_energy_j / total_tokens if total_tokens > 0 else 0.0

    # Find time range
    start_time = min(r.timestamps.start for r in raw_results)
    end_time = max(r.timestamps.end for r in raw_results)

    # Build aggregation metadata
    metadata = AggregationMetadata(
        method="sum_energy_avg_throughput",
        num_processes=num_processes,
        temporal_overlap_verified=temporal_overlap_ok,
        gpu_attribution_verified=gpu_attribution_ok,
        warnings=warnings,
    )

    logger.info(
        f"Aggregated {num_processes} processes: "
        f"tokens={total_tokens}, energy={total_energy_j:.2f}J, "
        f"throughput={avg_tokens_per_second:.2f} tok/s"
    )

    return AggregatedResult(
        experiment_id=experiment_id,
        aggregation=metadata,
        total_tokens=total_tokens,
        total_energy_j=total_energy_j,
        total_inference_time_sec=total_inference_time,
        avg_tokens_per_second=avg_tokens_per_second,
        avg_energy_per_token_j=avg_energy_per_token,
        total_flops=total_flops,
        process_results=raw_results,
        start_time=start_time,
        end_time=end_time,
    )


def _check_temporal_overlap(results: list[RawProcessResult]) -> bool:
    """Check if process execution times overlap.

    For valid distributed execution, all processes should run concurrently.
    Returns True if there's significant overlap between all processes.
    """
    if len(results) < 2:
        return True

    # Find the intersection of all time ranges
    max_start = max(r.timestamps.start for r in results)
    min_end = min(r.timestamps.end for r in results)

    # Check if there's positive overlap
    if max_start >= min_end:
        return False

    # Check that overlap is at least 50% of the shortest process duration
    overlap_duration = (min_end - max_start).total_seconds()
    min_process_duration = min(r.timestamps.duration_sec for r in results)

    return overlap_duration >= (min_process_duration * 0.5)


def _check_gpu_attribution(results: list[RawProcessResult]) -> bool:
    """Check that GPU IDs are unique across processes.

    Duplicate GPU IDs could indicate double-counting of energy.
    """
    gpu_ids = [r.gpu_id for r in results]
    return len(gpu_ids) == len(set(gpu_ids))


def calculate_efficiency_metrics(result: AggregatedResult) -> dict[str, float]:
    """Calculate derived efficiency metrics from aggregated result.

    Returns dictionary with:
    - tokens_per_joule: Energy efficiency
    - joules_per_token: Energy cost per token
    - flops_per_joule: Compute efficiency
    - tokens_per_second: Throughput
    - effective_batch_throughput: Total throughput across all GPUs
    """
    metrics: dict[str, float] = {}

    # Energy efficiency
    if result.total_energy_j > 0:
        metrics["tokens_per_joule"] = result.total_tokens / result.total_energy_j
        metrics["flops_per_joule"] = result.total_flops / result.total_energy_j
    else:
        metrics["tokens_per_joule"] = 0.0
        metrics["flops_per_joule"] = 0.0

    # Energy cost
    if result.total_tokens > 0:
        metrics["joules_per_token"] = result.total_energy_j / result.total_tokens
    else:
        metrics["joules_per_token"] = 0.0

    # Throughput
    metrics["tokens_per_second"] = result.avg_tokens_per_second
    metrics["effective_batch_throughput"] = (
        result.avg_tokens_per_second * result.aggregation.num_processes
    )

    return metrics
