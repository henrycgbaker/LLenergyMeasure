"""Aggregation logic for combining raw process results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from llm_energy_measure.constants import COMPLETION_MARKER_PREFIX
from llm_energy_measure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
)
from llm_energy_measure.exceptions import AggregationError


class CompletenessReport(BaseModel):
    """Report on process completion status."""

    expected_processes: int = Field(..., description="Expected number of processes")
    found_processes: int = Field(..., description="Number of results found")
    missing_indices: list[int] = Field(default_factory=list, description="Missing process indices")
    duplicate_indices: list[int] = Field(
        default_factory=list, description="Duplicate process indices"
    )
    marker_status: dict[int, bool] = Field(
        default_factory=dict, description="Process index -> has completion marker"
    )
    is_complete: bool = Field(default=False, description="Whether all processes completed")
    error_message: str | None = Field(default=None, description="Error description if incomplete")


def validate_process_completeness(
    experiment_id: str,
    raw_results: list[RawProcessResult],
    expected_processes: int,
    results_dir: Path,
) -> CompletenessReport:
    """Validate all expected processes completed successfully.

    Checks:
    1. Count: len(results) == expected_processes
    2. Index contiguity: indices are 0, 1, 2, ..., N-1 with no gaps
    3. No duplicates: each index appears exactly once
    4. Markers exist: each process has a completion marker

    Args:
        experiment_id: Experiment identifier.
        raw_results: List of raw process results.
        expected_processes: Number of processes expected.
        results_dir: Base results directory.

    Returns:
        CompletenessReport with validation results.
    """
    found_indices = [r.process_index for r in raw_results]
    expected_indices = set(range(expected_processes))
    found_set = set(found_indices)

    # Check for missing indices
    missing = sorted(expected_indices - found_set)

    # Check for duplicates
    duplicates = sorted({i for i in found_indices if found_indices.count(i) > 1})

    # Check markers
    marker_status: dict[int, bool] = {}
    raw_dir = results_dir / "raw" / experiment_id
    for i in range(expected_processes):
        marker_path = raw_dir / f"{COMPLETION_MARKER_PREFIX}{i}"
        marker_status[i] = marker_path.exists()

    # Determine completeness
    is_complete = (
        len(raw_results) == expected_processes
        and not missing
        and not duplicates
        and all(marker_status.values())
    )

    error_message = None
    if not is_complete:
        issues = []
        if len(raw_results) != expected_processes:
            issues.append(f"Expected {expected_processes} results, found {len(raw_results)}")
        if missing:
            issues.append(f"Missing process indices: {missing}")
        if duplicates:
            issues.append(f"Duplicate process indices: {duplicates}")
        missing_markers = [i for i, has in marker_status.items() if not has]
        if missing_markers:
            issues.append(f"Missing completion markers for processes: {missing_markers}")
        error_message = "; ".join(issues)

    return CompletenessReport(
        expected_processes=expected_processes,
        found_processes=len(raw_results),
        missing_indices=missing,
        duplicate_indices=duplicates,
        marker_status=marker_status,
        is_complete=is_complete,
        error_message=error_message,
    )


def aggregate_results(
    experiment_id: str,
    raw_results: list[RawProcessResult],
    verify_temporal_overlap: bool = True,
    verify_gpu_attribution: bool = True,
    expected_processes: int | None = None,
    results_dir: Path | None = None,
    strict: bool = True,
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
        expected_processes: Expected number of processes (for completeness validation).
        results_dir: Results directory (for marker checking).
        strict: If True, raise on incomplete; if False, warn and proceed.

    Returns:
        Aggregated result combining all process data.

    Raises:
        AggregationError: If raw_results is empty or (strict and incomplete).
    """
    if not raw_results:
        raise AggregationError("Cannot aggregate empty results list")

    warnings: list[str] = []
    num_processes = len(raw_results)

    # Completeness validation (Phase 5)
    if expected_processes is not None and results_dir is not None:
        report = validate_process_completeness(
            experiment_id, raw_results, expected_processes, results_dir
        )
        if not report.is_complete:
            if strict:
                raise AggregationError(f"Incomplete experiment results: {report.error_message}")
            else:
                warnings.append(f"Incomplete results: {report.error_message}")
                logger.warning(f"Proceeding with incomplete results: {report.error_message}")

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

    # Check for MIG instances and warn about energy measurement
    mig_instances = [r for r in raw_results if r.gpu_is_mig]
    if mig_instances:
        mig_profiles = {r.gpu_mig_profile for r in mig_instances if r.gpu_mig_profile}
        profile_str = ", ".join(sorted(mig_profiles)) if mig_profiles else "unknown"
        warnings.append(
            f"Experiment ran on {len(mig_instances)} MIG instance(s) ({profile_str}). "
            "Energy measurements reflect parent GPU total, not per-instance consumption."
        )
        logger.info(f"MIG instances detected: {len(mig_instances)} with profiles: {profile_str}")

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

    # Propagate effective_config and cli_overrides from first result (Phase 0)
    # All processes have the same config, so any result works
    effective_config: dict[str, Any] = {}
    cli_overrides: dict[str, Any] = {}
    if raw_results:
        effective_config = raw_results[0].effective_config
        cli_overrides = raw_results[0].cli_overrides

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
        effective_config=effective_config,
        cli_overrides=cli_overrides,
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
