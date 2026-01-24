"""Aggregation logic for combining raw process results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from llenergymeasure.constants import COMPLETION_MARKER_PREFIX
from llenergymeasure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
)
from llenergymeasure.domain.metrics import (
    ExtendedEfficiencyMetrics,
    LatencyMeasurements,
    LatencyStatistics,
)
from llenergymeasure.exceptions import AggregationError


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
    allow_mixed_backends: bool = False,
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
        allow_mixed_backends: If False (default), reject results from different backends.
            Mixed-backend aggregation produces statistically invalid comparisons.

    Returns:
        Aggregated result combining all process data.

    Raises:
        AggregationError: If raw_results is empty, strict and incomplete,
            or mixed backends without allow_mixed_backends=True.
    """
    if not raw_results:
        raise AggregationError("Cannot aggregate empty results list")

    warnings: list[str] = []
    num_processes = len(raw_results)

    # Backend consistency validation (Phase 3)
    backends = {r.backend for r in raw_results}
    if len(backends) > 1:
        backend_list = ", ".join(sorted(backends))
        msg = f"Mixed backends detected: {backend_list}. Aggregating results from different backends produces statistically invalid comparisons."
        if not allow_mixed_backends:
            raise AggregationError(
                f"{msg} Use --allow-mixed-backends to override (not recommended)."
            )
        warnings.append(msg)
        logger.warning(f"Proceeding with mixed-backend aggregation: {backend_list}")

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

    # Aggregate latency measurements if present (streaming mode)
    latency_stats: LatencyStatistics | None = None
    latency_measurements: list[LatencyMeasurements] = []
    for r in raw_results:
        lm = r.inference_metrics.latency_measurements
        if lm is not None:
            # Handle both LatencyMeasurements instance and dict (from JSON deserialization)
            if isinstance(lm, LatencyMeasurements):
                latency_measurements.append(lm)
            elif isinstance(lm, dict) and "ttft_ms" in lm:
                # Convert dict to LatencyMeasurements
                latency_measurements.append(LatencyMeasurements(**lm))

    if latency_measurements:
        latency_stats = aggregate_latency_measurements(latency_measurements)
        if latency_stats:
            logger.info(
                f"Latency stats: TTFT mean={latency_stats.ttft_mean_ms:.1f}ms, "
                f"ITL mean={latency_stats.itl_mean_ms:.1f}ms"
                if latency_stats.itl_mean_ms
                else ""
            )

    logger.info(
        f"Aggregated {num_processes} processes: "
        f"tokens={total_tokens}, energy={total_energy_j:.2f}J, "
        f"throughput={avg_tokens_per_second:.2f} tok/s"
    )

    # Check if any process had energy tracking failures
    energy_tracking_failed = any(r.energy_tracking_failed for r in raw_results)
    if energy_tracking_failed:
        warnings.append(
            "Energy tracking failed for one or more processes (metrics may be incomplete)"
        )
        logger.warning("Energy tracking failed for one or more processes")

    # Aggregate extended efficiency metrics (Phase 5)
    extended_metrics = _aggregate_extended_metrics_from_results(
        raw_results=raw_results,
        total_energy_j=total_energy_j,
        avg_tokens_per_second=avg_tokens_per_second,
        total_output_tokens=sum(r.inference_metrics.output_tokens for r in raw_results),
        latency_stats=latency_stats,
    )

    # Propagate effective_config, cli_overrides, config_warnings, and backend from first result
    # All processes have the same config/backend, so any result works
    effective_config: dict[str, Any] = {}
    cli_overrides: dict[str, Any] = {}
    config_warnings: list[str] = []
    backend = "pytorch"
    backend_version: str | None = None
    if raw_results:
        effective_config = raw_results[0].effective_config
        cli_overrides = raw_results[0].cli_overrides
        config_warnings = raw_results[0].config_warnings
        backend = raw_results[0].backend
        backend_version = raw_results[0].backend_version

    return AggregatedResult(
        experiment_id=experiment_id,
        backend=backend,
        backend_version=backend_version,
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
        config_warnings=config_warnings,
        latency_stats=latency_stats,
        energy_tracking_failed=energy_tracking_failed,
        extended_metrics=extended_metrics,
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


def _aggregate_extended_metrics_from_results(
    raw_results: list[RawProcessResult],
    total_energy_j: float,
    avg_tokens_per_second: float,
    total_output_tokens: int,
    latency_stats: LatencyStatistics | None,
) -> ExtendedEfficiencyMetrics:
    """Aggregate extended efficiency metrics from per-process results.

    Collects raw samples from all processes and computes aggregated statistics.
    Uses late aggregation pattern: raw samples stored per-process, stats computed here.

    Args:
        raw_results: List of raw results from each process.
        total_energy_j: Total energy consumption across all processes.
        avg_tokens_per_second: Average throughput.
        total_output_tokens: Total output tokens generated.
        latency_stats: Aggregated latency statistics (for ITL/TPOT).

    Returns:
        Aggregated ExtendedEfficiencyMetrics.
    """
    from llenergymeasure.core.extended_metrics import aggregate_extended_metrics
    from llenergymeasure.domain.metrics import ExtendedEfficiencyMetrics

    # Collect raw data from all processes for late aggregation
    all_request_latencies: list[float] = []
    all_gpu_samples: list[float] = []
    raw_extended_metrics: list[ExtendedEfficiencyMetrics] = []

    for r in raw_results:
        # Collect per-request latencies
        if r.per_request_latencies_ms:
            all_request_latencies.extend(r.per_request_latencies_ms)

        # Collect GPU utilisation samples
        if r.gpu_utilisation_samples:
            all_gpu_samples.extend(r.gpu_utilisation_samples)

        # Collect per-process extended metrics
        raw_extended_metrics.append(r.extended_metrics)

    # Get ITL mean for TPOT (from aggregated latency stats)
    itl_mean_ms: float | None = None
    if latency_stats and latency_stats.itl_mean_ms is not None:
        itl_mean_ms = latency_stats.itl_mean_ms

    # Get precision factor from first result's extended metrics (same across all)
    precision_factor = 1.0
    if raw_extended_metrics and raw_extended_metrics[0].token_efficiency_index is not None:
        # Reverse-engineer precision factor from existing TEI if available
        # TEI = throughput * tokens_per_joule * precision_factor
        # This is a best-effort approach; precision_factor should be same across processes
        pass  # Use default 1.0 for now; the per-process metrics capture precision correctly

    try:
        extended_metrics = aggregate_extended_metrics(
            raw_extended_metrics=raw_extended_metrics,
            all_request_latencies=all_request_latencies,
            all_gpu_samples=all_gpu_samples,
            aggregated_output_tokens=total_output_tokens,
            aggregated_energy_j=total_energy_j,
            aggregated_tokens_per_sec=avg_tokens_per_second,
            itl_mean_ms=itl_mean_ms,
            precision_factor=precision_factor,
        )
        logger.debug(
            f"Aggregated extended metrics: TPOT={extended_metrics.tpot_ms}, "
            f"TEI={extended_metrics.token_efficiency_index}"
        )
    except Exception as e:
        logger.warning(f"Extended metrics aggregation failed (non-fatal): {e}")
        extended_metrics = ExtendedEfficiencyMetrics()

    return extended_metrics


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


def aggregate_latency_measurements(
    measurements: list[LatencyMeasurements],
) -> LatencyStatistics | None:
    """Aggregate raw latency measurements from multiple processes.

    Concatenates raw samples from all processes, then computes statistics.
    This is the correct way to aggregate percentiles (not mean of percentiles).

    Args:
        measurements: List of LatencyMeasurements from each process.

    Returns:
        LatencyStatistics with computed percentiles, or None if no data.
    """
    if not measurements:
        return None

    # Concatenate all raw samples
    all_ttft: list[float] = []
    all_itl_full: list[float] = []
    all_itl_trimmed: list[float] = []

    for m in measurements:
        all_ttft.extend(m.ttft_ms)
        all_itl_full.extend(m.itl_full_ms)
        all_itl_trimmed.extend(m.itl_trimmed_ms)

    if not all_ttft:
        logger.warning("No TTFT samples to aggregate")
        return None

    # Compute TTFT statistics
    ttft_arr = np.array(all_ttft)

    # Compute ITL statistics (trimmed - primary metric)
    itl_mean_ms: float | None = None
    itl_median_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    itl_samples = 0

    if all_itl_trimmed:
        itl_arr = np.array(all_itl_trimmed)
        itl_mean_ms = float(np.mean(itl_arr))
        itl_median_ms = float(np.median(itl_arr))
        itl_p95_ms = float(np.percentile(itl_arr, 95))
        itl_p99_ms = float(np.percentile(itl_arr, 99))
        itl_samples = len(all_itl_trimmed)

    # Compute ITL full statistics (for comparison)
    itl_full_mean_ms: float | None = None
    itl_full_p99_ms: float | None = None

    if all_itl_full:
        itl_full_arr = np.array(all_itl_full)
        itl_full_mean_ms = float(np.mean(itl_full_arr))
        itl_full_p99_ms = float(np.percentile(itl_full_arr, 99))

    logger.info(
        f"Aggregated latency stats: TTFT samples={len(all_ttft)}, "
        f"ITL samples={itl_samples} (trimmed)"
    )

    return LatencyStatistics(
        ttft_mean_ms=float(np.mean(ttft_arr)),
        ttft_median_ms=float(np.median(ttft_arr)),
        ttft_p95_ms=float(np.percentile(ttft_arr, 95)),
        ttft_p99_ms=float(np.percentile(ttft_arr, 99)),
        ttft_min_ms=float(np.min(ttft_arr)),
        ttft_max_ms=float(np.max(ttft_arr)),
        ttft_samples=len(all_ttft),
        itl_mean_ms=itl_mean_ms,
        itl_median_ms=itl_median_ms,
        itl_p95_ms=itl_p95_ms,
        itl_p99_ms=itl_p99_ms,
        itl_samples=itl_samples,
        itl_full_mean_ms=itl_full_mean_ms,
        itl_full_p99_ms=itl_full_p99_ms,
    )
