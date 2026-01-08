"""Statistical calculations for multi-cycle experiments.

Implements academic benchmarking standards for statistical robustness:
- Mean and standard deviation
- 95% confidence intervals
- Coefficient of variation

References:
- TokenPowerBench: "repeated 10 times to ensure statistical robustness"
- MLPerf: Reports with confidence intervals
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from llm_energy_measure.domain.experiment import (
    CycleMetadata,
    CycleStatistics,
    MultiCycleResult,
)

if TYPE_CHECKING:
    from llm_energy_measure.domain.experiment import AggregatedResult


def calculate_statistics(values: list[float]) -> tuple[float, float, float, float]:
    """Calculate mean, std, and 95% CI for a list of values.

    Uses t-distribution critical values for small sample sizes.

    Args:
        values: List of measurements.

    Returns:
        Tuple of (mean, std, ci_lower, ci_upper).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean = sum(values) / n

    if n == 1:
        return mean, 0.0, mean, mean

    # Sample standard deviation
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    # t-distribution critical values for 95% CI
    # For small samples, use t-distribution; for n >= 30, normal approximation
    t_values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    t_crit = t_values.get(n, 1.96)  # Use 1.96 (z-score) for n >= 30

    # Standard error of the mean
    sem = std / math.sqrt(n)
    margin = t_crit * sem

    ci_lower = mean - margin
    ci_upper = mean + margin

    return mean, std, ci_lower, ci_upper


def calculate_cv(mean: float, std: float) -> float:
    """Calculate coefficient of variation.

    CV = std / mean, expressed as a percentage.
    Lower CV indicates more stable measurements.

    Args:
        mean: Mean value.
        std: Standard deviation.

    Returns:
        Coefficient of variation (as ratio, not percentage).
    """
    if mean == 0:
        return 0.0
    return std / abs(mean)


def create_cycle_statistics(results: list[AggregatedResult]) -> CycleStatistics:
    """Create statistical summary from multiple cycle results.

    Args:
        results: List of aggregated results from each cycle.

    Returns:
        CycleStatistics with mean, std, and CI for key metrics.
    """
    if not results:
        raise ValueError("Cannot calculate statistics from empty results list")

    n = len(results)
    logger.info(f"Calculating statistics for {n} cycles")

    # Extract metrics from each cycle
    energies = [r.total_energy_j for r in results]
    throughputs = [r.avg_tokens_per_second for r in results]
    efficiencies = [r.tokens_per_joule for r in results]

    # Calculate latency (ms per token) from each result
    latencies = []
    for r in results:
        if r.avg_tokens_per_second > 0:
            latencies.append(1000.0 / r.avg_tokens_per_second)
        else:
            latencies.append(0.0)

    # Calculate statistics for each metric
    energy_mean, energy_std, energy_ci_lo, energy_ci_hi = calculate_statistics(energies)
    tp_mean, tp_std, tp_ci_lo, tp_ci_hi = calculate_statistics(throughputs)
    eff_mean, eff_std, _, _ = calculate_statistics(efficiencies)
    lat_mean, lat_std, _, _ = calculate_statistics(latencies)

    # Calculate coefficients of variation
    energy_cv = calculate_cv(energy_mean, energy_std)
    throughput_cv = calculate_cv(tp_mean, tp_std)

    logger.debug(
        f"Energy: {energy_mean:.2f} ± {energy_std:.2f} J (CV={energy_cv:.2%}), "
        f"Throughput: {tp_mean:.2f} ± {tp_std:.2f} tok/s (CV={throughput_cv:.2%})"
    )

    return CycleStatistics(
        num_cycles=n,
        energy_mean_j=energy_mean,
        energy_std_j=energy_std,
        energy_ci_95_lower=energy_ci_lo,
        energy_ci_95_upper=energy_ci_hi,
        throughput_mean_tps=tp_mean,
        throughput_std_tps=tp_std,
        throughput_ci_95_lower=tp_ci_lo,
        throughput_ci_95_upper=tp_ci_hi,
        efficiency_mean_tpj=eff_mean,
        efficiency_std_tpj=eff_std,
        latency_mean_ms=lat_mean,
        latency_std_ms=lat_std,
        energy_cv=energy_cv,
        throughput_cv=throughput_cv,
    )


def create_cycle_metadata(
    cycle_id: int,
    timestamp: datetime | None = None,
    gpu_temperature_c: float | None = None,
    system_load: float | None = None,
) -> CycleMetadata:
    """Create metadata for a single cycle.

    Args:
        cycle_id: Cycle index (0-based).
        timestamp: Cycle start time (defaults to now).
        gpu_temperature_c: GPU temperature if available.
        system_load: CPU load if available.

    Returns:
        CycleMetadata for this cycle.
    """
    return CycleMetadata(
        cycle_id=cycle_id,
        timestamp=timestamp or datetime.now(),
        gpu_temperature_c=gpu_temperature_c,
        system_load=system_load,
    )


def try_get_gpu_temperature() -> float | None:
    """Try to get GPU temperature via nvidia-smi.

    Returns:
        GPU temperature in Celsius, or None if unavailable.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            # Take first GPU's temperature
            temps = result.stdout.strip().split("\n")
            if temps:
                return float(temps[0])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def try_get_system_load() -> float | None:
    """Try to get system CPU load average.

    Returns:
        1-minute load average, or None if unavailable.
    """
    try:
        import os

        load = os.getloadavg()
        return load[0]  # 1-minute average
    except (OSError, AttributeError):
        pass
    return None


def create_multi_cycle_result(
    experiment_id: str,
    cycle_results: list[AggregatedResult],
    cycle_metadata: list[CycleMetadata],
    effective_config: dict[str, Any] | None = None,
) -> MultiCycleResult:
    """Create a MultiCycleResult from cycle data.

    Args:
        experiment_id: Base experiment identifier.
        cycle_results: Aggregated results from each cycle.
        cycle_metadata: Metadata from each cycle.
        effective_config: Experiment configuration.

    Returns:
        MultiCycleResult with statistical aggregation.
    """
    if not cycle_results:
        raise ValueError("Cannot create multi-cycle result from empty results")

    # Calculate statistics
    statistics = create_cycle_statistics(cycle_results)

    # Timestamps
    start_time = min(r.start_time for r in cycle_results)
    end_time = max(r.end_time for r in cycle_results)
    total_duration = (end_time - start_time).total_seconds()

    logger.info(
        f"Multi-cycle result: {len(cycle_results)} cycles, "
        f"energy={statistics.energy_mean_j:.2f}±{statistics.energy_std_j:.2f}J, "
        f"throughput={statistics.throughput_mean_tps:.2f}±{statistics.throughput_std_tps:.2f}tok/s"
    )

    return MultiCycleResult(
        experiment_id=experiment_id,
        num_cycles=len(cycle_results),
        statistics=statistics,
        cycle_results=cycle_results,
        cycle_metadata=cycle_metadata,
        start_time=start_time,
        end_time=end_time,
        total_duration_sec=total_duration,
        effective_config=effective_config or {},
    )
