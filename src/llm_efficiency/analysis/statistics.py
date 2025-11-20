"""Statistical analysis utilities for experiment results."""

import statistics
from typing import Dict, List, Tuple

from llm_efficiency.storage.results import ExperimentResult


def calculate_statistics(experiments: List[ExperimentResult]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics across experiments.

    Args:
        experiments: List of experiment results

    Returns:
        Dictionary of statistics
    """
    if not experiments:
        return {}

    throughputs = [r.metrics.tokens_per_second for r in experiments]
    energies = [r.metrics.energy_per_token for r in experiments]
    co2_values = [r.metrics.co2_emissions for r in experiments]

    stats = {
        "throughput": {
            "mean": statistics.mean(throughputs),
            "median": statistics.median(throughputs),
            "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
            "min": min(throughputs),
            "max": max(throughputs),
        },
        "energy": {
            "mean": statistics.mean(energies),
            "median": statistics.median(energies),
            "stdev": statistics.stdev(energies) if len(energies) > 1 else 0.0,
            "min": min(energies),
            "max": max(energies),
        },
        "co2": {
            "mean": statistics.mean(co2_values),
            "median": statistics.median(co2_values),
            "stdev": statistics.stdev(co2_values) if len(co2_values) > 1 else 0.0,
            "min": min(co2_values),
            "max": max(co2_values),
        },
    }

    return stats


def detect_outliers(
    experiments: List[ExperimentResult],
    metric: str = "throughput",
    threshold: float = 2.0,
) -> Tuple[List[ExperimentResult], List[ExperimentResult]]:
    """
    Detect outliers using standard deviation method.

    Args:
        experiments: List of experiment results
        metric: Metric to check ("throughput", "energy", "co2")
        threshold: Number of standard deviations for outlier detection

    Returns:
        Tuple of (outliers, normal_experiments)
    """
    if len(experiments) <= 2:
        return [], experiments

    # Get metric values
    if metric == "throughput":
        values = [r.metrics.tokens_per_second for r in experiments]
    elif metric == "energy":
        values = [r.metrics.energy_per_token for r in experiments]
    elif metric == "co2":
        values = [r.metrics.co2_emissions for r in experiments]
    else:
        return [], experiments

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    outliers = []
    normal = []

    for exp, value in zip(experiments, values):
        if abs(value - mean) > threshold * stdev:
            outliers.append(exp)
        else:
            normal.append(exp)

    return outliers, normal


def calculate_efficiency_score(
    experiment: ExperimentResult,
    weights: Dict[str, float] = None,
) -> float:
    """
    Calculate overall efficiency score.

    Args:
        experiment: Experiment result
        weights: Metric weights (default: equal weights)

    Returns:
        Efficiency score (0-100, higher is better)
    """
    if weights is None:
        weights = {
            "throughput": 0.4,
            "energy": 0.4,
            "co2": 0.2,
        }

    # Normalize metrics (higher is better)
    # For throughput: higher is better (use as-is)
    # For energy/co2: lower is better (invert)

    # Simple normalization (would need baselines in production)
    throughput_score = min(experiment.metrics.tokens_per_second / 100.0, 1.0)
    energy_score = max(1.0 - experiment.metrics.energy_per_token * 10000, 0.0)
    co2_score = max(1.0 - experiment.metrics.co2_emissions * 1000, 0.0)

    score = (
        weights["throughput"] * throughput_score
        + weights["energy"] * energy_score
        + weights["co2"] * co2_score
    )

    return score * 100.0  # Scale to 0-100


def rank_experiments(
    experiments: List[ExperimentResult],
    metric: str = "efficiency",
    weights: Dict[str, float] = None,
) -> List[Tuple[ExperimentResult, float]]:
    """
    Rank experiments by metric.

    Args:
        experiments: List of experiment results
        metric: Metric to rank by
        weights: Weights for efficiency score calculation

    Returns:
        List of (experiment, score) tuples, sorted best to worst
    """
    if metric == "efficiency":
        scored = [
            (exp, calculate_efficiency_score(exp, weights)) for exp in experiments
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    elif metric == "throughput":
        scored = [(exp, exp.metrics.tokens_per_second) for exp in experiments]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    elif metric == "energy":
        scored = [(exp, exp.metrics.energy_per_token) for exp in experiments]
        return sorted(scored, key=lambda x: x[1])  # Lower is better

    elif metric == "co2":
        scored = [(exp, exp.metrics.co2_emissions) for exp in experiments]
        return sorted(scored, key=lambda x: x[1])  # Lower is better

    else:
        return [(exp, 0.0) for exp in experiments]
