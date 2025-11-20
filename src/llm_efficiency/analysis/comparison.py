"""Experiment comparison utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from llm_efficiency.storage.results import ExperimentResult, ResultsManager


@dataclass
class ComparisonResult:
    """Result of comparing experiments."""

    experiments: List[ExperimentResult]
    fastest: ExperimentResult
    most_efficient: ExperimentResult
    lowest_co2: ExperimentResult
    best_overall: ExperimentResult
    statistics: Dict[str, Dict[str, float]]


def compare_experiments(
    experiment_ids: List[str],
    results_dir: Path,
) -> ComparisonResult:
    """
    Compare multiple experiments.

    Args:
        experiment_ids: List of experiment IDs to compare
        results_dir: Directory containing results

    Returns:
        ComparisonResult with analysis
    """
    manager = ResultsManager(results_dir=results_dir)
    experiments = [manager.load_result(exp_id) for exp_id in experiment_ids]

    # Find best performers
    fastest = max(experiments, key=lambda r: r.metrics.tokens_per_second)
    most_efficient = min(experiments, key=lambda r: r.metrics.energy_per_token)
    lowest_co2 = min(experiments, key=lambda r: r.metrics.co2_emissions)

    # Calculate overall score
    def score(r):
        norm_throughput = r.metrics.tokens_per_second / fastest.metrics.tokens_per_second
        norm_energy = most_efficient.metrics.energy_per_token / r.metrics.energy_per_token
        return (norm_throughput + norm_energy) / 2

    best_overall = max(experiments, key=score)

    # Calculate statistics
    stats = {
        "throughput": {
            "mean": sum(r.metrics.tokens_per_second for r in experiments) / len(experiments),
            "min": min(r.metrics.tokens_per_second for r in experiments),
            "max": max(r.metrics.tokens_per_second for r in experiments),
        },
        "energy": {
            "mean": sum(r.metrics.energy_per_token for r in experiments) / len(experiments),
            "min": min(r.metrics.energy_per_token for r in experiments),
            "max": max(r.metrics.energy_per_token for r in experiments),
        },
    }

    return ComparisonResult(
        experiments=experiments,
        fastest=fastest,
        most_efficient=most_efficient,
        lowest_co2=lowest_co2,
        best_overall=best_overall,
        statistics=stats,
    )


def compare_models(
    models: List[str],
    results_dir: Path,
) -> Dict[str, List[ExperimentResult]]:
    """
    Compare results grouped by model.

    Args:
        models: List of model names
        results_dir: Results directory

    Returns:
        Dictionary mapping model names to results
    """
    manager = ResultsManager(results_dir=results_dir)
    all_experiments = manager.list_experiments()

    by_model: Dict[str, List[ExperimentResult]] = {model: [] for model in models}

    for exp_id in all_experiments:
        result = manager.load_result(exp_id)
        if result.config.model_name in models:
            by_model[result.config.model_name].append(result)

    return by_model


def compare_configurations(
    base_config: Dict,
    variations: List[Dict],
    results_dir: Path,
) -> List[ExperimentResult]:
    """
    Compare experiments with configuration variations.

    Args:
        base_config: Base configuration
        variations: List of configuration variations
        results_dir: Results directory

    Returns:
        List of matching experiment results
    """
    # Simplified implementation
    manager = ResultsManager(results_dir=results_dir)
    return [manager.load_result(exp_id) for exp_id in manager.list_experiments()]


def generate_comparison_report(
    comparison: ComparisonResult,
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate comparison report.

    Args:
        comparison: ComparisonResult
        output_file: Optional file to save report

    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append(f"\nExperiments compared: {len(comparison.experiments)}")

    lines.append("\n--- BEST PERFORMERS ---")
    lines.append(f"\nFastest: {comparison.fastest.config.model_name}")
    lines.append(f"  {comparison.fastest.metrics.tokens_per_second:.2f} tokens/sec")

    lines.append(f"\nMost energy efficient: {comparison.most_efficient.config.model_name}")
    lines.append(f"  {comparison.most_efficient.metrics.energy_per_token:.8f} kWh/token")

    lines.append(f"\nLowest CO2: {comparison.lowest_co2.config.model_name}")
    lines.append(f"  {comparison.lowest_co2.metrics.co2_emissions:.6f} kg")

    lines.append(f"\nBest overall: {comparison.best_overall.config.model_name}")

    lines.append("\n--- STATISTICS ---")
    lines.append(f"\nThroughput (tokens/sec):")
    lines.append(f"  Mean: {comparison.statistics['throughput']['mean']:.2f}")
    lines.append(f"  Range: {comparison.statistics['throughput']['min']:.2f} - {comparison.statistics['throughput']['max']:.2f}")

    lines.append(f"\nEnergy (kWh/token):")
    lines.append(f"  Mean: {comparison.statistics['energy']['mean']:.8f}")
    lines.append(f"  Range: {comparison.statistics['energy']['min']:.8f} - {comparison.statistics['energy']['max']:.8f}")

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)

    if output_file:
        output_file.write_text(report)

    return report
