"""
Results Analysis Example
=========================

This example demonstrates how to analyze and export experiment results.

Features demonstrated:
- Loading and reading results
- Analyzing efficiency metrics
- Exporting to different formats
- Generating custom reports
- Statistical analysis
"""

from pathlib import Path
from typing import List, Dict
import json
import statistics

from llm_efficiency.storage.results import (
    ResultsManager,
    ExperimentResult,
    EfficiencyMetrics,
)


def example_load_results():
    """Load and inspect experiment results."""

    print("=" * 70)
    print("Example 1: Loading Results")
    print("=" * 70)

    # Initialize results manager
    results_dir = Path("./results")
    manager = ResultsManager(results_dir=results_dir)

    # List all experiments
    experiments = manager.list_experiments()

    if not experiments:
        print("\nNo experiments found!")
        print(f"  Run experiments first and save to: {results_dir}")
        return None

    print(f"\nFound {len(experiments)} experiment(s)")

    # Show first few experiments
    for exp_id in experiments[:5]:
        result = manager.load_result(exp_id)
        print(f"\n  ID: {exp_id}")
        print(f"  Model: {result.config.model_name}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"  Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")

    return manager


def example_analyze_single_result(manager: ResultsManager):
    """Analyze a single experiment result in detail."""

    print("\n" + "=" * 70)
    print("Example 2: Analyzing Single Result")
    print("=" * 70)

    experiments = manager.list_experiments()
    if not experiments:
        print("\nNo experiments available for analysis")
        return

    # Load first experiment
    exp_id = experiments[0]
    result = manager.load_result(exp_id)

    print(f"\nAnalyzing: {exp_id}")
    print("\n--- Configuration ---")
    print(f"Model: {result.config.model_name}")
    print(f"Precision: {result.config.precision}")
    print(f"Quantization: {result.config.quantization.enabled}")
    print(f"Batch size: {result.config.batch_size}")
    print(f"Max length: {result.config.max_length}")

    print("\n--- Performance Metrics ---")
    metrics = result.metrics
    print(f"Total samples: {metrics.total_samples:,}")
    print(f"Total tokens: {metrics.total_tokens:,}")
    print(f"Throughput: {metrics.tokens_per_second:.2f} tokens/sec")
    print(f"Avg latency/token: {metrics.latency_per_token:.4f} sec")
    print(f"Avg latency/sample: {metrics.latency_per_sample:.4f} sec")

    print("\n--- Energy Metrics ---")
    print(f"Total energy: {metrics.total_energy_kwh:.6f} kWh")
    print(f"Energy per token: {metrics.energy_per_token:.8f} kWh")
    print(f"Energy per sample: {metrics.energy_per_sample:.8f} kWh")
    print(f"CO2 emissions: {metrics.co2_emissions:.6f} kg")

    print("\n--- Compute Metrics ---")
    print(f"Total FLOPs: {metrics.total_flops:,.0f}")
    print(f"FLOPs per token: {metrics.flops_per_token:,.0f}")
    print(f"FLOPs per sample: {metrics.flops_per_sample:,.0f}")

    # Calculate additional insights
    print("\n--- Efficiency Insights ---")

    # Energy cost estimation (assuming $0.12/kWh)
    energy_cost = metrics.total_energy_kwh * 0.12
    print(f"Estimated energy cost: ${energy_cost:.4f}")

    # Tokens per kWh
    tokens_per_kwh = 1.0 / metrics.energy_per_token if metrics.energy_per_token > 0 else 0
    print(f"Tokens per kWh: {tokens_per_kwh:,.0f}")

    # CO2 per million tokens
    co2_per_million = (metrics.co2_emissions / metrics.total_tokens) * 1_000_000
    print(f"CO2 per million tokens: {co2_per_million:.4f} kg")

    # Computational intensity
    comp_intensity = metrics.flops_per_token / metrics.latency_per_token if metrics.latency_per_token > 0 else 0
    print(f"Computational intensity: {comp_intensity:,.0f} FLOPs/sec")

    return result


def example_compare_results(manager: ResultsManager):
    """Compare multiple experiment results."""

    print("\n" + "=" * 70)
    print("Example 3: Comparing Multiple Results")
    print("=" * 70)

    experiments = manager.list_experiments()
    if len(experiments) < 2:
        print("\nNeed at least 2 experiments for comparison")
        return

    # Load all results
    results = [manager.load_result(exp_id) for exp_id in experiments]

    print(f"\nComparing {len(results)} experiments")

    # Find best performers
    print("\n--- Performance Leaders ---")

    fastest = max(results, key=lambda r: r.metrics.tokens_per_second)
    print(f"\nFastest throughput:")
    print(f"  Model: {fastest.config.model_name}")
    print(f"  Throughput: {fastest.metrics.tokens_per_second:.2f} tokens/sec")

    most_efficient = min(results, key=lambda r: r.metrics.energy_per_token)
    print(f"\nMost energy efficient:")
    print(f"  Model: {most_efficient.config.model_name}")
    print(f"  Energy per token: {most_efficient.metrics.energy_per_token:.8f} kWh")

    lowest_co2 = min(results, key=lambda r: r.metrics.co2_emissions)
    print(f"\nLowest CO2 emissions:")
    print(f"  Model: {lowest_co2.config.model_name}")
    print(f"  CO2: {lowest_co2.metrics.co2_emissions:.6f} kg")

    # Calculate statistics
    print("\n--- Aggregate Statistics ---")

    throughputs = [r.metrics.tokens_per_second for r in results]
    energies = [r.metrics.energy_per_token for r in results]
    co2_values = [r.metrics.co2_emissions for r in results]

    print(f"\nThroughput (tokens/sec):")
    print(f"  Mean: {statistics.mean(throughputs):.2f}")
    print(f"  Median: {statistics.median(throughputs):.2f}")
    print(f"  Std Dev: {statistics.stdev(throughputs) if len(throughputs) > 1 else 0:.2f}")
    print(f"  Range: {min(throughputs):.2f} - {max(throughputs):.2f}")

    print(f"\nEnergy per token (kWh):")
    print(f"  Mean: {statistics.mean(energies):.8f}")
    print(f"  Median: {statistics.median(energies):.8f}")
    print(f"  Range: {min(energies):.8f} - {max(energies):.8f}")

    total_co2 = sum(co2_values)
    print(f"\nTotal CO2 emissions: {total_co2:.6f} kg")

    return results


def example_export_results(manager: ResultsManager):
    """Export results to different formats."""

    print("\n" + "=" * 70)
    print("Example 4: Exporting Results")
    print("=" * 70)

    experiments = manager.list_experiments()
    if not experiments:
        print("\nNo experiments to export")
        return

    output_dir = Path("./exports")
    output_dir.mkdir(exist_ok=True)

    # Export to CSV
    csv_file = output_dir / "results.csv"
    manager.export_to_csv(csv_file)
    print(f"\n✓ Exported to CSV: {csv_file}")

    # Export to pickle (fastest)
    pickle_file = output_dir / "results.pkl"
    manager.export_to_pickle(pickle_file)
    print(f"✓ Exported to pickle: {pickle_file}")

    # Export to JSON
    json_file = output_dir / "results.json"
    manager.export(json_file, format="json")
    print(f"✓ Exported to JSON: {json_file}")

    # Export specific experiments only
    if len(experiments) >= 2:
        filtered_csv = output_dir / "filtered_results.csv"
        manager.export_to_csv(filtered_csv, experiment_ids=experiments[:2])
        print(f"✓ Exported 2 experiments to: {filtered_csv}")

    print(f"\nAll exports saved to: {output_dir}")


def example_custom_analysis(manager: ResultsManager):
    """Perform custom analysis on results."""

    print("\n" + "=" * 70)
    print("Example 5: Custom Analysis")
    print("=" * 70)

    experiments = manager.list_experiments()
    if not experiments:
        print("\nNo experiments for analysis")
        return

    results = [manager.load_result(exp_id) for exp_id in experiments]

    # Group by model
    by_model: Dict[str, List[ExperimentResult]] = {}
    for result in results:
        model = result.config.model_name
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(result)

    print(f"\nGrouping {len(results)} experiments by model")
    print(f"Unique models: {len(by_model)}")

    # Analyze each model
    for model, model_results in by_model.items():
        print(f"\n--- {model} ({len(model_results)} experiments) ---")

        throughputs = [r.metrics.tokens_per_second for r in model_results]
        energies = [r.metrics.energy_per_token for r in model_results]

        print(f"Avg throughput: {statistics.mean(throughputs):.2f} tokens/sec")
        print(f"Avg energy: {statistics.mean(energies):.8f} kWh/token")

        # Check consistency
        if len(throughputs) > 1:
            cv = statistics.stdev(throughputs) / statistics.mean(throughputs) * 100
            print(f"Throughput CV: {cv:.1f}% (consistency)")

    # Identify outliers (simple method: > 2 std devs from mean)
    print("\n--- Outlier Detection ---")

    all_throughputs = [r.metrics.tokens_per_second for r in results]
    if len(all_throughputs) > 2:
        mean_tp = statistics.mean(all_throughputs)
        std_tp = statistics.stdev(all_throughputs)
        threshold = 2 * std_tp

        outliers = [
            r for r in results
            if abs(r.metrics.tokens_per_second - mean_tp) > threshold
        ]

        if outliers:
            print(f"\nFound {len(outliers)} outlier(s):")
            for r in outliers:
                print(f"  {r.config.model_name}: {r.metrics.tokens_per_second:.2f} tokens/sec")
        else:
            print("\nNo significant outliers detected")

    # Calculate efficiency score (custom metric)
    print("\n--- Efficiency Score ---")
    print("Score = (Throughput / Mean) * (Mean Energy / Energy)")

    mean_throughput = statistics.mean(all_throughputs)
    all_energies = [r.metrics.energy_per_token for r in results]
    mean_energy = statistics.mean(all_energies)

    scored_results = []
    for r in results:
        score = (
            (r.metrics.tokens_per_second / mean_throughput) *
            (mean_energy / r.metrics.energy_per_token)
        )
        scored_results.append((r, score))

    # Sort by score
    scored_results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 by efficiency score:")
    for i, (r, score) in enumerate(scored_results[:3], 1):
        print(f"  {i}. {r.config.model_name}: {score:.3f}")


def main():
    """Run all results analysis examples."""

    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS EXAMPLES")
    print("=" * 70)

    # Note: This assumes you have run some experiments first
    print("\nNote: Run experiments first to generate results")
    print("      Examples 01-07 generate results for analysis")

    # Initialize manager
    manager = example_load_results()

    if manager is None:
        print("\n" + "=" * 70)
        print("No results available. Run experiments first:")
        print("  python examples/01_basic_experiment.py")
        print("  python examples/05_multi_model_comparison.py")
        print("=" * 70)
        return

    # Run analysis examples
    example_analyze_single_result(manager)
    example_compare_results(manager)
    example_export_results(manager)
    example_custom_analysis(manager)

    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Load results with ResultsManager")
    print("  2. Analyze individual and aggregate metrics")
    print("  3. Export to CSV, JSON, or pickle formats")
    print("  4. Perform custom analysis and grouping")
    print("  5. Calculate custom efficiency metrics")
    print("\nFor production use:")
    print("  - Automate report generation")
    print("  - Track metrics over time")
    print("  - Set up alerting for outliers")
    print("  - Create dashboards from exported data")
    print("=" * 70)


if __name__ == "__main__":
    main()
