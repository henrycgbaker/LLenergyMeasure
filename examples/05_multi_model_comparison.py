"""
Multi-Model Comparison
======================

This example demonstrates how to compare multiple language models
for efficiency metrics.

Features demonstrated:
- Running experiments with multiple models
- Collecting and comparing results
- Generating comparison reports
- Visualizing efficiency trade-offs
"""

from pathlib import Path
from typing import List
import json

from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.experiment import run_experiment
from llm_efficiency.storage.results import ExperimentResult


def run_model_comparison(models: List[str], output_dir: Path) -> List[ExperimentResult]:
    """
    Run efficiency experiments on multiple models.

    Args:
        models: List of model names to compare
        output_dir: Directory to save results

    Returns:
        List of experiment results
    """

    results = []

    for i, model_name in enumerate(models, 1):
        print(f"\n{'=' * 70}")
        print(f"Running Experiment {i}/{len(models)}: {model_name}")
        print(f"{'=' * 70}")

        # Create configuration for this model
        config = ExperimentConfig(
            model_name=model_name,
            precision="float16",
            quantization=QuantizationConfig(enabled=False),
            batch_size=4,
            num_batches=20,  # Enough for meaningful comparison
            max_length=128,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            output_dir=output_dir / model_name.replace("/", "_"),
        )

        try:
            result = run_experiment(config)
            results.append(result)

            print(f"\n✓ {model_name} complete!")
            print(f"  Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")
            print(f"  Energy: {result.metrics.total_energy_kwh:.6f} kWh")

        except Exception as e:
            print(f"\n✗ Failed to run {model_name}: {e}")
            continue

    return results


def generate_comparison_report(results: List[ExperimentResult], output_file: Path):
    """
    Generate a detailed comparison report.

    Args:
        results: List of experiment results
        output_file: Path to save report
    """

    if not results:
        print("No results to compare!")
        return

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LLM EFFICIENCY COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nModels compared: {len(results)}")
    report_lines.append(f"Timestamp: {results[0].timestamp}")

    # Table header
    report_lines.append("\n" + "-" * 80)
    report_lines.append(f"{'Model':<30} {'Tokens/sec':>12} {'Energy(kWh)':>12} {'CO2(kg)':>10} {'FLOPs':>14}")
    report_lines.append("-" * 80)

    # Sort by throughput (descending)
    sorted_results = sorted(
        results,
        key=lambda r: r.metrics.tokens_per_second,
        reverse=True
    )

    # Table rows
    for result in sorted_results:
        model_name = result.config.model_name[:28]
        tokens_per_sec = result.metrics.tokens_per_second
        energy = result.metrics.total_energy_kwh
        co2 = result.metrics.co2_emissions
        flops = result.metrics.total_flops

        report_lines.append(
            f"{model_name:<30} {tokens_per_sec:>12.2f} {energy:>12.6f} "
            f"{co2:>10.6f} {flops:>14,.0f}"
        )

    report_lines.append("-" * 80)

    # Performance rankings
    report_lines.append("\n--- PERFORMANCE RANKINGS ---")

    report_lines.append("\n1. Fastest (Throughput):")
    fastest = max(results, key=lambda r: r.metrics.tokens_per_second)
    report_lines.append(f"   {fastest.config.model_name}: {fastest.metrics.tokens_per_second:.2f} tokens/sec")

    report_lines.append("\n2. Most Energy Efficient (per token):")
    most_efficient = min(results, key=lambda r: r.metrics.energy_per_token)
    report_lines.append(f"   {most_efficient.config.model_name}: {most_efficient.metrics.energy_per_token:.8f} kWh/token")

    report_lines.append("\n3. Lowest CO2 Emissions:")
    lowest_co2 = min(results, key=lambda r: r.metrics.co2_emissions)
    report_lines.append(f"   {lowest_co2.config.model_name}: {lowest_co2.metrics.co2_emissions:.6f} kg")

    report_lines.append("\n4. Least Compute (FLOPs per token):")
    least_flops = min(results, key=lambda r: r.metrics.flops_per_token)
    report_lines.append(f"   {least_flops.config.model_name}: {least_flops.metrics.flops_per_token:,.0f} FLOPs/token")

    # Aggregate statistics
    report_lines.append("\n--- AGGREGATE STATISTICS ---")

    avg_throughput = sum(r.metrics.tokens_per_second for r in results) / len(results)
    total_energy = sum(r.metrics.total_energy_kwh for r in results)
    total_co2 = sum(r.metrics.co2_emissions for r in results)
    total_tokens = sum(r.metrics.total_tokens for r in results)

    report_lines.append(f"\nAverage throughput: {avg_throughput:.2f} tokens/sec")
    report_lines.append(f"Total energy consumed: {total_energy:.6f} kWh")
    report_lines.append(f"Total CO2 emissions: {total_co2:.6f} kg")
    report_lines.append(f"Total tokens generated: {total_tokens:,}")

    # Efficiency insights
    report_lines.append("\n--- EFFICIENCY INSIGHTS ---")

    # Throughput variance
    throughputs = [r.metrics.tokens_per_second for r in results]
    throughput_range = max(throughputs) - min(throughputs)
    report_lines.append(f"\nThroughput range: {throughput_range:.2f} tokens/sec")
    report_lines.append(f"  Fastest is {max(throughputs)/min(throughputs):.2f}x faster than slowest")

    # Energy variance
    energies = [r.metrics.energy_per_token for r in results]
    report_lines.append(f"\nEnergy efficiency range:")
    report_lines.append(f"  Most efficient uses {min(energies)/max(energies)*100:.1f}% energy of least efficient")

    # Recommendations
    report_lines.append("\n--- RECOMMENDATIONS ---")

    report_lines.append("\nFor maximum throughput:")
    report_lines.append(f"  → Use {fastest.config.model_name}")

    report_lines.append("\nFor minimum energy consumption:")
    report_lines.append(f"  → Use {most_efficient.config.model_name}")

    report_lines.append("\nFor lowest carbon footprint:")
    report_lines.append(f"  → Use {lowest_co2.config.model_name}")

    # Best overall (balanced score)
    def efficiency_score(r):
        # Normalize metrics (higher is better)
        norm_throughput = r.metrics.tokens_per_second / max(throughputs)
        norm_energy = min(energies) / r.metrics.energy_per_token
        return (norm_throughput + norm_energy) / 2

    best_overall = max(results, key=efficiency_score)
    report_lines.append("\nBest overall balance (throughput + efficiency):")
    report_lines.append(f"  → Use {best_overall.config.model_name}")

    report_lines.append("\n" + "=" * 80)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_text)
    print(f"\nReport saved to: {output_file}")

    # Also save as JSON for programmatic access
    json_output = output_file.with_suffix('.json')
    comparison_data = {
        "models": [r.config.model_name for r in results],
        "metrics": [
            {
                "model": r.config.model_name,
                "tokens_per_second": r.metrics.tokens_per_second,
                "total_energy_kwh": r.metrics.total_energy_kwh,
                "energy_per_token": r.metrics.energy_per_token,
                "co2_emissions": r.metrics.co2_emissions,
                "total_flops": r.metrics.total_flops,
                "flops_per_token": r.metrics.flops_per_token,
            }
            for r in results
        ],
        "rankings": {
            "fastest": fastest.config.model_name,
            "most_energy_efficient": most_efficient.config.model_name,
            "lowest_co2": lowest_co2.config.model_name,
            "best_overall": best_overall.config.model_name,
        },
    }

    json_output.write_text(json.dumps(comparison_data, indent=2))
    print(f"JSON data saved to: {json_output}")


def main():
    """Run multi-model comparison."""

    print("=" * 70)
    print("MULTI-MODEL EFFICIENCY COMPARISON")
    print("=" * 70)

    # Define models to compare
    # Note: Start with small models for quick testing
    models = [
        "gpt2",           # 124M parameters
        "gpt2-medium",    # 355M parameters
        # Add more models as needed:
        # "gpt2-large",   # 774M parameters
        # "facebook/opt-125m",
        # "EleutherAI/pythia-160m",
    ]

    print(f"\nComparing {len(models)} models:")
    for model in models:
        print(f"  - {model}")

    output_dir = Path("./results/model_comparison")
    print(f"\nResults will be saved to: {output_dir}")

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)

    results = run_model_comparison(models, output_dir)

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON REPORT")
    print("=" * 70)

    report_file = output_dir / "comparison_report.txt"
    generate_comparison_report(results, report_file)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the comparison report")
    print("  2. Analyze JSON data for deeper insights")
    print("  3. Add more models to the comparison")
    print("  4. Try different configurations (precision, quantization)")
    print("=" * 70)


if __name__ == "__main__":
    main()
