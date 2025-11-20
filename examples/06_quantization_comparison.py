"""
Quantization Comparison
=======================

This example demonstrates comparing different quantization methods
for LLM efficiency optimization.

Features demonstrated:
- 4-bit vs 8-bit quantization
- Quantization vs full precision comparison
- Performance/quality trade-offs
- Memory savings analysis
"""

from pathlib import Path
from typing import List
import json

from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.experiment import run_experiment
from llm_efficiency.storage.results import ExperimentResult


def run_quantization_experiments(
    model_name: str,
    output_dir: Path
) -> List[ExperimentResult]:
    """
    Run experiments with different quantization configurations.

    Args:
        model_name: Name of model to test
        output_dir: Directory to save results

    Returns:
        List of experiment results
    """

    # Define quantization configurations to test
    configs_to_test = [
        {
            "name": "Full Precision (FP16)",
            "precision": "float16",
            "quantization": QuantizationConfig(enabled=False),
        },
        {
            "name": "8-bit Quantization",
            "precision": "float16",
            "quantization": QuantizationConfig(
                enabled=True,
                load_in_8bit=True,
                compute_dtype="float16",
            ),
        },
        {
            "name": "4-bit Quantization (NF4)",
            "precision": "float16",
            "quantization": QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                quant_type="nf4",
                compute_dtype="float16",
            ),
        },
        {
            "name": "4-bit Quantization (FP4)",
            "precision": "float16",
            "quantization": QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                quant_type="fp4",
                compute_dtype="float16",
            ),
        },
    ]

    results = []

    for i, config_spec in enumerate(configs_to_test, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{len(configs_to_test)}: {config_spec['name']}")
        print(f"{'=' * 70}")

        config = ExperimentConfig(
            model_name=model_name,
            precision=config_spec["precision"],
            quantization=config_spec["quantization"],
            batch_size=4,
            num_batches=20,
            max_length=128,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            output_dir=output_dir / config_spec["name"].replace(" ", "_").lower(),
        )

        try:
            result = run_experiment(config)
            results.append(result)

            print(f"\n✓ {config_spec['name']} complete!")
            print(f"  Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")
            print(f"  Energy: {result.metrics.total_energy_kwh:.6f} kWh")
            print(f"  Memory: Peak GPU memory usage logged")

        except Exception as e:
            print(f"\n✗ Failed {config_spec['name']}: {e}")
            print("  Note: Quantization requires CUDA/GPU")
            continue

    return results


def generate_quantization_report(
    results: List[ExperimentResult],
    output_file: Path
):
    """
    Generate detailed quantization comparison report.

    Args:
        results: List of experiment results
        output_file: Path to save report
    """

    if not results:
        print("No results to compare!")
        return

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("QUANTIZATION COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nModel: {results[0].config.model_name}")
    report_lines.append(f"Configurations tested: {len(results)}")
    report_lines.append(f"Timestamp: {results[0].timestamp}")

    # Table header
    report_lines.append("\n" + "-" * 80)
    report_lines.append(
        f"{'Configuration':<25} {'Tokens/sec':>12} {'Energy':>12} {'Energy/Token':>14} {'Speedup':>8}"
    )
    report_lines.append("-" * 80)

    # Find baseline (full precision)
    baseline = results[0]  # Assuming first is full precision

    # Table rows
    for result in results:
        # Determine config name from quantization settings
        if not result.config.quantization.enabled:
            config_name = "Full Precision"
        elif result.config.quantization.load_in_8bit:
            config_name = "8-bit Quantization"
        elif result.config.quantization.load_in_4bit:
            quant_type = result.config.quantization.quant_type
            config_name = f"4-bit ({quant_type.upper()})"
        else:
            config_name = "Unknown"

        tokens_per_sec = result.metrics.tokens_per_second
        energy = result.metrics.total_energy_kwh
        energy_per_token = result.metrics.energy_per_token
        speedup = tokens_per_sec / baseline.metrics.tokens_per_second

        report_lines.append(
            f"{config_name:<25} {tokens_per_sec:>12.2f} {energy:>12.6f} "
            f"{energy_per_token:>14.8f} {speedup:>8.2f}x"
        )

    report_lines.append("-" * 80)

    # Performance comparison
    report_lines.append("\n--- PERFORMANCE COMPARISON ---")

    fastest = max(results, key=lambda r: r.metrics.tokens_per_second)
    slowest = min(results, key=lambda r: r.metrics.tokens_per_second)

    report_lines.append(f"\nFastest: {_get_config_name(fastest)}")
    report_lines.append(f"  {fastest.metrics.tokens_per_second:.2f} tokens/sec")

    report_lines.append(f"\nSlowest: {_get_config_name(slowest)}")
    report_lines.append(f"  {slowest.metrics.tokens_per_second:.2f} tokens/sec")

    speedup_range = fastest.metrics.tokens_per_second / slowest.metrics.tokens_per_second
    report_lines.append(f"\nSpeedup range: {speedup_range:.2f}x")

    # Energy comparison
    report_lines.append("\n--- ENERGY EFFICIENCY ---")

    most_efficient = min(results, key=lambda r: r.metrics.energy_per_token)
    least_efficient = max(results, key=lambda r: r.metrics.energy_per_token)

    report_lines.append(f"\nMost efficient: {_get_config_name(most_efficient)}")
    report_lines.append(f"  {most_efficient.metrics.energy_per_token:.8f} kWh/token")

    report_lines.append(f"\nLeast efficient: {_get_config_name(least_efficient)}")
    report_lines.append(f"  {least_efficient.metrics.energy_per_token:.8f} kWh/token")

    energy_savings = (
        1 - most_efficient.metrics.energy_per_token / least_efficient.metrics.energy_per_token
    ) * 100
    report_lines.append(f"\nEnergy savings: {energy_savings:.1f}%")

    # Quality vs Efficiency trade-off
    report_lines.append("\n--- TRADE-OFF ANALYSIS ---")

    report_lines.append("\nQuantization benefits:")
    report_lines.append("  ✓ Reduced memory footprint (4x-8x less for 4-bit)")
    report_lines.append("  ✓ Faster loading times")
    report_lines.append("  ✓ Can run larger models on same hardware")
    report_lines.append("  ✓ Lower energy consumption per token")

    report_lines.append("\nQuantization costs:")
    report_lines.append("  ✗ Potential quality degradation")
    report_lines.append("  ✗ May have slower inference speed")
    report_lines.append("  ✗ Requires CUDA/GPU (CPU not supported)")
    report_lines.append("  ✗ Additional quantization overhead")

    # Recommendations
    report_lines.append("\n--- RECOMMENDATIONS ---")

    report_lines.append("\nUse Full Precision when:")
    report_lines.append("  → Maximum quality is critical")
    report_lines.append("  → Plenty of GPU memory available")
    report_lines.append("  → Small models (< 1B parameters)")

    report_lines.append("\nUse 8-bit Quantization when:")
    report_lines.append("  → Good balance of speed and quality needed")
    report_lines.append("  → Memory constraints (2x savings)")
    report_lines.append("  → Running medium models (1B-7B parameters)")

    report_lines.append("\nUse 4-bit Quantization when:")
    report_lines.append("  → Extreme memory constraints")
    report_lines.append("  → Running large models (7B+ parameters)")
    report_lines.append("  → Quality degradation acceptable")
    report_lines.append("  → NF4 generally better than FP4 for quality")

    # Best configuration
    def balanced_score(r):
        # Balance throughput and energy efficiency
        norm_throughput = r.metrics.tokens_per_second / fastest.metrics.tokens_per_second
        norm_energy = most_efficient.metrics.energy_per_token / r.metrics.energy_per_token
        return (norm_throughput + norm_energy) / 2

    best_overall = max(results, key=balanced_score)
    report_lines.append("\n--- RECOMMENDED CONFIGURATION ---")
    report_lines.append(f"\nBest overall balance: {_get_config_name(best_overall)}")
    report_lines.append(f"  Throughput: {best_overall.metrics.tokens_per_second:.2f} tokens/sec")
    report_lines.append(f"  Energy: {best_overall.metrics.energy_per_token:.8f} kWh/token")

    report_lines.append("\n" + "=" * 80)

    # Print and save
    report_text = "\n".join(report_lines)
    print(report_text)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_text)
    print(f"\nReport saved to: {output_file}")

    # Save JSON data
    json_output = output_file.with_suffix('.json')
    comparison_data = {
        "model": results[0].config.model_name,
        "configurations": [
            {
                "name": _get_config_name(r),
                "precision": r.config.precision,
                "quantization": {
                    "enabled": r.config.quantization.enabled,
                    "load_in_4bit": r.config.quantization.load_in_4bit,
                    "load_in_8bit": r.config.quantization.load_in_8bit,
                    "quant_type": r.config.quantization.quant_type,
                },
                "metrics": {
                    "tokens_per_second": r.metrics.tokens_per_second,
                    "energy_per_token": r.metrics.energy_per_token,
                    "total_energy_kwh": r.metrics.total_energy_kwh,
                    "co2_emissions": r.metrics.co2_emissions,
                },
            }
            for r in results
        ],
        "best_configuration": _get_config_name(best_overall),
    }

    json_output.write_text(json.dumps(comparison_data, indent=2))
    print(f"JSON data saved to: {json_output}")


def _get_config_name(result: ExperimentResult) -> str:
    """Get human-readable configuration name."""
    if not result.config.quantization.enabled:
        return "Full Precision (FP16)"
    elif result.config.quantization.load_in_8bit:
        return "8-bit Quantization"
    elif result.config.quantization.load_in_4bit:
        quant_type = result.config.quantization.quant_type
        return f"4-bit {quant_type.upper()} Quantization"
    return "Unknown Configuration"


def main():
    """Run quantization comparison."""

    print("=" * 70)
    print("QUANTIZATION COMPARISON")
    print("=" * 70)

    # Use a small model for quick testing
    # For production, use larger models where quantization matters more
    model_name = "gpt2"  # Try "gpt2-large" or "facebook/opt-6.7b" for better demo

    print(f"\nModel: {model_name}")
    print("\nTesting configurations:")
    print("  1. Full Precision (FP16)")
    print("  2. 8-bit Quantization")
    print("  3. 4-bit NF4 Quantization")
    print("  4. 4-bit FP4 Quantization")

    print("\nNote: Quantization requires CUDA/GPU")
    print("      For CPU-only, only full precision will work")

    output_dir = Path("./results/quantization_comparison")
    print(f"\nResults will be saved to: {output_dir}")

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)

    results = run_quantization_experiments(model_name, output_dir)

    if not results:
        print("\n✗ No experiments completed successfully!")
        print("  Check that CUDA is available for quantization")
        return

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON REPORT")
    print("=" * 70)

    report_file = output_dir / "quantization_report.txt"
    generate_quantization_report(results, report_file)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the quantization report")
    print("  2. Test with larger models (7B+ parameters)")
    print("  3. Evaluate quality degradation")
    print("  4. Choose best configuration for your use case")
    print("=" * 70)


if __name__ == "__main__":
    main()
