"""
Basic LLM Efficiency Experiment
================================

This example demonstrates the simplest way to run an efficiency measurement
experiment with a small language model.

Features demonstrated:
- Creating basic experiment configuration
- Running inference with energy tracking
- Viewing and saving results
"""

from pathlib import Path
from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.experiment import run_experiment


def main():
    """Run a basic efficiency experiment."""

    # 1. Create experiment configuration
    # Using a small model for quick testing (GPT-2 small is ~124M parameters)
    config = ExperimentConfig(
        model_name="gpt2",  # Small model for testing
        precision="float16",  # Half precision for efficiency
        quantization=QuantizationConfig(enabled=False),  # No quantization for baseline
        batch_size=4,
        num_batches=10,  # Small number for quick test
        max_length=128,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        output_dir=Path("./results/basic_experiment"),
    )

    print("=" * 70)
    print("BASIC EXPERIMENT: GPT-2 Efficiency Measurement")
    print("=" * 70)
    print(f"\nModel: {config.model_name}")
    print(f"Precision: {config.precision}")
    print(f"Batch size: {config.batch_size}")
    print(f"Number of batches: {config.num_batches}")
    print(f"Max sequence length: {config.max_length}")
    print(f"\nResults will be saved to: {config.output_dir}")
    print("\n" + "=" * 70)

    # 2. Run the experiment
    # This will:
    # - Load the model and tokenizer
    # - Prepare the dataset
    # - Run inference with energy tracking
    # - Calculate efficiency metrics
    # - Save results to disk
    try:
        result = run_experiment(config)

        # 3. Display key results
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE!")
        print("=" * 70)
        print(f"\nExperiment ID: {result.experiment_id}")
        print(f"Timestamp: {result.timestamp}")

        print("\n--- Performance Metrics ---")
        print(f"Total samples: {result.metrics.total_samples}")
        print(f"Total tokens generated: {result.metrics.total_tokens}")
        print(f"Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")
        print(f"Latency per token: {result.metrics.latency_per_token:.4f} sec")

        print("\n--- Energy Metrics ---")
        print(f"Total energy: {result.metrics.total_energy_kwh:.6f} kWh")
        print(f"Energy per token: {result.metrics.energy_per_token:.8f} kWh")
        print(f"CO2 emissions: {result.metrics.co2_emissions:.6f} kg")

        print("\n--- Compute Metrics ---")
        print(f"Total FLOPs: {result.metrics.total_flops:,.0f}")
        print(f"FLOPs per token: {result.metrics.flops_per_token:,.0f}")

        print("\n--- Results Saved ---")
        print(f"JSON: {config.output_dir / f'{result.experiment_id}.json'}")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\nError running experiment: {e}")
        print("Make sure you have:")
        print("1. Internet connection (to download model)")
        print("2. Sufficient disk space (~500MB for GPT-2)")
        print("3. CUDA available (or set device_map='cpu' in config)")
        raise


if __name__ == "__main__":
    # Run the experiment
    result = main()

    # You can now analyze the results further
    print("\nTo view results later, use the CLI:")
    print(f"  llm-efficiency show {result.experiment_id}")
    print("\nTo export results:")
    print("  llm-efficiency export results.csv")
