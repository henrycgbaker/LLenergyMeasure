#!/usr/bin/env python3
"""
Complete workflow example demonstrating all v0.2.0 features.

This shows the clean, simple API for running LLM efficiency experiments.
"""

from pathlib import Path

from llm_efficiency.config import ExperimentConfig, BatchingConfig, DecoderConfig
from llm_efficiency.core import (
    setup_accelerator,
    generate_experiment_id,
    load_model_and_tokenizer,
    run_inference_experiment,
)
from llm_efficiency.data import load_prompts_from_dataset, filter_prompts_by_length
from llm_efficiency.metrics import FLOPsCalculator, track_energy, get_gpu_memory_stats
from llm_efficiency.storage import ResultsManager, create_results
from llm_efficiency.utils.logging import setup_logging


def main():
    """Run complete LLM efficiency experiment."""

    # Setup logging with rich output
    setup_logging(level="INFO", rich_output=True)

    print("=" * 70)
    print("LLM Efficiency Measurement - Complete Workflow")
    print("=" * 70)

    # 1. CONFIGURATION
    print("\n[1/7] Creating configuration...")
    config = ExperimentConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        precision="float16",
        num_input_prompts=20,
        max_input_tokens=128,
        max_output_tokens=128,
        batching=BatchingConfig(batch_size=4),
        decoder=DecoderConfig(mode="greedy"),
        results_dir="results",
    )
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Precision: {config.precision}")
    print(f"✓ Batch size: {config.batching.batch_size}")

    # 2. DISTRIBUTED SETUP
    print("\n[2/7] Setting up distributed environment...")
    accelerator = setup_accelerator(mixed_precision="no")
    experiment_id = generate_experiment_id(accelerator)
    print(f"✓ Experiment ID: {experiment_id}")
    print(f"✓ Process: {accelerator.process_index}/{accelerator.num_processes}")
    print(f"✓ Device: {accelerator.device}")

    # 3. LOAD MODEL
    print("\n[3/7] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    model = accelerator.prepare(model)
    device = accelerator.device

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {total_params:,}")
    print(f"✓ Model loaded on: {device}")

    # 4. PREPARE DATA
    print("\n[4/7] Loading and processing prompts...")
    prompts = load_prompts_from_dataset(
        dataset_name="AIEnergyScore/text_generation",
        split="train",
        num_prompts=config.num_input_prompts,
    )

    # Filter by length
    filtered_prompts = filter_prompts_by_length(
        prompts,
        tokenizer,
        max_tokens=config.max_input_tokens,
    )
    print(f"✓ Loaded {len(prompts)} prompts")
    print(f"✓ Filtered to {len(filtered_prompts)} prompts")

    # 5. RUN INFERENCE WITH ENERGY TRACKING
    print("\n[5/7] Running inference with energy tracking...")

    with track_energy(experiment_id, output_dir=Path("results/energy")) as energy_tracker:
        # Run inference
        outputs, inference_metrics = run_inference_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=filtered_prompts,
            config=config,
            accelerator=accelerator,
            warmup=True,
        )

    energy_results = energy_tracker.get_results()

    print(f"✓ Throughput: {inference_metrics['tokens_per_second']:.2f} tokens/s")
    print(f"✓ Latency: {inference_metrics['avg_latency_per_query']:.3f} s/query")
    print(f"✓ Energy: {energy_results['energy_consumed_kwh']:.6f} kWh")
    print(f"✓ Emissions: {energy_results['emissions_kg_co2']:.6f} kg CO2")

    # 6. COMPUTE METRICS
    print("\n[6/7] Computing additional metrics...")

    # FLOPs calculation
    calculator = FLOPsCalculator()
    flops = calculator.get_flops(
        model=model,
        model_name=config.model_name,
        sequence_length=config.max_input_tokens,
        device=device,
        is_quantized=config.quantization.enabled,
    )
    print(f"✓ FLOPs: {flops:,}")

    # Memory stats
    memory_stats = get_gpu_memory_stats(device)
    memory_mb = memory_stats.get("gpu_current_memory_allocated_bytes", 0) / 1e6
    print(f"✓ GPU Memory: {memory_mb:.2f} MB")

    # 7. SAVE RESULTS
    print("\n[7/7] Saving results...")

    # Create results object
    results = create_results(
        experiment_id=experiment_id,
        config=config.to_dict(),
        model_info={
            "model_name": config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "precision": config.precision,
        },
        inference_metrics=inference_metrics,
        compute_metrics={
            "flops": flops,
            "gpu_memory_allocated_mb": memory_mb,
            "gpu_memory_peak_mb": memory_stats.get("gpu_max_memory_allocated_bytes", 0) / 1e6,
        },
        energy_metrics=energy_results,
        outputs=outputs if config.save_outputs else None,
    )

    # Save using ResultsManager
    manager = ResultsManager(results_dir=Path(config.results_dir))
    saved_path = manager.save_experiment(results)

    print(f"✓ Results saved to: {saved_path}")

    # Export to CSV
    csv_path = Path(config.results_dir) / "all_experiments.csv"
    manager.export_to_csv(csv_path)
    print(f"✓ CSV exported to: {csv_path}")

    # Display summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"\nPerformance:")
    print(f"  • Throughput: {inference_metrics['tokens_per_second']:.2f} tokens/s")
    print(f"  • Latency: {inference_metrics['avg_latency_per_query']:.3f} s/query")
    print(f"  • Total time: {inference_metrics['total_time_seconds']:.2f} s")

    print(f"\nEnergy:")
    print(f"  • Total: {energy_results['energy_consumed_kwh']:.6f} kWh")
    print(f"  • CPU: {energy_results['cpu_energy_kwh']:.6f} kWh")
    print(f"  • GPU: {energy_results['gpu_energy_kwh']:.6f} kWh")
    print(f"  • RAM: {energy_results['ram_energy_kwh']:.6f} kWh")
    print(f"  • Emissions: {energy_results['emissions_kg_co2']:.6f} kg CO2")

    print(f"\nEfficiency:")
    if results.efficiency:
        for metric, value in results.efficiency.items():
            print(f"  • {metric}: {value:.2e}")

    print(f"\nResults location: {saved_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
