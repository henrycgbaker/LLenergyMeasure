#!/usr/bin/env python3
"""
Compare FLOPs and performance across different quantization levels.

This example demonstrates:
1. The fixed FLOPs calculator (no more hardcoded values!)
2. Comparing quantized vs non-quantized models
3. Using Pydantic configurations
"""

import torch
from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.model_loader import load_model_and_tokenizer
from llm_efficiency.metrics import FLOPsCalculator
from llm_efficiency.utils.logging import setup_logging


def run_flops_comparison(model_name: str = "hf-internal-testing/tiny-random-gpt2"):
    """Compare FLOPs across different precision/quantization settings."""

    setup_logging(level="INFO", rich_output=True)

    print("=" * 70)
    print("FLOPs Comparison: v1.0 Bug vs v2.0 Fix")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print("\nIn v1.0, ALL quantized models used the same FLOPs: 52,638,582,308,864")
    print("In v2.0, each model gets its accurate FLOPs!\n")

    configurations = [
        ("float32", QuantizationConfig()),
        ("float16", QuantizationConfig()),
        ("8-bit quantized", QuantizationConfig(load_in_8bit=True)),
        ("4-bit quantized", QuantizationConfig(load_in_4bit=True)),
    ]

    calculator = FLOPsCalculator()
    sequence_length = 128

    results = []

    for name, quant_config in configurations:
        print(f"\n{'─' * 70}")
        print(f"Testing: {name}")
        print(f"{'─' * 70}")

        try:
            # Create config
            if "float" in name:
                precision = name  # "float32" or "float16"
                config = ExperimentConfig(
                    model_name=model_name,
                    precision=precision,
                    quantization=quant_config,
                )
            else:
                config = ExperimentConfig(
                    model_name=model_name,
                    precision="float16",
                    quantization=quant_config,
                )

            # Load model
            print(f"  Loading model...")
            model, _ = load_model_and_tokenizer(config)
            device = next(model.parameters()).device

            # Calculate FLOPs
            print(f"  Computing FLOPs...")
            flops = calculator.get_flops(
                model=model,
                model_name=model_name,
                sequence_length=sequence_length,
                device=device,
                is_quantized=quant_config.enabled,
            )

            # Get model size
            param_count = sum(p.numel() for p in model.parameters())
            if torch.cuda.is_available():
                model_size_mb = torch.cuda.memory_allocated(device) / 1e6
            else:
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

            results.append(
                {
                    "name": name,
                    "flops": flops,
                    "params": param_count,
                    "size_mb": model_size_mb,
                }
            )

            print(f"  ✓ FLOPs: {flops:,}")
            print(f"  ✓ Parameters: {param_count:,}")
            print(f"  ✓ Model size: {model_size_mb:.2f} MB")

            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({"name": name, "flops": None, "params": None, "size_mb": None})

    # Summary table
    print(f"\n{'═' * 70}")
    print("SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'Configuration':<20} {'FLOPs':>20} {'Params':>15} {'Size (MB)':>12}")
    print(f"{'─' * 70}")

    for result in results:
        if result["flops"] is not None:
            print(
                f"{result['name']:<20} {result['flops']:>20,} "
                f"{result['params']:>15,} {result['size_mb']:>12.2f}"
            )
        else:
            print(f"{result['name']:<20} {'Failed':>20}")

    print(f"{'═' * 70}")

    # Show the difference
    if len([r for r in results if r["flops"] is not None]) > 1:
        print("\n💡 Key Insight:")
        print("   In v1.0, quantized models would all show: 52,638,582,308,864 FLOPs")
        print("   In v2.0, each configuration gets its accurate FLOPs value!")
        print("   This fixes energy efficiency calculations (FLOPs/Joule)")


if __name__ == "__main__":
    run_flops_comparison()
