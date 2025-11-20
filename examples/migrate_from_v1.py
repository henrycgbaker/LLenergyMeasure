#!/usr/bin/env python3
"""
Example: Migrating from v1.0 to v2.0 API.

This shows how to:
1. Use the legacy compatibility layer (temporary solution)
2. Properly migrate to v2.0 API (recommended)
"""

# Example 1: Using backward compatibility (works but deprecated)
def example_backward_compatibility():
    """Use v1.0 code with minimal changes."""
    print("=" * 60)
    print("Example 1: Backward Compatibility Layer")
    print("=" * 60)

    # Old v1.0 style config
    legacy_config = {
        "model_name": "hf-internal-testing/tiny-random-gpt2",
        "fp_precision": "float16",
        "num_processes": 1,
        "batching_options": {
            "batch_size___fixed_batching": 16,
            "adaptive_batching": False,
        },
        "decoder_config": {
            "decoding_mode": "greedy",
            "decoder_temperature": 1.0,
            "decoder_top_k": None,
            "decoder_top_p": None,
        },
        "quantization_config": {
            "quantization": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            # NO MORE cached_flops_for_quantised_models needed!
        },
    }

    # Import from legacy module (will show deprecation warning)
    from llm_efficiency.legacy import load_model_tokenizer

    print("\nLoading model using legacy API (v1.0 style)...")
    model, tokenizer = load_model_tokenizer(legacy_config)
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n⚠️  This works but shows deprecation warnings.")
    print("    Please migrate to v2.0 API (see Example 2)")


# Example 2: Proper migration to v2.0
def example_proper_migration():
    """Migrate to v2.0 API properly."""
    print("\n" + "=" * 60)
    print("Example 2: Proper v2.0 Migration (Recommended)")
    print("=" * 60)

    from llm_efficiency.config import (
        ExperimentConfig,
        BatchingConfig,
        DecoderConfig,
        QuantizationConfig,
    )
    from llm_efficiency.core.model_loader import load_model_and_tokenizer
    from llm_efficiency.metrics import FLOPsCalculator

    # v2.0 style: Type-safe Pydantic config
    config = ExperimentConfig(
        model_name="hf-internal-testing/tiny-random-gpt2",
        precision="float16",  # Note: 'precision' not 'fp_precision'
        num_processes=1,
        batching=BatchingConfig(
            batch_size=16,  # Note: 'batch_size' not 'batch_size___fixed_batching'
            adaptive=False,
        ),
        decoder=DecoderConfig(
            mode="greedy",  # Note: 'mode' not 'decoding_mode'
            temperature=1.0,
        ),
        quantization=QuantizationConfig(
            enabled=False,
            load_in_4bit=False,
            load_in_8bit=False,
            # NO cached_flops_for_quantised_models! It's automatic now!
        ),
    )

    print("\nLoading model using v2.0 API...")
    model, tokenizer = load_model_and_tokenizer(config)
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Use the new FLOPs calculator
    print("\nComputing FLOPs using v2.0 calculator...")
    calculator = FLOPsCalculator()
    device = next(model.parameters()).device

    flops = calculator.get_flops(
        model=model,
        model_name=config.model_name,
        sequence_length=128,
        device=device,
        is_quantized=config.quantization.enabled,
    )

    print(f"✓ FLOPs: {flops:,}")
    print("\n✅ This is the recommended approach!")


# Example 3: Automatic migration helper
def example_automatic_migration():
    """Use automatic migration from v1.0 dict to v2.0 config."""
    print("\n" + "=" * 60)
    print("Example 3: Automatic Config Migration")
    print("=" * 60)

    from llm_efficiency.config import ExperimentConfig

    # Old v1.0 dict config
    v1_config = {
        "model_name": "hf-internal-testing/tiny-random-gpt2",
        "fp_precision": "float16",
        "num_processes": 1,
        "batching_options": {"batch_size___fixed_batching": 32},
        "decoder_config": {"decoding_mode": "top_k", "decoder_top_k": 50},
    }

    print("\nMigrating v1.0 dict to v2.0 ExperimentConfig...")
    v2_config = ExperimentConfig.from_legacy_dict(v1_config)

    print(f"✓ Migration successful!")
    print(f"  Model: {v2_config.model_name}")
    print(f"  Precision: {v2_config.precision}")
    print(f"  Batch size: {v2_config.batching.batch_size}")
    print(f"  Decoder: {v2_config.decoder.mode} (k={v2_config.decoder.top_k})")

    # Save migrated config
    print("\nSaving migrated config to JSON...")
    import json

    config_dict = v2_config.model_dump(exclude_none=True)
    with open("migrated_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("✓ Saved to: migrated_config.json")


# Example 4: Side-by-side comparison
def example_comparison():
    """Compare v1.0 vs v2.0 code."""
    print("\n" + "=" * 60)
    print("Example 4: v1.0 vs v2.0 Comparison")
    print("=" * 60)

    print("\n━━━ v1.0 Code (Old) ━━━")
    print(
        """
# v1.0: Nested dictionaries, no validation
config = {
    "batching_options": {
        "batch_size___fixed_batching": 16  # Easy to typo!
    },
    "quantization_config": {
        "cached_flops_for_quantised_models": 52638582308864  # Same for ALL models!
    }
}

# Import from weird paths
from experiment_core_utils.b_model_loader import load_model_tokenizer
from experiment_core_utils.h_metrics_compute import get_flops

model, tokenizer = load_model_tokenizer(config)
flops = get_flops(model, inputs)  # Wrong for quantized!
    """
    )

    print("\n━━━ v2.0 Code (New) ━━━")
    print(
        """
# v2.0: Type-safe Pydantic models with validation
from llm_efficiency.config import ExperimentConfig, BatchingConfig
from llm_efficiency.core.model_loader import load_model_and_tokenizer
from llm_efficiency.metrics import FLOPsCalculator

config = ExperimentConfig(
    model_name="TinyLlama-1.1B",
    batching=BatchingConfig(batch_size=16)  # IDE autocomplete!
    # No cached_flops needed - automatic!
)

model, tokenizer = load_model_and_tokenizer(config)

calculator = FLOPsCalculator()  # Caching, accurate for quantized!
flops = calculator.get_flops(model, config.model_name, 128, device, is_quantized=False)
    """
    )

    print("\n✅ Benefits of v2.0:")
    print("   • Type safety and validation")
    print("   • IDE autocomplete")
    print("   • Accurate FLOPs for quantized models")
    print("   • Cleaner imports")
    print("   • Better error messages")


def main():
    """Run all examples."""
    print("\n" + "█" * 60)
    print("   Migration Guide: v1.0 → v2.0")
    print("█" * 60)

    # Example 1: Backward compatibility
    example_backward_compatibility()

    # Example 2: Proper migration
    example_proper_migration()

    # Example 3: Automatic migration
    example_automatic_migration()

    # Example 4: Comparison
    example_comparison()

    print("\n" + "█" * 60)
    print("For more info, see:")
    print("  • CHANGELOG.md - Full migration guide")
    print("  • README.md - Documentation")
    print("  • scripts/migrate_config.py - Batch migration tool")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
