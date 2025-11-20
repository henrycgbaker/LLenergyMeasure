"""
Custom Configuration Example
=============================

This example demonstrates advanced configuration options and
customization of the LLM efficiency measurement workflow.

Features demonstrated:
- Creating custom configurations programmatically
- Using YAML configuration files
- Overriding default settings
- Advanced configuration patterns
- Configuration validation
"""

from pathlib import Path
import yaml

from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.experiment import run_experiment


# Example 1: Programmatic Configuration
def example_programmatic_config():
    """Create configuration programmatically with all options."""

    print("=" * 70)
    print("Example 1: Programmatic Configuration")
    print("=" * 70)

    # Create detailed configuration
    config = ExperimentConfig(
        # Model settings
        model_name="gpt2",
        precision="float16",  # Options: float32, float16, bfloat16, float8

        # Quantization settings
        quantization=QuantizationConfig(
            enabled=True,
            load_in_4bit=True,
            load_in_8bit=False,
            quant_type="nf4",  # Options: nf4, fp4
            compute_dtype="float16",  # Options: float16, bfloat16
        ),

        # Inference settings
        batch_size=8,
        num_batches=50,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,

        # Dataset settings
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="test",

        # Performance settings
        use_cache=True,
        num_warmup_batches=3,
        seed=42,

        # Output settings
        output_dir=Path("./results/custom_experiment"),
        save_frequency=10,
    )

    print("\nConfiguration created:")
    print(f"  Model: {config.model_name}")
    print(f"  Precision: {config.precision}")
    print(f"  Quantization: {config.quantization.enabled}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max length: {config.max_length}")
    print(f"  Output: {config.output_dir}")

    return config


# Example 2: YAML Configuration
def example_yaml_config():
    """Create and load configuration from YAML file."""

    print("\n" + "=" * 70)
    print("Example 2: YAML Configuration")
    print("=" * 70)

    # Define YAML configuration
    yaml_config = """
# LLM Efficiency Experiment Configuration

# Model Configuration
model_name: "gpt2-medium"
precision: "float16"

# Quantization Configuration
quantization:
  enabled: false
  load_in_4bit: false
  load_in_8bit: false
  quant_type: null
  compute_dtype: "float16"

# Inference Configuration
batch_size: 16
num_batches: 100
max_length: 512
temperature: 1.0
top_p: 0.95
top_k: 50
repetition_penalty: 1.0

# Dataset Configuration
dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
dataset_split: "test"

# Performance Configuration
use_cache: true
num_warmup_batches: 5
seed: 42

# Output Configuration
output_dir: "./results/yaml_experiment"
save_frequency: 20
"""

    # Save to file
    config_file = Path("./experiment_config.yaml")
    config_file.write_text(yaml_config)
    print(f"\nYAML configuration saved to: {config_file}")

    # Load configuration
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Convert to ExperimentConfig
    config = ExperimentConfig(**config_dict)

    print("\nConfiguration loaded from YAML:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Num batches: {config.num_batches}")

    return config


# Example 3: Configuration Templates
def example_config_templates():
    """Demonstrate common configuration templates."""

    print("\n" + "=" * 70)
    print("Example 3: Configuration Templates")
    print("=" * 70)

    # Template 1: Quick Testing
    quick_test = ExperimentConfig(
        model_name="gpt2",
        precision="float16",
        batch_size=4,
        num_batches=5,  # Very few for quick test
        max_length=64,
        output_dir=Path("./results/quick_test"),
    )
    print("\n1. Quick Test Template:")
    print("   - Small model, few batches, short sequences")
    print("   - Use for: Development, debugging, CI/CD")

    # Template 2: Production Benchmark
    production = ExperimentConfig(
        model_name="facebook/opt-6.7b",
        precision="float16",
        quantization=QuantizationConfig(
            enabled=True,
            load_in_4bit=True,
            quant_type="nf4",
        ),
        batch_size=8,
        num_batches=100,
        max_length=512,
        num_warmup_batches=10,
        output_dir=Path("./results/production_benchmark"),
    )
    print("\n2. Production Benchmark Template:")
    print("   - Large model, quantized, many batches")
    print("   - Use for: Performance evaluation, reporting")

    # Template 3: Energy Optimization
    energy_focused = ExperimentConfig(
        model_name="gpt2",
        precision="float16",
        quantization=QuantizationConfig(
            enabled=True,
            load_in_8bit=True,
        ),
        batch_size=16,  # Larger batches for efficiency
        num_batches=50,
        max_length=256,
        use_cache=True,
        output_dir=Path("./results/energy_optimization"),
    )
    print("\n3. Energy Optimization Template:")
    print("   - Quantized, large batches, caching enabled")
    print("   - Use for: Minimizing energy consumption")

    # Template 4: Quality Testing
    quality_focused = ExperimentConfig(
        model_name="gpt2-large",
        precision="float32",  # Full precision for quality
        batch_size=1,  # Single sample for detailed analysis
        num_batches=100,
        max_length=1024,  # Long sequences
        temperature=0.8,
        top_p=0.92,
        output_dir=Path("./results/quality_testing"),
    )
    print("\n4. Quality Testing Template:")
    print("   - Full precision, long sequences, single batch")
    print("   - Use for: Quality evaluation, output analysis")

    return {
        "quick_test": quick_test,
        "production": production,
        "energy_focused": energy_focused,
        "quality_focused": quality_focused,
    }


# Example 4: Configuration Validation
def example_config_validation():
    """Demonstrate configuration validation and error handling."""

    print("\n" + "=" * 70)
    print("Example 4: Configuration Validation")
    print("=" * 70)

    # Valid configuration
    try:
        valid_config = ExperimentConfig(
            model_name="gpt2",
            batch_size=8,
            num_batches=10,
            output_dir=Path("./results/valid"),
        )
        print("\n✓ Valid configuration accepted")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")

    # Invalid: Negative batch size
    print("\nTesting invalid batch size...")
    try:
        invalid_config = ExperimentConfig(
            model_name="gpt2",
            batch_size=-1,  # Invalid!
            output_dir=Path("./results/invalid"),
        )
    except Exception as e:
        print(f"  ✓ Caught error: {type(e).__name__}")

    # Invalid: Both 4-bit and 8-bit enabled
    print("\nTesting conflicting quantization...")
    try:
        invalid_quant = ExperimentConfig(
            model_name="gpt2",
            quantization=QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                load_in_8bit=True,  # Can't have both!
            ),
            output_dir=Path("./results/invalid_quant"),
        )
    except Exception as e:
        print(f"  ✓ Caught error: {type(e).__name__}")

    print("\n✓ Configuration validation working correctly!")


# Example 5: Environment-Specific Configurations
def example_environment_configs():
    """Demonstrate configurations for different environments."""

    print("\n" + "=" * 70)
    print("Example 5: Environment-Specific Configurations")
    print("=" * 70)

    # Development environment
    dev_config = ExperimentConfig(
        model_name="gpt2",  # Small model
        precision="float16",
        batch_size=2,
        num_batches=3,  # Minimal
        max_length=64,
        output_dir=Path("./results/dev"),
    )
    print("\n1. Development Environment:")
    print("   - Fast iterations, minimal resources")

    # CI/CD environment
    ci_config = ExperimentConfig(
        model_name="gpt2",
        precision="float16",
        batch_size=4,
        num_batches=5,
        max_length=128,
        seed=42,  # Reproducible
        output_dir=Path("./results/ci"),
    )
    print("\n2. CI/CD Environment:")
    print("   - Reproducible, quick tests, no GPU required")

    # Staging environment
    staging_config = ExperimentConfig(
        model_name="gpt2-medium",
        precision="float16",
        quantization=QuantizationConfig(
            enabled=True,
            load_in_8bit=True,
        ),
        batch_size=8,
        num_batches=20,
        max_length=256,
        output_dir=Path("./results/staging"),
    )
    print("\n3. Staging Environment:")
    print("   - Medium-scale testing, GPU available")

    # Production environment
    prod_config = ExperimentConfig(
        model_name="facebook/opt-6.7b",
        precision="float16",
        quantization=QuantizationConfig(
            enabled=True,
            load_in_4bit=True,
            quant_type="nf4",
        ),
        batch_size=16,
        num_batches=100,
        max_length=512,
        num_warmup_batches=10,
        use_cache=True,
        output_dir=Path("./results/production"),
    )
    print("\n4. Production Environment:")
    print("   - Full-scale benchmark, optimized settings")

    return {
        "dev": dev_config,
        "ci": ci_config,
        "staging": staging_config,
        "production": prod_config,
    }


def main():
    """Run all configuration examples."""

    print("\n" + "=" * 70)
    print("CUSTOM CONFIGURATION EXAMPLES")
    print("=" * 70)

    # Run examples
    config1 = example_programmatic_config()
    config2 = example_yaml_config()
    templates = example_config_templates()
    example_config_validation()
    env_configs = example_environment_configs()

    print("\n" + "=" * 70)
    print("CONFIGURATION EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Create configs programmatically or from YAML")
    print("  2. Use templates for common scenarios")
    print("  3. Validation prevents invalid configurations")
    print("  4. Adapt configs to different environments")
    print("  5. Override defaults for specific needs")
    print("\nTo run an experiment with custom config:")
    print("  result = run_experiment(config)")
    print("\nTo use CLI with config file:")
    print("  llm-efficiency run experiment_config.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
