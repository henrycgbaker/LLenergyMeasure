"""
Error Handling and Recovery
============================

This example demonstrates error handling patterns and recovery strategies
for robust LLM efficiency measurement.

Features demonstrated:
- Catching and handling common errors
- Retry mechanisms
- Graceful degradation
- Logging and debugging
- Best practices for production use
"""

import logging
from pathlib import Path

from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.experiment import run_experiment
from llm_efficiency.utils.exceptions import (
    LLMEfficiencyError,
    ModelLoadingError,
    InferenceError,
    ConfigurationError,
    QuantizationError,
    NetworkError,
)
from llm_efficiency.utils.retry import retry_with_exponential_backoff


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example 1: Handling Model Loading Errors
def example_model_loading_errors():
    """Demonstrate handling model loading errors."""

    print("=" * 70)
    print("Example 1: Model Loading Error Handling")
    print("=" * 70)

    # Try loading non-existent model
    try:
        print("\nAttempting to load non-existent model...")
        config = ExperimentConfig(
            model_name="this-model-does-not-exist",
            batch_size=4,
            num_batches=10,
            output_dir=Path("./results/error_test"),
        )
        result = run_experiment(config)

    except ModelLoadingError as e:
        print(f"\n✓ Caught ModelLoadingError:")
        print(f"  {e}")
        print("\nRecovery strategy:")
        print("  1. Check model name spelling")
        print("  2. Verify model exists on Hugging Face")
        print("  3. Check internet connection")
        print("  4. Try with a known model (e.g., 'gpt2')")

    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")


# Example 2: Handling Quantization Errors
def example_quantization_errors():
    """Demonstrate handling quantization errors."""

    print("\n" + "=" * 70)
    print("Example 2: Quantization Error Handling")
    print("=" * 70)

    # Try quantization without GPU (will fail)
    try:
        print("\nAttempting quantization (may fail on CPU)...")
        config = ExperimentConfig(
            model_name="gpt2",
            precision="float16",
            quantization=QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
            ),
            batch_size=4,
            num_batches=10,
            output_dir=Path("./results/quant_error_test"),
        )

        # This will fail on CPU-only systems
        result = run_experiment(config)
        print("✓ Quantization successful (GPU available)")

    except QuantizationError as e:
        print(f"\n✓ Caught QuantizationError:")
        print(f"  {e}")
        print("\nRecovery strategy:")
        print("  1. Check CUDA availability")
        print("  2. Fall back to full precision")
        print("  3. Use CPU-compatible configuration")

        # Fallback: Try without quantization
        print("\nAttempting fallback to full precision...")
        try:
            fallback_config = ExperimentConfig(
                model_name="gpt2",
                precision="float16",
                quantization=QuantizationConfig(enabled=False),
                batch_size=4,
                num_batches=10,
                output_dir=Path("./results/fallback_test"),
            )
            print("✓ Fallback configuration created successfully")

        except Exception as e:
            print(f"✗ Fallback failed: {e}")

    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")


# Example 3: Handling Configuration Errors
def example_configuration_errors():
    """Demonstrate handling configuration errors."""

    print("\n" + "=" * 70)
    print("Example 3: Configuration Error Handling")
    print("=" * 70)

    # Invalid batch size
    try:
        print("\nAttempting invalid configuration (batch_size=-1)...")
        config = ExperimentConfig(
            model_name="gpt2",
            batch_size=-1,  # Invalid!
            output_dir=Path("./results/config_error"),
        )

    except (ConfigurationError, ValueError) as e:
        print(f"\n✓ Caught configuration error:")
        print(f"  {type(e).__name__}: {e}")
        print("\nRecovery strategy:")
        print("  1. Validate configuration before use")
        print("  2. Use default values for optional parameters")
        print("  3. Provide clear error messages to users")

    # Conflicting quantization settings
    try:
        print("\nAttempting conflicting quantization settings...")
        config = ExperimentConfig(
            model_name="gpt2",
            quantization=QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                load_in_8bit=True,  # Can't have both!
            ),
            output_dir=Path("./results/config_error2"),
        )

    except (ConfigurationError, ValueError) as e:
        print(f"\n✓ Caught configuration conflict:")
        print(f"  {type(e).__name__}: {e}")


# Example 4: Retry Mechanisms
@retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1.0,
    exceptions=(NetworkError, ConnectionError, TimeoutError),
)
def flaky_network_operation():
    """Simulate a network operation that might fail."""
    import random

    if random.random() < 0.7:  # 70% failure rate
        raise NetworkError("Simulated network failure")

    return "Success!"


def example_retry_mechanisms():
    """Demonstrate retry mechanisms for transient failures."""

    print("\n" + "=" * 70)
    print("Example 4: Retry Mechanisms")
    print("=" * 70)

    print("\nAttempting flaky network operation with retries...")
    print("(Will retry up to 3 times with exponential backoff)")

    try:
        result = flaky_network_operation()
        print(f"\n✓ Operation succeeded: {result}")

    except NetworkError as e:
        print(f"\n✗ Operation failed after all retries: {e}")
        print("\nRecovery strategy:")
        print("  1. Check network connectivity")
        print("  2. Verify API endpoints")
        print("  3. Use cached data if available")
        print("  4. Queue for later retry")


# Example 5: Comprehensive Error Handling
def run_experiment_with_error_handling(config: ExperimentConfig):
    """
    Run experiment with comprehensive error handling.

    This is a production-ready wrapper that handles all common errors.
    """

    logger.info(f"Starting experiment: {config.model_name}")

    try:
        # Validate configuration first
        logger.debug("Validating configuration...")
        # Config validation happens automatically in ExperimentConfig

        # Run experiment
        logger.info("Running experiment...")
        result = run_experiment(config)

        logger.info(f"✓ Experiment completed: {result.experiment_id}")
        return result

    except ModelLoadingError as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Suggestions:")
        logger.info("  - Verify model name")
        logger.info("  - Check internet connection")
        logger.info("  - Ensure sufficient disk space")
        raise

    except QuantizationError as e:
        logger.error(f"Quantization failed: {e}")
        logger.info("Falling back to full precision...")

        # Automatic fallback
        fallback_config = ExperimentConfig(
            model_name=config.model_name,
            precision=config.precision,
            quantization=QuantizationConfig(enabled=False),
            batch_size=config.batch_size,
            num_batches=config.num_batches,
            max_length=config.max_length,
            output_dir=config.output_dir,
        )

        logger.info("Retrying with fallback configuration...")
        return run_experiment(fallback_config)

    except InferenceError as e:
        logger.error(f"Inference failed: {e}")
        logger.info("Suggestions:")
        logger.info("  - Check GPU memory")
        logger.info("  - Reduce batch size")
        logger.info("  - Reduce max_length")
        raise

    except ConfigurationError as e:
        logger.error(f"Invalid configuration: {e}")
        logger.info("Review configuration parameters")
        raise

    except LLMEfficiencyError as e:
        logger.error(f"LLM Efficiency error: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        raise


def example_production_error_handling():
    """Demonstrate production-ready error handling."""

    print("\n" + "=" * 70)
    print("Example 5: Production Error Handling")
    print("=" * 70)

    # Test with valid configuration
    print("\n1. Testing with valid configuration...")
    try:
        config = ExperimentConfig(
            model_name="gpt2",
            precision="float16",
            batch_size=2,
            num_batches=2,  # Very small for quick test
            max_length=64,
            output_dir=Path("./results/production_test"),
        )

        result = run_experiment_with_error_handling(config)
        print(f"✓ Experiment succeeded: {result.experiment_id}")

    except Exception as e:
        print(f"✗ Experiment failed: {e}")

    # Test with quantization (will fallback on CPU)
    print("\n2. Testing with quantization (may fallback)...")
    try:
        config = ExperimentConfig(
            model_name="gpt2",
            precision="float16",
            quantization=QuantizationConfig(
                enabled=True,
                load_in_8bit=True,
            ),
            batch_size=2,
            num_batches=2,
            max_length=64,
            output_dir=Path("./results/quant_fallback_test"),
        )

        result = run_experiment_with_error_handling(config)
        print(f"✓ Experiment succeeded: {result.experiment_id}")

    except Exception as e:
        print(f"✗ Experiment failed even with fallback: {e}")


def main():
    """Run all error handling examples."""

    print("\n" + "=" * 70)
    print("ERROR HANDLING EXAMPLES")
    print("=" * 70)

    # Run examples
    example_model_loading_errors()
    example_quantization_errors()
    example_configuration_errors()
    example_retry_mechanisms()
    example_production_error_handling()

    print("\n" + "=" * 70)
    print("ERROR HANDLING EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Use specific exception types for targeted handling")
    print("  2. Implement retry logic for transient failures")
    print("  3. Provide fallback strategies (e.g., disable quantization)")
    print("  4. Log errors with context for debugging")
    print("  5. Give users clear error messages and suggestions")
    print("\nException hierarchy:")
    print("  LLMEfficiencyError (base)")
    print("  ├── ModelLoadingError")
    print("  ├── InferenceError")
    print("  ├── ConfigurationError")
    print("  ├── QuantizationError")
    print("  ├── DataError")
    print("  ├── MetricsError")
    print("  ├── StorageError")
    print("  └── NetworkError")
    print("\nFor production:")
    print("  - Always use try-except blocks")
    print("  - Implement automatic fallback strategies")
    print("  - Log errors for monitoring")
    print("  - Provide actionable error messages")
    print("  - Test error paths regularly")
    print("=" * 70)


if __name__ == "__main__":
    main()
