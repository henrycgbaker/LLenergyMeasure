#!/usr/bin/env python3
"""
Verify that v2.0 installation is working correctly.
"""

import sys


def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")

    try:
        # Core imports
        from llm_efficiency import __version__
        print(f"  ✓ llm_efficiency version: {__version__}")

        from llm_efficiency.config import ExperimentConfig
        print("  ✓ Config module")

        from llm_efficiency.metrics import FLOPsCalculator
        print("  ✓ Metrics module")

        from llm_efficiency.core.model_loader import load_model_and_tokenizer
        print("  ✓ Core module")

        from llm_efficiency.utils.logging import setup_logging
        print("  ✓ Utils module")

        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test creating configurations."""
    print("\nTesting configuration creation...")

    try:
        from llm_efficiency.config import ExperimentConfig, BatchingConfig

        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
            batching=BatchingConfig(batch_size=16),
        )

        assert config.model_name == "test-model"
        assert config.precision == "float16"
        assert config.batching.batch_size == 16

        print("  ✓ Configuration creation works")
        return True
    except Exception as e:
        print(f"  ✗ Config creation failed: {e}")
        return False


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")

    try:
        from llm_efficiency.config import ExperimentConfig
        from pydantic import ValidationError

        # Should fail - invalid batch size
        try:
            ExperimentConfig(model_name="test", batching={"batch_size": -1})
            print("  ✗ Validation should have failed")
            return False
        except (ValidationError, ValueError):
            print("  ✓ Validation works correctly")
            return True

    except Exception as e:
        print(f"  ✗ Validation test failed: {e}")
        return False


def test_legacy_migration():
    """Test v1.0 to v2.0 migration."""
    print("\nTesting legacy config migration...")

    try:
        from llm_efficiency.config import ExperimentConfig

        v1_config = {
            "model_name": "test-model",
            "fp_precision": "float16",
            "num_processes": 2,
            "batching_options": {"batch_size___fixed_batching": 32},
        }

        v2_config = ExperimentConfig.from_legacy_dict(v1_config)

        assert v2_config.model_name == "test-model"
        assert v2_config.precision == "float16"
        assert v2_config.num_processes == 2
        assert v2_config.batching.batch_size == 32

        print("  ✓ Legacy migration works")
        return True
    except Exception as e:
        print(f"  ✗ Migration failed: {e}")
        return False


def test_flops_calculator():
    """Test FLOPs calculator instantiation."""
    print("\nTesting FLOPs calculator...")

    try:
        from llm_efficiency.metrics import FLOPsCalculator

        calculator = FLOPsCalculator()
        assert calculator.cache == {}
        assert calculator.cache_dir.exists()

        print("  ✓ FLOPs calculator works")
        return True
    except Exception as e:
        print(f"  ✗ FLOPs calculator failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("LLM Efficiency v2.0 - Installation Verification")
    print("=" * 60)

    tests = [
        test_imports,
        test_config_creation,
        test_config_validation,
        test_legacy_migration,
        test_flops_calculator,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed! Installation is working correctly.")
        print("=" * 60)
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
