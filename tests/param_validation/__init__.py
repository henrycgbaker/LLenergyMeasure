"""Parameter Validation Testing Framework.

A systematic, programmatically-driven test framework that validates every
parameter across all backends (PyTorch, vLLM, TensorRT) - both for passthrough
and runtime behaviour.

Usage:
    # Run all param validation tests
    pytest tests/param_validation/ -v

    # Run only mockable (CI-safe) tests
    pytest tests/param_validation/ -v -m "not requires_gpu"

    # Run only vLLM tests
    pytest tests/param_validation/ -v -k "vllm"

    # Show coverage report
    python -m tests.param_validation.registry.discovery --coverage
"""
