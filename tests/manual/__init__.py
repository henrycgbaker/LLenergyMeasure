"""Manual e2e tests requiring GPU hardware.

These tests verify that experiment parameters are actually applied to
models and backends at runtime, not just parsed correctly. They require
a CUDA GPU to run.

Test Modules:
- test_parameter_validation_e2e.py: Comprehensive parameter validation suite
- test_pytorch_runtime_params.py: PyTorch backend runtime verification
- test_vllm_runtime_params.py: vLLM backend runtime verification
- test_backend_params_manual.py: Config model validation (no GPU required)

Running Tests:
    # All manual tests (requires GPU)
    pytest tests/manual/ -v

    # Just PyTorch runtime tests
    python tests/manual/test_pytorch_runtime_params.py

    # Just vLLM runtime tests
    python tests/manual/test_vllm_runtime_params.py

    # E2E parameter validation (pytest)
    pytest tests/manual/test_parameter_validation_e2e.py -v

Note: These tests are not included in CI as they require GPU hardware.
"""
