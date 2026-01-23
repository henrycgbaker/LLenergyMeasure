"""Runtime tests that require GPU hardware.

These tests verify that configuration parameters are actually applied at inference time,
not just parsed correctly. They require CUDA GPU access and take longer to run.

Run with:
    pytest tests/runtime/ -v                    # All runtime tests
    pytest tests/runtime/ -v -k pytorch         # PyTorch backend only
    pytest tests/runtime/ -v --backend vllm     # vLLM backend only
"""
