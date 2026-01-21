"""Pytest configuration for parameter validation tests.

Provides fixtures, markers, and hardware-conditional skip logic.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from .registry import (
    HardwareRequirement,
    check_requirements,
    detect_hardware,
    flash_attn_available,
    get_hardware_summary,
    gpu_available,
    is_ampere_or_newer,
    is_hopper_or_newer,
    multi_gpu_available,
    tensorrt_available,
    vllm_available,
)


def _load_env_file() -> None:
    """Load .env file from project root if it exists."""
    env_file = Path(__file__).parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


# Load env at module import
_load_env_file()


# =============================================================================
# Pytest Markers
# =============================================================================

requires_gpu = pytest.mark.skipif(not gpu_available(), reason="CUDA GPU not available")
requires_vllm = pytest.mark.skipif(not vllm_available(), reason="vLLM not installed")
requires_tensorrt = pytest.mark.skipif(
    not tensorrt_available(), reason="TensorRT-LLM not installed"
)
requires_hopper = pytest.mark.skipif(
    not is_hopper_or_newer(), reason="Requires Hopper (SM 9.0+) GPU for FP8"
)
requires_ampere = pytest.mark.skipif(
    not is_ampere_or_newer(), reason="Requires Ampere (SM 8.0+) GPU"
)
requires_flash_attn = pytest.mark.skipif(
    not flash_attn_available(), reason="Flash Attention not installed"
)
requires_multi_gpu = pytest.mark.skipif(not multi_gpu_available(), reason="Multiple GPUs required")


def skip_unless_requirements_met(requirements: set[HardwareRequirement]):
    """Create a skip marker based on hardware requirements."""
    met, unmet = check_requirements(requirements)
    if not met:
        return pytest.mark.skip(reason="; ".join(unmet))
    return pytest.mark.usefixtures()


# =============================================================================
# GPU Cleanup
# =============================================================================


def cleanup_gpu_memory() -> None:
    """Force GPU memory cleanup."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def gpu_cleanup() -> Generator[None, None, None]:
    """Auto-cleanup GPU memory before and after each test."""
    cleanup_gpu_memory()
    yield
    cleanup_gpu_memory()


# =============================================================================
# Test Models
# =============================================================================

# Small models for fast testing
SMALL_MODEL_GPT2 = "gpt2"
SMALL_MODEL_OPT = "facebook/opt-125m"
SMALL_MODEL_TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Default test prompts
DEFAULT_PROMPT = "The capital of France is"
BATCH_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "How does backpropagation work?",
    "What is overfitting?",
]


@pytest.fixture
def small_model() -> str:
    """Return the smallest model (gpt2) for quick tests."""
    return SMALL_MODEL_GPT2


@pytest.fixture
def opt_model() -> str:
    """Return OPT-125M for vLLM tests."""
    return SMALL_MODEL_OPT


@pytest.fixture
def tinyllama_model() -> str:
    """Return TinyLlama for LLaMA architecture tests."""
    return SMALL_MODEL_TINYLLAMA


@pytest.fixture
def test_prompt() -> str:
    """Return a simple test prompt."""
    return DEFAULT_PROMPT


@pytest.fixture
def batch_prompts() -> list[str]:
    """Return prompts for batch testing."""
    return BATCH_PROMPTS.copy()


# =============================================================================
# Config Builders
# =============================================================================


def build_base_config(
    model_name: str = SMALL_MODEL_GPT2,
    config_name: str = "param-test",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal experiment config dict for testing.

    Args:
        model_name: Model to use.
        config_name: Config identifier.
        **overrides: Additional config fields.

    Returns:
        Config dict suitable for ExperimentConfig.
    """
    base = {
        "config_name": config_name,
        "model_name": model_name,
        "max_input_tokens": 64,
        "max_output_tokens": 32,
        "num_input_prompts": 1,
        "gpus": [0],
    }
    base.update(overrides)
    return base


def build_pytorch_config(
    model_name: str = SMALL_MODEL_GPT2, **pytorch_params: Any
) -> dict[str, Any]:
    """Build a config dict with PyTorch backend settings."""
    base = build_base_config(model_name=model_name)
    base["backend"] = "pytorch"
    if pytorch_params:
        base["pytorch"] = pytorch_params
    return base


def build_vllm_config(model_name: str = SMALL_MODEL_OPT, **vllm_params: Any) -> dict[str, Any]:
    """Build a config dict with vLLM backend settings."""
    base = build_base_config(model_name=model_name)
    base["backend"] = "vllm"
    if vllm_params:
        base["vllm"] = vllm_params
    return base


def build_tensorrt_config(
    model_name: str = SMALL_MODEL_GPT2, **tensorrt_params: Any
) -> dict[str, Any]:
    """Build a config dict with TensorRT backend settings."""
    base = build_base_config(model_name=model_name)
    base["backend"] = "tensorrt"
    if tensorrt_params:
        base["tensorrt"] = tensorrt_params
    return base


@pytest.fixture
def base_config_factory():
    """Factory fixture for building test configs."""
    return build_base_config


@pytest.fixture
def pytorch_config_factory():
    """Factory fixture for building PyTorch test configs."""
    return build_pytorch_config


@pytest.fixture
def vllm_config_factory():
    """Factory fixture for building vLLM test configs."""
    return build_vllm_config


@pytest.fixture
def tensorrt_config_factory():
    """Factory fixture for building TensorRT test configs."""
    return build_tensorrt_config


# =============================================================================
# Hardware Info Fixture
# =============================================================================


@pytest.fixture(scope="session")
def hardware_profile():
    """Get detected hardware profile for the session."""
    return detect_hardware()


@pytest.fixture(scope="session")
def hardware_summary():
    """Get hardware summary string for the session."""
    return get_hardware_summary()


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring CUDA GPU")
    config.addinivalue_line("markers", "requires_vllm: mark test as requiring vLLM")
    config.addinivalue_line("markers", "requires_tensorrt: mark test as requiring TensorRT-LLM")
    config.addinivalue_line("markers", "requires_hopper: mark test as requiring Hopper GPU")
    config.addinivalue_line("markers", "requires_ampere: mark test as requiring Ampere GPU")
    config.addinivalue_line(
        "markers", "requires_flash_attn: mark test as requiring Flash Attention"
    )
    config.addinivalue_line("markers", "requires_multi_gpu: mark test as requiring multiple GPUs")
    config.addinivalue_line("markers", "backend(name): mark test for a specific backend")
    config.addinivalue_line("markers", "category(name): mark test for a specific category")


def pytest_collection_modifyitems(config, items):
    """Auto-apply skip markers based on hardware requirements."""
    for item in items:
        # Skip GPU tests if no GPU
        if "requires_gpu" in item.keywords and not gpu_available():
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))

        # Skip vLLM tests if not installed
        if "requires_vllm" in item.keywords and not vllm_available():
            item.add_marker(pytest.mark.skip(reason="vLLM not installed"))

        # Skip TensorRT tests if not installed
        if "requires_tensorrt" in item.keywords and not tensorrt_available():
            item.add_marker(pytest.mark.skip(reason="TensorRT-LLM not installed"))

        # Skip Hopper tests if not Hopper GPU
        if "requires_hopper" in item.keywords and not is_hopper_or_newer():
            item.add_marker(pytest.mark.skip(reason="Requires Hopper (SM 9.0+) GPU"))

        # Skip Ampere tests if not Ampere GPU
        if "requires_ampere" in item.keywords and not is_ampere_or_newer():
            item.add_marker(pytest.mark.skip(reason="Requires Ampere (SM 8.0+) GPU"))


def pytest_report_header(config):
    """Add hardware info to pytest report header."""
    lines = ["", "Parameter Validation Framework", "=" * 40]
    lines.append(get_hardware_summary())
    return lines
