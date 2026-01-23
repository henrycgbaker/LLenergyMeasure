"""Pytest fixtures and configuration for runtime tests.

These tests require actual GPU hardware and run real inference.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Test model - small, fast, no HF_TOKEN required
TEST_MODEL = "Qwen/Qwen2.5-0.5B"
TEST_SAMPLE_SIZE = 5
TEST_MAX_OUTPUT = 32


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["pytorch", "vllm", "tensorrt"],
        help="Run tests for specific backend only",
    )
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run quick subset of parameter tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring CUDA GPU")
    config.addinivalue_line("markers", "requires_vllm: mark test as requiring vLLM installation")
    config.addinivalue_line(
        "markers", "requires_tensorrt: mark test as requiring TensorRT-LLM installation"
    )
    config.addinivalue_line("markers", "slow: mark test as slow (>1 minute)")


def _check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_vllm_available() -> bool:
    """Check if vLLM is available."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def _check_tensorrt_available() -> bool:
    """Check if TensorRT-LLM is available."""
    try:
        import tensorrt_llm  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests based on hardware and installed packages."""
    cuda_available = _check_cuda_available()
    vllm_available = _check_vllm_available()
    tensorrt_available = _check_tensorrt_available()
    selected_backend = config.getoption("--backend")

    skip_no_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    skip_no_vllm = pytest.mark.skip(reason="vLLM not installed")
    skip_no_tensorrt = pytest.mark.skip(reason="TensorRT-LLM not installed")
    skip_wrong_backend = pytest.mark.skip(reason=f"Test not for {selected_backend}")

    for item in items:
        # Skip if no GPU
        if "requires_gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_no_gpu)

        # Skip if vLLM not available
        if "requires_vllm" in item.keywords and not vllm_available:
            item.add_marker(skip_no_vllm)

        # Skip if TensorRT not available
        if "requires_tensorrt" in item.keywords and not tensorrt_available:
            item.add_marker(skip_no_tensorrt)

        # Filter by backend if specified
        if selected_backend:
            # Check test name for backend
            test_name = item.name.lower()
            # Skip tests for other backends (but not shared tests)
            if selected_backend not in test_name and any(
                b in test_name for b in ["pytorch", "vllm", "tensorrt"]
            ):
                item.add_marker(skip_wrong_backend)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def test_model() -> str:
    """Get the test model name."""
    return TEST_MODEL


@pytest.fixture(scope="session")
def test_sample_size() -> int:
    """Get the test sample size."""
    return TEST_SAMPLE_SIZE


@pytest.fixture(scope="session")
def test_max_output() -> int:
    """Get the test max output tokens."""
    return TEST_MAX_OUTPUT


@pytest.fixture(scope="function")
def clean_results(project_root: Path) -> Generator[None, None, None]:
    """Clean results directories before and after test."""
    results_raw = project_root / "results" / "raw"
    results_agg = project_root / "results" / "aggregated"
    state_dir = project_root / ".state"

    # Clean before test
    for dir_path in [results_raw, results_agg, state_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)
        dir_path.mkdir(parents=True, exist_ok=True)

    yield

    # Clean after test (optional - can be removed if you want to inspect results)
    for dir_path in [results_raw, results_agg]:
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)
        dir_path.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="function")
def temp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test configs."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
