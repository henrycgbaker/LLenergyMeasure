"""Pytest configuration and shared fixtures."""

import pytest
import torch
from pathlib import Path
from typing import Generator
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_efficiency.config import ExperimentConfig, BatchingConfig
from llm_efficiency.metrics import FLOPsCalculator


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def flops_calculator(temp_cache_dir: Path) -> FLOPsCalculator:
    """FLOPs calculator with temporary cache."""
    return FLOPsCalculator(cache_dir=temp_cache_dir)


@pytest.fixture(scope="session")
def tiny_model() -> Generator:
    """
    Load a tiny GPT-2 model for testing.

    Uses a tiny random model for fast loading in tests.
    """
    model_name = "hf-internal-testing/tiny-random-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    yield model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def tiny_tokenizer() -> Generator:
    """Load tokenizer for tiny model."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    yield tokenizer


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """Sample experiment configuration for testing."""
    return ExperimentConfig(
        model_name="hf-internal-testing/tiny-random-gpt2",
        precision="float16",
        num_input_prompts=10,
        max_input_tokens=32,
        max_output_tokens=16,
        batching=BatchingConfig(batch_size=2),
    )


@pytest.fixture
def sample_prompts() -> list[str]:
    """Sample prompts for testing."""
    return [
        "Hello, how are you?",
        "What is the meaning of life?",
        "The quick brown fox",
        "Python is a programming language",
        "Machine learning is",
    ]


@pytest.fixture
def mock_device() -> torch.device:
    """Mock device for testing (CPU if CUDA not available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Reset CUDA memory between tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
