"""GPU-specific fixtures for manual e2e parameter validation tests.

These fixtures require actual GPU hardware and are designed for manual testing,
not CI. Run with: pytest tests/manual/ -v --tb=short

Small models used:
- gpt2 (124M params): Fast loading, good for decoder tests
- TinyLlama/TinyLlama-1.1B-Chat-v1.0: Small LLaMA architecture for backend tests
- facebook/opt-125m: Smallest OPT, good for vLLM tests

Note: Tests auto-load .env from project root for CUDA_VISIBLE_DEVICES, HF_TOKEN, etc.
On MIG systems, set CUDA_VISIBLE_DEVICES=0,1 in .env to avoid enumeration issues.
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path


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


_load_env_file()

import tempfile  # noqa: E402
from collections.abc import Generator  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

# =============================================================================
# Skip markers for GPU tests
# =============================================================================


def gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def vllm_available() -> bool:
    """Check if vLLM is installed."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def tensorrt_available() -> bool:
    """Check if TensorRT-LLM is installed."""
    try:
        import tensorrt_llm  # noqa: F401

        return True
    except ImportError:
        return False


def get_gpu_compute_capability() -> tuple[int, int] | None:
    """Get GPU compute capability (major, minor).

    Returns:
        Tuple of (major, minor) or None if not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return torch.cuda.get_device_capability(device)
    except Exception:
        pass
    return None


def is_hopper_or_newer() -> bool:
    """Check if GPU is Hopper (SM 9.0) or newer.

    Hopper GPUs support FP8 and other advanced features.
    """
    capability = get_gpu_compute_capability()
    if capability is None:
        return False
    major, minor = capability
    return major >= 9


def is_ampere_or_newer() -> bool:
    """Check if GPU is Ampere (SM 8.0) or newer.

    Ampere GPUs support BF16 natively.
    """
    capability = get_gpu_compute_capability()
    if capability is None:
        return False
    major, minor = capability
    return major >= 8


def flash_attention_available() -> bool:
    """Check if Flash Attention is installed."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def bf16_supported() -> bool:
    """Check if GPU supports bfloat16.

    This function handles MIG GPU configurations gracefully by catching
    CUDA device enumeration errors that can occur with MIG partitions.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()
    except Exception:
        # MIG GPUs or other CUDA device enumeration issues
        # Default to using Ampere check as fallback
        return is_ampere_or_newer()


def vllm_version() -> tuple[int, int, int] | None:
    """Get vLLM version as tuple (major, minor, patch).

    Returns:
        Tuple of (major, minor, patch) or None if not available.
    """
    try:
        import vllm

        version_str = vllm.__version__
        parts = version_str.split(".")[:3]
        return tuple(int(p) for p in parts)  # type: ignore
    except Exception:
        return None


def vllm_version_at_least(major: int, minor: int = 0, patch: int = 0) -> bool:
    """Check if vLLM version is at least the specified version."""
    version = vllm_version()
    if version is None:
        return False
    return version >= (major, minor, patch)


# Pytest markers
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
    not flash_attention_available(), reason="Flash Attention not installed"
)
requires_bf16 = pytest.mark.skipif(not bf16_supported(), reason="GPU does not support bfloat16")


def requires_vllm_version(major: int, minor: int = 0, patch: int = 0):
    """Create a skip marker for minimum vLLM version requirement."""
    return pytest.mark.skipif(
        not vllm_version_at_least(major, minor, patch),
        reason=f"Requires vLLM >= {major}.{minor}.{patch}",
    )


# =============================================================================
# GPU cleanup utilities
# =============================================================================


def cleanup_gpu_memory() -> None:
    """Force GPU memory cleanup between tests."""
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
# Test models (small enough for quick tests)
# =============================================================================

# GPT-2 base (124M params) - fastest for basic tests
SMALL_MODEL_GPT2 = "gpt2"

# TinyLlama (1.1B params) - LLaMA architecture, good for backend testing
SMALL_MODEL_TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# OPT-125M - smallest OPT, good for vLLM tests
SMALL_MODEL_OPT = "facebook/opt-125m"

# Model for speculative decoding draft
DRAFT_MODEL = "facebook/opt-125m"

# Quantized models for testing (may need to be verified they exist)
AWQ_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ"  # May not exist, placeholder
GPTQ_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"  # May not exist, placeholder


@pytest.fixture
def small_model_name() -> str:
    """Return the smallest available model for quick tests."""
    return SMALL_MODEL_GPT2


@pytest.fixture
def tinyllama_model_name() -> str:
    """Return TinyLlama model name for LLaMA architecture tests."""
    return SMALL_MODEL_TINYLLAMA


@pytest.fixture
def opt_model_name() -> str:
    """Return OPT-125M model name for vLLM tests."""
    return SMALL_MODEL_OPT


@pytest.fixture
def draft_model_name() -> str:
    """Return draft model name for speculative decoding tests."""
    return DRAFT_MODEL


# =============================================================================
# Test prompts
# =============================================================================

# Deterministic prompt (should produce consistent outputs with temp=0)
DETERMINISTIC_PROMPT = "The capital of France is"

# Longer prompt for batch/throughput testing
BATCH_TEST_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks in simple terms.",
    "How does backpropagation work?",
    "What is the difference between supervised and unsupervised learning?",
    "Describe a decision tree algorithm.",
    "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.",
    "What is gradient descent?",
]

# Prompts with common prefix for prefix caching tests
PREFIX_CACHE_PROMPTS = [
    "The following is a summary of machine learning: Machine learning is a subset of AI.",
    "The following is a summary of machine learning: Neural networks are computational models.",
    "The following is a summary of machine learning: Deep learning uses multiple layers.",
    "The following is a summary of machine learning: Supervised learning uses labels.",
]


@pytest.fixture
def deterministic_prompt() -> str:
    """Return a prompt for deterministic output testing."""
    return DETERMINISTIC_PROMPT


@pytest.fixture
def batch_test_prompts() -> list[str]:
    """Return prompts for batch/throughput testing."""
    return BATCH_TEST_PROMPTS.copy()


@pytest.fixture
def prefix_cache_prompts() -> list[str]:
    """Return prompts with common prefix for prefix caching tests."""
    return PREFIX_CACHE_PROMPTS.copy()


# =============================================================================
# Temporary directories for results
# =============================================================================


@pytest.fixture
def temp_results_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test results."""
    temp_dir = tempfile.mkdtemp(prefix="llm_bench_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_engine_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for TensorRT engine caching."""
    temp_dir = tempfile.mkdtemp(prefix="trt_engine_cache_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Config builders
# =============================================================================


def build_base_config(
    model_name: str = SMALL_MODEL_GPT2,
    config_name: str = "test-config",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a base config dict for testing.

    Args:
        model_name: Model to use.
        config_name: Config identifier.
        **overrides: Additional config fields to override.

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


def build_vllm_config(
    model_name: str = SMALL_MODEL_OPT,
    config_name: str = "test-vllm-config",
    **vllm_params: Any,
) -> dict[str, Any]:
    """Build a config dict with vLLM backend settings.

    Args:
        model_name: Model to use.
        config_name: Config identifier.
        **vllm_params: vLLM-specific parameters.

    Returns:
        Config dict suitable for ExperimentConfig with vLLM backend.
    """
    base = build_base_config(model_name=model_name, config_name=config_name)
    base["backend"] = "vllm"
    if vllm_params:
        base["vllm"] = vllm_params
    return base


def build_tensorrt_config(
    model_name: str = SMALL_MODEL_GPT2,
    config_name: str = "test-tensorrt-config",
    **tensorrt_params: Any,
) -> dict[str, Any]:
    """Build a config dict with TensorRT backend settings.

    Args:
        model_name: Model to use.
        config_name: Config identifier.
        **tensorrt_params: TensorRT-specific parameters.

    Returns:
        Config dict suitable for ExperimentConfig with TensorRT backend.
    """
    base = build_base_config(model_name=model_name, config_name=config_name)
    base["backend"] = "tensorrt"
    if tensorrt_params:
        base["tensorrt"] = tensorrt_params
    return base


@pytest.fixture
def base_config_factory():
    """Factory fixture for building test configs."""
    return build_base_config


@pytest.fixture
def vllm_config_factory():
    """Factory fixture for building vLLM test configs."""
    return build_vllm_config


@pytest.fixture
def tensorrt_config_factory():
    """Factory fixture for building TensorRT test configs."""
    return build_tensorrt_config


# =============================================================================
# Verification utilities
# =============================================================================


class ParameterVerifier:
    """Utility class for verifying parameter application.

    Provides methods to check that parameters are actually applied
    to models and backends at runtime.
    """

    @staticmethod
    def check_model_dtype(model: Any, expected_dtype: str) -> tuple[bool, str]:
        """Verify model is loaded in expected dtype.

        Args:
            model: HuggingFace model instance.
            expected_dtype: Expected dtype string (float16, bfloat16, float32).

        Returns:
            Tuple of (passed, message).
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        expected = dtype_map.get(expected_dtype)
        actual = model.dtype

        if actual == expected:
            return True, f"Model dtype matches: {actual}"
        return False, f"Model dtype mismatch: expected {expected}, got {actual}"

    @staticmethod
    def check_generation_config(model: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Verify model's generation_config has expected parameter value.

        Args:
            model: HuggingFace model instance.
            param: Parameter name in generation_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        gen_config = getattr(model, "generation_config", None)
        if gen_config is None:
            return False, "Model has no generation_config"

        actual = getattr(gen_config, param, None)
        if actual == expected:
            return True, f"generation_config.{param}={actual} (expected)"
        return False, f"generation_config.{param}={actual} (expected {expected})"

    @staticmethod
    def check_quantization_state(model: Any) -> dict[str, Any]:
        """Extract quantization state from model.

        Returns:
            Dict with quantization info (is_quantized, bits, method).
        """
        result = {
            "is_quantized": False,
            "bits": None,
            "method": None,
            "dtype": str(model.dtype),
        }

        # Check for BitsAndBytes quantization
        try:
            if hasattr(model, "is_loaded_in_4bit"):
                result["is_quantized"] = model.is_loaded_in_4bit
                if result["is_quantized"]:
                    result["bits"] = 4
                    result["method"] = "bitsandbytes"
            elif hasattr(model, "is_loaded_in_8bit"):
                result["is_quantized"] = model.is_loaded_in_8bit
                if result["is_quantized"]:
                    result["bits"] = 8
                    result["method"] = "bitsandbytes"
        except Exception:
            pass

        return result

    @staticmethod
    def check_outputs_identical(outputs: list[str]) -> tuple[bool, str]:
        """Check if all outputs in list are identical (for determinism test).

        Args:
            outputs: List of output strings.

        Returns:
            Tuple of (all_identical, message).
        """
        if not outputs:
            return False, "No outputs to compare"

        first = outputs[0]
        for i, out in enumerate(outputs[1:], 1):
            if out != first:
                return (
                    False,
                    f"Output {i} differs from output 0: '{out[:50]}...' vs '{first[:50]}...'",
                )

        return True, f"All {len(outputs)} outputs identical"

    @staticmethod
    def check_outputs_varied(outputs: list[str], min_unique: int = 2) -> tuple[bool, str]:
        """Check if outputs have sufficient variation (for sampling test).

        Args:
            outputs: List of output strings.
            min_unique: Minimum number of unique outputs required.

        Returns:
            Tuple of (has_variation, message).
        """
        unique = set(outputs)
        if len(unique) >= min_unique:
            return True, f"Found {len(unique)} unique outputs (required >= {min_unique})"
        return False, f"Only {len(unique)} unique outputs, expected >= {min_unique}"

    @staticmethod
    def get_gpu_memory_mb() -> float:
        """Get current GPU memory usage in MB.

        Returns:
            Memory usage in MB, or 0 if unavailable.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def get_gpu_memory_reserved_mb() -> float:
        """Get current GPU memory reserved in MB.

        Returns:
            Memory reserved in MB, or 0 if unavailable.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass
        return 0.0


class VLLMVerifier:
    """Utility class for verifying vLLM-specific parameter application."""

    @staticmethod
    def check_cache_config(llm: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check vLLM cache_config parameter.

        Args:
            llm: vLLM LLM instance.
            param: Parameter name in cache_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            cache_config = llm.llm_engine.cache_config
            actual = getattr(cache_config, param, None)
            if actual == expected:
                return True, f"cache_config.{param}={actual}"
            return False, f"cache_config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking cache_config: {e}"

    @staticmethod
    def check_model_config(llm: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check vLLM model_config parameter.

        Args:
            llm: vLLM LLM instance.
            param: Parameter name in model_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            model_config = llm.llm_engine.model_config
            actual = getattr(model_config, param, None)
            if actual == expected:
                return True, f"model_config.{param}={actual}"
            return False, f"model_config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking model_config: {e}"

    @staticmethod
    def check_scheduler_config(llm: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check vLLM scheduler_config parameter.

        Args:
            llm: vLLM LLM instance.
            param: Parameter name in scheduler_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            scheduler_config = llm.llm_engine.scheduler_config
            actual = getattr(scheduler_config, param, None)
            if actual == expected:
                return True, f"scheduler_config.{param}={actual}"
            return False, f"scheduler_config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking scheduler_config: {e}"

    @staticmethod
    def check_parallel_config(llm: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check vLLM parallel_config parameter.

        Args:
            llm: vLLM LLM instance.
            param: Parameter name in parallel_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            parallel_config = llm.llm_engine.parallel_config
            actual = getattr(parallel_config, param, None)
            if actual == expected:
                return True, f"parallel_config.{param}={actual}"
            return False, f"parallel_config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking parallel_config: {e}"

    @staticmethod
    def check_lora_config(llm: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check vLLM lora_config parameter.

        Args:
            llm: vLLM LLM instance.
            param: Parameter name in lora_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            lora_config = llm.llm_engine.lora_config
            if lora_config is None:
                if expected is None:
                    return True, "lora_config is None as expected"
                return False, "lora_config is None but expected a value"
            actual = getattr(lora_config, param, None)
            if actual == expected:
                return True, f"lora_config.{param}={actual}"
            return False, f"lora_config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking lora_config: {e}"

    @staticmethod
    def check_speculative_config(llm: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check vLLM speculative_config parameter.

        Args:
            llm: vLLM LLM instance.
            param: Parameter name in speculative_config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            # Speculative config may be in different locations depending on vLLM version
            spec_config = getattr(llm.llm_engine, "speculative_config", None)
            if spec_config is None:
                if expected is None:
                    return True, "speculative_config is None as expected"
                return False, "speculative_config is None but expected a value"
            actual = getattr(spec_config, param, None)
            if actual == expected:
                return True, f"speculative_config.{param}={actual}"
            return False, f"speculative_config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking speculative_config: {e}"


class TensorRTVerifier:
    """Utility class for verifying TensorRT-specific parameter application."""

    @staticmethod
    def check_engine_config(executor: Any, param: str, expected: Any) -> tuple[bool, str]:
        """Check TensorRT engine config parameter.

        Args:
            executor: TensorRT-LLM executor instance.
            param: Parameter name in config.
            expected: Expected value.

        Returns:
            Tuple of (passed, message).
        """
        try:
            # TensorRT-LLM config access may vary by version
            config = getattr(executor, "config", None)
            if config is None:
                return False, "Executor has no config attribute"
            actual = getattr(config, param, None)
            if actual == expected:
                return True, f"config.{param}={actual}"
            return False, f"config.{param}={actual} (expected {expected})"
        except Exception as e:
            return False, f"Error checking engine config: {e}"


@pytest.fixture
def verifier() -> ParameterVerifier:
    """Provide parameter verification utility."""
    return ParameterVerifier()


@pytest.fixture
def vllm_verifier() -> VLLMVerifier:
    """Provide vLLM verification utility."""
    return VLLMVerifier()


@pytest.fixture
def tensorrt_verifier() -> TensorRTVerifier:
    """Provide TensorRT verification utility."""
    return TensorRTVerifier()
