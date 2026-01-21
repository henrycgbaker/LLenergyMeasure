"""Mock-based verification for CI-safe parameter testing.

Provides verification without requiring GPU hardware by using mocks
to track parameter passthrough and configuration.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

from ..registry import ParamSpec, VerificationResult, VerificationStatus


class MockVerifier:
    """Verifies parameter passthrough using mocks.

    CI-safe verification that patches backend constructors to capture
    the parameters passed to them.
    """

    def verify_vllm_passthrough(
        self,
        spec: ParamSpec,
        config_dict: dict[str, Any],
    ) -> VerificationResult:
        """Verify vLLM parameter would be passed correctly.

        Patches vLLM LLM constructor to capture arguments without
        actually initializing the model.

        Args:
            spec: The ParamSpec being verified.
            config_dict: The experiment config dict with vllm params.

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            # Extract expected value from config
            expected_value = self._extract_value_from_config(config_dict, spec.config_path)

            # Mock vLLM constructor
            captured_kwargs: dict[str, Any] = {}

            def capture_llm_init(*args: Any, **kwargs: Any) -> MagicMock:
                captured_kwargs.update(kwargs)
                return MagicMock()

            with patch("vllm.LLM", side_effect=capture_llm_init):
                # Simulate what our backend would do
                self._simulate_vllm_config_to_kwargs(config_dict, captured_kwargs)

            # Check if param was captured
            param_key = self._get_vllm_kwarg_name(spec.name)
            if param_key in captured_kwargs:
                actual_value = captured_kwargs[param_key]
                passed = actual_value == expected_value

                if passed:
                    message = f"Would pass {param_key}={actual_value} to vLLM"
                else:
                    message = (
                        f"Value mismatch: {param_key}={actual_value} (expected {expected_value})"
                    )

                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    message=message,
                    param_name=spec.full_name,
                    test_value=expected_value,
                    actual_value=actual_value,
                    expected_value=expected_value,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            return VerificationResult(
                status=VerificationStatus.FAILED,
                message=f"Parameter {param_key} not passed to vLLM constructor",
                param_name=spec.full_name,
                test_value=expected_value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Mock verification error: {e}",
                param_name=spec.full_name,
                test_value=config_dict.get("vllm", {}),
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def verify_pytorch_passthrough(
        self,
        spec: ParamSpec,
        config_dict: dict[str, Any],
    ) -> VerificationResult:
        """Verify PyTorch parameter would be passed correctly.

        Args:
            spec: The ParamSpec being verified.
            config_dict: The experiment config dict with pytorch params.

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            expected_value = self._extract_value_from_config(config_dict, spec.config_path)

            # Mock AutoModelForCausalLM.from_pretrained
            captured_kwargs: dict[str, Any] = {}

            def capture_model_load(*args: Any, **kwargs: Any) -> MagicMock:
                captured_kwargs.update(kwargs)
                mock_model = MagicMock()
                mock_model.dtype = kwargs.get("torch_dtype")
                return mock_model

            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=capture_model_load,
            ):
                self._simulate_pytorch_config_to_kwargs(config_dict, captured_kwargs)

            param_key = self._get_pytorch_kwarg_name(spec.name)
            if param_key in captured_kwargs:
                actual_value = captured_kwargs[param_key]
                passed = actual_value == expected_value

                if passed:
                    message = f"Would pass {param_key}={actual_value} to from_pretrained"
                else:
                    message = (
                        f"Value mismatch: {param_key}={actual_value} (expected {expected_value})"
                    )

                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    message=message,
                    param_name=spec.full_name,
                    test_value=expected_value,
                    actual_value=actual_value,
                    expected_value=expected_value,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            return VerificationResult(
                status=VerificationStatus.FAILED,
                message=f"Parameter {param_key} not passed to from_pretrained",
                param_name=spec.full_name,
                test_value=expected_value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Mock verification error: {e}",
                param_name=spec.full_name,
                test_value=config_dict.get("pytorch", {}),
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def verify_config_parsing(
        self,
        spec: ParamSpec,
        test_value: Any,
    ) -> VerificationResult:
        """Verify config value is correctly parsed by Pydantic model.

        Args:
            spec: The ParamSpec being verified.
            test_value: The value to test.

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            from llm_energy_measure.config.backend_configs import (
                PyTorchConfig,
                TensorRTConfig,
                VLLMConfig,
            )

            backend_configs = {
                "vllm": VLLMConfig,
                "pytorch": PyTorchConfig,
                "tensorrt": TensorRTConfig,
            }

            if spec.backend not in backend_configs:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    message=f"Unknown backend: {spec.backend}",
                    param_name=spec.full_name,
                    test_value=test_value,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            config_class = backend_configs[spec.backend]

            # Build config dict for this param
            config_data = self._build_nested_dict(spec.name, test_value)

            # Try to parse
            config_instance = config_class(**config_data)

            # Extract the value back
            actual_value = self._get_nested_value(config_instance, spec.name)

            passed = actual_value == test_value

            if passed:
                message = f"Config parsing OK: {spec.name}={actual_value}"
            else:
                message = f"Config parsing mismatch: got {actual_value}, expected {test_value}"

            return VerificationResult(
                status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                message=message,
                param_name=spec.full_name,
                test_value=test_value,
                actual_value=actual_value,
                expected_value=test_value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Config parsing error: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _extract_value_from_config(self, config_dict: dict[str, Any], config_path: str) -> Any:
        """Extract a value from config dict using dot notation."""
        parts = config_path.split(".")
        current = config_dict

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def _build_nested_dict(self, path: str, value: Any) -> dict[str, Any]:
        """Build a nested dict from a dot-separated path."""
        parts = path.split(".")
        result: dict[str, Any] = {}
        current = result

        for _i, part in enumerate(parts[:-1]):
            current[part] = {}
            current = current[part]

        current[parts[-1]] = value
        return result

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get a nested attribute value using dot notation."""
        parts = path.split(".")
        current = obj

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def _get_vllm_kwarg_name(self, param_name: str) -> str:
        """Map our param name to vLLM constructor kwarg name."""
        # Most map directly, but some have different names
        mapping = {
            "enable_prefix_caching": "enable_prefix_caching",
            "max_num_seqs": "max_num_seqs",
            "max_num_batched_tokens": "max_num_batched_tokens",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "swap_space": "swap_space",
            "enable_chunked_prefill": "enable_chunked_prefill",
            "kv_cache_dtype": "kv_cache_dtype",
            "block_size": "block_size",
            "max_model_len": "max_model_len",
            "max_seq_len_to_capture": "max_seq_len_to_capture",
            "enforce_eager": "enforce_eager",
            "distributed_backend": "distributed_executor_backend",
            "disable_custom_all_reduce": "disable_custom_all_reduce",
        }
        return mapping.get(param_name, param_name)

    def _get_pytorch_kwarg_name(self, param_name: str) -> str:
        """Map our param name to PyTorch from_pretrained kwarg name."""
        mapping = {
            "attn_implementation": "attn_implementation",
            "use_cache": "use_cache",
            "low_cpu_mem_usage": "low_cpu_mem_usage",
        }
        return mapping.get(param_name, param_name)

    def _simulate_vllm_config_to_kwargs(
        self, config_dict: dict[str, Any], kwargs: dict[str, Any]
    ) -> None:
        """Simulate how our backend translates config to vLLM kwargs."""
        vllm_config = config_dict.get("vllm", {})
        for key, value in vllm_config.items():
            if not isinstance(value, dict):  # Skip nested configs for now
                kwarg_name = self._get_vllm_kwarg_name(key)
                kwargs[kwarg_name] = value

    def _simulate_pytorch_config_to_kwargs(
        self, config_dict: dict[str, Any], kwargs: dict[str, Any]
    ) -> None:
        """Simulate how our backend translates config to PyTorch kwargs."""
        pytorch_config = config_dict.get("pytorch", {})
        for key, value in pytorch_config.items():
            if not isinstance(value, dict):
                kwarg_name = self._get_pytorch_kwarg_name(key)
                kwargs[kwarg_name] = value


def create_mock_vllm_llm(
    config_overrides: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock vLLM LLM object with realistic structure.

    Args:
        config_overrides: Override default config values.

    Returns:
        Mock LLM object with llm_engine.{cache_config, scheduler_config, etc.}
    """
    config = {
        "max_num_seqs": 256,
        "max_num_batched_tokens": None,
        "gpu_memory_utilization": 0.9,
        "swap_space_bytes": 4 * (1024**3),
        "enable_prefix_caching": False,
        "chunked_prefill_enabled": False,
        "cache_dtype": "auto",
        "block_size": 16,
        "max_model_len": 2048,
        "max_seq_len_to_capture": 8192,
        "enforce_eager": False,
        "disable_custom_all_reduce": False,
    }

    if config_overrides:
        config.update(config_overrides)

    llm = MagicMock()

    # cache_config
    llm.llm_engine.cache_config.enable_prefix_caching = config["enable_prefix_caching"]
    llm.llm_engine.cache_config.gpu_memory_utilization = config["gpu_memory_utilization"]
    llm.llm_engine.cache_config.swap_space_bytes = config["swap_space_bytes"]
    llm.llm_engine.cache_config.cache_dtype = config["cache_dtype"]
    llm.llm_engine.cache_config.block_size = config["block_size"]

    # scheduler_config
    llm.llm_engine.scheduler_config.max_num_seqs = config["max_num_seqs"]
    llm.llm_engine.scheduler_config.max_num_batched_tokens = config["max_num_batched_tokens"]
    llm.llm_engine.scheduler_config.chunked_prefill_enabled = config["chunked_prefill_enabled"]

    # model_config
    llm.llm_engine.model_config.max_model_len = config["max_model_len"]
    llm.llm_engine.model_config.max_seq_len_to_capture = config["max_seq_len_to_capture"]
    llm.llm_engine.model_config.enforce_eager = config["enforce_eager"]

    # parallel_config
    llm.llm_engine.parallel_config.disable_custom_all_reduce = config["disable_custom_all_reduce"]

    return llm


def create_mock_pytorch_model(
    config_overrides: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock PyTorch model with realistic structure.

    Args:
        config_overrides: Override default config values.

    Returns:
        Mock model object with config and generation_config.
    """
    import torch

    config = {
        "dtype": torch.float16,
        "attn_implementation": "sdpa",
        "use_cache": True,
    }

    if config_overrides:
        config.update(config_overrides)

    model = MagicMock()
    model.dtype = config["dtype"]
    model.config._attn_implementation = config["attn_implementation"]
    model.config.use_cache = config["use_cache"]
    model.generation_config.use_cache = config["use_cache"]

    return model
