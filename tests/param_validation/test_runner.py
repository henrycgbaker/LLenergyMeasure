"""Test runner for parameter validation framework.

Provides utilities to generate and run tests from ParamSpec definitions,
supporting both GPU-based and mock-based verification.
"""

from __future__ import annotations

import gc
from typing import Any

from .registry import (
    ParamSpec,
    RunSummary,
    VerificationResult,
    VerificationStatus,
    VerificationType,
    check_requirements,
    get_registry,
)
from .verifiers import BehaviourVerifier, IntrospectionVerifier, MockVerifier, PassthroughVerifier


def cleanup_gpu() -> None:
    """Clean up GPU memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


class ParamTestRunner:
    """Runs parameter validation tests based on ParamSpec definitions."""

    def __init__(self) -> None:
        self.passthrough_verifier = PassthroughVerifier()
        self.behaviour_verifier = BehaviourVerifier()
        self.introspection_verifier = IntrospectionVerifier()
        self.mock_verifier = MockVerifier()

    def run_all(
        self,
        backend: str | None = None,
        gpu_available: bool = True,
        cleanup_between: bool = True,
    ) -> RunSummary:
        """Run all applicable tests.

        Args:
            backend: Filter to specific backend (vllm, pytorch, tensorrt).
            gpu_available: Whether GPU is available for tests.
            cleanup_between: Clean GPU memory between tests.

        Returns:
            RunSummary with all results.
        """
        registry = get_registry()
        summary = RunSummary()

        specs = registry.by_backend(backend) if backend else registry.all_specs

        for spec in specs:
            if spec.skip_reason:
                result = VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    message=spec.skip_reason,
                    param_name=spec.full_name,
                    test_value=None,
                )
                summary.add_result(result)
                continue

            # Check hardware requirements
            met, unmet = check_requirements(spec.hardware_requirements)
            if not met and gpu_available:
                result = VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    message=f"Missing requirements: {', '.join(unmet)}",
                    param_name=spec.full_name,
                    test_value=None,
                )
                summary.add_result(result)
                continue

            # Run test for each test value
            for test_value in spec.test_values:
                if gpu_available and met:
                    result = self._run_gpu_test(spec, test_value)
                else:
                    result = self._run_mock_test(spec, test_value)

                summary.add_result(result)

                if cleanup_between:
                    cleanup_gpu()

        return summary

    def run_spec(
        self,
        spec: ParamSpec,
        test_value: Any,
        use_gpu: bool = True,
    ) -> VerificationResult:
        """Run a single ParamSpec test.

        Args:
            spec: The ParamSpec to test.
            test_value: The value to test.
            use_gpu: Whether to use GPU for this test.

        Returns:
            VerificationResult.
        """
        if spec.skip_reason:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                message=spec.skip_reason,
                param_name=spec.full_name,
                test_value=test_value,
            )

        if use_gpu:
            return self._run_gpu_test(spec, test_value)
        return self._run_mock_test(spec, test_value)

    def _run_gpu_test(self, spec: ParamSpec, test_value: Any) -> VerificationResult:
        """Run a GPU-based test."""
        match spec.verification_type:
            case VerificationType.PASSTHROUGH:
                return self._run_passthrough_test(spec, test_value)
            case VerificationType.BEHAVIOUR:
                return self._run_behaviour_test(spec, test_value)
            case VerificationType.INTROSPECTION:
                return self._run_introspection_test(spec, test_value)
            case VerificationType.MOCK:
                return self._run_mock_test(spec, test_value)
            case _:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    message=f"Unknown verification type: {spec.verification_type}",
                    param_name=spec.full_name,
                    test_value=test_value,
                )

    def _run_passthrough_test(self, spec: ParamSpec, test_value: Any) -> VerificationResult:
        """Run a passthrough verification test."""
        try:
            instance = self._create_backend_instance(spec, test_value)
            result = self.passthrough_verifier.verify(spec, instance, test_value)
            self._cleanup_instance(instance)
            return result
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Passthrough test error: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
            )

    def _run_behaviour_test(self, spec: ParamSpec, test_value: Any) -> VerificationResult:
        """Run a behaviour verification test."""
        try:
            # Get baseline output
            baseline_instance = self._create_backend_instance(spec, spec.default_value)
            baseline_output = self._generate_output(baseline_instance, spec)
            self._cleanup_instance(baseline_instance)
            cleanup_gpu()

            # Get test output
            test_instance = self._create_backend_instance(spec, test_value)
            test_output = self._generate_output(test_instance, spec)
            self._cleanup_instance(test_instance)

            return self.behaviour_verifier.verify(spec, baseline_output, test_output, test_value)
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Behaviour test error: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
            )

    def _run_introspection_test(self, spec: ParamSpec, test_value: Any) -> VerificationResult:
        """Run an introspection verification test."""
        try:
            instance = self._create_backend_instance(spec, test_value)

            # Determine what to introspect based on category
            if spec.category == "precision":
                result = self.introspection_verifier.verify_dtype(spec, instance, test_value)
            elif spec.category == "attention":
                result = self.introspection_verifier.verify_attention_impl(
                    spec, instance, test_value
                )
            else:
                result = VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    message=f"No introspection handler for category: {spec.category}",
                    param_name=spec.full_name,
                    test_value=test_value,
                )

            self._cleanup_instance(instance)
            return result
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Introspection test error: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
            )

    def _run_mock_test(self, spec: ParamSpec, test_value: Any) -> VerificationResult:
        """Run a mock-based verification test."""
        return self.mock_verifier.verify_config_parsing(spec, test_value)

    def _create_backend_instance(self, spec: ParamSpec, test_value: Any) -> Any:
        """Create a backend instance with the specified parameter value."""
        if spec.backend == "vllm":
            return self._create_vllm_instance(spec, test_value)
        elif spec.backend == "pytorch":
            return self._create_pytorch_instance(spec, test_value)
        elif spec.backend == "tensorrt":
            return self._create_tensorrt_instance(spec, test_value)
        else:
            raise ValueError(f"Unknown backend: {spec.backend}")

    def _create_vllm_instance(self, spec: ParamSpec, test_value: Any) -> Any:
        """Create a vLLM LLM instance."""
        from vllm import LLM

        # Build kwargs based on param path
        kwargs = {
            "model": "facebook/opt-125m",
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
        }

        # Map param name to vLLM constructor arg
        param_name = spec.name.split(".")[-1]  # Handle nested params
        if spec.name.startswith("lora."):
            # LoRA params need special handling
            if param_name == "enabled":
                kwargs["enable_lora"] = test_value
            elif param_name == "max_loras":
                kwargs["enable_lora"] = True
                kwargs["max_loras"] = test_value
            elif param_name == "max_rank":
                kwargs["enable_lora"] = True
                kwargs["max_lora_rank"] = test_value
        else:
            kwargs[param_name] = test_value

        return LLM(**kwargs)

    def _create_pytorch_instance(self, spec: ParamSpec, test_value: Any) -> Any:
        """Create a PyTorch model instance."""
        import torch
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": "gpt2",
            "torch_dtype": torch.float16,
            "device_map": "cuda",
            "low_cpu_mem_usage": True,
        }

        param_name = spec.name.split(".")[-1]
        if param_name == "attn_implementation":
            kwargs["attn_implementation"] = test_value
        elif param_name == "precision":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            kwargs["torch_dtype"] = dtype_map.get(test_value, torch.float16)

        return AutoModelForCausalLM.from_pretrained(**kwargs)

    def _create_tensorrt_instance(self, spec: ParamSpec, test_value: Any) -> Any:
        """Create a TensorRT-LLM executor instance."""
        # TensorRT-LLM instantiation is complex and requires engine building
        # For now, return None and skip actual TensorRT tests
        raise NotImplementedError("TensorRT instance creation not yet implemented")

    def _generate_output(self, instance: Any, spec: ParamSpec) -> Any:
        """Generate output from an instance for behaviour testing."""
        prompt = "The capital of France is"

        if spec.backend == "vllm":
            from vllm import SamplingParams

            sampling = SamplingParams(max_tokens=32, temperature=0)
            return instance.generate([prompt], sampling)
        elif spec.backend == "pytorch":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = instance.generate(**inputs, max_new_tokens=32)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            return None

    def _cleanup_instance(self, instance: Any) -> None:
        """Clean up a backend instance."""
        del instance
        cleanup_gpu()


def get_test_params_for_pytest(
    backend: str | None = None,
    mockable_only: bool = False,
    skip_skipped: bool = True,
) -> list[tuple[ParamSpec, Any]]:
    """Get (spec, value) tuples for pytest parametrization.

    Args:
        backend: Filter to specific backend.
        mockable_only: Only include specs that can be tested via mocking.
        skip_skipped: Exclude specs with skip_reason.

    Returns:
        List of (ParamSpec, test_value) tuples.
    """
    registry = get_registry()

    params = []
    for spec in registry.filter(
        backend=backend, mockable_only=mockable_only, exclude_skipped=skip_skipped
    ):
        for test_value in spec.test_values:
            params.append((spec, test_value))

    return params


def get_pytest_id(spec: ParamSpec, test_value: Any) -> str:
    """Generate a pytest ID for a (spec, value) pair."""
    return f"{spec.full_name}[{test_value}]"
