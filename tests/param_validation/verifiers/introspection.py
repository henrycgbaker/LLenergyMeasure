"""Introspection verification for parameter validation.

Verifies parameter effects by inspecting internal model/engine state,
such as model dtype, quantization status, cache configuration, etc.
"""

from __future__ import annotations

import time
from typing import Any

from ..registry import ParamSpec, VerificationResult, VerificationStatus


class IntrospectionVerifier:
    """Verifies parameter effects via model/engine state inspection.

    Examines internal state that may not be directly exposed as config
    attributes but reflects the effect of parameter settings.
    """

    def verify_dtype(
        self,
        spec: ParamSpec,
        model: Any,
        expected_dtype: str,
    ) -> VerificationResult:
        """Verify model is loaded with expected dtype.

        Args:
            spec: The ParamSpec being verified.
            model: Model instance to check.
            expected_dtype: Expected dtype string (float16, bfloat16, float32).

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            import torch

            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }

            expected = dtype_map.get(expected_dtype.lower())
            if expected is None:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    message=f"Unknown dtype: {expected_dtype}",
                    param_name=spec.full_name,
                    test_value=expected_dtype,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            actual = model.dtype
            passed = actual == expected

            if passed:
                message = f"Model dtype matches: {actual}"
            else:
                message = f"Model dtype mismatch: expected {expected}, got {actual}"

            return VerificationResult(
                status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                message=message,
                param_name=spec.full_name,
                test_value=expected_dtype,
                actual_value=str(actual),
                expected_value=str(expected),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Error checking dtype: {e}",
                param_name=spec.full_name,
                test_value=expected_dtype,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def verify_quantization(
        self,
        spec: ParamSpec,
        model: Any,
        expected_bits: int | None,
        expected_method: str | None = None,
    ) -> VerificationResult:
        """Verify model quantization state.

        Args:
            spec: The ParamSpec being verified.
            model: Model instance to check.
            expected_bits: Expected quantization bits (4, 8, or None for no quant).
            expected_method: Expected quantization method (bitsandbytes, gptq, etc.).

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            state = self._extract_quantization_state(model)

            if expected_bits is None:
                # Expect no quantization
                if not state["is_quantized"]:
                    return VerificationResult(
                        status=VerificationStatus.PASSED,
                        message="Model is not quantized as expected",
                        param_name=spec.full_name,
                        test_value=None,
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    message=f"Model is quantized (expected no quantization): {state}",
                    param_name=spec.full_name,
                    test_value=None,
                    actual_value=state,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            # Expect quantization
            if not state["is_quantized"]:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    message=f"Model not quantized (expected {expected_bits}-bit)",
                    param_name=spec.full_name,
                    test_value=expected_bits,
                    actual_value=state,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            bits_match = state["bits"] == expected_bits
            method_match = expected_method is None or state["method"] == expected_method

            if bits_match and method_match:
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    message=f"Quantization matches: {state['bits']}-bit {state['method']}",
                    param_name=spec.full_name,
                    test_value=expected_bits,
                    actual_value=state,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            return VerificationResult(
                status=VerificationStatus.FAILED,
                message=f"Quantization mismatch: expected {expected_bits}-bit "
                f"{expected_method or 'any'}, got {state}",
                param_name=spec.full_name,
                test_value=expected_bits,
                actual_value=state,
                expected_value={"bits": expected_bits, "method": expected_method},
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Error checking quantization: {e}",
                param_name=spec.full_name,
                test_value=expected_bits,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def verify_attention_impl(
        self,
        spec: ParamSpec,
        model: Any,
        expected_impl: str,
    ) -> VerificationResult:
        """Verify attention implementation.

        Args:
            spec: The ParamSpec being verified.
            model: Model instance to check.
            expected_impl: Expected implementation (sdpa, flash_attention_2, eager).

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            actual = self._detect_attention_impl(model)

            # Normalise names
            expected_normalised = expected_impl.lower().replace("-", "_").replace(" ", "_")
            actual_normalised = actual.lower().replace("-", "_").replace(" ", "_") if actual else ""

            passed = (
                expected_normalised in actual_normalised or actual_normalised == expected_normalised
            )

            if passed:
                message = f"Attention implementation: {actual}"
            else:
                message = f"Attention mismatch: expected {expected_impl}, got {actual}"

            return VerificationResult(
                status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                message=message,
                param_name=spec.full_name,
                test_value=expected_impl,
                actual_value=actual,
                expected_value=expected_impl,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Error checking attention: {e}",
                param_name=spec.full_name,
                test_value=expected_impl,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def verify_gpu_memory_usage(
        self,
        spec: ParamSpec,
        baseline_mb: float,
        test_mb: float,
        expected_direction: str = "different",
        min_diff_mb: float = 10.0,
    ) -> VerificationResult:
        """Verify GPU memory usage change.

        Args:
            spec: The ParamSpec being verified.
            baseline_mb: Baseline memory usage in MB.
            test_mb: Test memory usage in MB.
            expected_direction: "higher", "lower", or "different".
            min_diff_mb: Minimum difference threshold.

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        diff = test_mb - baseline_mb

        if expected_direction == "higher":
            passed = diff > min_diff_mb
            msg = f"Memory {diff:+.1f}MB (expected increase > {min_diff_mb}MB)"
        elif expected_direction == "lower":
            passed = diff < -min_diff_mb
            msg = f"Memory {diff:+.1f}MB (expected decrease > {min_diff_mb}MB)"
        else:  # different
            passed = abs(diff) > min_diff_mb
            msg = f"Memory {diff:+.1f}MB (expected |diff| > {min_diff_mb}MB)"

        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            message=msg,
            param_name=spec.full_name,
            test_value=expected_direction,
            actual_value=diff,
            expected_value=min_diff_mb,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    def _extract_quantization_state(self, model: Any) -> dict[str, Any]:
        """Extract quantization state from model."""
        result = {
            "is_quantized": False,
            "bits": None,
            "method": None,
            "dtype": str(getattr(model, "dtype", "unknown")),
        }

        # Check BitsAndBytes 4-bit
        if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
            result["is_quantized"] = True
            result["bits"] = 4
            result["method"] = "bitsandbytes"
            return result

        # Check BitsAndBytes 8-bit
        if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
            result["is_quantized"] = True
            result["bits"] = 8
            result["method"] = "bitsandbytes"
            return result

        # Check for GPTQ
        if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
            qconfig = model.config.quantization_config
            if hasattr(qconfig, "bits"):
                result["is_quantized"] = True
                result["bits"] = qconfig.bits
                result["method"] = getattr(qconfig, "quant_method", "gptq")

        return result

    def _detect_attention_impl(self, model: Any) -> str | None:
        """Detect which attention implementation is being used."""
        # Check config
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "_attn_implementation"):
                return config._attn_implementation
            if hasattr(config, "attn_implementation"):
                return config.attn_implementation

        # Check model attributes
        if hasattr(model, "_attn_implementation"):
            return model._attn_implementation

        return None


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    """Get current reserved GPU memory in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception:
        pass
    return 0.0
