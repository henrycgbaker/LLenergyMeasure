"""Passthrough verification for parameter validation.

Verifies that parameter values are correctly passed through to backend
configuration objects by checking nested attributes.
"""

from __future__ import annotations

import time
from typing import Any

from ..registry import ParamSpec, VerificationResult, VerificationStatus


class PassthroughVerifier:
    """Verifies parameter passthrough to backend configuration.

    Checks that a configured parameter value actually reaches the
    expected location in the instantiated backend (e.g., vLLM's
    llm_engine.scheduler_config.max_num_seqs).
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize the verifier.

        Args:
            tolerance: Tolerance for float comparisons.
        """
        self.tolerance = tolerance

    def verify(
        self,
        spec: ParamSpec,
        instance: Any,
        test_value: Any,
    ) -> VerificationResult:
        """Verify a parameter was passed through correctly.

        Args:
            spec: The ParamSpec being verified.
            instance: The backend instance to check (e.g., vLLM LLM object).
            test_value: The value that was configured.

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            # Use custom checker if provided
            if spec.passthrough_checker is not None:
                passed, message = spec.passthrough_checker(instance, test_value)
                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    message=message,
                    param_name=spec.full_name,
                    test_value=test_value,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            # Use passthrough path for attribute lookup
            if spec.passthrough_path is None:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    message="No passthrough_path or checker defined",
                    param_name=spec.full_name,
                    test_value=test_value,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            # Navigate the attribute path
            actual_value = self._get_nested_attr(instance, spec.passthrough_path)

            # Apply expected transform if defined
            expected_value = test_value
            if spec.expected_transform is not None:
                expected_value = spec.expected_transform(test_value)

            # Compare values
            passed = self._values_equal(actual_value, expected_value)

            if passed:
                message = f"{spec.passthrough_path}={actual_value} (matches expected)"
            else:
                message = f"{spec.passthrough_path}={actual_value} (expected {expected_value})"

            return VerificationResult(
                status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                message=message,
                param_name=spec.full_name,
                test_value=test_value,
                actual_value=actual_value,
                expected_value=expected_value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except AttributeError as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Attribute not found: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Verification error: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _get_nested_attr(self, obj: Any, path: str) -> Any:
        """Get a nested attribute using dot notation.

        Args:
            obj: The object to traverse.
            path: Dot-separated attribute path (e.g., "engine.config.value").

        Returns:
            The attribute value.

        Raises:
            AttributeError: If any part of the path doesn't exist.
        """
        parts = path.split(".")
        current = obj

        for part in parts:
            # Handle array indexing like "items[0]"
            if "[" in part and "]" in part:
                attr_name = part[: part.index("[")]
                index = int(part[part.index("[") + 1 : part.index("]")])
                current = getattr(current, attr_name)
                current = current[index]
            else:
                current = getattr(current, part)

        return current

    def _values_equal(self, actual: Any, expected: Any) -> bool:
        """Compare two values with tolerance for floats.

        Args:
            actual: The actual value found.
            expected: The expected value.

        Returns:
            True if values are equal (within tolerance for floats).
        """
        # None comparison
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False

        # Float comparison with tolerance
        if isinstance(actual, float) and isinstance(expected, float):
            return abs(actual - expected) <= self.tolerance

        # Handle torch dtype comparison
        if hasattr(actual, "__class__") and "torch" in str(type(actual)):
            return str(actual) == str(expected) or actual == expected

        # Direct comparison
        return actual == expected


def verify_vllm_passthrough(
    llm: Any,
    config_section: str,
    param_name: str,
    expected_value: Any,
) -> tuple[bool, str]:
    """Convenience function for vLLM passthrough verification.

    Args:
        llm: vLLM LLM instance.
        config_section: Config section name (cache_config, scheduler_config, etc.).
        param_name: Parameter name within the section.
        expected_value: Expected value.

    Returns:
        Tuple of (passed, message).
    """
    try:
        config = getattr(llm.llm_engine, config_section)
        actual = getattr(config, param_name)

        if isinstance(actual, float) and isinstance(expected_value, float):
            passed = abs(actual - expected_value) < 0.01
        else:
            passed = actual == expected_value

        if passed:
            return True, f"{config_section}.{param_name}={actual}"
        return False, f"{config_section}.{param_name}={actual} (expected {expected_value})"

    except Exception as e:
        return False, f"Error checking {config_section}.{param_name}: {e}"


def verify_pytorch_passthrough(
    model: Any,
    attr_path: str,
    expected_value: Any,
) -> tuple[bool, str]:
    """Convenience function for PyTorch passthrough verification.

    Args:
        model: PyTorch model instance.
        attr_path: Dot-separated attribute path.
        expected_value: Expected value.

    Returns:
        Tuple of (passed, message).
    """
    verifier = PassthroughVerifier()
    try:
        actual = verifier._get_nested_attr(model, attr_path)
        passed = verifier._values_equal(actual, expected_value)

        if passed:
            return True, f"{attr_path}={actual}"
        return False, f"{attr_path}={actual} (expected {expected_value})"

    except Exception as e:
        return False, f"Error checking {attr_path}: {e}"
