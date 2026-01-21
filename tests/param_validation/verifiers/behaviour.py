"""Behaviour verification for parameter validation.

Verifies that parameter values produce observable changes in model
output or performance characteristics.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..registry import ParamSpec, VerificationResult, VerificationStatus


class BehaviourVerifier:
    """Verifies parameter effects on observable behaviour.

    Tests that changing a parameter produces measurable differences
    in model outputs, timing, memory usage, or other characteristics.
    """

    def verify(
        self,
        spec: ParamSpec,
        baseline_output: Any,
        test_output: Any,
        test_value: Any,
    ) -> VerificationResult:
        """Verify a parameter affects behaviour as expected.

        Args:
            spec: The ParamSpec being verified.
            baseline_output: Output with default/baseline parameter value.
            test_output: Output with test parameter value.
            test_value: The test value that was configured.

        Returns:
            VerificationResult with verification outcome.
        """
        start = time.perf_counter()

        try:
            if spec.behaviour_assertion is None:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    message="No behaviour_assertion defined",
                    param_name=spec.full_name,
                    test_value=test_value,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            passed, message = spec.behaviour_assertion(baseline_output, test_output, test_value)

            return VerificationResult(
                status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                message=message,
                param_name=spec.full_name,
                test_value=test_value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Behaviour verification error: {e}",
                param_name=spec.full_name,
                test_value=test_value,
                error=e,
                duration_ms=(time.perf_counter() - start) * 1000,
            )


# =============================================================================
# Common Behaviour Assertions
# =============================================================================


def outputs_differ(
    baseline: Any,
    test: Any,
    test_value: Any,
) -> tuple[bool, str]:
    """Assert that outputs are different (for sampling params like temperature).

    Args:
        baseline: Baseline output.
        test: Test output.
        test_value: The test parameter value.

    Returns:
        Tuple of (passed, message).
    """
    if baseline != test:
        return True, f"Outputs differ as expected with value={test_value}"
    return False, "Outputs are identical (expected difference)"


def outputs_identical(
    baseline: Any,
    test: Any,
    test_value: Any,
) -> tuple[bool, str]:
    """Assert that outputs are identical (for determinism tests).

    Args:
        baseline: Baseline output.
        test: Test output.
        test_value: The test parameter value.

    Returns:
        Tuple of (passed, message).
    """
    if baseline == test:
        return True, f"Outputs identical as expected with value={test_value}"
    return False, "Outputs differ (expected identical)"


def output_length_bounded(
    max_tokens: int,
) -> Callable[[Any, Any, Any], tuple[bool, str]]:
    """Create an assertion that checks output length is bounded.

    Args:
        max_tokens: Maximum expected token count.

    Returns:
        Assertion function.
    """

    def assertion(baseline: Any, test: Any, test_value: Any) -> tuple[bool, str]:
        # Assume test is a string or has a text attribute
        if hasattr(test, "text"):
            text = test.text
        elif isinstance(test, str):
            text = test
        else:
            return False, f"Cannot determine text length from {type(test)}"

        # Rough token estimation (words ≈ tokens for English)
        words = len(text.split())
        if words <= max_tokens * 1.5:  # Allow some margin
            return True, f"Output length {words} within bounds (max_tokens={max_tokens})"
        return False, f"Output length {words} exceeds expected (max_tokens={max_tokens})"

    return assertion


def timing_differs(
    min_ratio: float = 0.8,
    max_ratio: float = 1.2,
) -> Callable[[Any, Any, Any], tuple[bool, str]]:
    """Create an assertion that checks timing differs.

    Args:
        min_ratio: Minimum acceptable ratio (test_time / baseline_time).
        max_ratio: Maximum acceptable ratio.

    Returns:
        Assertion function that expects (baseline_time, test_time) tuples.
    """

    def assertion(baseline: Any, test: Any, test_value: Any) -> tuple[bool, str]:
        if not isinstance(baseline, int | float) or not isinstance(test, int | float):
            return False, "Expected numeric timing values"

        if baseline <= 0:
            return False, "Baseline timing is zero or negative"

        ratio = test / baseline
        if ratio < min_ratio or ratio > max_ratio:
            return True, f"Timing ratio {ratio:.2f} outside [{min_ratio}, {max_ratio}]"
        return False, f"Timing ratio {ratio:.2f} within expected range"

    return assertion


def memory_differs(
    min_diff_mb: float = 10.0,
) -> Callable[[Any, Any, Any], tuple[bool, str]]:
    """Create an assertion that checks memory usage differs.

    Args:
        min_diff_mb: Minimum expected difference in MB.

    Returns:
        Assertion function that expects memory values in MB.
    """

    def assertion(baseline: Any, test: Any, test_value: Any) -> tuple[bool, str]:
        if not isinstance(baseline, int | float) or not isinstance(test, int | float):
            return False, "Expected numeric memory values in MB"

        diff = abs(test - baseline)
        if diff >= min_diff_mb:
            return True, f"Memory differs by {diff:.1f}MB (threshold {min_diff_mb}MB)"
        return False, f"Memory difference {diff:.1f}MB below threshold {min_diff_mb}MB"

    return assertion


def logprobs_present(
    baseline: Any,
    test: Any,
    test_value: Any,
) -> tuple[bool, str]:
    """Assert that logprobs are present in output.

    Args:
        baseline: Baseline output (ignored).
        test: Test output (should have logprobs).
        test_value: The logprobs value configured.

    Returns:
        Tuple of (passed, message).
    """
    # Handle vLLM output format
    if hasattr(test, "outputs") and len(test.outputs) > 0:
        output = test.outputs[0]
        if hasattr(output, "logprobs") and output.logprobs:
            return True, f"Logprobs present ({len(output.logprobs)} tokens)"

    # Handle raw logprobs
    if hasattr(test, "logprobs") and test.logprobs:
        return True, "Logprobs present in output"

    return False, "Logprobs not found in output"


def sampling_varied(
    min_unique: int = 2,
) -> Callable[[Any, Any, Any], tuple[bool, str]]:
    """Create an assertion that checks sampling produces varied outputs.

    Args:
        min_unique: Minimum number of unique outputs required.

    Returns:
        Assertion function that expects list of outputs.
    """

    def assertion(baseline: Any, test: Any, test_value: Any) -> tuple[bool, str]:
        if not isinstance(test, list):
            return False, "Expected list of outputs for sampling variation check"

        unique = set(str(o) for o in test)
        if len(unique) >= min_unique:
            return True, f"Found {len(unique)} unique outputs (required >= {min_unique})"
        return False, f"Only {len(unique)} unique outputs (required >= {min_unique})"

    return assertion
