"""Tests for warmup convergence detection module.

Tests warmup_until_converged with various configurations:
disabled, convergence, non-convergence, fixed mode, exceptions.
"""

import pytest

from llenergymeasure.config.models import WarmupConfig
from llenergymeasure.core.warmup import warmup_until_converged
from llenergymeasure.domain.metrics import WarmupResult


class TestDisabledWarmup:
    """Tests for warmup when disabled."""

    def test_disabled_warmup_returns_immediately(self):
        config = WarmupConfig(enabled=False)
        call_count = 0

        def run_inference() -> float:
            nonlocal call_count
            call_count += 1
            return 100.0

        result = warmup_until_converged(run_inference, config, show_progress=False)
        assert isinstance(result, WarmupResult)
        assert result.converged is True
        assert result.iterations_completed == 0
        assert call_count == 0  # inference was never called


class TestConvergenceDetection:
    """Tests for CV-based convergence detection."""

    def test_convergence_with_stable_latencies(self):
        """Stable latencies (~100ms, <2% noise) should converge before max."""
        import random

        random.seed(42)
        config = WarmupConfig(
            enabled=True,
            convergence_detection=True,
            cv_threshold=0.05,
            min_prompts=5,
            max_prompts=100,
            window_size=5,
        )
        # Latencies with very small noise
        latencies = iter([100.0 + random.uniform(-1.0, 1.0) for _ in range(100)])

        result = warmup_until_converged(lambda: next(latencies), config, show_progress=False)
        assert result.converged is True
        assert result.iterations_completed < config.max_prompts
        assert result.final_cv < config.cv_threshold

    def test_non_convergence_with_noisy_latencies(self):
        """Very noisy latencies with tight threshold should hit max_prompts."""
        import random

        random.seed(42)
        config = WarmupConfig(
            enabled=True,
            convergence_detection=True,
            cv_threshold=0.01,  # very tight
            min_prompts=5,
            max_prompts=20,
            window_size=5,
        )
        # Highly variable latencies
        latencies = iter([random.uniform(50.0, 200.0) for _ in range(20)])

        result = warmup_until_converged(lambda: next(latencies), config, show_progress=False)
        assert result.converged is False
        assert result.iterations_completed == config.max_prompts


class TestFixedMode:
    """Tests for fixed iteration warmup (convergence_detection=False)."""

    def test_fixed_mode_runs_exact_count(self):
        config = WarmupConfig(
            enabled=True,
            convergence_detection=False,
            max_prompts=10,
        )
        call_count = 0

        def run_inference() -> float:
            nonlocal call_count
            call_count += 1
            return 100.0

        result = warmup_until_converged(run_inference, config, show_progress=False)
        assert result.converged is True  # fixed mode always "converges"
        assert result.iterations_completed == 10
        assert call_count == 10


class TestExceptionHandling:
    """Tests for exception handling during warmup."""

    def test_exception_in_inference_continues(self):
        """Warmup should continue past exceptions in individual prompts."""
        config = WarmupConfig(
            enabled=True,
            convergence_detection=False,
            max_prompts=5,
        )
        call_count = 0

        def run_inference() -> float:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated inference failure")
            return 100.0

        result = warmup_until_converged(run_inference, config, show_progress=False)
        assert call_count == 5  # all 5 calls made
        # 4 successful latencies (call 3 raised an exception)
        assert result.iterations_completed == 4
        assert len(result.latencies_ms) == 4


class TestWarmupResultFields:
    """Tests for WarmupResult field population."""

    def test_all_fields_populated(self):
        config = WarmupConfig(
            enabled=True,
            convergence_detection=False,
            max_prompts=5,
            window_size=5,
            cv_threshold=0.05,
        )
        result = warmup_until_converged(lambda: 100.0, config, show_progress=False)

        assert isinstance(result, WarmupResult)
        assert result.converged is True
        assert result.iterations_completed == 5
        assert result.target_cv == pytest.approx(0.05)
        assert result.max_prompts == 5
        assert len(result.latencies_ms) == 5
        assert all(lat == pytest.approx(100.0) for lat in result.latencies_ms)
        # final_cv should be 0 for identical latencies
        assert result.final_cv == pytest.approx(0.0, abs=1e-10)
