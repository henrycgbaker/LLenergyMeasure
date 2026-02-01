"""Unit tests for bootstrap confidence interval estimation."""

from __future__ import annotations

import pytest

from llenergymeasure.results.bootstrap import BootstrapResult, bootstrap_ci


class TestBootstrapCI:
    """Tests for bootstrap_ci function — pure logic, no mocks needed."""

    def test_bootstrap_ci_basic(self) -> None:
        """5 samples produces valid CI with lower < mean < upper."""
        result = bootstrap_ci([10.0, 12.0, 11.0, 13.0, 10.5])
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower < result.mean < result.ci_upper
        assert result.n_samples == 5
        assert result.confidence == 0.95
        assert result.warning is None

    def test_bootstrap_ci_deterministic(self) -> None:
        """Same inputs + seed produce identical outputs."""
        samples = [10.0, 12.0, 11.0, 13.0, 10.5]
        r1 = bootstrap_ci(samples, seed=42)
        r2 = bootstrap_ci(samples, seed=42)
        assert r1.mean == r2.mean
        assert r1.std == r2.std
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_bootstrap_ci_single_sample(self) -> None:
        """Single sample returns no CI (None) with warning."""
        result = bootstrap_ci([42.0])
        assert result.ci_lower is None
        assert result.ci_upper is None
        assert result.n_samples == 1
        assert result.warning is not None
        assert "< 2 samples" in result.warning

    def test_bootstrap_ci_two_samples(self) -> None:
        """Two samples returns CI with 'unreliable' warning."""
        result = bootstrap_ci([10.0, 12.0])
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.n_samples == 2
        assert result.warning is not None
        assert "unreliable" in result.warning

    def test_bootstrap_ci_many_samples(self) -> None:
        """100 samples produces tight CI relative to the data range."""
        import numpy as np

        rng = np.random.default_rng(99)
        samples = rng.normal(loc=50.0, scale=2.0, size=100).tolist()
        result = bootstrap_ci(samples, seed=123)

        assert result.ci_lower is not None
        assert result.ci_upper is not None
        ci_width = result.ci_upper - result.ci_lower
        # Tight CI: width should be well under 2.0 for 100 samples from N(50, 2)
        assert ci_width < 2.0
        assert result.warning is None

    def test_bootstrap_ci_confidence_levels(self) -> None:
        """0.99 CI is wider than 0.90 CI."""
        samples = [10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 9.5, 14.0, 10.0]
        r_90 = bootstrap_ci(samples, confidence=0.90, seed=42)
        r_99 = bootstrap_ci(samples, confidence=0.99, seed=42)

        assert r_90.ci_lower is not None and r_90.ci_upper is not None
        assert r_99.ci_lower is not None and r_99.ci_upper is not None

        width_90 = r_90.ci_upper - r_90.ci_lower
        width_99 = r_99.ci_upper - r_99.ci_lower
        assert width_99 > width_90

    def test_bootstrap_ci_constant_values(self) -> None:
        """All same value produces zero-width CI."""
        result = bootstrap_ci([5.0, 5.0, 5.0, 5.0, 5.0])
        assert result.mean == pytest.approx(5.0)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_upper - result.ci_lower == pytest.approx(0.0, abs=1e-10)


class TestBootstrapResultModel:
    """Tests for BootstrapResult Pydantic model."""

    def test_bootstrap_result_serializes_to_dict(self) -> None:
        """BootstrapResult serializes to dict correctly."""
        result = BootstrapResult(
            mean=10.0,
            std=1.5,
            ci_lower=9.0,
            ci_upper=11.0,
            n_samples=5,
            confidence=0.95,
            warning=None,
        )
        d = result.model_dump()
        assert d["mean"] == 10.0
        assert d["std"] == 1.5
        assert d["ci_lower"] == 9.0
        assert d["ci_upper"] == 11.0
        assert d["n_samples"] == 5
        assert d["confidence"] == 0.95
        assert d["warning"] is None

    def test_bootstrap_result_with_warning(self) -> None:
        """BootstrapResult with warning serializes correctly."""
        result = BootstrapResult(
            mean=10.0,
            std=0.0,
            ci_lower=None,
            ci_upper=None,
            n_samples=1,
            confidence=0.95,
            warning="< 2 samples, CI not computable",
        )
        d = result.model_dump()
        assert d["warning"] == "< 2 samples, CI not computable"
        assert d["ci_lower"] is None
        assert d["ci_upper"] is None
