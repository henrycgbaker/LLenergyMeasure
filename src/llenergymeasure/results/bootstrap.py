"""Bootstrap resampling for confidence interval estimation.

Provides bootstrap CI computation for campaign-level aggregation,
where multiple cycles of the same config need statistically rigorous
confidence intervals (MEAS-08).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

__all__ = ["BootstrapResult", "bootstrap_ci", "compute_metric_ci"]


class BootstrapResult(BaseModel):
    """Result of bootstrap confidence interval estimation."""

    mean: float = Field(..., description="Sample mean")
    std: float = Field(..., description="Sample standard deviation")
    ci_lower: float | None = Field(default=None, description="Lower bound of confidence interval")
    ci_upper: float | None = Field(default=None, description="Upper bound of confidence interval")
    n_samples: int = Field(..., description="Number of input samples")
    confidence: float = Field(..., description="Confidence level (e.g. 0.95)")
    warning: str | None = Field(
        default=None, description="Warning when CI is unreliable or not computable"
    )


def bootstrap_ci(
    samples: list[float] | np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval using the percentile method.

    Args:
        samples: Input data samples (e.g. energy readings across cycles).
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with mean, std, and CI bounds.
    """
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # Too few samples for any CI
    if n < 2:
        return BootstrapResult(
            mean=mean,
            std=std,
            ci_lower=None,
            ci_upper=None,
            n_samples=n,
            confidence=confidence,
            warning="< 2 samples, CI not computable",
        )

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_iterations)
    for i in range(n_iterations):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    warning = None
    if n < 3:
        warning = "< 3 samples, CI unreliable (recommend >= 3 cycles)"

    return BootstrapResult(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n,
        confidence=confidence,
        warning=warning,
    )


def compute_metric_ci(
    values: list[float],
    metric_name: str = "",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute CI for a named metric, returning a dict suitable for JSON serialization.

    Args:
        values: Metric values across cycles.
        metric_name: Optional name for logging/debugging.
        confidence: Confidence level.

    Returns:
        Dict with keys: mean, std, ci_lower, ci_upper, n_samples, confidence, warning.
    """
    result = bootstrap_ci(values, confidence=confidence)
    d: dict[str, Any] = {
        "mean": result.mean,
        "std": result.std,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "n_samples": result.n_samples,
        "confidence": result.confidence,
    }
    if result.warning:
        d["warning"] = result.warning
    return d
