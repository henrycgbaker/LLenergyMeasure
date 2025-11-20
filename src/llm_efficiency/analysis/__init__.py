"""Analysis utilities for experiment results."""

from llm_efficiency.analysis.comparison import (
    compare_experiments,
    compare_models,
    compare_configurations,
    generate_comparison_report,
    ComparisonResult,
)
from llm_efficiency.analysis.statistics import (
    calculate_statistics,
    detect_outliers,
    calculate_efficiency_score,
    rank_experiments,
)

__all__ = [
    # Comparison
    "compare_experiments",
    "compare_models",
    "compare_configurations",
    "generate_comparison_report",
    "ComparisonResult",
    # Statistics
    "calculate_statistics",
    "detect_outliers",
    "calculate_efficiency_score",
    "rank_experiments",
]
