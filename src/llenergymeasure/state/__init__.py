"""State management for llenergymeasure experiments.

This package redirects to llenergymeasure.core.state, which is the canonical
location since v2.0. Direct imports from this package are supported for
backwards compatibility.
"""

from llenergymeasure.core.state import (
    ExperimentPhase,
    ExperimentState,
    StateManager,
    compute_config_hash,
)

__all__ = [
    "ExperimentPhase",
    "ExperimentState",
    "StateManager",
    "compute_config_hash",
]
