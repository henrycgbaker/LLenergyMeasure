"""Utility functions for CLI operations.

This module contains helper functions for config merging, duration parsing,
state management, and other shared CLI utilities.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_energy_measure.state.experiment_state import ExperimentState, StateManager


# Re-export deep_merge from config.loader for backwards compatibility
from llm_energy_measure.config.loader import deep_merge

__all__ = [
    "apply_cli_overrides",
    "deep_merge",
    "parse_duration",
    "update_process_state_from_markers",
]


def _get_nested_value(d: dict[str, Any], path: list[str]) -> Any:
    """Get a value from a nested dict using a path list.

    Args:
        d: Dictionary to traverse.
        path: List of keys forming the path.

    Returns:
        The value at the path, or None if not found.
    """
    current = d
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested_value(d: dict[str, Any], path: list[str], value: Any) -> None:
    """Set a value in a nested dict using a path list, creating intermediate dicts as needed.

    Args:
        d: Dictionary to modify.
        path: List of keys forming the path.
        value: Value to set.
    """
    current = d
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            # Override non-dict with dict if we need to nest further
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def apply_cli_overrides(
    config_dict: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply CLI overrides to config dict, tracking what was overridden.

    Supports arbitrary nesting depth using dot notation (e.g., "vllm.speculative.num_tokens").

    Args:
        config_dict: Configuration dictionary to modify.
        overrides: Dictionary of override paths and values.

    Returns:
        Tuple of (merged config dict, dict of overridden values with original values)
    """
    tracked_overrides: dict[str, Any] = {}

    for key, value in overrides.items():
        if value is None:
            continue

        # Parse dot-separated path
        path = key.split(".")

        # Get original value before override
        original = _get_nested_value(config_dict, path)

        # Set the new value
        _set_nested_value(config_dict, path, value)

        # Track the override
        tracked_overrides[key] = {"new": value, "original": original}

    return config_dict, tracked_overrides


def update_process_state_from_markers(
    state: ExperimentState,
    state_manager: StateManager,
    results_dir: Path,
) -> ExperimentState:
    """Update experiment state by scanning completion markers.

    Args:
        state: Current experiment state.
        state_manager: State manager for persistence.
        results_dir: Base results directory.

    Returns:
        Updated experiment state.
    """
    from llm_energy_measure.constants import COMPLETION_MARKER_PREFIX
    from llm_energy_measure.state.experiment_state import ProcessProgress, ProcessStatus

    raw_dir = results_dir / "raw" / state.experiment_id

    for i in range(state.num_processes):
        marker_path = raw_dir / f"{COMPLETION_MARKER_PREFIX}{i}"
        if marker_path.exists():
            try:
                marker_data = json.loads(marker_path.read_text())
                state.process_progress[i] = ProcessProgress(
                    process_index=i,
                    status=ProcessStatus.COMPLETED,
                    gpu_id=marker_data.get("gpu_id"),
                    completed_at=datetime.fromisoformat(marker_data["timestamp"]),
                )
            except Exception:
                # Marker exists but couldn't parse - still mark as completed
                state.process_progress[i] = ProcessProgress(
                    process_index=i,
                    status=ProcessStatus.COMPLETED,
                )
        elif i not in state.process_progress:
            # No marker and not previously tracked = failed or didn't start
            state.process_progress[i] = ProcessProgress(
                process_index=i,
                status=ProcessStatus.FAILED,
                error_message="No completion marker found",
            )

    state_manager.save(state)
    return state


def parse_duration(duration_str: str) -> float:
    """Parse duration string to seconds.

    Supports: '30s', '5m', '2h', '1d', '1w'
    """
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([smhdw])$", duration_str.lower().strip())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Use e.g., '30m', '6h', '1d'")

    value = float(match.group(1))
    unit = match.group(2)

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return value * multipliers[unit]
