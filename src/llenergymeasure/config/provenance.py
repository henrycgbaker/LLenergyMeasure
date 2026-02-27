"""Parameter provenance tracking for configuration resolution.

This module provides models and utilities for tracking where each configuration
parameter value comes from (Pydantic defaults, presets, config files, CLI overrides).

Full provenance tracking enables:
- Debugging configuration issues
- Reproducibility auditing
- Understanding parameter inheritance chains
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ParameterSource(str, Enum):
    """Source of a configuration parameter value.

    Sources are listed in order of precedence (lowest to highest):
    - PYDANTIC_DEFAULT: Default value from Pydantic field definition
    - PRESET: Value from a preset configuration
    - CONFIG_FILE: Value from YAML config file
    - CLI: Value from command-line override
    """

    PYDANTIC_DEFAULT = "pydantic_default"
    PRESET = "preset"
    CONFIG_FILE = "config_file"
    CLI = "cli"


class ParameterProvenance(BaseModel):
    """Provenance information for a single parameter.

    Tracks the source and value of a configuration parameter, enabling
    full audit trail of how the final configuration was determined.

    Attributes:
        path: Dot-separated parameter path (e.g., "decoder.temperature").
        value: The resolved parameter value.
        source: Where the value came from.
        source_detail: Additional context (preset name, config file path, etc.).
    """

    path: str = Field(..., description="Dot-separated parameter path")
    value: Any = Field(..., description="Resolved parameter value")
    source: ParameterSource = Field(..., description="Source of the value")
    source_detail: str | None = Field(
        default=None,
        description="Additional context (preset name, config file path, CLI flag)",
    )

    def __str__(self) -> str:
        """Human-readable representation."""
        detail = f" ({self.source_detail})" if self.source_detail else ""
        return f"{self.path}: {self.value!r} [from {self.source.value}{detail}]"


class ResolvedConfig(BaseModel):
    """Configuration with full provenance tracking.

    Pairs the resolved ExperimentConfig with provenance information
    for each parameter, enabling full traceability.

    Attributes:
        config: The resolved experiment configuration.
        provenance: Map of parameter paths to their provenance.
        preset_chain: Ordered list of presets applied (for preset inheritance).
        config_file_path: Path to the config file, if any.
    """

    config: Any = Field(..., description="Resolved ExperimentConfig")
    provenance: dict[str, ParameterProvenance] = Field(
        default_factory=dict,
        description="Provenance for each parameter path",
    )
    preset_chain: list[str] = Field(
        default_factory=list,
        description="Presets applied in order",
    )
    config_file_path: str | None = Field(
        default=None,
        description="Path to config file, if used",
    )

    def get_provenance(self, path: str) -> ParameterProvenance | None:
        """Get provenance for a specific parameter path.

        Args:
            path: Dot-separated parameter path.

        Returns:
            ParameterProvenance if found, None otherwise.
        """
        return self.provenance.get(path)

    def get_parameters_by_source(self, source: ParameterSource) -> list[ParameterProvenance]:
        """Get all parameters from a specific source.

        Args:
            source: The parameter source to filter by.

        Returns:
            List of ParameterProvenance objects from that source.
        """
        return [p for p in self.provenance.values() if p.source == source]

    def get_non_default_parameters(self) -> list[ParameterProvenance]:
        """Get all parameters that differ from Pydantic defaults.

        Returns:
            List of ParameterProvenance objects with non-default values.
        """
        return [p for p in self.provenance.values() if p.source != ParameterSource.PYDANTIC_DEFAULT]

    def get_cli_overrides(self) -> list[ParameterProvenance]:
        """Get all parameters overridden via CLI.

        Returns:
            List of ParameterProvenance objects from CLI overrides.
        """
        return self.get_parameters_by_source(ParameterSource.CLI)

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to a summary dictionary for serialisation.

        Returns:
            Dictionary suitable for JSON serialisation with provenance info.
        """
        return {
            "preset_chain": self.preset_chain,
            "config_file": self.config_file_path,
            "non_default_params": {
                p.path: {"value": p.value, "source": p.source.value, "detail": p.source_detail}
                for p in self.get_non_default_parameters()
            },
            "cli_overrides": [p.path for p in self.get_cli_overrides()],
        }


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary into dot-separated keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).
        sep: Separator for nested keys.

    Returns:
        Flattened dictionary with dot-separated keys.

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    """
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflatten a dot-separated dictionary into nested structure.

    Args:
        d: Dictionary with dot-separated keys.
        sep: Separator used in keys.

    Returns:
        Nested dictionary structure.

    Example:
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {"a": {"b": 1, "c": 2}}
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def compare_dicts(
    base: dict[str, Any],
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare two flattened dicts, returning changed and unchanged values.

    Args:
        base: Base dictionary (flattened).
        overlay: Overlay dictionary (flattened).

    Returns:
        Tuple of (changed_values, unchanged_values) where changed_values
        contains keys that differ between base and overlay.
    """
    changed: dict[str, Any] = {}
    unchanged: dict[str, Any] = {}

    all_keys = set(base.keys()) | set(overlay.keys())
    for key in all_keys:
        base_val = base.get(key)
        overlay_val = overlay.get(key)

        if overlay_val is not None and overlay_val != base_val:
            changed[key] = overlay_val
        elif base_val is not None:
            unchanged[key] = base_val

    return changed, unchanged
