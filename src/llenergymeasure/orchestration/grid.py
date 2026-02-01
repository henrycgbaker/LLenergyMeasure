"""Campaign grid expansion and validation for multi-config experiments.

Generates experiment configs from a two-level grid definition:
- Shared parameters applied across all backends
- Backend-specific overrides per backend

Validation uses Pydantic dry-run instantiation (Tier 1) and SSOT introspection
for hardware capability warnings (Tier 2).
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ValidationError

from llenergymeasure.config.campaign_config import CampaignGridConfig


class GridValidationIssue(BaseModel):
    """A validation issue found during grid expansion."""

    config_desc: str
    """Human-readable description of the config (e.g., 'pytorch/float16/batch_size=4')."""

    reason: str
    """Why it was filtered or flagged."""

    severity: Literal["error", "warning"]
    """Error = invalid (filtered), Warning = may fail at runtime (kept)."""


class GridExpansionResult(BaseModel):
    """Result of expanding and validating a campaign grid."""

    valid_configs: list[dict[str, Any]]
    """Config dicts that passed validation."""

    filtered_configs: list[GridValidationIssue]
    """Configs that failed Pydantic validation."""

    warnings: list[GridValidationIssue]
    """Configs with hardware/env warnings (still included in valid)."""

    total_generated: int
    """Total configs before filtering."""

    @property
    def summary(self) -> str:
        """Human-readable summary of expansion results."""
        parts = [f"Generated {self.total_generated} experiments"]

        n_valid = len(self.valid_configs)
        n_filtered = len(self.filtered_configs)
        n_warnings = len(self.warnings)

        if n_filtered > 0:
            reasons = sorted({f.reason for f in self.filtered_configs})
            reason_str = "; ".join(reasons)
            parts.append(f"{n_filtered} filtered as invalid: {reason_str}")
        else:
            parts.append(f"{n_valid} valid")

        if n_warnings > 0:
            parts.append(f"{n_warnings} have hardware warnings")

        return " | ".join(parts)


def _set_nested(d: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation.

    Args:
        d: Target dictionary to modify in-place.
        key_path: Dot-separated key path (e.g., 'decoder.preset').
        value: Value to set.
    """
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _describe_config(config_dict: dict[str, Any]) -> str:
    """Generate a human-readable description of a config dict.

    Args:
        config_dict: Experiment config dictionary.

    Returns:
        Short description string like 'pytorch/float16/batch_size=4'.
    """
    parts: list[str] = []
    backend = config_dict.get("backend", "unknown")
    parts.append(backend)

    if fp := config_dict.get("fp_precision"):
        parts.append(str(fp))

    # model_name is a top-level field in ExperimentConfig
    model_name = config_dict.get("model_name", "")
    if model_name:
        # Use short model name (last part after /)
        parts.append(str(model_name).rsplit("/", 1)[-1])

    # Add any backend-specific params that were set
    backend_section = config_dict.get(backend, {})
    if isinstance(backend_section, dict):
        for k, v in sorted(backend_section.items()):
            parts.append(f"{k}={v}")

    return "/".join(parts)


def _generate_config_name(config_dict: dict[str, Any], index: int) -> str:
    """Generate a unique config_name for a grid-expanded config.

    Uses backend, model short name, and key params to build a descriptive name.
    Falls back to index-based naming if description is too short.

    Args:
        config_dict: Experiment config dictionary.
        index: Config index in the expansion.

    Returns:
        Config name string like 'pytorch-gpt2-float16'.
    """
    parts: list[str] = []
    backend = config_dict.get("backend", "unknown")
    parts.append(backend)

    model_name = config_dict.get("model_name", "")
    if model_name:
        parts.append(str(model_name).rsplit("/", 1)[-1])

    if fp := config_dict.get("fp_precision"):
        parts.append(str(fp))

    # Add backend-specific params for uniqueness
    backend_section = config_dict.get(backend, {})
    if isinstance(backend_section, dict):
        for k, v in sorted(backend_section.items()):
            parts.append(f"{k}{v}")

    name = "-".join(parts)
    return name if len(parts) > 1 else f"{name}-{index}"


def expand_campaign_grid(
    grid: CampaignGridConfig,
    base_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand campaign grid into individual experiment config dicts.

    Generates cartesian product of:
    - backends x models x shared params x backend-specific params

    Args:
        grid: Campaign grid configuration.
        base_config: Optional base config dict to merge into each variation.

    Returns:
        List of config dicts (one per experiment).
    """
    base = base_config or {}
    results: list[dict[str, Any]] = []

    # Build shared axes: list of (key, values) pairs
    shared_axes: list[tuple[str, list[Any]]] = []
    for key, values in grid.shared.items():
        if isinstance(values, list):
            shared_axes.append((key, values))
        else:
            # Single value treated as list of one
            shared_axes.append((key, [values]))

    # Models axis: list of model names, or [None] to iterate once without model override
    models_to_sweep: Sequence[str | None]
    if grid.models:
        models_to_sweep = grid.models
    else:
        no_model: list[str | None] = [None]
        models_to_sweep = no_model

    # Generate shared combinations
    if shared_axes:
        shared_keys = [k for k, _ in shared_axes]
        shared_values = [v for _, v in shared_axes]
        shared_combos = list(itertools.product(*shared_values))
    else:
        shared_keys = []
        shared_combos = [()]  # Single empty combo

    config_index = 0
    for backend in grid.backends:
        # Build backend-specific override axes
        override_axes: list[tuple[str, list[Any]]] = []
        if backend in grid.backend_overrides:
            for key, val in grid.backend_overrides[backend].items():
                if isinstance(val, list):
                    override_axes.append((key, val))
                else:
                    override_axes.append((key, [val]))

        if override_axes:
            override_keys = [k for k, _ in override_axes]
            override_values = [v for _, v in override_axes]
            override_combos = list(itertools.product(*override_values))
        else:
            override_keys = []
            override_combos = [()]  # Single empty combo

        for model in models_to_sweep:
            for shared_combo in shared_combos:
                for override_combo in override_combos:
                    config = copy.deepcopy(base)

                    # Set backend
                    config["backend"] = backend

                    # Set model_name (top-level field for ExperimentConfig)
                    if model is not None:
                        config["model_name"] = model

                    # Set shared params (handle nested keys with dot notation)
                    for key, value in zip(shared_keys, shared_combo, strict=False):
                        _set_nested(config, key, value)

                    # Set backend-specific overrides under the backend section
                    if override_keys:
                        if backend not in config or not isinstance(config.get(backend), dict):
                            config[backend] = {}
                        for key, value in zip(override_keys, override_combo, strict=False):
                            config[backend][key] = value

                    # Auto-generate config_name if not already in base
                    if "config_name" not in config:
                        config["config_name"] = _generate_config_name(config, config_index)

                    config_index += 1
                    results.append(config)

    logger.debug(
        "Grid expansion: {} backends x {} models x {} shared combos = {} configs",
        len(grid.backends),
        len(models_to_sweep),
        len(shared_combos),
        len(results),
    )

    return results


def validate_campaign_grid(
    config_dicts: list[dict[str, Any]],
) -> GridExpansionResult:
    """Validate expanded grid configs via Pydantic dry-run instantiation.

    Two-tier validation:
    1. Pydantic: Attempt ExperimentConfig(**config_dict) - catches invalid combos
    2. SSOT: Check backend capabilities and skip conditions - adds warnings

    Args:
        config_dicts: List of experiment config dicts from expand_campaign_grid.

    Returns:
        GridExpansionResult with valid configs, filtered configs, and warnings.
    """
    from llenergymeasure.config.introspection import get_param_skip_conditions
    from llenergymeasure.config.models import ExperimentConfig

    valid_configs: list[dict[str, Any]] = []
    filtered_configs: list[GridValidationIssue] = []
    warnings: list[GridValidationIssue] = []

    skip_conditions = get_param_skip_conditions()

    for config_dict in config_dicts:
        desc = _describe_config(config_dict)

        # Tier 1: Pydantic validation
        try:
            ExperimentConfig(**config_dict)
        except ValidationError as e:
            # Extract first error message for readability
            errors = e.errors()
            reason = errors[0]["msg"] if errors else str(e)
            filtered_configs.append(
                GridValidationIssue(
                    config_desc=desc,
                    reason=reason,
                    severity="error",
                )
            )
            logger.debug("Grid config filtered: {} - {}", desc, reason)
            continue

        # Tier 2: SSOT hardware/env warnings
        backend = config_dict.get("backend", "")
        backend_section = config_dict.get(backend, {})

        config_warnings: list[str] = []
        if isinstance(backend_section, dict):
            for param, value in backend_section.items():
                # Check param=value skip conditions
                skip_key = f"{backend}.{param}={value}"
                if skip_key in skip_conditions:
                    config_warnings.append(skip_conditions[skip_key])

                # Check param>value skip conditions (for numeric params)
                if isinstance(value, int | float) and value > 1:
                    gt_key = f"{backend}.{param}>{1}"
                    if gt_key in skip_conditions:
                        config_warnings.append(skip_conditions[gt_key])

        # Check top-level params too
        fp_precision = config_dict.get("fp_precision", "")
        fp_skip_key = f"{backend}.fp_precision={fp_precision}"
        if fp_skip_key in skip_conditions:
            config_warnings.append(skip_conditions[fp_skip_key])

        # Add to valid (warnings don't filter)
        valid_configs.append(config_dict)

        for warning_msg in config_warnings:
            warnings.append(
                GridValidationIssue(
                    config_desc=desc,
                    reason=warning_msg,
                    severity="warning",
                )
            )

    result = GridExpansionResult(
        valid_configs=valid_configs,
        filtered_configs=filtered_configs,
        warnings=warnings,
        total_generated=len(config_dicts),
    )

    logger.info(
        "Grid validation: {} valid, {} filtered, {} warnings",
        len(valid_configs),
        len(filtered_configs),
        len(warnings),
    )

    return result


__all__ = [
    "GridExpansionResult",
    "GridValidationIssue",
    "expand_campaign_grid",
    "validate_campaign_grid",
]
