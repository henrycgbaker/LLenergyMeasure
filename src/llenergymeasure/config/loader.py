"""YAML/JSON configuration loader for experiment configs (v2.0).

Implements the v2.0 loading contract:
- Collect ALL errors before raising (not one-at-a-time)
- ConfigError with file path + did-you-mean for unknown fields
- CLI override merging at highest priority
- Native YAML anchor support via yaml.safe_load

Priority (highest wins): cli_overrides > path YAML > user_config_defaults
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.exceptions import ConfigError

__all__ = ["deep_merge", "load_experiment_config"]


# =============================================================================
# Public API
# =============================================================================


def load_experiment_config(
    path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    user_config_defaults: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and validate experiment configuration.

    Priority (highest wins): cli_overrides > path YAML > user_config_defaults

    Args:
        path: Path to YAML or JSON config file. None = only CLI/defaults.
        cli_overrides: Dict of CLI flag overrides (e.g. {"model": "gpt2", "backend": "pytorch"}).
            Keys match ExperimentConfig field names. None values are ignored (unset flags).
        user_config_defaults: Dict of user config defaults to apply as lowest priority.
            Only fields valid on ExperimentConfig (e.g. output_dir, backend defaults).

    Returns:
        Validated ExperimentConfig.

    Raises:
        ConfigError: File not found, parse error, unknown fields, or structural validation failure.
            Includes all errors collected at once (not one-at-a-time).
        ValidationError: Pydantic field-level validation errors pass through unchanged.
            (Bad values like n=-1 are Pydantic's domain; unknown keys become ConfigError.)
    """
    # Start with user config defaults (lowest priority)
    merged: dict[str, Any] = {}
    if user_config_defaults:
        merged = deep_merge(
            merged, {k: v for k, v in user_config_defaults.items() if v is not None}
        )

    # Load and apply YAML/JSON file
    if path is not None:
        file_dict = _load_file(path)  # raises ConfigError on missing/parse error
        merged = deep_merge(merged, file_dict)

    # Apply CLI overrides (highest priority, skip None values)
    if cli_overrides:
        overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        merged = deep_merge(
            merged, _unflatten(overrides)
        )  # handle "pytorch.batch_size" dotted keys

    # Strip optional version field — not an ExperimentConfig field
    merged.pop("version", None)

    # Collect unknown field errors before handing to Pydantic
    known_fields = set(ExperimentConfig.model_fields.keys())
    unknown = set(merged.keys()) - known_fields
    if unknown:
        errors = []
        for key in sorted(unknown):
            suggestion = _did_you_mean(key, known_fields)
            msg = f"Unknown field '{key}'"
            if suggestion:
                msg += f" — did you mean '{suggestion}'?"
            if path:
                msg += f" (in {path})"
            errors.append(msg)
        raise ConfigError("\n".join(errors))

    # Construct ExperimentConfig — let ValidationError pass through unchanged
    try:
        return ExperimentConfig(**merged)
    except ValidationError:
        raise  # Pydantic field-level errors are not our domain to wrap
    except Exception as e:
        context = f" (in {path})" if path else ""
        raise ConfigError(f"Config construction failed{context}: {e}") from e


# =============================================================================
# Utility: deep merge
# =============================================================================


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with overlay taking precedence.

    Args:
        base: Base dictionary.
        overlay: Dictionary to overlay on base.

    Returns:
        Merged dictionary (new object, originals unchanged).
    """
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


# =============================================================================
# Private helpers
# =============================================================================


def _load_file(path: Path | str) -> dict[str, Any]:
    """Load YAML or JSON config file into a dict.

    Args:
        path: Path to config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ConfigError: If file not found, unsupported format, parse error, or not a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            result = yaml.safe_load(content)  # native YAML anchors (&/*) handled automatically
        elif path.suffix == ".json":
            result = json.loads(content)
        else:
            raise ConfigError(f"Unsupported config format '{path.suffix}': use .yaml or .json")
        if not isinstance(result, dict):
            raise ConfigError(f"Config must be a mapping (got {type(result).__name__}): {path}")
        return result
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigError(f"Parse error in {path}: {e}") from e


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested dicts. Non-dotted keys pass through.

    Example:
        {"pytorch.batch_size": 8, "model": "gpt2"}
        -> {"pytorch": {"batch_size": 8}, "model": "gpt2"}
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if "." in key:
            parts = key.split(".", 1)
            if parts[0] not in result:
                result[parts[0]] = {}
            if isinstance(result[parts[0]], dict):
                result[parts[0]][parts[1]] = value
        else:
            result[key] = value
    return result


def _did_you_mean(key: str, candidates: set[str], max_distance: int = 3) -> str | None:
    """Return the closest candidate if within max_distance edits, else None.

    Args:
        key: Unknown key to find a suggestion for.
        candidates: Set of valid field names.
        max_distance: Maximum Levenshtein distance to suggest (default 3).

    Returns:
        Closest candidate string, or None if nothing is close enough.
    """
    best: str | None = None
    best_dist = max_distance + 1
    for candidate in candidates:
        dist = _levenshtein(key, candidate)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best if best_dist <= max_distance else None


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]
