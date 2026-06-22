"""Shared dictionary utilities: unflatten dotted keys, deep merge.

Canonical home for these utilities, imported by config/loader.py
and config/grid.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from llenergymeasure.utils.exceptions import ConfigError


def is_sweep_group(value: object) -> bool:
    """True if a sweep entry is a group (list of dicts), not an independent axis.

    Disambiguation: a list of scalars is an independent axis (Cartesian product);
    a list of dicts (or containing ``{}``) is a dependent group (union of variants).

    Raises ``ConfigError`` for mixed lists (some dicts, some scalars).
    """
    if not isinstance(value, list) or len(value) == 0:
        return False
    has_dicts = any(isinstance(e, dict) for e in value)
    if not has_dicts:
        return False
    all_dicts = all(isinstance(e, dict) for e in value)
    if not all_dicts:
        raise ConfigError(
            "Sweep entry mixes dicts and scalars. Group entries must all be "
            "dicts; independent axes must all be scalars."
        )
    return True


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested dicts. Non-dotted keys pass through.

    Example:
        {"engine.block_size": 16}        -> {"engine": {"block_size": 16}}
        {"task.dataset.n_prompts": 50}   -> {"task": {"dataset": {"n_prompts": 50}}}
        {"batch_size": 4}                -> {"batch_size": 4}
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if "." not in key:
            result[key] = value
            continue
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return result


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. overlay values take precedence over base values."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
