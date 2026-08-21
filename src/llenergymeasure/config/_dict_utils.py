"""Shared dictionary utilities: unflatten dotted keys, deep merge, leaf paths.

Canonical home for these utilities, imported by config/loader.py
and config/grid.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


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


def dotted_leaf_paths(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths of every leaf in ``data`` - a non-dict value, or an empty dict.

    An empty dict is a leaf because it carries no leaves of its own to name. A key
    that is already dotted (``{"task.model": "m"}``) is left as it is, so a nested
    and a dotted spelling of the same override normalise to the same path.

    Example:
        {"output": {"results_dir": "/d"}} -> ["output.results_dir"]
        {"runners": {}}                   -> ["runners"]
    """
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(dotted_leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. overlay values take precedence over base values."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
