"""Per-engine version-bundle SSOT loader for miners.

The SSOT lives at ``<repo_root>/engine_versions/{engine}.yaml``. This
module exposes a single helper, :func:`load_miner_pin`, that returns the
:class:`~packaging.specifiers.SpecifierSet` for one ``(engine, producer)``
pair, where ``producer`` is one of the keys under ``miner_pins:`` in the
SSOT (``static`` | ``dynamic`` | ``discovery``).

The SSOT path is resolved by walking up from this file until a
``pyproject.toml`` marker is found - the canonical project-root marker
across the codebase. Failing to find the SSOT raises ``FileNotFoundError``
loud (no silent fallback to a hard-coded default).

Cached via :func:`functools.lru_cache` so repeated calls within one CI run
re-parse the YAML at most once per engine.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from packaging.specifiers import SpecifierSet

Producer = Literal["static", "dynamic", "discovery"]


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a ``pyproject.toml`` marker appears."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root (no pyproject.toml above {start}).")


def ssot_path(engine: str) -> Path:
    """Return the absolute path to ``engine_versions/{engine}.yaml``."""
    return _find_repo_root(Path(__file__).resolve()) / "engine_versions" / f"{engine}.yaml"


@cache
def load_miner_pin(engine: str, producer: Producer) -> SpecifierSet:
    """Return the SpecifierSet for ``miner_pins.{producer}`` in the engine's SSOT.

    Raises :class:`FileNotFoundError` if the SSOT file is missing,
    :class:`KeyError` if ``miner_pins.{producer}`` is unset.
    """
    path = ssot_path(engine)
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Engine SSOT not found at {path}. "
            f"Every engine miner reads its version pin from "
            f"engine_versions/{engine}.yaml miner_pins.{producer}."
        ) from exc
    data = yaml.safe_load(text)
    pins = (data or {}).get("miner_pins") or {}
    if producer not in pins:
        raise KeyError(f"miner_pins.{producer} missing from {path}; present keys: {sorted(pins)}.")
    return SpecifierSet(str(pins[producer]))


@cache
def load_ssot(engine: str) -> dict[str, object]:
    """Read + parse ``engine_versions/{engine}.yaml`` into a dict.

    Cached (lru) so repeated calls within one process re-parse at most once
    per engine. Used by producer modules to discover ``library.current_version``
    when dispatching to per-version machinery in
    ``llenergymeasure._engine_archive``.

    Raises :class:`FileNotFoundError` if the SSOT is missing,
    :class:`ValueError` if it does not parse to a mapping.
    """
    path = ssot_path(engine)
    text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"SSOT at {path} did not parse to a mapping.")
    return data


def safe_version(version: str) -> str:
    """Map a dotted PEP-440 version string to a Python-identifier-safe form.

    ``"0.7.3"`` -> ``"v0_7_3"``. Used to derive subpackage names under
    ``llenergymeasure._engine_archive.<engine>.<safe_version>.machinery.*``.
    Raises :class:`ValueError` if the resulting identifier would be illegal
    (e.g. version contains characters other than ``[0-9a-zA-Z._-]``).
    """
    safe = "v" + version.replace(".", "_").replace("-", "_")
    if not safe.replace("_", "").isalnum():
        raise ValueError(
            f"Cannot derive a Python identifier from version {version!r}; "
            f"resulting candidate {safe!r} contains non-alphanumeric chars."
        )
    return safe
