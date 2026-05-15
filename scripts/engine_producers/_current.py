"""Per-engine version-bundle current-state loader.

The current-state file lives at ``<repo_root>/engine_versions/{engine}/current.yaml``.
This module exposes the lookup helpers that read it:

- :func:`current_path` - absolute path to the engine's current.yaml.
- :func:`load_current` - parse the YAML to a dict.
- :func:`safe_version` - identifier-safe mangling of a PEP 440 version.

The current-state path is resolved by walking up from this file until a
``pyproject.toml`` marker is found - the canonical project-root marker
across the codebase. Failing to find the file raises ``FileNotFoundError``
loud (no silent fallback to a hard-coded default).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a ``pyproject.toml`` marker appears."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root (no pyproject.toml above {start}).")


def current_path(engine: str) -> Path:
    """Return the absolute path to ``engine_versions/{engine}/current.yaml``."""
    return _find_repo_root(Path(__file__).resolve()) / "engine_versions" / engine / "current.yaml"


def load_current(engine: str) -> dict[str, object]:
    """Read + parse ``engine_versions/{engine}/current.yaml`` into a dict.

    Intentionally uncached: callers (probe, producer modules) all run at
    most once per cell invocation, and tests rely on hermetic per-test
    fixtures that an ``@cache`` decorator would silently shadow across
    test cases.

    Raises :class:`FileNotFoundError` if current.yaml is missing,
    :class:`ValueError` if it does not parse to a mapping.
    """
    path = current_path(engine)
    text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"current.yaml at {path} did not parse to a mapping.")
    return data


def safe_version(version: str) -> str:
    """Map a dotted PEP-440 version string to a Python-identifier-safe form.

    ``"0.7.3"`` -> ``"v0_7_3"``. Used to derive subpackage names under
    ``engine_versions.<engine>.<safe_version>.producers.*``.
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


def current_outputs_dir(engine: str) -> Path:
    """Return ``engine_versions/{engine}/v<safe>/outputs/`` for the current version.

    Resolves the current ``library.current_version`` from
    ``engine_versions/<engine>/current.yaml``, derives the safe form, and
    returns the per-version outputs directory that hosts the bot-written
    machine artefacts (``invariants.proposed.yaml``,
    ``invariants.validated.yaml``, ``schema.discovered.json``).

    The directory is the bundling source for hatchling's force-include
    (see ``pyproject.toml``) and the read target for the doc-generators
    and refresh shell scripts that previously read from
    ``<pkg_dir>/llenergymeasure/engines/<engine>/`` (via hatchling force-include from the outputs/ source).
    """
    raw_version = load_current(engine).get("library", {})
    if not isinstance(raw_version, dict):
        raise ValueError(
            f"current.yaml for {engine!r} has no 'library' mapping; cannot resolve current version."
        )
    version = raw_version.get("current_version")
    if not isinstance(version, str) or not version:
        raise ValueError(
            f"current.yaml for {engine!r} has no 'library.current_version' string; "
            f"cannot resolve current version."
        )
    return current_path(engine).parent / safe_version(version) / "outputs"
