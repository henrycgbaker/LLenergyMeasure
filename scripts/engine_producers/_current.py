"""Per-engine version-bundle current-state loader.

The current-state file lives at ``<repo_root>/engine_versions/{engine}/current.yaml``.
This module exposes the lookup helpers that read it:

- :func:`current_path` - absolute path to the engine's current.yaml.
- :func:`load_current` - parse the YAML to a dict.
- :func:`safe_version` - identifier-safe mangling of a PEP 440 version.
- :func:`current_outputs_dir` - directory holding the engine's per-version
  mined outputs (``engine_versions/{engine}/v{safe}/outputs/``).

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
    """Return ``engine_versions/{engine}/v{safe}/outputs/`` for the current pin.

    Resolves ``library.current_version`` from the engine's ``current.yaml``,
    mangles it via :func:`safe_version`, and joins under the engine's
    version-bundle root. The returned path is the SSOT directory holding the
    per-version mined corpus artefacts (``invariants.proposed.yaml``,
    ``invariants.validated.yaml``, ``schema.discovered.json``).

    Used by sync scripts that mirror the per-version archive into the loader's
    expected location under ``src/llenergymeasure/engines/<engine>/`` (the
    "data shadow" the wheel ships).
    """
    data = load_current(engine)
    library = data.get("library")
    if not isinstance(library, dict):
        raise ValueError(f"current.yaml for {engine!r} missing required 'library' mapping.")
    version = library.get("current_version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"current.yaml for {engine!r} missing 'library.current_version' string.")
    return (
        _find_repo_root(Path(__file__).resolve())
        / "engine_versions"
        / engine
        / safe_version(version)
        / "outputs"
    )
