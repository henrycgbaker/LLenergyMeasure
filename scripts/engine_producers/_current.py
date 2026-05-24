"""Per-engine version-bundle current-state loader.

The current-state file lives at ``<repo_root>/engine_versions/{engine}/current.toml``.
This module exposes the lookup helpers that read it:

- :func:`current_path` - absolute path to the engine's current.toml.
- :func:`load_current` - parse the TOML to a dict.
- :func:`safe_version` - identifier-safe mangling of a PEP 440 version.
- :func:`current_outputs_dir` - resolved archive outputs/ directory for the
  engine's currently-pinned version (the canonical write target for miners).

The current-state path is resolved by walking up from this file until a
``pyproject.toml`` marker is found - the canonical project-root marker
across the codebase. Failing to find the file raises ``FileNotFoundError``
loud (no silent fallback to a hard-coded default).
"""

from __future__ import annotations

from pathlib import Path

import tomllib


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a ``pyproject.toml`` marker appears."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root (no pyproject.toml above {start}).")


def current_path(engine: str) -> Path:
    """Return the absolute path to ``engine_versions/{engine}/current.toml``."""
    return _find_repo_root(Path(__file__).resolve()) / "engine_versions" / engine / "current.toml"


def load_current(engine: str) -> dict[str, object]:
    """Read + parse ``engine_versions/{engine}/current.toml`` into a dict.

    Intentionally uncached: callers (probe, producer modules) all run at
    most once per cell invocation, and tests rely on hermetic per-test
    fixtures that an ``@cache`` decorator would silently shadow across
    test cases.

    Raises :class:`FileNotFoundError` if current.toml is missing,
    :class:`ValueError` if it does not parse to a mapping.
    """
    path = current_path(engine)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"current.toml at {path} did not parse to a mapping.")
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
    """Return the archive outputs/ directory for the engine's pinned version.

    Resolves ``engine_versions/{engine}/current.toml``, reads
    ``library.current_version``, and returns
    ``engine_versions/{engine}/v{safe_version}/outputs/`` - the canonical
    write target for miners (the version-bundle archive is the SSOT; the
    shadow under ``src/llenergymeasure/engines/<engine>/`` is derived
    via the CI cell's archive -> shadow mirror).

    Raises :class:`FileNotFoundError` if current.toml is missing (delegated
    from :func:`load_current`), :class:`ValueError` if
    ``library.current_version`` is missing or not a non-empty string.
    """
    current = load_current(engine)
    library = current.get("library")
    if not isinstance(library, dict):
        raise ValueError(
            f"engine_versions/{engine}/current.toml: 'library' must be a mapping, "
            f"got {type(library).__name__}."
        )
    # F#15: split missing vs wrong-type into distinct error messages so the
    # operator can tell the cases apart (e.g. a YAML-parsed bare float 4.73
    # is present-but-non-string, not missing).
    if "current_version" not in library:
        raise ValueError(
            f"engine_versions/{engine}/current.toml: 'library.current_version' "
            f"key is missing."
        )
    version = library["current_version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"engine_versions/{engine}/current.toml: 'library.current_version' "
            f"must be a non-empty string, got {version!r} (type {type(version).__name__})."
        )
    safe = safe_version(version)
    return current_path(engine).parent / safe / "outputs"
