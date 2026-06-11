"""Per-engine version-bundle current-state loader.

The current-state file lives at ``<repo_root>/engine_versions/{engine}/current.yaml``.
This module exposes the lookup helpers that read it:

- :func:`current_path` - absolute path to the engine's current.yaml.
- :func:`load_current` - parse the YAML to a dict.
- :func:`safe_version` - identifier-safe mangling of a PEP 440 version.
- :func:`current_version` - the pinned version string for an engine.
- :func:`current_outputs_dir` - the SSOT outputs/ directory for the active pin.
- :func:`previous_pin_outputs_dir` - the most-recent prior pin's outputs/
  directory (the carried-input source for the decay alarm + surface trend),
  or ``None`` when no prior vendored pin exists yet.
- :func:`is_major_bump` - whether the current pin crosses a semver MAJOR
  over the previous pin (drives the major-bump label + churn warning).

The current-state path is resolved by walking up from this file until a
``pyproject.toml`` marker is found - the canonical project-root marker
across the codebase. Failing to find the file raises ``FileNotFoundError``
loud (no silent fallback to a hard-coded default).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version


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


def current_version(engine: str) -> str:
    """Return the pinned version string from ``engine_versions/{engine}/current.yaml``.

    Reads ``library.current_version``. Raises :class:`ValueError` if the key
    is missing or is not a string (a malformed pin must fail loud, not
    silently resolve to a default path).
    """
    data = load_current(engine)
    library = data.get("library")
    version = library.get("current_version") if isinstance(library, dict) else None
    if not isinstance(version, str):
        raise ValueError(
            f"current.yaml for {engine!r} has no string library.current_version (got {version!r})."
        )
    return version


def current_outputs_dir(engine: str) -> Path:
    """Return the SSOT outputs/ directory for the engine's active pin.

    ``engine_versions/{engine}/v<safe>/outputs/`` where ``<safe>`` is the
    identifier-safe form of ``library.current_version``. This is the canonical
    locus the sync script copies into the ``src/`` data shadow.
    """
    root = _find_repo_root(Path(__file__).resolve())
    safe = safe_version(current_version(engine))
    return root / "engine_versions" / engine / safe / "outputs"


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


def previous_pin_outputs_dir(engine: str) -> Path | None:
    """Return the outputs/ directory of the most-recent prior vendored pin.

    "Prior" means: a ``engine_versions/{engine}/v<safe>/`` version directory
    that (a) carries a populated ``outputs/`` directory and (b) is a STRICTLY
    LOWER semver than the current pin. The newest such version wins. Returns
    ``None`` when no qualifying prior exists - which is the state for every
    engine today (each has exactly one outputs-bearing version dir, the
    current pin), so the decay alarm and surface-trend steps are a structural
    no-op until C10 populates the trailing window.

    The carried-input source for the decay alarm (``rules.proposed.yaml`` in
    the returned directory; the validated envelope records gate output only)
    and the OLD side of the surface-trend diff.
    """
    root = _find_repo_root(Path(__file__).resolve())
    engine_dir = root / "engine_versions" / engine
    if not engine_dir.is_dir():
        return None

    try:
        current = Version(current_version(engine))
    except InvalidVersion:
        return None

    candidates: list[tuple[Version, Path]] = []
    for version_dir in engine_dir.iterdir():
        if not version_dir.is_dir() or not version_dir.name.startswith("v"):
            continue
        outputs = version_dir / "outputs"
        if not outputs.is_dir() or not any(outputs.iterdir()):
            continue
        parsed = _dir_version(version_dir.name)
        if parsed is not None and parsed < current:
            candidates.append((parsed, outputs))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _dir_version(version_dir_name: str) -> Version | None:
    """Recover the semver from a ``v<safe>`` version-directory name.

    ``safe_version`` maps both ``.`` and ``-`` to ``_``, and the engine
    versions in scope are dotted release segments, so ``_`` -> ``.``
    round-trips the comparison key. Returns ``None`` when it does not parse.
    """
    try:
        return Version(version_dir_name[1:].replace("_", "."))
    except InvalidVersion:
        return None


def is_major_bump(engine: str) -> bool:
    """True when the current pin crosses a semver MAJOR over the previous pin.

    False when there is no prior, when neither version parses, or when the
    major component is unchanged. Drives the major-bump label + the expected-
    churn warning the report leads with (design section 5 step 6).
    """
    outputs = previous_pin_outputs_dir(engine)
    if outputs is None:
        return False
    prev = _dir_version(outputs.parent.name)
    if prev is None:
        return False
    try:
        current = Version(current_version(engine))
    except InvalidVersion:
        return False
    return current.major != prev.major


def _main(argv: list[str] | None = None) -> int:
    """Resolve the prior-pin facts the engine-pipeline cell needs and print
    them as ``KEY=value`` lines for ``$GITHUB_OUTPUT``: the previous pin's
    outputs/ directory (empty when no prior exists) and whether the current
    pin is a major bump over it. The shell skips both alarm steps gracefully
    on an empty ``prev_outputs``.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="Engine name")
    args = parser.parse_args(argv)

    prev = previous_pin_outputs_dir(args.engine)
    print(f"prev_outputs={prev if prev is not None else ''}")
    print(f"major_bump={'true' if is_major_bump(args.engine) else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
