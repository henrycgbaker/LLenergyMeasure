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
- :func:`carry_forward_inputs` - seed the current pin's outputs/ with the
  maintainer-owned input files (curated.yaml always; overlay.yaml when the
  prior pin had one) from the most-recent prior pin, so a Renovate bump that
  only touches current.yaml still produces a complete SSOT for the pipeline
  to mine + derive against.
- :func:`is_major_bump` - whether the current pin crosses a semver MAJOR
  over the previous pin (drives the major-bump label + churn warning).

The current-state path is resolved by walking up from this file until a
``pyproject.toml`` marker is found - the canonical project-root marker
across the codebase. Failing to find the file raises ``FileNotFoundError``
loud (no silent fallback to a hard-coded default).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

# Maintainer-owned input files (design section 4). curated.yaml is required:
# it is the exposure allowlist every derivation needs. overlay.yaml is
# optional: hand-authored narrowings that only some pins carry.
REQUIRED_INPUT_FILES: tuple[str, ...] = ("curated.yaml",)
OPTIONAL_INPUT_FILES: tuple[str, ...] = ("overlay.yaml",)


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


def carry_forward_inputs(engine: str) -> list[str]:
    """Seed the current pin's outputs/ with the maintainer-owned input files.

    A real Renovate bump edits only ``engine_versions/{engine}/current.yaml``,
    so the new pin's ``outputs/`` directory starts empty - it has no
    ``curated.yaml`` (the exposure allowlist every derivation needs) and no
    ``overlay.yaml`` (hand-authored narrowings). These are maintainer-owned
    inputs that "carry forward unchanged" across a bump (design section 5
    step 5), so the pipeline copies them from the most-recent prior pin's
    outputs/ before any mining/derivation step reads them. They then appear in
    the bump PR diff as the starting point the maintainer reviews/edits
    (design step 7).

    Carry rules:

    - ``curated.yaml`` is REQUIRED. If the current pin already has one, it is
      left untouched (never clobber maintainer edits). If it is missing, it is
      copied from the prior pin. If neither the current pin nor any prior pin
      has one, raise ``FileNotFoundError`` - a brand-new engine needs a
      bootstrap curated.yaml, which is out of scope for the per-bump loop.
    - ``overlay.yaml`` is OPTIONAL. Carried only when the current pin lacks one
      and the prior pin has one; its absence everywhere is fine, not an error.

    Returns the list of file names (e.g. ``["curated.yaml"]``) actually copied,
    for the caller to log. Idempotent: a second call is a no-op (returns
    ``[]``) because the files now exist on the current pin.
    """
    current = current_outputs_dir(engine)
    current.mkdir(parents=True, exist_ok=True)
    prior = previous_pin_outputs_dir(engine)
    copied: list[str] = []

    for name in REQUIRED_INPUT_FILES:
        dst = current / name
        if dst.exists():
            continue
        src = (prior / name) if prior is not None else None
        if src is None or not src.exists():
            raise FileNotFoundError(
                f"{engine}: required input {name} missing from the current pin "
                f"({current}) and no prior pin carries one to forward. A brand-new "
                f"engine needs a bootstrap {name} (out of scope for the per-bump loop)."
            )
        shutil.copy2(src, dst)
        copied.append(name)

    for name in OPTIONAL_INPUT_FILES:
        dst = current / name
        if dst.exists() or prior is None:
            continue
        src = prior / name
        if not src.exists():
            continue
        shutil.copy2(src, dst)
        copied.append(name)

    return copied


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

    ``prev_outputs`` is printed REPO-RELATIVE, not absolute: the decay-alarm
    re-gate consumes it inside the engine container where the checkout mounts
    at ``/repo`` (``-w /repo``), so an absolute runner path would dangle. The
    host-side consumers (surface trend, gate-report comment) run from the
    checkout root, where the relative form resolves identically.

    With ``--carry-forward`` the command instead seeds the current pin's
    maintainer-owned input files from the prior pin (see
    :func:`carry_forward_inputs`) and prints what it copied; this is the
    pipeline's pre-mine step on a fresh-version bump.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="Engine name")
    parser.add_argument(
        "--carry-forward",
        action="store_true",
        help=(
            "Seed the current pin's outputs/ with curated.yaml (always) and "
            "overlay.yaml (when the prior pin had one) from the most-recent "
            "prior pin, then exit. Idempotent."
        ),
    )
    args = parser.parse_args(argv)

    if args.carry_forward:
        copied = carry_forward_inputs(args.engine)
        if copied:
            print(f"[carry-forward] {args.engine}: seeded {', '.join(copied)} from the prior pin.")
        else:
            print(
                f"[carry-forward] {args.engine}: maintainer inputs already present; nothing carried."
            )
        return 0

    prev = previous_pin_outputs_dir(args.engine)
    if prev is not None:
        prev = prev.relative_to(_find_repo_root(Path(__file__).resolve()))
    print(f"prev_outputs={prev if prev is not None else ''}")
    print(f"major_bump={'true' if is_major_bump(args.engine) else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
