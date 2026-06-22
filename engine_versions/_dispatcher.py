"""Version-aware loader for per-engine producers.

Resolves the importable subpackage
``engine_versions.<engine>.<safe_version>.producers.<producer>``
and returns it. Producer modules in ``scripts/engine_producers/`` use this
dispatcher (via PEP 562 ``__getattr__`` on the module level) to delegate
``LANDMARKS`` exports to the version-pinned archive.

Producer kinds match the producer-archive subpackage names:

- ``static_invariant_miner``: validation-invariant miners (probe contract: ``invariants``)
- ``schema_introspector``: schema introspectors (probe contract: ``schemas``)
- ``dynamic_invariant_miner``: combinatorial miners (no probe contract today; reserved)

The version string is the dotted form from
``engine_versions/<engine>/current.yaml`` ``library.current_version`` (e.g.
``"0.7.3"``); the dispatcher converts it to a Python-identifier-safe form
(``v0_7_3``) via :func:`scripts.engine_producers._current.safe_version`.

Auto-fallback semantics
-----------------------

When the exact-match subpackage does not exist
(``engine_versions/<engine>/v<safe(current_version)>/producers/``), the
dispatcher scans sibling ``vN`` directories, picks the highest version
``<= target`` that contains a ``producers/<producer>.py`` file, and imports
it. A stderr log line names the fallback so the cell's run log surfaces
which producer was used.

This makes patch-version bumps zero-touch when LANDMARKS still resolve
under the new library: Renovate bumps ``current_version`` to ``0.7.4``, the
dispatcher falls back to ``v0_7_3/producers/``, and the probe (run by
``scripts._drift``) verifies the LANDMARKS still resolve under the live
``vllm==0.7.4``. The probe is the runtime gate; if a fallback producer's
landmarks no longer resolve at the new library version, the probe fails
red and the maintainer either patches LANDMARKS in the fallback producer
or vendors a fresh ``vN/producers/`` directory.

When no vendored archive exists at or below the target, the dispatcher
raises ``ModuleNotFoundError`` with the path the maintainer needs to create.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal

from packaging.version import InvalidVersion, Version

ProducerKind = Literal["static_invariant_miner", "dynamic_invariant_miner", "schema_introspector"]

_SAFE_VERSION_RE = re.compile(r"^v[0-9_a-zA-Z]+$")


def _safe_to_version(safe: str) -> Version | None:
    """Inverse of ``safe_version``: ``"v0_7_3"`` -> ``Version("0.7.3")``.

    Returns ``None`` for paths that don't match the safe-version shape
    (e.g. ``__pycache__``, ``outputs``) and for inputs whose recovered
    dotted form isn't a valid PEP 440 version.
    """
    if not _SAFE_VERSION_RE.match(safe):
        return None
    try:
        return Version(safe[1:].replace("_", "."))
    except InvalidVersion:
        return None


def find_fallback_safe_version(
    *, engine_root: Path, target_version: str, required_rel: Path
) -> str | None:
    """Return ``safe`` of highest vendored archive at or below ``target_version``.

    Scans ``engine_root`` for ``v[0-9_a-zA-Z]+`` directories, parses each via
    :func:`_safe_to_version`, filters to those ``<= target`` AND that contain a
    ``required_rel`` file (relative to the version directory), returns the
    highest match. Returns ``None`` when no candidate exists, when
    ``engine_root`` is not a directory, or when ``target_version`` does not
    parse as PEP 440.

    Engine-root path is taken as a parameter so the dispatch logic is unit-
    testable against tmp directories. :func:`load_producer` computes the real
    path from this module's location. ``required_rel`` is the per-resource
    presence marker: ``producers/<producer>.py`` for code dispatch,
    ``landmarks.yaml`` for the landmark-data loader.
    """
    try:
        target = Version(target_version)
    except InvalidVersion:
        return None
    if not engine_root.is_dir():
        return None

    candidates: list[tuple[Version, str]] = []
    for child in engine_root.iterdir():
        if not child.is_dir():
            continue
        version = _safe_to_version(child.name)
        if version is None or version > target:
            continue
        if not (child / required_rel).is_file():
            continue
        candidates.append((version, child.name))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def _find_fallback_safe_version(
    *, engine_root: Path, target_version: str, producer: ProducerKind
) -> str | None:
    """Code-dispatch wrapper around :func:`find_fallback_safe_version`.

    Requires a ``producers/<producer>.py`` file in each candidate version
    directory. The landmark-data loader uses the generic helper directly with
    a ``landmarks.yaml`` marker instead.
    """
    return find_fallback_safe_version(
        engine_root=engine_root,
        target_version=target_version,
        required_rel=Path("producers") / f"{producer}.py",
    )


def find_forward_safe_version(
    *, engine_root: Path, target_version: str, required_rel: Path
) -> str | None:
    """Return ``safe`` of the lowest vendored archive at or above ``target_version``.

    The forward counterpart of :func:`find_fallback_safe_version`, used only as
    a last resort when no archive exists at or below the target. In the
    single-producer steady state the one kept producer sits ABOVE the pin (e.g.
    a ``v0_19_1`` producer kept while ``current_version`` is still ``0.7.3``,
    because the algorithm is version-stable and only the per-version landmark
    DATA differs), so code dispatch must fall FORWARD to it. The runtime drift
    probe stays the gate: if the forward producer's ``LANDMARKS`` no longer
    resolve under the live library, the probe fails red.

    Returns ``None`` when no candidate exists, when ``engine_root`` is not a
    directory, or when ``target_version`` does not parse as PEP 440.
    """
    try:
        target = Version(target_version)
    except InvalidVersion:
        return None
    if not engine_root.is_dir():
        return None

    candidates: list[tuple[Version, str]] = []
    for child in engine_root.iterdir():
        if not child.is_dir():
            continue
        version = _safe_to_version(child.name)
        if version is None or version < target:
            continue
        if not (child / required_rel).is_file():
            continue
        candidates.append((version, child.name))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def _find_forward_safe_version(
    *, engine_root: Path, target_version: str, producer: ProducerKind
) -> str | None:
    """Code-dispatch wrapper around :func:`find_forward_safe_version`."""
    return find_forward_safe_version(
        engine_root=engine_root,
        target_version=target_version,
        required_rel=Path("producers") / f"{producer}.py",
    )


def load_producer(*, engine: str, version: str, producer: ProducerKind) -> ModuleType:
    """Return the ``producers.<producer>`` submodule for ``(engine, version)``.

    Args:
        engine: Engine name (``"vllm"`` | ``"tensorrt"`` | ``"transformers"``).
        version: Dotted PEP-440 version string (e.g. ``"0.7.3"``).
        producer: One of ``"static_invariant_miner"``, ``"dynamic_invariant_miner"``,
            ``"schema_introspector"``.

    Returns:
        The imported producer module. Exact match is preferred:
        ``engine_versions.<engine>.v<safe(version)>.producers.<producer>``.
        On miss, falls back to the highest vendored archive ``<= version``, or
        if none exists below, falls forward to the lowest archive ``>= version``
        (the single-producer steady state, where the kept producer is newer than
        the pin). A stderr log line names the resolved archive. The caller reads
        ``LANDMARKS`` (and optionally ``_CLASS_TARGETS`` etc.) from the
        returned module.

    Raises:
        ModuleNotFoundError: when no vendored archive exists in either direction
            with a ``producers/<producer>.py`` file. The error message names the
            exact file path to create.
    """
    # Late import: ``_current`` imports yaml/packaging which we'd rather not
    # pay at every ``__getattr__`` call site. Importing here keeps module-
    # load cheap.
    from scripts.engine_producers._current import safe_version

    safe = safe_version(version)
    qualified = f"engine_versions.{engine}.{safe}.producers.{producer}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError:
        # Fall through to fallback. Distinct from "import succeeded but raised":
        # a syntax error in an existing producer module would surface a
        # different exception (SyntaxError / ImportError without
        # ModuleNotFoundError), which propagates rather than triggering a
        # silent fallback to a different version.
        pass

    engine_root = Path(__file__).resolve().parent / engine
    resolved_safe = _find_fallback_safe_version(
        engine_root=engine_root, target_version=version, producer=producer
    )
    direction = "falling back to"
    if resolved_safe is None:
        # No archive at or below the target: fall FORWARD to the lowest archive
        # above it. This is the single-producer steady state, where the one kept
        # producer is newer than ``current_version`` (the algorithm is version-
        # stable; only the per-version landmark DATA differs). The drift probe
        # remains the runtime gate.
        resolved_safe = _find_forward_safe_version(
            engine_root=engine_root, target_version=version, producer=producer
        )
        direction = "falling forward to"
    if resolved_safe is None:
        raise ModuleNotFoundError(
            f"No archived producer for {engine}=={version} ({producer}), and "
            f"no vendored archive exists in either direction "
            f"with engine_versions/{engine}/v*/producers/{producer}.py. "
            f"Create engine_versions/{engine}/{safe}/producers/{producer}.py "
            f"(with sibling __init__.py files as needed) by copying the nearest "
            f"version's producers and rewriting LANDMARKS for the new library "
            f"API. See the version-bump chunk pattern in the engine archive "
            f"package docstring."
        )

    resolved_qualified = f"engine_versions.{engine}.{resolved_safe}.producers.{producer}"
    print(
        f"engine_versions._dispatcher: no exact match for {engine}=={version}; "
        f"{direction} {resolved_qualified}",
        file=sys.stderr,
    )
    return importlib.import_module(resolved_qualified)
