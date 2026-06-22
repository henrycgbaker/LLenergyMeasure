"""Per-version landmark-data loader for engine producers.

A producer's walking algorithm is version-stable; only the DATA it walks with
(probe landmarks, class/method targets, StrEnum field map, source-tree layout)
changes across library versions. This loader externalises that data to a
per-version ``engine_versions/<engine>/v<safe>/landmarks.yaml`` and parses it
into a typed :class:`Landmarks` the producer consumes in place of module-level
tuples.

Resolution mirrors the producer dispatcher
(:mod:`engine_versions._dispatcher`): exact-match
``v<safe(version)>/landmarks.yaml`` is preferred; on miss the highest vendored
version ``<= target`` that carries a ``landmarks.yaml`` is used. The dispatcher
helpers (:func:`safe_version`, :func:`find_fallback_safe_version`) are reused
so the data loader and the code dispatcher cannot drift apart.

Validation is at the edge: a missing file or a malformed document raises
loud. Internal callers (the producer) trust the parsed object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from engine_versions._dispatcher import find_fallback_safe_version
from scripts.engine_producers._current import safe_version

_LANDMARKS_FILE = Path("landmarks.yaml")


@dataclass(frozen=True)
class Landmarks:
    """Parsed per-version landmark data for one engine producer.

    ``probe_landmarks`` is the drift-tool surface; ``class_targets`` and
    ``method_landmarks`` are ``(name, file_rel_path)`` / ``(class, method)``
    tuples the miner fails loud on; ``strenum_fields`` maps a StrEnum class to
    the field it constrains; ``source_root`` is the resolved source tree (the
    ``{version}`` token already substituted) and ``llm_args_rel`` /
    ``builder_rel`` are the relative paths the miner AST-walks.
    """

    probe_landmarks: tuple[str, ...]
    class_targets: tuple[tuple[str, Path], ...]
    method_landmarks: tuple[tuple[str, str], ...]
    strenum_fields: tuple[tuple[str, str], ...]
    source_root: Path
    llm_args_rel: Path
    builder_rel: Path


def _engine_root(engine: str) -> Path:
    """Absolute path to ``engine_versions/<engine>/`` from this file's location."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "engine_versions" / engine


def _resolve_landmarks_path(engine: str, version: str) -> Path:
    """Resolve the landmarks.yaml for ``(engine, version)`` with ``<=`` fallback.

    Exact ``v<safe(version)>/landmarks.yaml`` wins; otherwise the highest
    vendored version at or below ``version`` that carries a ``landmarks.yaml``.
    Raises :class:`FileNotFoundError` when neither exists.
    """
    engine_root = _engine_root(engine)
    exact = engine_root / safe_version(version) / _LANDMARKS_FILE
    if exact.is_file():
        return exact

    fallback_safe = find_fallback_safe_version(
        engine_root=engine_root,
        target_version=version,
        required_rel=_LANDMARKS_FILE,
    )
    if fallback_safe is None:
        raise FileNotFoundError(
            f"No landmarks.yaml for {engine}=={version}, and no vendored "
            f"landmarks.yaml exists at or below {version} under "
            f"{engine_root}/v*/landmarks.yaml."
        )
    return engine_root / fallback_safe / _LANDMARKS_FILE


def load_landmarks(engine: str, version: str) -> Landmarks:
    """Load + parse the landmark data for ``(engine, version)``.

    ``version`` is the requested dotted library version (e.g. ``"1.0.0"``); it
    selects which ``landmarks.yaml`` to read (with ``<=`` fallback) AND is
    substituted into the ``source.root`` template's ``{version}`` token so the
    returned ``source_root`` points at the matching extracted source tree.
    """
    path = _resolve_landmarks_path(engine, version)
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path} did not parse to a mapping.")

    source = doc["source"]
    files = source["files"]
    llm_args_rel = Path(files["llm_args"])
    builder_rel = Path(files["builder"])
    source_root = Path(str(source["root"]).format(version=version))

    return Landmarks(
        probe_landmarks=tuple(doc["probe_landmarks"]),
        class_targets=tuple(_class_target(entry, files) for entry in doc["class_targets"]),
        method_landmarks=tuple(
            (entry["class"], entry["method"]) for entry in doc["method_landmarks"]
        ),
        strenum_fields=tuple((entry["enum"], entry["field"]) for entry in doc["strenum_fields"]),
        source_root=source_root,
        llm_args_rel=llm_args_rel,
        builder_rel=builder_rel,
    )


def _class_target(entry: dict[str, Any], files: dict[str, str]) -> tuple[str, Path]:
    """Map a class_targets entry ``{class, file}`` to ``(class_name, rel_path)``."""
    return entry["class"], Path(files[entry["file"]])
