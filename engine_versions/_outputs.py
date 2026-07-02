"""Locate the mined-schema outputs for each engine version snapshot.

The SSOT workspace stores one snapshot per engine version under
``engine_versions/<engine>/v<safe_version>/outputs/``. Each snapshot holds
two files:

- ``schema.discovered.json``: the full configuration surface discovered for
  that engine version (typed models, bounds, enums).
- ``curated.yaml``: the maintainer allowlist naming which discovered fields
  llem exposes as first-class typed fields.

This module is the one place that knows how those paths are laid out. It is a
plain path resolver: no imports of engine libraries, no dynamic module
loading, no magic attribute delegation. Callers pass an engine name and a
dotted version string and get back a path.

Version strings are dotted PEP 440 (``"0.7.3"``); directory names are the
identifier-safe form (``"v0_7_3"``) produced by :func:`safe_version`.
"""

from __future__ import annotations

from pathlib import Path

from packaging.version import Version

#: Engines that ship a mined-schema workspace.
ENGINES: tuple[str, ...] = ("vllm", "tensorrt", "transformers")

SCHEMA_FILENAME = "schema.discovered.json"
CURATED_FILENAME = "curated.yaml"

_WORKSPACE_ROOT = Path(__file__).resolve().parent


def safe_version(version: str) -> str:
    """Map a dotted PEP 440 version to its identifier-safe directory name.

    ``"0.7.3"`` -> ``"v0_7_3"``. Raises :class:`ValueError` when the result
    would not be a legal Python identifier fragment (any character other than
    a digit, letter, dot, or dash in the input).
    """
    safe = "v" + version.replace(".", "_").replace("-", "_")
    if not safe.replace("_", "").isalnum():
        raise ValueError(
            f"Cannot derive a directory name from version {version!r}: "
            f"candidate {safe!r} contains non-alphanumeric characters."
        )
    return safe


def outputs_dir(engine: str, version: str) -> Path:
    """Return the ``outputs/`` directory for one engine version snapshot."""
    return _WORKSPACE_ROOT / engine / safe_version(version) / "outputs"


def schema_path(engine: str, version: str) -> Path:
    """Return the discovered-schema JSON path for one engine version."""
    return outputs_dir(engine, version) / SCHEMA_FILENAME


def curated_path(engine: str, version: str) -> Path:
    """Return the curated-allowlist YAML path for one engine version."""
    return outputs_dir(engine, version) / CURATED_FILENAME


def workspace_versions(engine: str) -> list[str]:
    """Return the dotted versions of *engine* present in the workspace.

    A version counts as present when its ``outputs/`` directory holds both a
    ``schema.discovered.json`` and a ``curated.yaml`` (an incomplete snapshot
    is ignored). The result is sorted low-to-high by version number.
    """
    engine_root = _WORKSPACE_ROOT / engine
    if not engine_root.is_dir():
        return []

    found: list[tuple[Version, str]] = []
    for child in sorted(engine_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("v"):
            continue
        outputs = child / "outputs"
        if not (outputs / SCHEMA_FILENAME).is_file():
            continue
        if not (outputs / CURATED_FILENAME).is_file():
            continue
        dotted = child.name[1:].replace("_", ".")
        try:
            parsed = Version(dotted)
        except ValueError:
            continue
        found.append((parsed, dotted))

    found.sort(key=lambda pair: pair[0])
    return [dotted for _, dotted in found]
