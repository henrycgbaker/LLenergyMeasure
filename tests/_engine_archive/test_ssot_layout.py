"""Integrity tests for the engine_versions SSOT workspace and its path resolver.

Covers two things this layer ships:

- Every active version snapshot holds both mined files
  (``schema.discovered.json`` + ``curated.yaml``), each parses, and the
  schema's ``engine_version`` matches the directory it lives in.
- ``engine_versions._outputs`` resolves those files by (engine, version) and
  reports the present versions by scanning the tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine_versions import _outputs

# The version snapshots this layer establishes, per engine. Kept explicit here
# (rather than derived) so the test pins the expected workspace shape; the
# resolver derives the same set from the directory tree at runtime.
ACTIVE_PINS: dict[str, tuple[str, ...]] = {
    "vllm": ("0.7.3", "0.19.1"),
    "tensorrt": ("0.21.0", "1.0.0"),
    "transformers": ("5.7.0",),
}

_PINS = [(engine, version) for engine, versions in ACTIVE_PINS.items() for version in versions]


def test_engines_match_resolver() -> None:
    assert set(ACTIVE_PINS) == set(_outputs.ENGINES)


@pytest.mark.parametrize("version", ["0.7.3", "5.11.0", "1.0.0"])
def test_safe_version_maps_dots_to_underscores(version: str) -> None:
    assert _outputs.safe_version(version) == "v" + version.replace(".", "_")


def test_safe_version_rejects_illegal_input() -> None:
    with pytest.raises(ValueError):
        _outputs.safe_version("0.7.3/../etc")


@pytest.mark.parametrize(("engine", "version"), _PINS)
def test_snapshot_has_both_mined_files(engine: str, version: str) -> None:
    schema = _outputs.schema_path(engine, version)
    curated = _outputs.curated_path(engine, version)
    assert schema.is_file(), f"missing schema for {engine} {version}: {schema}"
    assert curated.is_file(), f"missing curated for {engine} {version}: {curated}"


@pytest.mark.parametrize(("engine", "version"), _PINS)
def test_schema_parses_and_self_identifies(engine: str, version: str) -> None:
    data = json.loads(_outputs.schema_path(engine, version).read_text())
    assert data.get("engine") == engine
    assert data.get("engine_version") == version


@pytest.mark.parametrize(("engine", "version"), _PINS)
def test_curated_parses_and_self_identifies(engine: str, version: str) -> None:
    data = yaml.safe_load(_outputs.curated_path(engine, version).read_text())
    assert isinstance(data, dict)
    assert data.get("engine") == engine
    assert data.get("engine_version") == version


@pytest.mark.parametrize(("engine", "versions"), ACTIVE_PINS.items())
def test_workspace_versions_matches_active_pins(engine: str, versions: tuple[str, ...]) -> None:
    assert _outputs.workspace_versions(engine) == sorted(versions, key=_version_key)


def test_outputs_dir_is_under_the_workspace() -> None:
    resolved = _outputs.outputs_dir("vllm", "0.7.3").resolve()
    workspace = (Path(_outputs.__file__).resolve().parent).resolve()
    assert workspace in resolved.parents


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))
