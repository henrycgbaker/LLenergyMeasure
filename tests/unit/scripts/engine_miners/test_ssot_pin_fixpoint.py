"""Structural fixpoint tests for the per-engine version-bundle SSOT.

These tests pin the contract that miner version pins live exclusively in
``engine_versions/{engine}.yaml`` and are read via
:func:`scripts.engine_miners._ssot.load_miner_pin`. If a future change
re-introduces a hard-coded ``TESTED_AGAINST_VERSIONS`` constant, drops a
required SSOT key, or silently swallows a missing-engine lookup, one of
these assertions will fail.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_miners import _ssot  # noqa: E402

ENGINES = ("transformers", "vllm", "tensorrt")
PRODUCERS = ("static", "dynamic", "discovery")

_MINERS_DIR = _PROJECT_ROOT / "scripts" / "engine_miners"
_ENGINE_VERSIONS_DIR = _PROJECT_ROOT / "engine_versions"


def _miner_python_files() -> list[Path]:
    return sorted(p for p in _MINERS_DIR.rglob("*.py") if p.is_file())


def _module_assign_target_names(tree: ast.AST) -> set[str]:
    """Collect every module-level assignment target name in ``tree``."""
    if not isinstance(tree, ast.Module):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_no_tested_against_versions_constant_survives() -> None:
    """No miner module may define ``TESTED_AGAINST_VERSIONS`` at module scope."""
    offenders: list[Path] = []
    for path in _miner_python_files():
        tree = ast.parse(path.read_text())
        if "TESTED_AGAINST_VERSIONS" in _module_assign_target_names(tree):
            offenders.append(path)
    assert not offenders, (
        "TESTED_AGAINST_VERSIONS was reintroduced as a module-level constant in: "
        f"{[str(p.relative_to(_PROJECT_ROOT)) for p in offenders]}. "
        f"Pins live in engine_versions/<engine>.yaml miner_pins.<producer>; "
        f"read via scripts.engine_miners._ssot.load_miner_pin."
    )


REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "engine")
REQUIRED_LIBRARY_KEYS = ("pep503_name", "current_version")
REQUIRED_IMAGE_KEYS = ("base_image_ref",)
REQUIRED_ARTEFACT_PATH_KEYS = ("engine_invariants", "validated_invariants", "discovered_schemas")


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_versions_yaml_parses_and_has_required_keys(engine: str) -> None:
    """Every engine SSOT parses as YAML and exposes the documented schema."""
    path = _ENGINE_VERSIONS_DIR / f"{engine}.yaml"
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict), f"{path} did not parse to a mapping."

    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in data, f"{path} missing top-level key {key!r}."

    library = data.get("library")
    assert isinstance(library, dict), f"{path} library: must be a mapping."
    for key in REQUIRED_LIBRARY_KEYS:
        assert key in library, f"{path} library.{key} missing."

    image = data.get("image")
    assert isinstance(image, dict), f"{path} image: must be a mapping."
    for key in REQUIRED_IMAGE_KEYS:
        assert key in image, f"{path} image.{key} missing."

    miner_pins = data.get("miner_pins")
    assert isinstance(miner_pins, dict), f"{path} miner_pins: must be a mapping."
    for producer in PRODUCERS:
        assert producer in miner_pins, f"{path} miner_pins.{producer} missing."

    artefact_paths = data.get("artefact_paths")
    assert isinstance(artefact_paths, dict), f"{path} artefact_paths: must be a mapping."
    for key in REQUIRED_ARTEFACT_PATH_KEYS:
        assert key in artefact_paths, f"{path} artefact_paths.{key} missing."


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("producer", PRODUCERS)
def test_miner_pins_parse_as_valid_specifier_sets(engine: str, producer: str) -> None:
    """Each ``miner_pins.<producer>`` value is a valid PEP 440 specifier set."""
    data = yaml.safe_load((_ENGINE_VERSIONS_DIR / f"{engine}.yaml").read_text())
    raw = data["miner_pins"][producer]
    SpecifierSet(str(raw))  # raises InvalidSpecifier on malformed input


def test_load_miner_pin_returns_specifier_set() -> None:
    """``load_miner_pin`` returns a ``SpecifierSet`` matching the SSOT value."""
    pin = _ssot.load_miner_pin("transformers", "static")
    assert isinstance(pin, SpecifierSet)
    assert pin == SpecifierSet(">=4.56,<4.57")


def test_load_miner_pin_raises_on_unknown_engine() -> None:
    """A missing engine SSOT surfaces as ``FileNotFoundError`` — no silent fallback."""
    with pytest.raises(FileNotFoundError):
        _ssot.load_miner_pin("does_not_exist", "static")


def test_load_miner_pin_raises_on_unknown_producer() -> None:
    """An unknown ``miner_pins`` key surfaces as ``KeyError`` — no silent default."""
    with pytest.raises(KeyError):
        _ssot.load_miner_pin("transformers", "not_a_producer")  # type: ignore[arg-type]
