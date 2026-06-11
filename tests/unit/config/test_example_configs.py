"""Parse-and-path-resolution tests for shipped example/tutorial configs.

These guard against two failure modes that the loose ``extra="allow"`` engine
sub-configs make easy to introduce silently:

1. A config that no longer parses through ``load_study_config``.
2. A swept dotted path (sweep axis or group-variant key) that points at a field
   which does not exist on the resolved config models. Because the engine
   sections use ``extra="allow"``, a typo'd or mis-nested path
   (e.g. ``vllm.attention.backend`` instead of ``vllm.engine.attention.backend``)
   is accepted at parse time and then never consumed - the swept axis is a no-op.
   We resolve each path against the *declared* ``model_fields`` so such paths fail
   loudly here instead of producing wasted experiment cells.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from llenergymeasure.api import load_study
from llenergymeasure.config.models import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"

# Study configs that should parse and whose swept paths must all resolve.
STUDY_CONFIGS = [
    CONFIGS_DIR / "example-study-full.yaml",
    CONFIGS_DIR / "tutorials" / "tutorial-multi-engine.yaml",
]


def _field_submodels(annotation: object) -> list[type[BaseModel]]:
    """Return BaseModel subclasses referenced by a field annotation.

    Handles ``X | None`` and other unions by walking the annotation args.
    """
    candidates = list(typing.get_args(annotation)) or [annotation]
    return [c for c in candidates if isinstance(c, type) and issubclass(c, BaseModel)]


def _resolve_path(path: str) -> str | None:
    """Resolve a dotted sweep path against declared ExperimentConfig fields.

    Returns ``None`` if every segment is a declared field, otherwise a string
    describing the segment that failed (for an actionable assertion message).
    """
    model: type[BaseModel] = ExperimentConfig
    parts = path.split(".")
    for i, part in enumerate(parts):
        fields = model.model_fields
        if part not in fields:
            declared = ", ".join(sorted(fields))
            return f"'{part}' is not a declared field on {model.__name__} (declared: {declared})"
        if i < len(parts) - 1:
            submodels = _field_submodels(fields[part].annotation)
            if not submodels:
                return f"'{part}' on {model.__name__} is a leaf field but path continues"
            model = submodels[0]
    return None


def _swept_paths(raw_study: dict) -> set[str]:
    """Collect every dotted path referenced by the study's sweep block.

    Independent axes contribute their key; dependent groups (lists of dicts)
    contribute the keys of every variant dict.
    """
    paths: set[str] = set()
    for key, value in (raw_study.get("sweep") or {}).items():
        is_group = isinstance(value, list) and value and all(isinstance(e, dict) for e in value)
        if is_group:
            for variant in value:
                paths.update(variant.keys())
        else:
            paths.add(key)
    # Only dotted paths can be mis-routed into an extra="allow" section.
    return {p for p in paths if "." in p}


@pytest.mark.parametrize("config_path", STUDY_CONFIGS, ids=lambda p: p.name)
def test_study_config_parses(config_path: Path) -> None:
    """The config loads through the study loader and yields >=1 experiment."""
    study = load_study(config_path)
    assert len(study.experiments) >= 1


@pytest.mark.parametrize("config_path", STUDY_CONFIGS, ids=lambda p: p.name)
def test_swept_paths_resolve_to_real_fields(config_path: Path) -> None:
    """Every swept dotted path targets a declared field (no extra='allow' no-ops)."""
    raw = yaml.safe_load(config_path.read_text())
    failures: list[str] = []
    for path in sorted(_swept_paths(raw)):
        problem = _resolve_path(path)
        if problem is not None:
            failures.append(f"{path}: {problem}")
    assert not failures, "Swept paths that do not resolve to real fields:\n" + "\n".join(failures)


def test_example_full_resolver_self_check() -> None:
    """Sanity-check the resolver itself: a known-good and known-bad path."""
    assert _resolve_path("transformers.batch_size") is None
    assert _resolve_path("vllm.engine.attention.backend") is None
    # Mis-nested attention path (the bug this module guards against).
    assert _resolve_path("vllm.attention.backend") is not None
