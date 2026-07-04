"""Tests for the study-scaffold renderer (``llem study init`` backend).

Two things are under test:

1. Rendering: annotations are derived only from the generated Pydantic models,
   bare and defaults modes differ solely by a comment prefix, field order is the
   model declaration order, and output is byte-deterministic.
2. Round-trip: every emitted file loads through the public study loader (the
   same path ``llem run`` uses) with the expected experiment count and no
   ``ConfigValidationWarning``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from llenergymeasure.api import load_study
from llenergymeasure.config.introspection import get_engine_config_model
from llenergymeasure.config.scaffold import all_engine_names, render_study_scaffold
from llenergymeasure.config.warnings import ConfigValidationWarning

MODEL = "Qwen/Qwen2.5-0.5B"

# A defaults-mode field line: "        <name>: <value>  # <annotation>".
_DEFAULTS_FIELD = re.compile(r"^        (\w+): (.*)$")
# A bare-mode field line: same, but commented out.
_BARE_FIELD = re.compile(r"^        # (\w+): (.*)$")


def _field_entries(text: str, *, defaults: bool) -> list[tuple[str, str]]:
    """Return (name, 'value  # annotation') for every rendered field line."""
    pattern = _DEFAULTS_FIELD if defaults else _BARE_FIELD
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


# ---------------------------------------------------------------------------
# Annotation formatting - derived from the generated models only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        # engine_params
        ("dtype", "auto  # one of [auto, half, float16, bfloat16, float, float32], default auto"),
        ("gpu_memory_utilization", "0.9  # float >0.0 <=1.0, default 0.9"),
        ("cpu_offload_gb", "0  # float >=0.0, default 0"),
        ("max_num_seqs", "null  # int >=1, default null"),
        ("enforce_eager", "false  # bool, default false"),
        ("speculative_config", "null  # object, default null"),
        ("distributed_executor_backend", "null  # any, default null"),
        # sampling_params
        ("temperature", "1.0  # float, default 1.0"),
        ("top_k", "0  # int, default 0"),
    ],
)
def test_annotation_formats(field_name: str, expected: str) -> None:
    """Each field annotation reflects its type, bounds, and default verbatim."""
    text = render_study_scaffold(MODEL, ["vllm"], defaults=True)
    entries = dict(_field_entries(text, defaults=True))
    assert entries[field_name] == expected


# ---------------------------------------------------------------------------
# Bare vs defaults - identical but for the comment prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", all_engine_names())
def test_bare_and_defaults_differ_only_by_comment(engine: str) -> None:
    """Bare fields are exactly the defaults fields, commented out (same values)."""
    bare = render_study_scaffold(MODEL, [engine], defaults=False)
    defaults = render_study_scaffold(MODEL, [engine], defaults=True)
    assert _field_entries(bare, defaults=False) == _field_entries(defaults, defaults=True)


def test_bare_has_no_uncommented_field_lines() -> None:
    """In bare mode the tuning fields are all commented (only structure is live)."""
    bare = render_study_scaffold(MODEL, ["vllm"], defaults=False)
    assert _field_entries(bare, defaults=True) == []


# ---------------------------------------------------------------------------
# Field order == model declaration order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", all_engine_names())
def test_field_order_matches_model_declaration(engine: str) -> None:
    """Rendered field order equals the generated model's declaration order."""
    config = get_engine_config_model(engine)
    expected: list[str] = []
    for sub_name in ("engine_params", "sampling_params"):
        field = config.model_fields[sub_name]
        sub_model = next(
            a
            for a in (field.annotation.__args__)  # type: ignore[union-attr]
            if isinstance(a, type)
        )
        expected.extend(sub_model.model_fields)

    text = render_study_scaffold(MODEL, [engine], defaults=True)
    rendered = [name for name, _ in _field_entries(text, defaults=True)]
    assert rendered == expected


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_render_is_byte_deterministic() -> None:
    """Identical inputs produce byte-identical output (no timestamps, stable order)."""
    a = render_study_scaffold(MODEL, all_engine_names(), defaults=False)
    b = render_study_scaffold(MODEL, all_engine_names(), defaults=False)
    assert a == b
    assert a.endswith("\n")


def test_study_name_slug() -> None:
    """Study name is the model tail slug, suffixed with the engine when single."""
    single = render_study_scaffold(MODEL, ["vllm"], defaults=False)
    multi = render_study_scaffold(MODEL, all_engine_names(), defaults=False)
    assert "study_name: qwen2.5-0.5b-vllm" in single
    assert "study_name: qwen2.5-0.5b\n" in multi


def test_all_engine_names() -> None:
    """The engine list is the SSOT engine set in declaration order."""
    assert all_engine_names() == ["transformers", "vllm", "tensorrt"]


# ---------------------------------------------------------------------------
# Round-trip through the public study loader
# ---------------------------------------------------------------------------


def _load_without_config_warnings(path: Path):
    """Load a study, turning any ConfigValidationWarning into a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigValidationWarning)
        return load_study(path)


@pytest.mark.parametrize("defaults", [False, True], ids=["bare", "defaults"])
@pytest.mark.parametrize("engine", all_engine_names())
def test_single_engine_round_trips(engine: str, defaults: bool, tmp_path: Path) -> None:
    """A single-engine file loads to exactly one experiment, no config warnings."""
    text = render_study_scaffold(MODEL, [engine], defaults=defaults)
    path = tmp_path / "study.yaml"
    path.write_text(text)
    study = _load_without_config_warnings(path)
    assert len(study.experiments) == 1
    assert study.experiments[0].engine == engine


@pytest.mark.parametrize("defaults", [False, True], ids=["bare", "defaults"])
def test_all_engines_round_trips(defaults: bool, tmp_path: Path) -> None:
    """An all-engines file loads to one experiment per engine, no config warnings."""
    engines = all_engine_names()
    text = render_study_scaffold(MODEL, engines, defaults=defaults)
    path = tmp_path / "study.yaml"
    path.write_text(text)
    study = _load_without_config_warnings(path)
    assert [e.engine for e in study.experiments] == engines
