"""Tests for the study-scaffold renderer (``llem study init`` backend).

Three things are under test:

1. Rendering: annotations are derived only from the generated Pydantic models,
   bare and defaults modes differ solely by a comment prefix, field order is the
   model declaration order, and output is byte-deterministic.
2. Round-trip: every emitted file loads through the public study loader (the
   same path ``llem run`` uses) with the expected experiment count and no
   ``ConfigValidationWarning``.
3. Bounds mode: sweepable fields become explicit value lists under ``sweep:``,
   known-rejected values are dropped, curated overrides apply, and the file
   expands to exactly the product of the emitted series lengths.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from llenergymeasure.api import load_study
from llenergymeasure.config.introspection import get_engine_config_model
from llenergymeasure.config.scaffold import (
    all_engine_names,
    render_study_bounds,
    render_study_scaffold,
)
from llenergymeasure.config.series import OVERRIDES_FILENAME, StudyOverridesError
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


# ---------------------------------------------------------------------------
# Bounds mode - value lists under sweep:, pinned fields, round-trip counts
# ---------------------------------------------------------------------------


def _sweep_block(text: str) -> dict[str, list[Any]]:
    """Parse the rendered YAML and return its sweep: mapping."""
    sweep = yaml.safe_load(text)["sweep"]
    assert isinstance(sweep, dict)
    return sweep


def test_bounds_emits_series_for_derivable_fields() -> None:
    """Literals get all members, bools both values, curated lists win."""
    sweep = _sweep_block(render_study_bounds(MODEL, ["vllm"]))
    assert sweep["vllm.engine_params.dtype"] == [
        "auto",
        "half",
        "float16",
        "bfloat16",
        "float",
        "float32",
    ]
    assert sweep["vllm.engine_params.enforce_eager"] == [False, True]
    # Curated override (shipped study_overrides.yaml) beats the span floor.
    assert sweep["vllm.engine_params.gpu_memory_utilization"] == [0.5, 0.7, 0.8, 0.9]


def test_bounds_pins_fields_without_derivable_range() -> None:
    """One-sided/unbounded fields stay pinned at their defaults, not swept."""
    text = render_study_bounds(MODEL, ["vllm"])
    raw = yaml.safe_load(text)
    assert "experiments" not in raw
    sweep = raw["sweep"]
    assert "vllm.engine_params.max_num_seqs" not in sweep  # one-sided bound
    assert "vllm.sampling_params.temperature" not in sweep  # unbounded float
    assert raw["vllm"]["engine_params"]["max_num_seqs"] is None
    assert raw["vllm"]["sampling_params"]["temperature"] == 1.0


def test_bounds_emits_both_bool_values_for_early_stopping() -> None:
    """early_stopping is a valid boolean knob, so it sweeps both values.

    early_stopping=True is the documented way to enable beam-search early
    stopping; it must not be pruned from the sweep. (At the default num_beams=1
    the two values are measurement-equivalent and collapse at dedup time - see
    the round-trip tests - but the sweep axis itself is emitted.)
    """
    text = render_study_bounds(MODEL, ["transformers"])
    raw = yaml.safe_load(text)
    sweep = raw["sweep"]
    assert sweep["transformers.engine_params.early_stopping"] == [False, True]
    assert sweep["transformers.engine_params.use_cache"] == [False, True]
    assert sweep["transformers.sampling_params.do_sample"] == [False, True]


def test_bounds_render_is_byte_deterministic() -> None:
    a = render_study_bounds(MODEL, all_engine_names())
    b = render_study_bounds(MODEL, all_engine_names())
    assert a == b
    assert a.endswith("\n")


@pytest.mark.parametrize("engine", all_engine_names())
def test_bounds_round_trips_to_series_product(engine: str, tmp_path: Path) -> None:
    """A bounds file expands to the product of its series lengths.

    The full Cartesian product is enumerated pre-dedup (one equivalence group
    per combination member); measurement-equivalent combinations then collapse
    via the resolved-config-hash dedup, so the number of executed experiments
    equals the group count (<= product). For transformers this is a strict
    collapse: early_stopping is dormant at the default num_beams=1, so its two
    values are measurement-equivalent.
    """
    text = render_study_bounds(MODEL, [engine])
    path = tmp_path / "study.yaml"
    path.write_text(text)
    sweep = _sweep_block(text)
    product = math.prod(len(v) for v in sweep.values())
    study = _load_without_config_warnings(path)
    assert product > 1
    # Every sweep combination is accounted for in exactly one equivalence group.
    assert sum(g["member_count"] for g in study.pre_run_equivalence_groups) == product
    # Executed experiments are the post-dedup distinct resolved configs.
    assert len(study.experiments) == len(study.pre_run_equivalence_groups)
    assert 1 < len(study.experiments) <= product
    assert all(e.engine == engine for e in study.experiments)


def test_bounds_all_engines_round_trips_to_sum_of_products(tmp_path: Path) -> None:
    """A multi-engine bounds file expands to the sum of per-engine products.

    As in the single-engine case, the sum-of-products is the pre-dedup member
    total; executed experiments are the post-dedup group count.
    """
    text = render_study_bounds(MODEL, all_engine_names())
    path = tmp_path / "study.yaml"
    path.write_text(text)
    per_engine: dict[str, list[int]] = {}
    for key, values in _sweep_block(text).items():
        per_engine.setdefault(key.split(".")[0], []).append(len(values))
    assert set(per_engine) == set(all_engine_names())
    expected_members = sum(math.prod(lengths) for lengths in per_engine.values())
    study = _load_without_config_warnings(path)
    assert sum(g["member_count"] for g in study.pre_run_equivalence_groups) == expected_members
    assert len(study.experiments) == len(study.pre_run_equivalence_groups)


# ---------------------------------------------------------------------------
# Curated overrides in the renderers
# ---------------------------------------------------------------------------


def _write_vllm_overrides(tmp_path: Path, text: str) -> Path:
    engine_dir = tmp_path / "vllm"
    engine_dir.mkdir()
    (engine_dir / OVERRIDES_FILENAME).write_text(text)
    return tmp_path


def test_study_default_applies_to_every_mode(tmp_path: Path) -> None:
    """A curated study_default pins the field in bare, defaults, and bounds."""
    root = _write_vllm_overrides(
        tmp_path, "engine_params.tensor_parallel_size:\n  study_default: 2\n"
    )
    expected = "tensor_parallel_size: 2  # int, default 1"
    bare = render_study_scaffold(MODEL, ["vllm"], defaults=False, overrides_root=root)
    live = render_study_scaffold(MODEL, ["vllm"], defaults=True, overrides_root=root)
    bounds = render_study_bounds(MODEL, ["vllm"], overrides_root=root)
    assert f"        # {expected}" in bare
    assert f"        {expected}" in live
    assert f"    {expected}" in bounds
    # The bare/defaults invariant still holds with overrides applied.
    assert _field_entries(bare, defaults=False) == _field_entries(live, defaults=True)


def test_sweep_values_override_wins_in_bounds(tmp_path: Path) -> None:
    root = _write_vllm_overrides(tmp_path, "engine_params.dtype:\n  sweep_values: [auto, half]\n")
    sweep = _sweep_block(render_study_bounds(MODEL, ["vllm"], overrides_root=root))
    assert sweep["vllm.engine_params.dtype"] == ["auto", "half"]


def test_unknown_override_path_fails_loud(tmp_path: Path) -> None:
    root = _write_vllm_overrides(tmp_path, "engine_params.bogus:\n  study_default: 1\n")
    with pytest.raises(StudyOverridesError, match="unknown field path"):
        render_study_bounds(MODEL, ["vllm"], overrides_root=root)
    with pytest.raises(StudyOverridesError, match="unknown field path"):
        render_study_scaffold(MODEL, ["vllm"], defaults=True, overrides_root=root)
