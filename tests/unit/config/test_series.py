"""Tests for value-series derivation (``llem study init --bounds`` backend).

Covers the deterministic floor (bounded numerics, enums, bools, unbounded
fields), the named series policies (span / log / pow2), curated override
loading and precedence, and known-rejected exclusion against both synthetic
rules and the shipped corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from llenergymeasure.config.engine_rules.loader import EngineRulesLoader, Provenance, Rule
from llenergymeasure.config.series import (
    OVERRIDES_FILENAME,
    FieldOverride,
    StudyOverridesError,
    derive_series,
    drop_known_rejected,
    load_study_overrides,
)


class _Probe(BaseModel):
    open_float: Annotated[float | None, Field(gt=0.0, le=1.0)] = 0.9
    closed_float: Annotated[float | None, Field(ge=0.0, le=1.0)] = 0.5
    closed_int: Annotated[int | None, Field(ge=1, le=8)] = 1
    narrow_int: Annotated[int | None, Field(ge=1, le=2)] = 1
    log_float: Annotated[float | None, Field(ge=1.0, le=10000.0)] = 1.0
    pow_int: Annotated[int | None, Field(ge=1, le=32)] = 1
    strict_pow_int: Annotated[int | None, Field(gt=1, lt=32)] = 2
    lower_only: Annotated[int | None, Field(ge=1)] = None
    unbounded: float | None = None
    flag: bool | None = None
    choice: Literal["a", "b", "c"] | None = "a"
    text: str | None = None


def _series(
    name: str, override: FieldOverride | None = None, rules: tuple[Rule, ...] = ()
) -> list[Any] | None:
    return derive_series(
        _Probe.model_fields[name],
        field_path=f"eng.engine_params.{name}",
        override=override,
        rules=rules,
    )


def _rule(match_fields: dict[str, Any], severity: str = "error") -> Rule:
    return Rule(
        id="test_rule",
        engine="eng",
        severity=severity,
        match_fields=match_fields,
        provenance=Provenance(source="manual", verified="human", engine_version="0"),
    )


# ---------------------------------------------------------------------------
# Floor series: bounded numerics, enums, bools; nothing invented otherwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("open_float", [0.25, 0.5, 0.75, 1.0]),  # strict lower endpoint dropped
        ("closed_float", [0.0, 0.25, 0.5, 0.75, 1.0]),  # closed range keeps endpoints
        ("closed_int", [1, 3, 4, 6, 8]),  # int quantiles round to ints
        ("narrow_int", [1, 2]),  # rounding collisions dedupe
        ("flag", [False, True]),
        ("choice", ["a", "b", "c"]),
        ("lower_only", None),  # one-sided bound: no series
        ("unbounded", None),
        ("text", None),
    ],
)
def test_floor_series(field_name: str, expected: list[Any] | None) -> None:
    """The deterministic floor yields the documented series (or none)."""
    assert _series(field_name) == expected


def test_floor_default_policy_is_span() -> None:
    """Without a hint, a bounded numeric gets linear quantile points."""
    assert _series("log_float") == [1.0, 2500.75, 5000.5, 7500.25, 10000.0]


def test_series_is_deterministic() -> None:
    """The same field always yields the same series."""
    assert _series("open_float") == _series("open_float")


# ---------------------------------------------------------------------------
# Named policies via curated sweep_hint
# ---------------------------------------------------------------------------


def test_log_hint_spaces_points_geometrically() -> None:
    assert _series("log_float", FieldOverride(sweep_hint="log")) == [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    ]


def test_pow2_hint_emits_all_powers_in_range() -> None:
    assert _series("pow_int", FieldOverride(sweep_hint="pow2")) == [1, 2, 4, 8, 16, 32]


def test_pow2_strict_bounds_exclude_matching_powers() -> None:
    assert _series("strict_pow_int", FieldOverride(sweep_hint="pow2")) == [2, 4, 8, 16]


def test_log_needs_positive_lower_bound() -> None:
    """open_float's lower bound is 0.0 (strict), which log cannot span."""
    with pytest.raises(StudyOverridesError, match="positive lower bound"):
        _series("open_float", FieldOverride(sweep_hint="log"))


@pytest.mark.parametrize("field_name", ["flag", "choice", "unbounded", "lower_only", "text"])
def test_hint_without_derivable_range_fails_loud(field_name: str) -> None:
    """A sweep_hint on anything but a two-sided numeric is a curation error."""
    with pytest.raises(StudyOverridesError, match="sweep_hint"):
        _series(field_name, FieldOverride(sweep_hint="span"))


def test_sweep_values_win_over_hint_and_floor() -> None:
    """Explicit sweep_values beat both the policy hint and the floor."""
    override = FieldOverride(sweep_hint="log", sweep_values=[2.0, 3.0])
    assert _series("log_float", override) == [2.0, 3.0]


# ---------------------------------------------------------------------------
# Known-rejected exclusion
# ---------------------------------------------------------------------------


def test_error_rule_drops_matching_candidates() -> None:
    rules = (_rule({"eng.engine_params.x": {">": 10}}),)
    assert drop_known_rejected([5, 20], "eng.engine_params.x", rules) == [5]


def test_dormant_rule_does_not_drop() -> None:
    rules = (_rule({"eng.engine_params.x": {">": 10}}, severity="dormant"),)
    assert drop_known_rejected([5, 20], "eng.engine_params.x", rules) == [5, 20]


def test_multi_field_rule_is_ignored() -> None:
    """Cross-field rules are C2's job; a bare candidate cannot satisfy them."""
    rules = (_rule({"eng.engine_params.x": {">": 10}, "eng.engine_params.y": {"present": True}}),)
    assert drop_known_rejected([5, 20], "eng.engine_params.x", rules) == [5, 20]


def test_other_field_rule_is_ignored() -> None:
    rules = (_rule({"eng.engine_params.other": {">": 10}}),)
    assert drop_known_rejected([5, 20], "eng.engine_params.x", rules) == [5, 20]


def test_field_ref_spec_is_skipped() -> None:
    """A single-key rule whose spec references another field is cross-field."""
    rules = (_rule({"eng.engine_params.x": {"<": "@sibling"}}),)
    assert drop_known_rejected([5, 20], "eng.engine_params.x", rules) == [5, 20]


def test_series_collapsing_below_two_values_is_pinned() -> None:
    """When exclusion leaves fewer than two candidates, no series is emitted."""
    rules = (_rule({"eng.engine_params.flag": True}),)
    assert _series("flag", rules=rules) is None


def test_real_corpus_keeps_early_stopping_true() -> None:
    """early_stopping=true is valid (enables beam-search early stopping).

    Two independent facts now keep both boolean values. First, the mis-mined
    ``{'>': 0}`` bound that rejected ``True`` (``True > 0`` is true in Python)
    was deleted from the shipped transformers corpus. Second, even a re-mined
    ordering bound could not reject booleans: ordering predicates do not fire on
    bool operands (see ``_ordered``), so the value pruner never drops ``True``.
    """
    rules = EngineRulesLoader().load_rules("transformers").rules
    kept = drop_known_rejected([False, True], "transformers.engine_params.early_stopping", rules)
    assert kept == [False, True]


def test_real_corpus_rejects_vllm_top_k_below_neg1() -> None:
    """The shipped vllm corpus rejects top_k < -1 and nothing else here."""
    rules = EngineRulesLoader().load_rules("vllm").rules
    assert drop_known_rejected([-2, -1, 0, 50], "vllm.sampling_params.top_k", rules) == [
        -1,
        0,
        50,
    ]


# ---------------------------------------------------------------------------
# Overrides file loading + validation
# ---------------------------------------------------------------------------


def _write_overrides(tmp_path: Path, text: str) -> Path:
    engine_dir = tmp_path / "eng"
    engine_dir.mkdir()
    (engine_dir / OVERRIDES_FILENAME).write_text(text)
    return tmp_path


def test_absent_overrides_file_is_valid(tmp_path: Path) -> None:
    assert load_study_overrides("eng", tmp_path) == {}


def test_empty_overrides_file_is_valid(tmp_path: Path) -> None:
    assert load_study_overrides("eng", _write_overrides(tmp_path, "")) == {}


def test_valid_overrides_parse(tmp_path: Path) -> None:
    root = _write_overrides(
        tmp_path,
        "a.b:\n  study_default: null\n  sweep_hint: pow2\nc.d:\n  sweep_values: [1, 2]\n",
    )
    overrides = load_study_overrides("eng", root)
    assert overrides["a.b"].has_study_default
    assert overrides["a.b"].study_default is None
    assert overrides["a.b"].sweep_hint == "pow2"
    assert overrides["c.d"].sweep_values == [1, 2]
    assert not overrides["c.d"].has_study_default


@pytest.mark.parametrize(
    "text",
    [
        "a.b:\n  bogus_key: 1\n",  # unknown entry key
        "a.b: {}\n",  # empty entry
        "a.b: 3\n",  # entry not a mapping
        "a.b:\n  sweep_hint: cubic\n",  # unknown series shape
        "a.b:\n  sweep_values: 5\n",  # sweep_values not a list
        "a.b:\n  sweep_values: []\n",  # sweep_values empty
        "- a\n",  # top level not a mapping
    ],
)
def test_malformed_overrides_fail_loud(tmp_path: Path, text: str) -> None:
    with pytest.raises(StudyOverridesError):
        load_study_overrides("eng", _write_overrides(tmp_path, text))


def test_shipped_vllm_overrides_load() -> None:
    """The seeded package-data file parses and carries the curated gpu series."""
    overrides = load_study_overrides("vllm")
    gpu = overrides["engine_params.gpu_memory_utilization"]
    assert gpu.sweep_values == [0.5, 0.7, 0.8, 0.9]
    assert not gpu.has_study_default
