"""Tests for cross-field ``@field_path`` references and divisibility operators.

Covers the loader-grammar extensions that close gaps surfaced by PR #387:

- ``@field_path`` substitution on the right-hand side of any operator,
  with sibling and dotted-from-root resolution semantics.
- ``divisible_by`` / ``not_divisible_by`` operators with strict
  non-bool integer operands and zero-divisor guards.
- Spec walking through nested lists / dicts so refs anywhere in the
  predicate tree get resolved before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenergymeasure.config.engine_rules import (
    Provenance,
    Rule,
    evaluate_predicate,
)
from llenergymeasure.config.engine_rules.loader import (
    _is_int_pair,
    _ordered,
    _resolve_field_refs_in_spec,
)

# ---------------------------------------------------------------------------
# Config stubs (mirror test_rule_matching.py shape)
# ---------------------------------------------------------------------------


@dataclass
class _Sampling:
    num_beams: int | None = None
    num_beam_groups: int | None = None
    num_return_sequences: int | None = None
    diversity_penalty: float | None = None


@dataclass
class _Transformers:
    sampling: _Sampling


@dataclass
class _Config:
    transformers: _Transformers


def _make_rule(*, match_fields: dict[str, Any]) -> Rule:
    return Rule(
        id="rule_x",
        engine="transformers",
        severity="error",
        match_fields=match_fields,
        provenance=Provenance(
            source="manual",
            verified="human",
            engine_version="4.57.3",
            citation=None,
            date="2026-04-25",
        ),
        message_template="msg {declared_value}",
    )


# ---------------------------------------------------------------------------
# @field_ref resolution - sibling
# ---------------------------------------------------------------------------


def test_field_ref_sibling_substitutes_value() -> None:
    config = {"a": {"b": {"x": 5, "y": 3}}}
    spec = {">": "@y"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {">": 3}


def test_field_ref_sibling_resolves_to_none_when_missing() -> None:
    config = {"a": {"b": {"x": 5}}}
    spec = {">": "@y"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {">": None}


@pytest.mark.parametrize(
    "x_value, y_value, expected_fires",
    [
        (5, 3, True),  # x > y → fires
        (3, 3, False),  # x == y → does not fire
        (2, 3, False),  # x < y → does not fire
        (5, None, False),  # y missing → comparison is None-safe, does not fire
    ],
)
def test_field_ref_sibling_via_evaluate_predicate(
    x_value: int, y_value: int | None, expected_fires: bool
) -> None:
    config = {"a": {"b": {"x": x_value, "y": y_value}}}
    spec = {">": "@y"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert evaluate_predicate(x_value, resolved) is expected_fires


# ---------------------------------------------------------------------------
# @field_ref resolution - dotted from root
# ---------------------------------------------------------------------------


def test_field_ref_dotted_resolves_from_root() -> None:
    config = {"deep": {"nested": {"path": 42}}, "a": {"b": {"x": 1}}}
    spec = {">": "@deep.nested.path"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {">": 42}


def test_field_ref_dotted_resolves_through_attribute_chains() -> None:
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(num_beams=4, num_return_sequences=6))
    )
    spec = {">": "@transformers.sampling.num_beams"}
    resolved = _resolve_field_refs_in_spec(
        spec, config, "transformers.sampling.num_return_sequences"
    )
    assert resolved == {">": 4}


# ---------------------------------------------------------------------------
# Spec walk - recursion through lists and nested dicts
# ---------------------------------------------------------------------------


def test_spec_walk_resolves_refs_inside_list() -> None:
    config = {"a": {"b": {"x": 1, "y": 7, "z": 9}}}
    spec = {"in": ["@y", "@z"]}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {"in": [7, 9]}


def test_spec_walk_leaves_non_ref_strings_alone() -> None:
    config = {"a": {"x": 1, "name": "foo"}}
    spec = {"==": "literal_value"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.x")
    assert resolved == {"==": "literal_value"}


def test_spec_walk_passes_through_bare_value() -> None:
    # Bare-value spec (equality) is returned unchanged when not a ref.
    assert _resolve_field_refs_in_spec(0.5, {"a": 1}, "a") == 0.5


# ---------------------------------------------------------------------------
# divisible_by / not_divisible_by operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, actual, expected",
    [
        # not_divisible_by: positive case (rule fires)
        ({"not_divisible_by": 4}, 6, True),  # 6 % 4 == 2 → fires
        ({"not_divisible_by": 3}, 6, False),  # 6 % 3 == 0 → does not fire
        # divisible_by: positive case (rule fires)
        ({"divisible_by": 3}, 6, True),  # 6 % 3 == 0 → fires
        ({"divisible_by": 4}, 6, False),  # 6 % 4 == 2 → does not fire
    ],
)
def test_divisibility_basic(spec: dict[str, Any], actual: int, expected: bool) -> None:
    assert evaluate_predicate(actual, spec) is expected


def test_divisibility_zero_divisor_does_not_fire() -> None:
    # b == 0: never fires (avoids ZeroDivisionError, no rule should match).
    assert evaluate_predicate(6, {"not_divisible_by": 0}) is False
    assert evaluate_predicate(6, {"divisible_by": 0}) is False


@pytest.mark.parametrize(
    "actual, divisor",
    [
        (None, 3),  # missing field - predicate must not fire
        (6.0, 3),  # float operand - strict int-only
        ("6", 3),  # str operand - strict int-only
        (6, 3.0),  # float divisor - strict int-only
    ],
)
def test_divisibility_rejects_non_int_operands(actual: Any, divisor: Any) -> None:
    assert evaluate_predicate(actual, {"not_divisible_by": divisor}) is False
    assert evaluate_predicate(actual, {"divisible_by": divisor}) is False


@pytest.mark.parametrize(
    "actual, divisor",
    [
        (True, 1),  # bool actual: would otherwise pass via bool < int
        (False, 1),
        (6, True),  # bool divisor
        (True, True),
    ],
)
def test_divisibility_rejects_bool_operands(actual: Any, divisor: Any) -> None:
    assert evaluate_predicate(actual, {"not_divisible_by": divisor}) is False
    assert evaluate_predicate(actual, {"divisible_by": divisor}) is False


def test_is_int_pair_helper() -> None:
    # Direct unit on the helper for a tight regression guard.
    assert _is_int_pair(6, 3) is True
    assert _is_int_pair(0, 1) is True
    assert _is_int_pair(True, 1) is False
    assert _is_int_pair(1, False) is False
    assert _is_int_pair(6.0, 3) is False
    assert _is_int_pair(None, 3) is False


# ---------------------------------------------------------------------------
# Ordering operators - type-incomparable operands are no-match, not a crash
#
# A mined numeric bound can land on a field that naturally holds a non-numeric
# value (e.g. transformers ``compile_config``, a dict / CompileConfig shape).
# The raw ``<`` / ``<=`` / ``>`` / ``>=`` comparison raises TypeError on such a
# pair; the handler must treat it as no-match so config construction never
# crashes on a corpus artifact. The generated pydantic model stays the
# authority on the field's type validity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("actual", [{"backend": "inductor"}, "inductor", ["a"], (1,)])
def test_ordering_incomparable_actual_is_no_match(op: str, actual: Any) -> None:
    # dict / str / list / tuple vs an int bound: no TypeError, rule does not fire.
    assert evaluate_predicate(actual, {op: 0}) is False


@pytest.mark.parametrize(
    "op, actual, bound, expected",
    [
        (">", 5, 0, True),
        (">", 0, 5, False),
        ("<", 0, 5, True),
        ("<", 5, 0, False),
        (">=", 5, 5, True),
        ("<=", 5, 5, True),
    ],
)
def test_ordering_numeric_still_matches(op: str, actual: Any, bound: Any, expected: bool) -> None:
    # Guarding incomparable types must not weaken real numeric matching.
    assert evaluate_predicate(actual, {op: bound}) is expected


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_ordering_none_operands_are_no_match(op: str) -> None:
    # None on either side (unset field / unresolved @field_ref) never fires.
    assert evaluate_predicate(None, {op: 0}) is False
    assert evaluate_predicate(5, {op: None}) is False


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("actual", [True, False])
def test_ordering_rejects_bool_actual(op: str, actual: bool) -> None:
    # bool subclasses int, so ``True > 0`` compares cleanly; the ordering ops
    # must treat a boolean-valued field as non-comparable and never fire (the
    # generated pydantic model stays the authority on bool-field validity).
    # This is what keeps a mined ``{'>': 0}`` bound off ``early_stopping=True``.
    assert evaluate_predicate(actual, {op: 0}) is False


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_ordering_rejects_bool_bound(op: str) -> None:
    # A bool on the bound side (e.g. an @field_ref resolving to a boolean) is
    # equally non-comparable.
    assert evaluate_predicate(5, {op: True}) is False


def test_ordering_comparable_non_numeric_still_compares() -> None:
    # Same-type ordering (str vs str) stays live - only cross-type pairs no-match.
    assert evaluate_predicate("b", {">": "a"}) is True
    assert evaluate_predicate("a", {">": "b"}) is False


def test_ordered_helper() -> None:
    # Direct unit on the helper for a tight regression guard.
    import operator

    assert _ordered(5, 0, operator.gt) is True
    assert _ordered(0, 5, operator.gt) is False
    assert _ordered(None, 0, operator.gt) is False
    assert _ordered(5, None, operator.gt) is False
    # bool operands are non-comparable (mirrors the divisibility exclusion).
    assert _ordered(True, 0, operator.gt) is False
    assert _ordered(5, True, operator.gt) is False
    # Type-incomparable pair: TypeError swallowed, treated as no-match.
    assert _ordered({"backend": "inductor"}, 0, operator.gt) is False
    assert _ordered("inductor", 0, operator.lt) is False


# ---------------------------------------------------------------------------
# End-to-end via Rule.try_match - corpus-shape predicate
# ---------------------------------------------------------------------------


def test_try_match_with_field_ref_fires_when_left_exceeds_right() -> None:
    # Mirrors the rewritten transformers_num_return_sequences_exceeds_num_beams
    # rule: fires when num_return_sequences > num_beams.
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_return_sequences": {">": "@num_beams"},
        }
    )
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(num_beams=2, num_return_sequences=4))
    )
    match = rule.try_match(config)
    assert match is not None
    assert match.declared_value == 4


def test_try_match_with_field_ref_does_not_fire_when_left_le_right() -> None:
    # The valid case (num_return_sequences=2, num_beams=4) - rule must not fire.
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_return_sequences": {">": "@num_beams"},
        }
    )
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(num_beams=4, num_return_sequences=2))
    )
    assert rule.try_match(config) is None


def test_try_match_with_not_divisible_by_field_ref_fires() -> None:
    # Mirrors the new transformers_num_beams_not_divisible_by_groups rule.
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_beam_groups": {">": 1},
            "transformers.sampling.num_beams": {"not_divisible_by": "@num_beam_groups"},
        }
    )
    config = _Config(transformers=_Transformers(sampling=_Sampling(num_beams=6, num_beam_groups=4)))
    match = rule.try_match(config)
    assert match is not None
    # Last predicate's field is the subject (num_beams).
    assert match.declared_value == 6


def test_try_match_with_not_divisible_by_field_ref_does_not_fire_on_valid() -> None:
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_beam_groups": {">": 1},
            "transformers.sampling.num_beams": {"not_divisible_by": "@num_beam_groups"},
        }
    )
    # 6 % 3 == 0 → divisible → rule does not fire.
    config = _Config(transformers=_Transformers(sampling=_Sampling(num_beams=6, num_beam_groups=3)))
    assert rule.try_match(config) is None
