"""Tests for scripts/check_pydantic_matches_discovered.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from check_pydantic_matches_discovered import (
    CLASSIFICATION_CONTRADICTION,
    CLASSIFICATION_EXTENSION,
    CLASSIFICATION_SUBSET,
    _canonicalise_discovered_type,
    _canonicalise_pydantic_type,
    _is_subset,
    check_engine,
)

from llenergymeasure.config.ssot import ALL_ENGINES

_ENGINES = sorted(ALL_ENGINES)

# ---------------------------------------------------------------------------
# Canonicalisation unit tests
# ---------------------------------------------------------------------------


class TestCanonicaliseDiscoveredType:
    def test_simple_python_type(self) -> None:
        assert _canonicalise_discovered_type("bool") == "bool"

    def test_json_schema_integer(self) -> None:
        assert _canonicalise_discovered_type("integer") == "int"

    def test_json_schema_boolean(self) -> None:
        assert _canonicalise_discovered_type("boolean") == "bool"

    def test_json_schema_number(self) -> None:
        assert _canonicalise_discovered_type("number") == "float"

    def test_json_schema_string(self) -> None:
        assert _canonicalise_discovered_type("string") == "str"

    def test_strips_none_suffix(self) -> None:
        assert _canonicalise_discovered_type("int | None") == "int"

    def test_literal_sorted(self) -> None:
        result = _canonicalise_discovered_type("Literal['c', 'a', 'b']")
        assert result == "Literal['a', 'b', 'c']"

    def test_compound_sorted(self) -> None:
        result = _canonicalise_discovered_type("str | int")
        assert result == "int | str"

    def test_compound_with_json_names(self) -> None:
        result = _canonicalise_discovered_type("string | integer")
        assert result == "int | str"

    def test_unknown_passthrough(self) -> None:
        assert _canonicalise_discovered_type("unknown") == "unknown"


class TestCanonicalisePydanticType:
    def test_simple_integer(self) -> None:
        prop = {"type": "integer"}
        assert _canonicalise_pydantic_type(prop, {}) == "int"

    def test_simple_boolean(self) -> None:
        prop = {"type": "boolean"}
        assert _canonicalise_pydantic_type(prop, {}) == "bool"

    def test_optional_strips_null(self) -> None:
        prop = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        assert _canonicalise_pydantic_type(prop, {}) == "int"

    def test_enum_becomes_literal(self) -> None:
        prop = {"enum": ["b", "a", "c"], "type": "string"}
        assert _canonicalise_pydantic_type(prop, {}) == "Literal['a', 'b', 'c']"

    def test_ref_enum(self) -> None:
        prop = {"$ref": "#/$defs/MyEnum"}
        defs = {"MyEnum": {"enum": ["x", "y"], "title": "MyEnum"}}
        assert _canonicalise_pydantic_type(prop, defs) == "Literal['x', 'y']"

    def test_array_type(self) -> None:
        prop = {"type": "array", "items": {"type": "string"}}
        assert _canonicalise_pydantic_type(prop, {}) == "list[str]"


# ---------------------------------------------------------------------------
# Subset relation
# ---------------------------------------------------------------------------


class TestSubsetRelation:
    def test_equal_scalars(self) -> None:
        assert _is_subset("int", "int") is True
        assert _is_subset("str", "str") is True
        assert _is_subset("bool", "bool") is True

    def test_concrete_subset_of_any(self) -> None:
        assert _is_subset("any", "int") is True
        assert _is_subset("any", "Literal['a', 'b']") is True
        assert _is_subset("any", "ClassName") is True

    def test_literal_to_literal_narrowing_fails(self) -> None:
        # Engine enumerated the value set; llem must match it.
        # Narrowing is a CONTRADICTION (engine owns the value contract).
        assert _is_subset("Literal['a', 'b', 'c']", "Literal['a', 'b']") is False
        assert _is_subset("Literal['a', 'b']", "Literal['a']") is False

    def test_literal_to_literal_equal_passes(self) -> None:
        # Equal Literals pass via clause 1 (equal types), not via narrowing.
        assert _is_subset("Literal['a', 'b']", "Literal['a', 'b']") is True

    def test_literal_widening_fails(self) -> None:
        # Pydantic Literal accepts MORE than discovered: not a subset.
        assert _is_subset("Literal['a', 'b']", "Literal['a', 'b', 'c']") is False
        # Pydantic Literal disjoint from discovered: not a subset.
        assert _is_subset("Literal['a', 'b']", "Literal['c', 'd']") is False

    def test_str_to_literal_enrichment(self) -> None:
        # Engine signature uninformative (str/int); llem enriches with the
        # runtime-valid value set. Engine signature didn't enumerate, so
        # this is type enrichment, not narrowing.
        assert _is_subset("str", "Literal['mp', 'ray']") is True
        assert _is_subset("int", "Literal['8', '16']") is True

    def test_compound_containing_str_to_literal_enrichment(self) -> None:
        # Discovered exposes a union including str; pydantic enriches to specific values.
        assert _is_subset("str | type[Foo]", "Literal['mp', 'ray']") is True

    def test_class_to_primitive_abstraction(self) -> None:
        # Pydantic abstracts a complex upstream class to a primitive.
        assert _is_subset("CompilationConfig", "dict") is True
        assert _is_subset("CompilationConfig", "str") is True
        assert _is_subset("CompilationConfig", "list") is True

    def test_widening_to_str_fails(self) -> None:
        # Pydantic str accepts MORE than a discovered Literal: not a subset.
        assert _is_subset("Literal['a']", "str") is False

    def test_unrelated_types_fail(self) -> None:
        assert _is_subset("int", "str") is False
        assert _is_subset("bool", "int") is False
        assert _is_subset("list", "dict") is False


# ---------------------------------------------------------------------------
# 3-state classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", _ENGINES)
def test_live_schemas_have_no_contradictions(engine: str) -> None:
    """Real repo schemas should classify with zero CONTRADICTIONs.

    SUBSET-COMPATIBLE and LLEM-EXTENSION classifications are permitted
    (they represent type enrichment + whitelisted llem-native fields).
    Only CONTRADICTION fails the gate.
    """
    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    records = check_engine(engine, schema)
    contradictions = [r for r in records if r["classification"] == CLASSIFICATION_CONTRADICTION]
    assert contradictions == [], f"Contradictions detected for {engine}: {contradictions}"


@pytest.mark.parametrize("engine", _ENGINES)
def test_live_schemas_emit_full_classifications(engine: str) -> None:
    """Each (engine, field) record carries a recognised classification.

    Downstream consumers (the audit-bot in #655) walk the JSON output and
    expect every record to be one of the three known classifications.
    """
    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    records = check_engine(engine, schema)
    assert records, f"check_engine emitted no classifications for {engine}"
    valid = {CLASSIFICATION_SUBSET, CLASSIFICATION_EXTENSION, CLASSIFICATION_CONTRADICTION}
    for record in records:
        assert record["classification"] in valid, (
            f"Unknown classification {record['classification']!r} for {engine}.{record['field']}"
        )
