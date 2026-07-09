"""Tests for scripts/check_pydantic_matches_discovered.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from check_pydantic_matches_discovered import (
    _canonicalise_discovered_type,
    _canonicalise_pydantic_type,
    _collect_props,
    _is_intentional_narrowing,
    check_engine,
)

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
# Narrowing detection
# ---------------------------------------------------------------------------


class TestIntentionalNarrowing:
    def test_str_to_literal(self) -> None:
        assert _is_intentional_narrowing("str", "Literal['a', 'b']") is True

    def test_int_to_literal(self) -> None:
        assert _is_intentional_narrowing("int", "Literal['8', '16']") is True

    def test_compound_str_to_literal(self) -> None:
        assert _is_intentional_narrowing("str | type[Foo]", "Literal['mp', 'ray']") is True

    def test_class_to_dict(self) -> None:
        assert _is_intentional_narrowing("CompilationConfig", "dict") is True

    def test_same_type_not_narrowing(self) -> None:
        assert _is_intentional_narrowing("int", "int") is False

    def test_widening_not_allowed(self) -> None:
        assert _is_intentional_narrowing("Literal['a']", "str") is False

    def test_opaque_passthrough_allowed(self) -> None:
        # llem declines to type a complex field and passes it through as Any;
        # that maximal non-narrowing is intentional, not drift.
        assert _is_intentional_narrowing("QuantConfig", "any") is True
        assert _is_intentional_narrowing("set[str]", "any") is True


# ---------------------------------------------------------------------------
# Structural $defs walk (rename-immune; must not go dormant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
def test_collect_props_reaches_generated_config(engine: str) -> None:
    """The walk resolves the generated ...__config__Config def and reaches its
    fields - the live comparison the pre-codegen name map silently stopped doing."""
    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    assert _collect_props(engine, schema), f"no props reachable for {engine}"


def test_bogus_config_def_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A root resolution that matches nothing must raise, not silently compare zero
    fields - the guard that keeps the type check from going dormant again. Point the
    engine property at a $defs entry that does not exist and confirm the loud fail."""
    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    bogus = {p: dict(v) for p, v in schema["properties"].items()}
    bogus["vllm"] = {"anyOf": [{"$ref": "#/$defs/NoSuchConfigDef"}, {"type": "null"}]}
    monkeypatch.setitem(schema, "properties", bogus)
    with pytest.raises(SystemExit, match="not in schema"):
        _collect_props("vllm", schema)


# ---------------------------------------------------------------------------
# Live schema alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
def test_live_schemas_align(engine: str) -> None:
    """Real repo schemas should have zero unexplained drift."""
    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    drifts = check_engine(engine, schema)
    assert drifts == [], f"Drift detected for {engine}: {drifts}"
