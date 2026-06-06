"""Tests for :mod:`scripts.validate_schema` pure logic (no container needed).

The live-reflection + construct-probe paths require the engine container and
are exercised by the schema-gate smoke run inside CI; these unit tests cover
the reflection-diff and enum-probe-value helpers deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import validate_schema  # noqa: E402


def test_type_signature_ignores_description_and_default():
    a = {"type": "string", "enum": ["a", "b"], "default": "a", "description": "x"}
    b = {"type": "string", "enum": ["a", "b"], "default": "b", "description": "y"}
    assert validate_schema._type_signature(a) == validate_schema._type_signature(b)


def test_type_signature_distinguishes_type():
    assert validate_schema._type_signature({"type": "string"}) != validate_schema._type_signature(
        {"type": "integer"}
    )


def test_diff_section_all_match():
    stored = {"f": {"type": "string", "default": "a"}}
    live = {"f": {"type": "string", "default": "a"}}
    results, divergences = validate_schema._diff_section("engine_params", stored, live)
    assert divergences == []
    assert results[0]["exists"] and results[0]["type_match"] and results[0]["default_match"]


def test_diff_section_type_default_and_missing():
    stored = {"f": {"type": "string", "default": "a"}, "gone": {"type": "integer"}}
    live = {"f": {"type": "integer", "default": "b"}, "added": {"type": "string"}}
    _results, divergences = validate_schema._diff_section("engine_params", stored, live)
    kinds = {d["check"] for d in divergences}
    assert "type" in kinds  # f type changed
    assert "default" in kinds  # f default changed
    assert "exists" in kinds  # gone absent in live
    assert "new_in_live" in kinds  # added present in live only


def test_enum_probe_values_string():
    valid, invalid = validate_schema._enum_probe_values({"enum": ["auto", "slow"]})
    assert valid == "auto"
    assert invalid not in ("auto", "slow")


def test_enum_probe_values_numeric():
    valid, invalid = validate_schema._enum_probe_values({"enum": [1, 2, 4]})
    assert valid == 1
    assert invalid == 5  # max + 1, out of domain


def test_enum_probe_values_none_for_non_enum():
    assert validate_schema._enum_probe_values({"type": "string"}) is None
    assert validate_schema._enum_probe_values({"enum": []}) is None


def test_semantic_type_normalizes_representation_skew():
    # Same engine truth, different introspector renderings must compare equal:
    # enum+string vs Literal, and string vs anyOf[string, string+path].
    assert validate_schema._semantic_type(
        {"enum": ["auto", "slow"], "type": "string"}
    ) == validate_schema._semantic_type(
        {"enum": ["auto", "slow"], "type": "Literal['auto', 'slow']"}
    )
    assert validate_schema._semantic_type({"type": "string"}) == validate_schema._semantic_type(
        {"anyOf": [{"type": "string"}, {"format": "path", "type": "string"}]}
    )
    # Genuine type drift must still differ.
    assert validate_schema._semantic_type({"type": "string"}) != validate_schema._semantic_type(
        {"type": "integer"}
    )


def test_diff_section_ignores_representation_skew():
    stored = {"model": {"type": "string", "default": None}}
    live = {
        "model": {
            "anyOf": [{"type": "string"}, {"format": "path", "type": "string"}],
            "default": None,
        }
    }
    _results, divergences = validate_schema._diff_section("engine_params", stored, live)
    assert divergences == []  # representation differs, semantic type identical


def test_semantic_type_resolves_ref_to_target_name():
    # A nested-config field ($ref) must canonicalise to its target type, not "".
    assert validate_schema._semantic_type({"$ref": "#/$defs/CompileConfig"}) == "ref:CompileConfig"
    # Distinct nested types must differ (the latent bug: both used to be "").
    assert validate_schema._semantic_type(
        {"$ref": "#/$defs/LoraConfig"}
    ) != validate_schema._semantic_type({"$ref": "#/$defs/QuantConfig"})
    # Optional[Nested] (anyOf[$ref, null]) drops null and keeps the ref target.
    assert (
        validate_schema._semantic_type(
            {"anyOf": [{"$ref": "#/$defs/LoraConfig"}, {"type": "null"}]}
        )
        == "ref:LoraConfig"
    )


def test_diff_section_flags_ref_vs_flattened_object():
    # The flatten-vs-nested drift the schema gate must catch: stored flattened a
    # nested config to a bare object; live now points at the proper nested type.
    stored = {"lora_config": {"type": "object", "default": None}}
    live = {"lora_config": {"$ref": "#/$defs/LoraConfig", "default": None}}
    _results, divergences = validate_schema._diff_section("engine_params", stored, live)
    type_divs = [d for d in divergences if d["check"] == "type"]
    assert len(type_divs) == 1
    assert type_divs[0]["declared"] == "dict"
    assert type_divs[0]["observed"] == "ref:LoraConfig"
