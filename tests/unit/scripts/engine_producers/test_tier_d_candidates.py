"""Tests for :mod:`scripts.engine_producers.tier_d_candidates`.

Two layers:

- Pure-logic unit tests over hand-built schema/rules fixtures, pinning each
  filter stage of the recipe (composite-ref, schema-constrained, corpus-covered,
  bool noise, free-form noise, numeric keep) and both type vocabularies
  (python-spelled + JSON-native).
- Committed-artifact smoke tests asserting the known anchor cases against the
  real schemas: vllm ``gpu_memory_utilization`` (bounded -> excluded), a
  corpus-covered leaf (excluded), a bool leaf (excluded), and a genuinely
  unconstrained numeric leaf (included). Counts are determinism-pinned: the same
  committed bytes must yield the same candidate set on every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import tier_d_candidates as tdc  # noqa: E402
from scripts.engine_producers._current import (  # noqa: E402
    current_outputs_dir,
)

# --------------------------------------------------------------------------- #
# Pure-logic helpers
# --------------------------------------------------------------------------- #


def test_base_types_splits_unions_and_collapses_generics():
    assert tdc._base_types("int | None") == {"int"}
    assert tdc._base_types("str | list[str] | None") == {"str", "list"}
    assert tdc._base_types("integer | None") == {"integer"}
    assert tdc._base_types(None) == set()
    assert tdc._base_types("Literal['a', 'b']") == {"Literal"}


def test_is_schema_constrained_detects_each_form():
    defs = {"MyEnum": {"enum": ["a", "b"]}, "MyObj": {"properties": {}}}
    # constraint key on the shape
    assert tdc._is_schema_constrained({"type": "float", "maximum": 1}, defs)
    assert tdc._is_schema_constrained({"type": "int", "exclusiveMinimum": 0}, defs)
    assert tdc._is_schema_constrained({"type": "str", "enum": ["x"]}, defs)
    # Literal in the type string (tensorrt / transformers style)
    assert tdc._is_schema_constrained({"type": "Literal['auto', 'slow']"}, defs)
    # $ref to an enum $def
    assert tdc._is_schema_constrained({"$ref": "#/$defs/MyEnum"}, defs)
    # plain numeric leaf, no constraint -> not constrained
    assert not tdc._is_schema_constrained({"type": "int"}, defs)
    # $ref to an object $def is not a machine constraint
    assert not tdc._is_schema_constrained({"$ref": "#/$defs/MyObj"}, defs)


def test_is_composite_ref_only_for_object_defs():
    defs = {"MyEnum": {"enum": ["a"]}, "MyObj": {"properties": {}}}
    assert tdc._is_composite_ref({"$ref": "#/$defs/MyObj"}, defs)
    assert not tdc._is_composite_ref({"$ref": "#/$defs/MyEnum"}, defs)
    assert not tdc._is_composite_ref({"type": "int"}, defs)


def test_corpus_covered_leaves_takes_dotted_tail():
    rules = {
        "invariants": [
            {"match": {"fields": {"vllm.engine_params.all2all_backend": {"in": ["x"]}}}},
            {"match": {"fields": {"vllm.sampling_params.num_beams": 1}}},
            {"match": {"fields": {}}},
            {},
        ]
    }
    assert tdc._corpus_covered_leaves(rules) == {"all2all_backend", "num_beams"}


# --------------------------------------------------------------------------- #
# Recipe over a synthetic engine (monkeypatched artifact loader)
# --------------------------------------------------------------------------- #


def _fixture_schema() -> dict:
    return {
        "$defs": {
            "SomeEnum": {"enum": ["a", "b"]},
            "NestedCfg": {"properties": {"x": {}}},
        },
        "engine_params": {
            # numeric, unconstrained -> INCLUDED
            "tensor_parallel_size": {"type": "int", "default": 1},
            # numeric JSON-native, unconstrained -> INCLUDED
            "max_tokens": {"type": "integer", "default": 32},
            # bounded numeric -> excluded (schema_constrained)
            "gpu_mem": {"type": "float", "exclusiveMinimum": 0, "maximum": 1},
            # enum -> excluded (schema_constrained)
            "dtype": {"type": "str", "enum": ["auto", "half"]},
            # Literal in type -> excluded (schema_constrained)
            "tokenizer_mode": {"type": "Literal['auto', 'slow']"},
            # $ref to enum -> excluded (schema_constrained)
            "mode_ref": {"$ref": "#/$defs/SomeEnum"},
            # $ref to object -> excluded (composite_ref)
            "nested": {"$ref": "#/$defs/NestedCfg"},
            # bool -> excluded (bool noise)
            "trust_remote_code": {"type": "bool", "default": False},
            # JSON-native bool -> excluded (bool noise)
            "skip_init": {"type": "boolean", "default": False},
            # free-form str -> excluded (freeform noise)
            "model": {"type": "str", "default": "x"},
            # opaque / unknown -> excluded (freeform noise)
            "weird": {"type": "unknown"},
            # corpus-covered numeric -> excluded (corpus_covered, beats numeric keep)
            "covered_num": {"type": "int", "default": 0},
        },
        "sampling_params": {
            "temperature": {"type": "number", "default": 1.0},  # INCLUDED
        },
    }


def _fixture_rules() -> dict:
    return {"invariants": [{"match": {"fields": {"vllm.engine_params.covered_num": {"<": 0}}}}]}


def test_enumerate_recipe_over_fixture(monkeypatch):
    monkeypatch.setattr(
        tdc, "_load_artifacts", lambda engine: (_fixture_schema(), _fixture_rules())
    )
    candidates, counts = tdc.enumerate_class3_candidates("vllm")

    leaves = {c.leaf for c in candidates}
    # genuine unbounded numeric kept across both vocabularies + sections
    assert leaves == {"tensor_parallel_size", "max_tokens", "temperature"}
    assert all(c.reason_included == "unbounded_numeric" for c in candidates)

    assert counts.total_leaves == 13
    assert counts.excluded_composite_ref == 1
    assert counts.excluded_schema_constrained == 4  # gpu_mem, dtype, tokenizer_mode, mode_ref
    assert counts.excluded_corpus_covered == 1  # covered_num
    assert counts.excluded_bool == 2  # trust_remote_code, skip_init
    assert counts.excluded_freeform == 2  # model, weird
    assert counts.included == 3
    # every leaf accounted for by exactly one stage
    accounted = (
        counts.excluded_composite_ref
        + counts.excluded_schema_constrained
        + counts.excluded_corpus_covered
        + counts.excluded_bool
        + counts.excluded_freeform
        + counts.included
    )
    assert accounted == counts.total_leaves


def test_candidates_sorted_deterministically(monkeypatch):
    monkeypatch.setattr(
        tdc, "_load_artifacts", lambda engine: (_fixture_schema(), _fixture_rules())
    )
    a, _ = tdc.enumerate_class3_candidates("vllm")
    b, _ = tdc.enumerate_class3_candidates("vllm")
    assert [(c.section, c.leaf) for c in a] == [(c.section, c.leaf) for c in b]
    # sort order: section then leaf
    keys = [(c.section, c.leaf) for c in a]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# Committed-artifact anchor cases (real schemas)
# --------------------------------------------------------------------------- #


def test_vllm_anchor_cases_real_schema():
    candidates, counts = tdc.enumerate_class3_candidates("vllm")
    leaves = {c.leaf for c in candidates}

    # bounded float (exclusiveMinimum/maximum) -> EXCLUDED
    assert "gpu_memory_utilization" not in leaves
    # enum + corpus-covered -> EXCLUDED
    assert "all2all_backend" not in leaves
    # genuinely-unconstrained numeric -> INCLUDED
    assert "tensor_parallel_size" in leaves
    # bool leaf -> EXCLUDED
    assert "enable_return_routed_experts" not in leaves
    # free-form identifier -> EXCLUDED
    assert "model" not in leaves

    assert counts.included == len(candidates) > 0


def test_real_counts_are_deterministic():
    # Re-running over committed bytes must reproduce the exact candidate set.
    for engine in tdc._ENGINES:
        first, c1 = tdc.enumerate_class3_candidates(engine)
        second, c2 = tdc.enumerate_class3_candidates(engine)
        assert [c.leaf for c in first] == [c.leaf for c in second]
        assert c1 == c2


def test_no_candidate_carries_machine_constraint():
    # Invariant: nothing in the output can already be schema-constrained.
    for engine in tdc._ENGINES:
        schema = json.loads((current_outputs_dir(engine) / "schema.discovered.json").read_text())
        defs = schema.get("$defs", {}) or {}
        candidates, _ = tdc.enumerate_class3_candidates(engine)
        for c in candidates:
            shape = schema[c.section][c.leaf]
            assert not tdc._is_schema_constrained(shape, defs)
            assert not tdc._is_composite_ref(shape, defs)
