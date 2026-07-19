"""Tests for ``scripts.engine_producers`` - common helpers only. - pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
in :mod:`scripts.engine_producers._common` without requiring any engine
package.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Literal

import pydantic
import pytest
from pydantic import BaseModel

# Make the top-level ``scripts`` package importable from tests.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.engine_producers import _common  # noqa: E402

# ---------------------------------------------------------------------------
# annotation_to_type_str
# ---------------------------------------------------------------------------


def test_type_str_simple_primitives() -> None:
    assert _common.annotation_to_type_str(int) == "int"
    assert _common.annotation_to_type_str(str) == "str"
    assert _common.annotation_to_type_str(bool) == "bool"


def test_type_str_none_type() -> None:
    assert _common.annotation_to_type_str(type(None)) == "None"


def test_type_str_pep604_union() -> None:
    assert _common.annotation_to_type_str(int | None) == "int | None"
    assert _common.annotation_to_type_str(int | str | None) == "int | str | None"


def test_type_str_typing_optional() -> None:
    # Deliberately exercising the legacy typing.Optional form - discovery sees
    # this syntax in third-party code even if we prefer X | None ourselves.
    from typing import Optional

    assert _common.annotation_to_type_str(Optional[int]) == "int | None"  # noqa: UP045


def test_type_str_typing_union() -> None:
    # Ditto for typing.Union - third-party engine packages still use it.
    from typing import Union

    assert _common.annotation_to_type_str(Union[int, str]) == "int | str"  # noqa: UP007


def test_type_str_generic_list_dict() -> None:
    assert _common.annotation_to_type_str(list[str]) == "list[str]"
    assert _common.annotation_to_type_str(dict[str, int]) == "dict[str, int]"
    assert _common.annotation_to_type_str(list[dict[str, int]]) == "list[dict[str, int]]"


def test_type_str_literal() -> None:
    assert _common.annotation_to_type_str(Literal["a", "b"]) == "Literal['a', 'b']"


def test_type_str_empty_means_unknown() -> None:
    import inspect

    assert _common.annotation_to_type_str(inspect.Parameter.empty) == "unknown"
    assert _common.annotation_to_type_str(inspect.Signature.empty) == "unknown"


# ---------------------------------------------------------------------------
# read_dockerfile_from
# ---------------------------------------------------------------------------


def test_read_dockerfile_single_stage(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ARG FOO_VERSION=1.2.3\nFROM foo/foo:${FOO_VERSION}\n")
    assert _common.read_dockerfile_from(df) == "foo/foo:1.2.3"


def test_read_dockerfile_prefers_runtime_stage(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text(
        "ARG DEVEL=a:1-devel\n"
        "ARG RUNTIME=a:1-runtime\n"
        "FROM foo:${DEVEL} AS builder\n"
        "FROM foo:${RUNTIME} AS runtime\n"
        "FROM runtime AS dev\n"
    )
    assert _common.read_dockerfile_from(df) == "foo:a:1-runtime"


def test_read_dockerfile_no_runtime_stage_falls_back(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM foo:1 AS builder\n"
        "FROM bar:2 AS packager\n"
        "FROM builder\n"  # references prior stage - should be skipped
    )
    # No `AS runtime` -> first external FROM wins (foo:1)
    assert _common.read_dockerfile_from(df) == "foo:1"


def test_read_dockerfile_expands_only_default_args(tmp_path: Path, monkeypatch) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ARG VER=default\nFROM foo:${VER} AS runtime\n")
    monkeypatch.setenv("VER", "from-env")  # must be ignored
    assert _common.read_dockerfile_from(df) == "foo:default"


def test_read_dockerfile_no_from_raises(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ARG X=1\n# no FROM\n")
    with pytest.raises(ValueError, match="No FROM directive"):
        _common.read_dockerfile_from(df)


def test_read_dockerfile_against_real_dockerfiles() -> None:
    # Only the transformers engine has a first-party Dockerfile post the
    # mount-pivot; vllm and tensorrt run inside upstream images directly.
    tx_from = _common.read_dockerfile_from(REPO_ROOT / "docker/Dockerfile.transformers")
    # runtime stage uses non-devel tag
    assert "pytorch/pytorch:" in tx_from and "devel" not in tx_from


# ---------------------------------------------------------------------------
# jsonable
# ---------------------------------------------------------------------------


def test_jsonable_primitives_passthrough() -> None:
    for v in (None, True, 1, 1.5, "x"):
        assert _common.jsonable(v) == v


def test_jsonable_sets_sorted_list() -> None:
    assert _common.jsonable({3, 1, 2}) == [1, 2, 3]


def test_jsonable_tuple_to_list() -> None:
    assert _common.jsonable((1, "a", None)) == [1, "a", None]


def test_jsonable_nested_dict() -> None:
    got = _common.jsonable({"k": (1, {2, 3})})
    assert got == {"k": [1, [2, 3]]}


def test_jsonable_type_to_name() -> None:
    assert _common.jsonable(int) == "int"


def test_jsonable_fallback_to_str() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert _common.jsonable(Opaque()) == "<opaque>"


# ---------------------------------------------------------------------------
# Envelope shape
# ---------------------------------------------------------------------------


def test_make_envelope_fills_required_keys() -> None:
    env = _common.make_envelope(
        engine="vllm",
        engine_version="0.7.3",
        engine_commit_sha=None,
        image_ref="foo:1",
        base_image_ref="foo:1",
        discovery_method="unit test",
        discovery_limitations=[],
        engine_params={"a": {"type": "int", "default": 0}},
        sampling_params={"b": {"type": "str", "default": ""}},
    )
    assert env["schema_version"] == _common.SCHEMA_VERSION
    assert env["engine"] == "vllm"
    assert env["discovered_at"]  # ISO string
    assert "engine_params" in env and "sampling_params" in env


def test_schema_version_is_semver_with_major_one() -> None:
    major = int(_common.SCHEMA_VERSION.split(".")[0])
    assert major == 1, "ships schema_version 1.x; bumping major requires loader update"


# ---------------------------------------------------------------------------
# Per-engine LANDMARKS contracts
# ---------------------------------------------------------------------------
#
# ``LANDMARKS`` names the dotted attribute paths each producer module must
# resolve against the installed library before discovery runs. These tests
# resolve the transformers introspector's landmarks host-side so a
# landmark-name drift in an upstream transformers bump is caught at
# unit-test time.


def test_transformers_introspector_landmarks_resolve() -> None:
    pytest.importorskip("transformers")
    import importlib

    from scripts.engine_producers import transformers_schema_introspector

    landmarks = transformers_schema_introspector.LANDMARKS
    assert isinstance(landmarks, tuple) and landmarks, "LANDMARKS must be a non-empty tuple"

    missing: list[str] = []
    for landmark in landmarks:
        parts = landmark.split(".")
        module = None
        module_idx = 0
        for split in range(len(parts), 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:split]))
                module_idx = split
                break
            except ImportError:
                continue
        if module is None:
            missing.append(landmark)
            continue
        try:
            obj: object = module
            for attr in parts[module_idx:]:
                obj = getattr(obj, attr)
        except AttributeError:
            missing.append(landmark)
    assert not missing, f"Unresolvable transformers landmarks: {missing}"


# ===========================================================================
# Schema-substrate additions: docstring type recovery, $defs recursion, source
# overlay, opaque-default sanitisation, flat-Literal enum capture, msgspec type
# recovery.
# ===========================================================================


# ---------------------------------------------------------------------------
# Module-scope fixtures for the $defs recursion tests. Defined at module scope
# (not inside the test functions) so ``typing.get_type_hints`` can resolve the
# forward-referenced names.
# ---------------------------------------------------------------------------


class _NestedModel(BaseModel):
    depth: int = 2


class _SubConfig(BaseModel):
    x: int = 1
    nested: _NestedModel = _NestedModel()


@dataclasses.dataclass
class _EngineArgsLike:
    sub: _SubConfig | None = None
    plain: int = 0


@dataclasses.dataclass
class _ListOfPydanticArgs:
    subs: list[_SubConfig] = dataclasses.field(default_factory=list)


# Non-Pydantic (stdlib) dataclass sub-configs - the vLLM CompilationConfig /
# AttentionConfig shape that exposes no ``model_json_schema``.
@dataclasses.dataclass
class _DcLeaf:
    mode: str = "default"
    size: int | None = None


@dataclasses.dataclass
class _DcMid:
    leaf: _DcLeaf = dataclasses.field(default_factory=_DcLeaf)
    flag: bool = False
    self_ref: _DcMid | None = None  # self-reference -> cycle guard


@dataclasses.dataclass
class _DcEngineArgsLike:
    mid: _DcMid | None = None
    pyd: _SubConfig | None = None  # mixed: Pydantic sub-config alongside dataclass
    plain: str = "p"
    leaves: list[_DcLeaf] | None = None  # generic -> no single class to $ref


@dataclasses.dataclass
class _DictTypedArgs:
    spec: dict | None = None  # dict-typed (like vLLM speculative_config) -> needs a hint
    plain: int = 0


# A pydantic dataclass sub-config - the vLLM @config shape: is_dataclass True,
# __pydantic_fields__ present, model_json_schema absent. Its Field bounds + Literal
# enum must be captured (the bare-dataclass walk loses them).
@pydantic.dataclasses.dataclass
class _PydDcLeaf:
    frac: float = pydantic.Field(default=0.9, gt=0, le=1)
    mode: Literal["a", "b", "c"] = "a"
    unbounded: int = 0  # no Field bound -> stays plain


@dataclasses.dataclass
class _ArgsWithPydDc:
    leaf: _PydDcLeaf | None = None
    plain: str = "p"


# ---------------------------------------------------------------------------
# docstring_arg_types
# ---------------------------------------------------------------------------


class _DocStyleConfig:
    """A HuggingFace-style configurable.

    Args:
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search.
        temperature (`float`, *optional*):
            The sampling temperature.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to sample.
        early_stopping (`bool` or `str`, *optional*):
            Beam-search stopping condition.
        cache_config (`Dict`, *optional*):
            Arguments used in the key/value cache.
        not_a_field:
            A line with no parenthesised type.
    """


def test_docstring_arg_types_recovers_scalar_types() -> None:
    types = _common.docstring_arg_types(_DocStyleConfig)
    assert types["num_beams"] == "int"
    assert types["temperature"] == "float"
    assert types["do_sample"] == "bool"


def test_docstring_arg_types_takes_first_member_of_or_union() -> None:
    # ``bool` or `str`` documents a union; the first member matches the
    # value-inference baseline the older pins produced.
    assert _common.docstring_arg_types(_DocStyleConfig)["early_stopping"] == "bool"


def test_docstring_arg_types_omits_non_scalar_and_untyped() -> None:
    types = _common.docstring_arg_types(_DocStyleConfig)
    # Non-scalar documented type (Dict) is dropped so the caller falls back to
    # default-inference; an untyped arg line is never captured.
    assert "cache_config" not in types
    assert "not_a_field" not in types


def test_docstring_arg_types_empty_when_no_docstring() -> None:
    class _NoDoc:
        pass

    assert _common.docstring_arg_types(_NoDoc) == {}


# ---------------------------------------------------------------------------
# make_envelope - $defs block
# ---------------------------------------------------------------------------


def test_make_envelope_omits_defs_when_empty() -> None:
    env = _common.make_envelope(
        engine="vllm",
        engine_version="0.7.3",
        engine_commit_sha=None,
        image_ref="foo:1",
        base_image_ref="foo:1",
        discovery_method="unit test",
        discovery_limitations=[],
        engine_params={},
        sampling_params={},
    )
    assert "$defs" not in env
    # Explicit empty dict is also treated as "nothing to emit".
    env_empty = _common.make_envelope(
        engine="vllm",
        engine_version="0.7.3",
        engine_commit_sha=None,
        image_ref="foo:1",
        base_image_ref="foo:1",
        discovery_method="unit test",
        discovery_limitations=[],
        engine_params={},
        sampling_params={},
        defs={},
    )
    assert "$defs" not in env_empty


def test_make_envelope_emits_and_jsonifies_defs() -> None:
    env = _common.make_envelope(
        engine="tensorrt",
        engine_version="1.0.0",
        engine_commit_sha=None,
        image_ref="foo:1",
        base_image_ref="foo:1",
        discovery_method="unit test",
        discovery_limitations=[],
        engine_params={"kv_cache_config": {"$ref": "#/$defs/KvCacheConfig", "default": None}},
        sampling_params={},
        defs={
            "KvCacheConfig": {
                "type": "object",
                "properties": {"max_tokens": {"type": "integer"}},
                "required": {"max_tokens"},  # set -> jsonable should sort to a list
            }
        },
    )
    assert "$defs" in env
    kv = env["$defs"]["KvCacheConfig"]
    assert kv["properties"]["max_tokens"]["type"] == "integer"
    # ``jsonable`` coerces the set to a sorted list so json.dumps stays clean.
    assert kv["required"] == ["max_tokens"]


# ---------------------------------------------------------------------------
# dataclass_fields_to_specs - $defs recursion (Pydantic + stdlib dataclass)
# ---------------------------------------------------------------------------


def test_dataclass_specs_without_defs_flatten_pydantic_to_type_str() -> None:
    """Without a ``defs`` accumulator the walk keeps the legacy flat shape."""
    specs = _common.dataclass_fields_to_specs(_EngineArgsLike)
    # No $ref emitted; nested pydantic flattens to a readable type string.
    assert "$ref" not in specs["sub"]
    assert specs["plain"]["type"] == "int"


def test_dataclass_specs_recurse_into_pydantic_field_emitting_ref_and_defs() -> None:
    defs: dict = {}
    specs = _common.dataclass_fields_to_specs(_EngineArgsLike, defs=defs)

    # Pydantic-typed field becomes a $ref into $defs; default still recorded.
    assert specs["sub"]["$ref"] == "#/$defs/_SubConfig"
    # Plain field is untouched.
    assert specs["plain"]["type"] == "int"
    # The sub-model AND its transitively-referenced nested model are folded in,
    # so every $ref in the envelope resolves.
    assert "_SubConfig" in defs and "_NestedModel" in defs
    assert defs["_SubConfig"]["properties"]["x"]["default"] == 1
    # The nested model is reachable via the standard #/$defs/ ref template.
    assert defs["_SubConfig"]["properties"]["nested"]["$ref"] == "#/$defs/_NestedModel"


def test_dataclass_specs_list_of_pydantic_stays_object_not_ref() -> None:
    """``list[SubConfig]`` has no single class to $ref - stays a type string."""
    defs: dict = {}
    specs = _common.dataclass_fields_to_specs(_ListOfPydanticArgs, defs=defs)
    assert "$ref" not in specs["subs"]
    assert defs == {}


def test_dataclass_specs_recurse_into_non_pydantic_dataclass_field() -> None:
    """A stdlib-dataclass sub-config field becomes a $ref with its leaves in $defs.

    This is the vLLM ``EngineArgs -> CompilationConfig`` shape: a nested
    ``@dataclass`` that exposes no ``model_json_schema`` and would otherwise
    flatten to a bare type-name string.
    """
    defs: dict = {}
    specs = _common.dataclass_fields_to_specs(_DcEngineArgsLike, defs=defs)

    # Dataclass-typed field -> $ref; the def carries the leaves with types/defaults.
    assert specs["mid"]["$ref"] == "#/$defs/_DcMid"
    assert "_DcMid" in defs and "_DcLeaf" in defs
    assert defs["_DcLeaf"]["properties"]["mode"] == {"type": "str", "default": "default"}
    # Transitive: _DcMid.leaf folds _DcLeaf as a nested $ref.
    assert defs["_DcMid"]["properties"]["leaf"]["$ref"] == "#/$defs/_DcLeaf"
    # Cycle: _DcMid.self_ref -> _DcMid terminates via the placeholder guard.
    assert defs["_DcMid"]["properties"]["self_ref"]["$ref"] == "#/$defs/_DcMid"
    # Mixed: a Pydantic sub-config in the same class still routes via model_json_schema.
    assert specs["pyd"]["$ref"] == "#/$defs/_SubConfig"
    assert "_SubConfig" in defs
    # Plain scalar untouched; generic list[dataclass] not recursed.
    assert specs["plain"] == {"type": "str", "default": "p"}
    assert "$ref" not in specs["leaves"]


def test_dataclass_specs_without_defs_flatten_dataclass_to_type_str() -> None:
    """The legacy (no-``defs``) path keeps the flat type string for dataclasses too."""
    specs = _common.dataclass_fields_to_specs(_DcEngineArgsLike)
    assert "$ref" not in specs["mid"]
    assert "type" in specs["mid"]


def test_pydantic_dataclass_field_captures_bounds_and_enums() -> None:
    """A pydantic-dataclass sub-config folds via the rich path, keeping Field bounds + Literal enums.

    This is the vLLM ``@config`` shape (is_dataclass + __pydantic_fields__, no
    model_json_schema). The bare-dataclass walk would lose ``gt=0, le=1`` and the
    ``Literal`` membership; the pydantic JSON-schema path keeps them.
    """
    defs: dict = {}
    specs = _common.dataclass_fields_to_specs(_ArgsWithPydDc, defs=defs)
    assert specs["leaf"]["$ref"] == "#/$defs/_PydDcLeaf"
    leaf = defs["_PydDcLeaf"]["properties"]
    # Numeric bound preserved (Field(gt=0, le=1) -> exclusiveMinimum/maximum).
    assert leaf["frac"].get("exclusiveMinimum") == 0
    assert leaf["frac"].get("maximum") == 1
    # Literal enum preserved.
    assert leaf["mode"].get("enum") == ["a", "b", "c"]
    # An unbounded field stays plain (no bound invented).
    assert "minimum" not in leaf["unbounded"] and "maximum" not in leaf["unbounded"]


# ---------------------------------------------------------------------------
# fold_dict_typed_subconfig
# ---------------------------------------------------------------------------


def test_fold_dict_typed_subconfig_rewrites_dict_field_to_ref() -> None:
    """A dict-typed field is rewritten to a $ref once the real class is supplied.

    Mirrors vLLM ``speculative_config`` (annotated ``dict[str, Any]`` but coerced
    to ``SpeculativeConfig``): the introspector carries the field->class hint.
    """
    defs: dict = {}
    specs = _common.dataclass_fields_to_specs(_DictTypedArgs, defs=defs)
    # Pre-hint: dict-typed field is a bare type string, no $ref, no def.
    assert "$ref" not in specs["spec"]
    assert defs == {}

    folded = _common.fold_dict_typed_subconfig(specs, "spec", _DcLeaf, defs)
    assert folded is True
    assert specs["spec"] == {"$ref": "#/$defs/_DcLeaf", "default": None}
    assert defs["_DcLeaf"]["properties"]["mode"]["type"] == "str"


def test_fold_dict_typed_subconfig_noop_on_missing_field() -> None:
    """No hinted field -> returns False, no mutation, no crash."""
    defs: dict = {}
    specs: dict = {"plain": {"type": "int", "default": 0}}
    assert _common.fold_dict_typed_subconfig(specs, "absent", _DcLeaf, defs) is False
    assert specs == {"plain": {"type": "int", "default": 0}}
    assert defs == {}


def test_fold_dict_typed_subconfig_returns_false_for_plain_class() -> None:
    """A plain (non-dataclass, non-Pydantic) class cannot be lifted -> False, no mutation.

    This is the tensorrt ``DecodingConfig`` shape: the caller uses the False
    return to record an honest discovery limitation instead of a silent gap.
    """

    class _PlainCls:
        def __init__(self, a: int = 1) -> None:
            self.a = a

    defs: dict = {}
    specs: dict = {"spec": {"type": "object", "default": None}}
    assert _common.fold_dict_typed_subconfig(specs, "spec", _PlainCls, defs) is False
    assert specs == {"spec": {"type": "object", "default": None}}  # untouched
    assert defs == {}


# ---------------------------------------------------------------------------
# exposable_default - opaque object defaults must NOT leak as stringified reprs
# ---------------------------------------------------------------------------


class _OpaqueBlob:
    def __repr__(self) -> str:
        return "<opaque config object repr blob>"


@dataclasses.dataclass
class _OpaqueDefaultArgs:
    blob: object = dataclasses.field(default_factory=_OpaqueBlob)
    plain: int = 0


def test_exposable_default_nulls_opaque_objects_keeps_clean_structures() -> None:
    assert _common.exposable_default(_OpaqueBlob()) is None
    assert _common.exposable_default(5) == 5
    assert _common.exposable_default("auto") == "auto"
    assert _common.exposable_default([1, 2]) == [1, 2]
    assert _common.exposable_default({"a": 1}) == {"a": 1}
    # An opaque value nested inside a clean container is nulled in place rather
    # than stringified.
    assert _common.exposable_default({"k": _OpaqueBlob()}) == {"k": None}


def test_exposable_default_set_with_opaque_elements_does_not_raise() -> None:
    """A set default with opaque elements must not crash discovery.

    Opaque elements null to None, leaving a non-orderable mix; sorting must fall
    back to a stable key rather than raising TypeError. Clean homogeneous sets
    keep their natural sort order.
    """
    assert _common.exposable_default({"b", "a"}) == ["a", "b"]
    assert _common.exposable_default({3, 1, 2}) == [1, 2, 3]
    # opaque + primitive: opaque -> None, no crash, deterministic order
    assert _common.exposable_default({1, _OpaqueBlob()}) == [1, None]
    # all-opaque: every element nulls to None, still deterministic
    assert _common.exposable_default({_OpaqueBlob(), _OpaqueBlob()}) == [None, None]


def test_dataclass_specs_null_opaque_object_default_not_stringified() -> None:
    """A field whose default is an opaque object is recorded as None, never its
    repr string - else codegen would emit a bogus non-None default and forward
    that stringified blob to the engine on every unset run. Pin the contract so
    the str() fallback can't return silently.
    """
    specs = _common.dataclass_fields_to_specs(_OpaqueDefaultArgs)
    assert specs["blob"]["default"] is None
    assert specs["plain"]["default"] == 0


# ---------------------------------------------------------------------------
# _literal_enum_spec - flat Literal fields captured as structured enums
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ArgsWithFlatLiteral:
    """A flat (non-nested) dataclass carrying ``Literal`` scalar fields.

    Mirrors vLLM ``EngineArgs.dtype`` / ``kv_cache_dtype``: the membership set
    lives directly on the engine-args dataclass, not on a nested sub-config.
    """

    dtype: Literal["auto", "half", "bfloat16"] = "auto"
    block: Literal[8, 16, 32] = 16
    backend: Literal["a", "b"] | None = None
    plain: int = 3


def test_flat_literal_field_captures_enum_not_type_string() -> None:
    """A flat ``Literal`` field becomes a structured ``enum`` (not an opaque string)."""
    specs = _common.dataclass_fields_to_specs(_ArgsWithFlatLiteral, defs={})
    assert specs["dtype"] == {
        "enum": ["auto", "half", "bfloat16"],
        "type": "str",
        "default": "auto",
    }
    assert specs["block"] == {"enum": [8, 16, 32], "type": "int", "default": 16}
    # ``X | None`` unwraps to the sole Literal; the enum is captured (default kept).
    assert specs["backend"]["enum"] == ["a", "b"]
    assert specs["backend"]["type"] == "str"
    # A non-Literal scalar is untouched (no enum invented).
    assert specs["plain"] == {"type": "int", "default": 3}
    assert "enum" not in specs["plain"]


def test_literal_enum_spec_ignores_non_literal() -> None:
    """``_literal_enum_spec`` returns None for non-Literal annotations (no false enums)."""
    assert _common._literal_enum_spec(int) is None
    assert _common._literal_enum_spec(str | None) is None
    assert _common._literal_enum_spec(Literal["x", "y"]) == {"enum": ["x", "y"], "type": "str"}


# ---------------------------------------------------------------------------
# merge_source_constraints - source-text bounds/enums fold onto discovered fields
# ---------------------------------------------------------------------------


def test_merge_source_constraints_overlays_bounds_and_enums(tmp_path: Path) -> None:
    src = tmp_path / "cache.py"
    src.write_text(
        "from typing import Literal\n"
        "from pydantic import Field\n"
        "class CacheConfig:\n"
        "    block_size: int = Field(1, ge=1, le=256)\n"
        "    cache_dtype: Literal['auto', 'fp8'] = 'auto'\n"
    )
    # Discovery is the source of truth for which fields exist; only fields already
    # present gain constraint keys, and type/default are never overwritten.
    schema_fields: dict[str, dict[str, object]] = {
        "block_size": {"type": "int", "default": 16},
        "cache_dtype": {"type": "str", "default": "auto"},
        "unrelated": {"type": "bool", "default": False},
    }
    touched = _common.merge_source_constraints(schema_fields, [src])
    assert touched == 2
    assert schema_fields["block_size"] == {
        "type": "int",
        "default": 16,
        "minimum": 1,
        "maximum": 256,
    }
    assert schema_fields["cache_dtype"]["enum"] == ["auto", "fp8"]
    # A field absent from the source walk is untouched.
    assert schema_fields["unrelated"] == {"type": "bool", "default": False}


def test_merge_source_constraints_skips_fields_not_in_schema(tmp_path: Path) -> None:
    src = tmp_path / "cfg.py"
    src.write_text(
        "from pydantic import Field\nclass FooConfig:\n    only_in_source: int = Field(0, ge=0)\n"
    )
    schema_fields: dict = {"other": {"type": "int", "default": 1}}
    assert _common.merge_source_constraints(schema_fields, [src]) == 0
    assert schema_fields == {"other": {"type": "int", "default": 1}}


def test_merge_source_constraints_ignores_missing_paths(tmp_path: Path) -> None:
    schema_fields: dict = {"x": {"type": "int", "default": 0}}
    assert _common.merge_source_constraints(schema_fields, [tmp_path / "nope.py"]) == 0


# ---------------------------------------------------------------------------
# recover_field_types - msgspec type recovery for the schema product
# ---------------------------------------------------------------------------

msgspec = pytest.importorskip("msgspec")


class _UntypedByJsonSchema(msgspec.Struct):  # type: ignore[name-defined,misc]
    """Fields msgspec.json.schema renders as an untyped anyOf / bare token.

    Mirrors the vLLM ``SamplingParams`` shape: optional unions, a nested-list
    union, a dict union, and a genuinely-opaque ``Any | None``.
    ``recover_field_types`` must resolve every concrete one and omit only the
    opaque ``Any | None``.
    """

    seed: int | None = None
    stop_token_ids: list[int] | None = None
    logit_bias: dict[int, float] | None = None
    mode: Literal["greedy", "beam"] = "greedy"
    logits_processors: object | None = None  # opaque - stays unrecovered


def test_recover_field_types_resolves_unions_and_containers() -> None:
    recovered = _common.recover_field_types(_UntypedByJsonSchema)
    assert recovered["seed"] == "int | None"
    assert recovered["stop_token_ids"] == "list[int] | None"
    assert recovered["logit_bias"] == "dict[int, float] | None"
    # msgspec sorts Literal values when building its type info; mirror that order.
    assert recovered["mode"] == "Literal['beam', 'greedy']"


def test_recover_field_types_empty_for_non_struct() -> None:
    class _NotStruct:
        pass

    assert _common.recover_field_types(_NotStruct) == {}
