"""Tests for ``scripts.engine_producers`` - common helpers only. - pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
in :mod:`scripts.engine_producers._common` without requiring any engine
package.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import inspect
import sys
import textwrap
import uuid
from pathlib import Path
from types import ModuleType
from typing import Literal, Optional

import pytest

# Make the top-level ``scripts`` package importable from tests.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.engine_producers import _common  # noqa: E402

# ---------------------------------------------------------------------------
# Module-scope fixtures for Pydantic-in-dataclass recursion tests
# (TestDataclassFieldsToSpecs + TestAnnotationToJsonSchema). Module-scope so
# ``typing.get_type_hints`` can resolve PEP 563 string annotations against
# this module's namespace - test-method-local classes don't work because
# they're not in ``__module__``'s globals.
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel as _BaseModel  # noqa: E402
except ImportError:  # pragma: no cover - pydantic is a hard dep
    _BaseModel = None  # type: ignore[assignment]


if _BaseModel is not None:

    class _FixtureKvCacheConfig(_BaseModel):
        enable_block_reuse: bool = False
        dtype: str = "auto"

    @dataclasses.dataclass
    class _FixtureEngineArgs:
        kv_cache_config: _FixtureKvCacheConfig | None = None

    class _FixtureInner(_BaseModel):
        level: int = 0

    class _FixtureOuter(_BaseModel):
        inner: _FixtureInner = _FixtureInner()
        name: str = "x"

    @dataclasses.dataclass
    class _FixtureNestedArgs:
        outer: _FixtureOuter | None = None

    class _FixtureForwardConf(_BaseModel):
        x: int = 1

    @dataclasses.dataclass
    class _FixtureForwardArgs:
        conf: _FixtureForwardConf | None = None


def _load_synthetic_module(tmp_path: Path, source: str) -> ModuleType:
    """Write ``source`` to a uniquely named file under ``tmp_path`` and import it.

    The walker calls :func:`inspect.getsource` on the passed module; a file-
    backed module is the simplest way to satisfy that contract without
    monkeypatching :mod:`inspect`. A uuid-suffix avoids collisions when
    pytest re-orders tests.
    """
    mod_name = f"_synthetic_{uuid.uuid4().hex}"
    path = tmp_path / f"{mod_name}.py"
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


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
        discovery_limitations=[],
        engine_params={"a": {"type": "integer", "default": 0}},
        sampling_params={"b": {"type": "string", "default": ""}},
    )
    assert env["schema_version"] == _common.SCHEMA_VERSION
    assert env["engine"] == "vllm"
    assert env["discovered_at"]  # ISO string
    assert "engine_params" in env and "sampling_params" in env


def test_envelope_omits_discovery_method() -> None:
    """``discovery_method`` was dropped in schema_version 2.0.0."""
    env = _common.make_envelope(
        engine="vllm",
        engine_version="0.7.3",
        engine_commit_sha=None,
        image_ref="foo:1",
        base_image_ref="foo:1",
        discovery_limitations=[],
        engine_params={},
        sampling_params={},
    )
    assert "discovery_method" not in env


def test_schema_version_is_semver_with_major_two() -> None:
    """v2.0.0 introduces canonical JSON Schema per-field shapes."""
    major = int(_common.SCHEMA_VERSION.split(".")[0])
    assert major == 2, "ships schema_version 2.x; bumping major requires loader update"


# ---------------------------------------------------------------------------
# $defs propagation (post-#540): make_envelope + canonicalize_defs
# ---------------------------------------------------------------------------


class TestMakeEnvelopeWithDefs:
    """The optional ``defs`` parameter carries JSON Schema ``$defs``
    reusable sub-schemas from Pydantic / msgspec discovery through to the
    envelope so consumers (datamodel-code-generator) can resolve ``$ref``
    pointers to nested config classes."""

    def _envelope_with_defs(self, defs: dict[str, dict[str, _common.Any]] | None) -> dict[str, _common.Any]:
        return _common.make_envelope(
            engine="tensorrt",
            engine_version="0.21.0",
            engine_commit_sha=None,
            image_ref="foo:1",
            base_image_ref="foo:1",
            discovery_limitations=[],
            engine_params={"kv_cache_config": {"$ref": "#/$defs/KvCacheConfig"}},
            sampling_params={},
            defs=defs,
        )

    def test_populated_defs_round_trip(self) -> None:
        env = self._envelope_with_defs(
            {"KvCacheConfig": {"type": "object", "properties": {"enable": {"type": "boolean"}}}}
        )
        assert "$defs" in env
        assert env["$defs"]["KvCacheConfig"]["properties"]["enable"] == {"type": "boolean"}

    def test_empty_defs_omits_key(self) -> None:
        """Pre-existing envelopes round-trip byte-identical: no $defs key
        when caller passes None or empty dict."""
        env_none = self._envelope_with_defs(None)
        env_empty = self._envelope_with_defs({})
        assert "$defs" not in env_none
        assert "$defs" not in env_empty


class TestCanonicalizeDefs:
    """``canonicalize_defs`` strips ``title`` (Pydantic restates the class
    name; noise) and canonicalizes inner ``properties`` so the envelope's
    $defs entries land in the same leaf shape as engine_params /
    sampling_params entries."""

    def test_none_input_returns_empty(self) -> None:
        assert _common.canonicalize_defs(None) == {}

    def test_empty_input_returns_empty(self) -> None:
        assert _common.canonicalize_defs({}) == {}

    def test_drops_title_at_def_level(self) -> None:
        out = _common.canonicalize_defs(
            {"Inner": {"title": "Inner", "type": "object", "properties": {}}}
        )
        assert "title" not in out["Inner"]
        assert out["Inner"]["type"] == "object"

    def test_canonicalizes_inner_properties(self) -> None:
        """Inner property dicts get jsonschema_property_to_canonical
        treatment: ``title`` dropped on each property too."""
        out = _common.canonicalize_defs(
            {
                "Inner": {
                    "type": "object",
                    "properties": {
                        "foo": {"title": "Foo", "type": "integer", "default": 0},
                    },
                }
            }
        )
        assert "title" not in out["Inner"]["properties"]["foo"]
        assert out["Inner"]["properties"]["foo"]["type"] == "integer"

    def test_exclude_drops_named_entries(self) -> None:
        """msgspec's ``msgspec.json.schema(SamplingParams)`` returns
        SamplingParams as a root entry in $defs whose properties have
        already been unpacked into the envelope's sampling_params section;
        exclude drops it so the envelope's $defs isn't a redundant
        duplicate."""
        out = _common.canonicalize_defs(
            {
                "SamplingParams": {"type": "object", "properties": {}},
                "BeamSearchParams": {"type": "object", "properties": {}},
            },
            exclude=["SamplingParams"],
        )
        assert "SamplingParams" not in out
        assert "BeamSearchParams" in out

    def test_idempotent(self) -> None:
        """Calling twice produces the same shape (already-canonical input
        round-trips byte-identical)."""
        defs = {
            "Inner": {
                "type": "object",
                "properties": {"foo": {"type": "integer", "default": 0}},
            }
        }
        once = _common.canonicalize_defs(defs)
        twice = _common.canonicalize_defs(once)
        assert once == twice


# ---------------------------------------------------------------------------
# annotation_to_json_schema (canonical JSON Schema 2020-12 emitters)
# ---------------------------------------------------------------------------


class TestAnnotationToJsonSchema:
    def test_primitive_int(self) -> None:
        assert _common.annotation_to_json_schema(int) == {"type": "integer"}

    def test_primitive_str(self) -> None:
        assert _common.annotation_to_json_schema(str) == {"type": "string"}

    def test_primitive_float(self) -> None:
        assert _common.annotation_to_json_schema(float) == {"type": "number"}

    def test_primitive_bool(self) -> None:
        assert _common.annotation_to_json_schema(bool) == {"type": "boolean"}

    def test_none_type(self) -> None:
        assert _common.annotation_to_json_schema(type(None)) == {"type": "null"}

    def test_optional_collapses_to_type_array(self) -> None:
        """X | None -> {"type": ["X", "null"]} (canonical JSON Schema 2020-12)."""
        assert _common.annotation_to_json_schema(int | None) == {"type": ["integer", "null"]}
        assert _common.annotation_to_json_schema(str | None) == {"type": ["string", "null"]}

    def test_typing_optional_collapses_to_type_array(self) -> None:
        assert _common.annotation_to_json_schema(Optional[int]) == {  # noqa: UP045
            "type": ["integer", "null"]
        }

    def test_multi_branch_union_anyof(self) -> None:
        """X | Y (no None) -> {"anyOf": [{"type": "X"}, {"type": "Y"}]}."""
        result = _common.annotation_to_json_schema(int | str)
        assert result == {"anyOf": [{"type": "integer"}, {"type": "string"}]}

    def test_multi_branch_optional_union(self) -> None:
        """X | Y | None -> anyOf with null branch."""
        result = _common.annotation_to_json_schema(int | str | None)
        assert result == {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]}

    def test_literal_emits_enum(self) -> None:
        result = _common.annotation_to_json_schema(Literal["a", "b"])
        assert result == {"type": "string", "enum": ["a", "b"]}

    def test_literal_int(self) -> None:
        result = _common.annotation_to_json_schema(Literal[1, 2, 3])
        assert result == {"type": "integer", "enum": [1, 2, 3]}

    def test_enum_subclass(self) -> None:
        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        result = _common.annotation_to_json_schema(Color)
        assert result == {"type": "string", "enum": ["red", "blue"]}

    def test_list_of_str(self) -> None:
        result = _common.annotation_to_json_schema(list[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_dict_str_int(self) -> None:
        result = _common.annotation_to_json_schema(dict[str, int])
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_class_becomes_object_with_description(self) -> None:
        class MyConfig:
            pass

        result = _common.annotation_to_json_schema(MyConfig)
        assert result == {"type": "object", "description": "MyConfig"}

    def test_class_union_with_string_and_none(self) -> None:
        """The motivating real case: ``PretrainedConfig | str | PathLike | None``."""

        class PretrainedConfig:
            pass

        result = _common.annotation_to_json_schema(PretrainedConfig | str | None)
        assert result == {
            "anyOf": [
                {"type": "object", "description": "PretrainedConfig"},
                {"type": "string"},
                {"type": "null"},
            ]
        }

    def test_parameter_empty_emits_empty_with_description(self) -> None:
        result = _common.annotation_to_json_schema(inspect.Parameter.empty)
        # No 'type' key (any-shape) and a description noting opacity.
        assert "type" not in result
        assert "no annotation" in result["description"]

    def test_pydantic_class_emits_ref_when_defs_acc_provided(self) -> None:
        """A bare Pydantic class annotation (post-#540) emits ``$ref`` and
        populates the accumulator with the class's full schema.
        Module-scope fixture used so we don't depend on test-local class
        scoping (irrelevant for this direct-call test, but consistent
        with the dataclass-walker tests below)."""
        defs_acc: dict[str, _common.Any] = {}
        result = _common.annotation_to_json_schema(
            _FixtureKvCacheConfig, defs_acc=defs_acc
        )
        assert result == {"$ref": "#/$defs/_FixtureKvCacheConfig"}
        assert "_FixtureKvCacheConfig" in defs_acc

    def test_pydantic_class_falls_back_without_defs_acc(self) -> None:
        """Legacy path: without an accumulator, a Pydantic class collapses
        to an opaque object schema with the class name as description.
        Producers that don't opt in get pre-#540 behaviour."""
        result = _common.annotation_to_json_schema(_FixtureKvCacheConfig)
        assert result == {
            "type": "object",
            "description": "_FixtureKvCacheConfig",
        }


# ---------------------------------------------------------------------------
# dataclass_fields_to_specs (canonical output)
# ---------------------------------------------------------------------------


class TestDataclassFieldsToSpecs:
    def test_canonical_types_emitted(self) -> None:
        @dataclasses.dataclass
        class M:
            x: int = 0
            y: str | None = None
            z: bool = True

        specs = _common.dataclass_fields_to_specs(M)
        assert specs["x"] == {"type": "integer", "default": 0}
        assert specs["y"] == {"type": ["string", "null"], "default": None}
        assert specs["z"] == {"type": "boolean", "default": True}

    def test_literal_field_lifts_enum(self) -> None:
        @dataclasses.dataclass
        class M:
            color: Literal["red", "blue"] = "red"

        specs = _common.dataclass_fields_to_specs(M)
        assert specs["color"] == {
            "type": "string",
            "enum": ["red", "blue"],
            "default": "red",
        }

    def test_default_factory_is_evaluated(self) -> None:
        @dataclasses.dataclass
        class M:
            tags: list[str] = dataclasses.field(default_factory=list)

        specs = _common.dataclass_fields_to_specs(M)
        assert specs["tags"] == {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        }

    def test_skip_private(self) -> None:
        @dataclasses.dataclass
        class M:
            public: int = 0
            _private: int = 0

        skipped = _common.dataclass_fields_to_specs(M, skip_private=True)
        kept = _common.dataclass_fields_to_specs(M, skip_private=False)
        assert "_private" not in skipped
        assert "_private" in kept

    def test_pydantic_typed_field_surfaces_as_ref_when_defs_acc_provided(self) -> None:
        """A stdlib dataclass with a Pydantic-typed field (the vllm
        ``EngineArgs`` pattern) emits ``$ref`` and populates ``defs_acc``
        when the accumulator is provided. The fixture classes are
        module-level so ``typing.get_type_hints`` can resolve them through
        PEP 563 stringification."""
        defs_acc: dict[str, _common.Any] = {}
        specs = _common.dataclass_fields_to_specs(_FixtureEngineArgs, defs_acc=defs_acc)
        # The field surfaces as anyOf [$ref, null] (the Optional shape).
        assert specs["kv_cache_config"] == {
            "anyOf": [{"$ref": "#/$defs/_FixtureKvCacheConfig"}, {"type": "null"}],
            "default": None,
        }
        # Accumulator populated with the Pydantic class's schema.
        assert "_FixtureKvCacheConfig" in defs_acc
        assert defs_acc["_FixtureKvCacheConfig"]["type"] == "object"
        assert set(defs_acc["_FixtureKvCacheConfig"]["properties"]) == {
            "enable_block_reuse",
            "dtype",
        }

    def test_pydantic_typed_field_legacy_fallback_without_defs_acc(self) -> None:
        """When ``defs_acc`` is None (legacy path), a Pydantic-typed field
        falls back to ``{type: object, description: ClassName}``. This is
        the pre-#540 behaviour; producers that don't opt in get the same
        output as before."""
        specs = _common.dataclass_fields_to_specs(_FixtureEngineArgs)
        # No $ref; opaque object with class-name hint.
        assert specs["kv_cache_config"] == {
            "anyOf": [
                {"type": "object", "description": "_FixtureKvCacheConfig"},
                {"type": "null"},
            ],
            "default": None,
        }

    def test_pydantic_recursion_captures_transitive_defs(self) -> None:
        """When a Pydantic class itself references other Pydantic classes,
        ``model_json_schema()`` emits its own ``$defs`` block; the walker
        flattens those into ``defs_acc`` so the envelope has every
        transitively-referenced sub-class."""
        defs_acc: dict[str, _common.Any] = {}
        _common.dataclass_fields_to_specs(_FixtureNestedArgs, defs_acc=defs_acc)
        # Both Outer and its transitive Inner land in defs_acc.
        assert "_FixtureOuter" in defs_acc
        assert "_FixtureInner" in defs_acc

    def test_resolves_string_annotations_via_get_type_hints(self) -> None:
        """PEP 563 / ``from __future__ import annotations`` makes
        ``fld.type`` a literal string. The walker uses
        ``typing.get_type_hints`` to resolve to actual classes so the
        Pydantic-detection branch fires for forward-annotated fields.
        This whole test file uses PEP 563; if resolution failed we'd see
        opaque-object output with no ``$ref``."""
        defs_acc: dict[str, _common.Any] = {}
        specs = _common.dataclass_fields_to_specs(_FixtureForwardArgs, defs_acc=defs_acc)
        assert specs["conf"]["anyOf"][0] == {"$ref": "#/$defs/_FixtureForwardConf"}
        assert "_FixtureForwardConf" in defs_acc


# ---------------------------------------------------------------------------
# signature_param_to_spec
# ---------------------------------------------------------------------------


class TestSignatureParamToSpec:
    def test_typed_param(self) -> None:
        def f(x: int = 5) -> None: ...

        param = inspect.signature(f).parameters["x"]
        assert _common.signature_param_to_spec(param) == {"type": "integer", "default": 5}

    def test_optional_param(self) -> None:
        def f(x: str | None = None) -> None: ...

        param = inspect.signature(f).parameters["x"]
        assert _common.signature_param_to_spec(param) == {
            "type": ["string", "null"],
            "default": None,
        }

    def test_untyped_param(self) -> None:
        def f(x=42) -> None: ...

        param = inspect.signature(f).parameters["x"]
        spec = _common.signature_param_to_spec(param)
        # No annotation -> no 'type'; default still surfaces.
        assert "type" not in spec
        assert spec["default"] == 42

    def test_no_default(self) -> None:
        def f(x: int) -> None: ...

        param = inspect.signature(f).parameters["x"]
        assert _common.signature_param_to_spec(param) == {"type": "integer", "default": None}


# ---------------------------------------------------------------------------
# runtime_value_to_spec (used by transformers GenerationConfig walker)
# ---------------------------------------------------------------------------


class TestRuntimeValueToSpec:
    def test_int_value(self) -> None:
        assert _common.runtime_value_to_spec(5) == {"type": "integer", "default": 5}

    def test_str_value(self) -> None:
        assert _common.runtime_value_to_spec("hi") == {"type": "string", "default": "hi"}

    def test_bool_value(self) -> None:
        assert _common.runtime_value_to_spec(True) == {"type": "boolean", "default": True}

    def test_list_value(self) -> None:
        assert _common.runtime_value_to_spec([1, 2]) == {
            "type": "array",
            "default": [1, 2],
        }

    def test_dict_value(self) -> None:
        assert _common.runtime_value_to_spec({"a": 1}) == {
            "type": "object",
            "default": {"a": 1},
        }

    def test_none_value_has_no_type(self) -> None:
        """None default with no annotation -> no 'type' (untyped)."""
        spec = _common.runtime_value_to_spec(None)
        assert "type" not in spec
        assert spec["default"] is None
        assert "no type annotation" in spec["description"]


# ---------------------------------------------------------------------------
# jsonschema_property_to_canonical (msgspec / Pydantic JSON Schema cleanup)
# ---------------------------------------------------------------------------


class TestJsonschemaPropertyToCanonical:
    def test_primitive_passthrough(self) -> None:
        result = _common.jsonschema_property_to_canonical({"type": "integer", "default": 0})
        assert result == {"type": "integer", "default": 0}

    def test_title_dropped(self) -> None:
        result = _common.jsonschema_property_to_canonical({"type": "integer", "title": "My Field"})
        assert result == {"type": "integer"}

    def test_anyof_with_null_collapses_to_type_array(self) -> None:
        result = _common.jsonschema_property_to_canonical(
            {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}
        )
        assert result == {"type": ["string", "null"], "default": None}

    def test_anyof_multi_branch_preserved(self) -> None:
        spec = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert _common.jsonschema_property_to_canonical(spec) == spec

    def test_anyof_with_ref_not_collapsed(self) -> None:
        """anyOf with $ref branch keeps the anyOf shape (not a simple primitive)."""
        spec = {"anyOf": [{"$ref": "#/$defs/Foo"}, {"type": "null"}]}
        assert _common.jsonschema_property_to_canonical(spec) == spec

    def test_x_source_passes_through(self) -> None:
        """PR-0.5 extension keys (x-source, x-source-ref, enum) survive intact."""
        spec = {"type": "string", "enum": ["a", "b"], "x-source": "validation_collection"}
        assert _common.jsonschema_property_to_canonical(spec) == spec


# ---------------------------------------------------------------------------
# Per-engine LANDMARKS contracts
# ---------------------------------------------------------------------------
#
# The drift tool (``scripts._drift``) reads ``LANDMARKS`` from each
# producer module and resolves every dotted path against the installed
# library; a single missing landmark flips the probe verdict to ``fail``
# and skips downstream discovery. These tests exercise the same
# resolution logic for the transformers introspector so a landmark-name
# drift in upstream transformers is caught at unit-test time rather than
# only on the next drift-tool run.


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


# ---------------------------------------------------------------------------
# discover_validation_collections
# ---------------------------------------------------------------------------
#
# Two-pass AST walker (collect module-scope value-set constants; then verify
# each is actually used in a validator-body ``in`` test). Tests cover the
# four supported literal shapes, the validator detection axes, the false-
# positive guard, the output schema, and the field-name attribution rules.


def test_discover_lifts_constant_used_in_membership_check(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        from __future__ import annotations

        ALLOWED = ("a", "b", "c")

        class Cfg:
            def __post_init__(self) -> None:
                if self.kind not in ALLOWED:
                    raise ValueError("bad")
        """,
    )

    out = _common.discover_validation_collections(module)
    assert "kind" in out
    assert out["kind"]["enum"] == ["a", "b", "c"]
    assert out["kind"]["x-source"] == "module_validation_collection"
    assert out["kind"]["x-source-ref"].endswith(".ALLOWED")


def test_discover_handles_positive_in_test(tmp_path: Path) -> None:
    # ``if v in CONST`` (positive) should lift identically to ``not in``.
    module = _load_synthetic_module(
        tmp_path,
        """
        ALLOWED = ("a", "b")

        class Cfg:
            def validate(self) -> None:
                if self.mode in ALLOWED:
                    return
                raise ValueError("bad mode")
        """,
    )
    out = _common.discover_validation_collections(module)
    assert "mode" in out
    assert out["mode"]["enum"] == ["a", "b"]


def test_discover_skips_constant_with_no_membership_reference(tmp_path: Path) -> None:
    # The load-bearing false-positive guard. ``_VALID_X`` looks like a
    # validation collection by name but is never referenced in a validator
    # ``in`` test, so the walker must NOT lift it.
    module = _load_synthetic_module(
        tmp_path,
        """
        _VALID_THINGS = ("a", "b", "c")
        ANOTHER_CONST = {1, 2, 3}

        class Cfg:
            def __post_init__(self) -> None:
                if self.x is None:
                    raise ValueError("missing")
        """,
    )
    out = _common.discover_validation_collections(module)
    assert out == {}


def test_discover_handles_set_frozenset_tuple_list(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        AS_SET = {"a", "b"}
        AS_FROZENSET = frozenset({"c", "d"})
        AS_TUPLE = ("e", "f")
        AS_LIST = ["g", "h"]

        class Cfg:
            def __post_init__(self) -> None:
                if self.s not in AS_SET:
                    raise ValueError()
                if self.fs not in AS_FROZENSET:
                    raise ValueError()
                if self.t not in AS_TUPLE:
                    raise ValueError()
                if self.lst not in AS_LIST:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    # Sets are sorted-and-deduped (deterministic enum); tuple/list preserve order.
    assert out["s"]["enum"] == ["a", "b"]
    assert out["fs"]["enum"] == ["c", "d"]
    assert out["t"]["enum"] == ["e", "f"]
    assert out["lst"]["enum"] == ["g", "h"]


def test_discover_lifts_dict_keys_as_enum(tmp_path: Path) -> None:
    # Per the design: ``_STR_DTYPE_TO_TORCH_DTYPE`` keys are the valid dtype
    # strings; the values (torch.float16, etc.) are NOT what the user sets.
    module = _load_synthetic_module(
        tmp_path,
        """
        _STR_DTYPE_TO_TORCH_DTYPE = {
            "half": 1,
            "float16": 2,
            "bfloat16": 3,
        }

        class ModelConfig:
            def _verify_args(self) -> None:
                if self.dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                    raise ValueError("bad dtype")
        """,
    )
    out = _common.discover_validation_collections(module)
    assert out["dtype"]["enum"] == ["half", "float16", "bfloat16"]
    assert out["dtype"]["x-source-ref"].endswith("._STR_DTYPE_TO_TORCH_DTYPE")


def test_discover_module_with_no_validators_returns_empty(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        ALLOWED = ("a", "b")

        class Helper:
            def some_method(self) -> int:
                return 1
        """,
    )
    out = _common.discover_validation_collections(module)
    assert out == {}


def test_discover_module_with_no_constants_returns_empty(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        class Cfg:
            def __post_init__(self) -> None:
                if self.x is None:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    assert out == {}


def test_discover_multiple_constants_each_get_separate_entries(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        DTYPES = ("half", "float", "double")
        BACKENDS = {"a", "b"}

        class Cfg:
            def validate(self) -> None:
                if self.dtype not in DTYPES:
                    raise ValueError("bad dtype")
                if self.backend not in BACKENDS:
                    raise ValueError("bad backend")
        """,
    )
    out = _common.discover_validation_collections(module)
    assert set(out) == {"dtype", "backend"}
    assert out["dtype"]["enum"] == ["half", "float", "double"]
    assert out["backend"]["enum"] == ["a", "b"]
    assert out["dtype"]["x-source-ref"].endswith(".DTYPES")
    assert out["backend"]["x-source-ref"].endswith(".BACKENDS")


def test_discover_pydantic_field_validator(tmp_path: Path) -> None:
    # Field name from ``@field_validator("dtype")`` rather than ``self.<x>``;
    # the function body uses ``v`` as the value-under-test.
    module = _load_synthetic_module(
        tmp_path,
        """
        SUPPORTED_DTYPES = ("half", "float", "double")

        def field_validator(*args, **kwargs):
            def deco(fn):
                fn.__validator_args__ = args
                return fn
            return deco

        class Cfg:
            @field_validator("dtype")
            def _check_dtype(cls, v):
                if v not in SUPPORTED_DTYPES:
                    raise ValueError("bad dtype")
                return v
        """,
    )
    out = _common.discover_validation_collections(module)
    assert "dtype" in out
    assert out["dtype"]["enum"] == ["half", "float", "double"]


def test_discover_local_alias_chain(tmp_path: Path) -> None:
    # ``local = self.<field>`` then ``if local not in CONST`` -> attribution
    # walks the alias chain back to ``<field>``.
    module = _load_synthetic_module(
        tmp_path,
        """
        VALID = ("a", "b")

        class Cfg:
            def validate(self) -> None:
                fmt = self.format
                if fmt not in VALID:
                    raise ValueError("bad fmt")
        """,
    )
    out = _common.discover_validation_collections(module)
    assert "format" in out
    assert out["format"]["enum"] == ["a", "b"]


def test_discover_concatenated_constants(tmp_path: Path) -> None:
    # Transformers' ALL_CACHE_IMPLEMENTATIONS pattern: literal tuples are
    # concatenated at module scope before being used as the gate.
    module = _load_synthetic_module(
        tmp_path,
        """
        STATIC = ("static", "offloaded_static")
        DYNAMIC = ("dynamic", "quantized")
        ALL = STATIC + DYNAMIC

        class Cfg:
            def validate(self) -> None:
                if self.cache_impl not in ALL:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    assert "cache_impl" in out
    assert out["cache_impl"]["enum"] == [
        "static",
        "offloaded_static",
        "dynamic",
        "quantized",
    ]


def test_discover_skips_constants_referenced_outside_validators(tmp_path: Path) -> None:
    # ``in CONST`` used inside a non-validator helper does NOT count -- the
    # walker is intentionally conservative about what counts as enforcement.
    module = _load_synthetic_module(
        tmp_path,
        """
        ALLOWED = ("a", "b")

        class Cfg:
            def helper(self, x):
                # Not a validator method by name or decorator
                return x in ALLOWED
        """,
    )
    out = _common.discover_validation_collections(module)
    assert out == {}


def test_discover_one_constant_multiple_fields(tmp_path: Path) -> None:
    # When the same constant gates two distinct fields, both get an entry
    # that share the ``x-source-ref``.
    module = _load_synthetic_module(
        tmp_path,
        """
        DTYPES = ("half", "float")

        class Cfg:
            def validate(self) -> None:
                if self.dtype not in DTYPES:
                    raise ValueError()
                if self.head_dtype not in DTYPES:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    assert set(out) == {"dtype", "head_dtype"}
    assert out["dtype"]["x-source-ref"] == out["head_dtype"]["x-source-ref"]


def test_discover_output_schema_shape(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        OPTS = ("x", "y")

        class Cfg:
            def __post_init__(self) -> None:
                if self.opt not in OPTS:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    spec = out["opt"]
    # Required keys per the design.
    assert set(spec) >= {"enum", "x-source", "x-source-ref"}
    assert spec["x-source"] == "module_validation_collection"
    assert isinstance(spec["enum"], list)
    assert spec["x-source-ref"].endswith(".OPTS")


def test_discover_handles_verify_prefix_methods(tmp_path: Path) -> None:
    # vLLM convention: ``_verify_<thing>``.
    module = _load_synthetic_module(
        tmp_path,
        """
        SUPPORTED = ("auto", "fp8")

        class Cfg:
            def _verify_dtype(self) -> None:
                if self.kv_dtype not in SUPPORTED:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    assert "kv_dtype" in out


def test_discover_handles_validate_prefix_methods(tmp_path: Path) -> None:
    module = _load_synthetic_module(
        tmp_path,
        """
        OPTIONS = {"a", "b"}

        class Cfg:
            def validate_thing(self) -> None:
                if self.thing not in OPTIONS:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    assert "thing" in out


def test_merge_validation_collections_into_existing_specs() -> None:
    field_specs = {
        "dtype": {"type": "str", "default": "auto"},
        "other": {"type": "int", "default": 1},
    }
    collections = {
        "dtype": {
            "enum": ["half", "float"],
            "x-source": "module_validation_collection",
            "x-source-ref": "vllm.config.model._STR_DTYPE_TO_TORCH_DTYPE",
        },
    }
    merged = _common.merge_validation_collections(field_specs, collections)
    assert merged is field_specs  # in-place
    assert merged["dtype"]["type"] == "str"
    assert merged["dtype"]["default"] == "auto"
    assert merged["dtype"]["enum"] == ["half", "float"]
    assert merged["dtype"]["x-source"] == "module_validation_collection"
    assert merged["dtype"]["x-source-ref"].endswith("._STR_DTYPE_TO_TORCH_DTYPE")
    # Untouched fields stay as-is
    assert merged["other"] == {"type": "int", "default": 1}


def test_merge_validation_collections_records_unknown_fields() -> None:
    # A constant the walker found whose field isn't in field_specs (e.g. the
    # introspector skipped it) is still recorded so the information isn't lost.
    field_specs: dict[str, dict] = {}
    collections = {
        "novel_field": {
            "enum": ["a", "b"],
            "x-source": "module_validation_collection",
            "x-source-ref": "mod.CONST",
        },
    }
    merged = _common.merge_validation_collections(field_specs, collections)
    assert "novel_field" in merged
    assert merged["novel_field"]["enum"] == ["a", "b"]


def test_discover_handles_dict_constructor_call(tmp_path: Path) -> None:
    # ``X = dict(a=1, b=2)`` should lift KEYS as the enum.
    module = _load_synthetic_module(
        tmp_path,
        """
        MAP = dict({"a": 1, "b": 2})

        class Cfg:
            def validate(self) -> None:
                if self.k not in MAP:
                    raise ValueError()
        """,
    )
    out = _common.discover_validation_collections(module)
    assert out["k"]["enum"] == ["a", "b"]


def test_discover_skips_module_with_no_extractable_source(tmp_path: Path, monkeypatch) -> None:
    # ``inspect.getsource`` raises ``OSError`` for builtin / C-extension
    # modules. The walker must return ``{}`` rather than blow up.
    module = _load_synthetic_module(
        tmp_path,
        """
        ALLOWED = ("a",)
        class Cfg:
            def __post_init__(self):
                if self.x not in ALLOWED:
                    raise ValueError()
        """,
    )

    def _raise(_module):
        raise OSError("synthetic")

    monkeypatch.setattr(_common.inspect, "getsource", _raise)
    assert _common.discover_validation_collections(module) == {}
