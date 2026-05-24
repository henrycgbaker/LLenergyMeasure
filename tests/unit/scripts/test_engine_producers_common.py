"""Tests for ``scripts.engine_producers`` - common helpers only. - pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
in :mod:`scripts.engine_producers._common` without requiring any engine
package.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import sys
from pathlib import Path
from typing import Literal, Optional

import pytest

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
