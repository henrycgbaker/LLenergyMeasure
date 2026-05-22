"""Tests for ``scripts.engine_producers`` - common helpers only. - pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
in :mod:`scripts.engine_producers._common` without requiring any engine
package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

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
        discovery_method="unit test",
        discovery_limitations=[],
        engine_params={"a": {"type": "int", "default": 0}},
        sampling_params={"b": {"type": "str", "default": ""}},
    )
    assert env["schema_version"] == _common.SCHEMA_VERSION
    assert env["engine"] == "vllm"
    assert env["discovered_at"]  # ISO string
    assert "engine_params" in env and "sampling_params" in env


def test_schema_version_is_semver_at_supported_major() -> None:
    major = int(_common.SCHEMA_VERSION.split(".")[0])
    # The loader supports majors 1 and 2 (see SchemaLoader.SUPPORTED_MAJOR_VERSIONS);
    # bumping the introspector past 2 requires extending that allowlist.
    assert major in {1, 2}, "ships schema_version <=2.x; bumping major requires loader update"


# ---------------------------------------------------------------------------
# literal_to_json_schema
# ---------------------------------------------------------------------------


def test_literal_to_json_schema_string_literal() -> None:
    got = _common.literal_to_json_schema(Literal["a", "b", "c"])
    assert got == {"type": "string", "enum": ["a", "b", "c"]}


def test_literal_to_json_schema_int_literal() -> None:
    got = _common.literal_to_json_schema(Literal[1, 2, 3])
    assert got == {"type": "integer", "enum": [1, 2, 3]}


def test_literal_to_json_schema_bool_literal() -> None:
    # Bool checked first because bool is subclass of int in Python.
    got = _common.literal_to_json_schema(Literal[True, False])
    assert got == {"type": "boolean", "enum": [True, False]}


def test_literal_to_json_schema_enum_subclass() -> None:
    import enum as _enum

    class Color(_enum.Enum):
        RED = "red"
        BLUE = "blue"

    got = _common.literal_to_json_schema(Color)
    assert got == {"type": "string", "enum": ["red", "blue"]}


def test_literal_to_json_schema_returns_none_for_non_literal() -> None:
    assert _common.literal_to_json_schema(int) is None
    assert _common.literal_to_json_schema(str | None) is None
    assert _common.literal_to_json_schema(list[int]) is None


def test_literal_to_json_schema_mixed_types_falls_back_to_string() -> None:
    got = _common.literal_to_json_schema(Literal["one", 2])
    # Mixed types are rare; helper falls back to string + jsonable values.
    assert got is not None
    assert got["type"] == "string"
    assert got["enum"] == ["one", 2]


# ---------------------------------------------------------------------------
# pydantic_to_json_schema + pydantic_properties_to_specs
# ---------------------------------------------------------------------------


def test_pydantic_to_json_schema_preserves_constraints() -> None:
    from pydantic import BaseModel, Field

    class _Model(BaseModel):
        ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="A ratio")
        mode: Literal["fast", "slow"] = Field(default="fast")

    raw = _common.pydantic_to_json_schema(_Model)
    props = raw["properties"]
    assert props["ratio"]["minimum"] == 0.0
    assert props["ratio"]["maximum"] == 1.0
    assert props["ratio"]["description"] == "A ratio"
    assert props["ratio"]["x-source"] == "pydantic_field"
    assert props["mode"]["enum"] == ["fast", "slow"]


def test_pydantic_to_json_schema_emits_defs_for_nested_models() -> None:
    from pydantic import BaseModel

    class _Sub(BaseModel):
        inner: int = 1

    class _Outer(BaseModel):
        child: _Sub | None = None

    raw = _common.pydantic_to_json_schema(_Outer)
    assert "$defs" in raw
    assert "_Sub" in raw["$defs"]
    assert raw["$defs"]["_Sub"]["properties"]["inner"]["type"] == "integer"


def test_pydantic_to_json_schema_rejects_non_pydantic() -> None:
    class _Plain:
        x: int = 0

    import pytest as _pytest  # ensure no cross-import noise

    with _pytest.raises(TypeError, match="not a Pydantic"):
        _common.pydantic_to_json_schema(_Plain)


def test_pydantic_properties_to_specs_preserves_legacy_type_string() -> None:
    from pydantic import BaseModel

    class _Inner(BaseModel):
        v: int = 0

    class _M(BaseModel):
        child: _Inner | None = None
        flag: bool = False
        mode: Literal["a", "b"] = "a"

    props, defs = _common.pydantic_properties_to_specs(_M)
    # legacy compact type string preserved
    assert props["child"]["type"] == "_Inner | None"
    assert props["flag"]["type"] == "boolean"
    assert props["mode"]["type"] == "string"
    # enum carried alongside the type string
    assert props["mode"]["enum"] == ["a", "b"]
    # default backfill when Pydantic omits it
    for spec in props.values():
        assert "default" in spec, "v2 contract: every leaf carries default"
        assert "deprecated" in spec, "v2 contract: every leaf carries deprecated"
    # nested $defs surfaced
    assert "_Inner" in defs


def test_pydantic_properties_skip_private_default() -> None:
    from pydantic import BaseModel

    class _M(BaseModel):
        x: int = 1
        # Pydantic doesn't typically expose ``_y`` in schema, but the helper
        # still honours the skip_private flag at the property level.

    props, _ = _common.pydantic_properties_to_specs(_M)
    assert "x" in props


# ---------------------------------------------------------------------------
# msgspec_properties_to_specs
# ---------------------------------------------------------------------------


def test_msgspec_properties_to_specs_flattens_ref_envelope() -> None:
    msgspec = pytest.importorskip("msgspec")

    class _M(msgspec.Struct):
        x: int = 1
        mode: str = "auto"

    props, defs = _common.msgspec_properties_to_specs(_M)
    assert "x" in props
    assert props["x"]["type"] == "integer"
    assert props["x"]["x-source"] == "msgspec_field"
    # defs envelope is preserved (msgspec wraps in $ref by default)
    assert defs


# ---------------------------------------------------------------------------
# dataclass_fields_to_specs enrichment
# ---------------------------------------------------------------------------


def test_dataclass_fields_surfaces_literal_enum() -> None:
    # Build the dataclass via dataclasses.make_dataclass so the type
    # annotations stay as live typing objects rather than the stringified
    # form ``from __future__ import annotations`` produces at module top.
    # The helper only resolves Literal/Enum from live annotations; engines
    # like vLLM's EngineArgs use that shape in practice.
    import dataclasses

    _D = dataclasses.make_dataclass(
        "_D",
        [
            ("mode", Literal["fast", "slow"], dataclasses.field(default="fast")),
            ("count", int, dataclasses.field(default=0)),
        ],
    )

    specs = _common.dataclass_fields_to_specs(_D)
    assert specs["mode"]["enum"] == ["fast", "slow"]
    # Legacy compact ``type`` string is preserved (v1 backward compat);
    # the JSON Schema primitive sits alongside under ``json_schema_type``.
    assert specs["mode"]["type"] == "Literal['fast', 'slow']"
    assert specs["mode"]["json_schema_type"] == "string"
    assert specs["count"].get("enum") is None
    # All leaves carry x-source provenance.
    assert specs["mode"]["x-source"] == "dataclass_field"
    assert specs["count"]["x-source"] == "dataclass_field"


def test_dataclass_fields_surfaces_description_metadata() -> None:
    import dataclasses

    _D = dataclasses.make_dataclass(
        "_D",
        [
            (
                "name",
                str,
                dataclasses.field(default="anon", metadata={"description": "The user's name"}),
            ),
            ("age", int, dataclasses.field(default=0)),
        ],
    )

    specs = _common.dataclass_fields_to_specs(_D)
    assert specs["name"]["description"] == "The user's name"
    assert "description" not in specs["age"]


# ---------------------------------------------------------------------------
# signature_param_to_spec
# ---------------------------------------------------------------------------


def test_signature_param_to_spec_basic() -> None:
    # Build via inspect.Parameter directly so annotations stay as live
    # typing objects rather than the stringified form the file-level
    # ``from __future__ import annotations`` produces.
    import inspect as _inspect

    x_param = _inspect.Parameter(
        "x",
        kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=5,
        annotation=int,
    )
    mode_param = _inspect.Parameter(
        "mode",
        kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default="a",
        annotation=Literal["a", "b"],
    )

    x_spec = _common.signature_param_to_spec(x_param)
    assert x_spec == {
        "type": "int",
        "default": 5,
        "x-source": "signature_param",
    }
    mode_spec = _common.signature_param_to_spec(mode_param)
    assert mode_spec["enum"] == ["a", "b"]
    # Legacy compact ``type`` string preserved (v1 backward compat).
    assert mode_spec["type"] == "Literal['a', 'b']"
    assert mode_spec["json_schema_type"] == "string"


# ---------------------------------------------------------------------------
# compact_type_from_jsonschema
# ---------------------------------------------------------------------------


def test_compact_type_from_jsonschema_ref() -> None:
    spec = {"$ref": "#/$defs/Foo"}
    assert _common.compact_type_from_jsonschema(spec) == "Foo"


def test_compact_type_from_jsonschema_anyof_with_null() -> None:
    spec = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert _common.compact_type_from_jsonschema(spec) == "integer | None"


def test_compact_type_from_jsonschema_anyof_with_ref() -> None:
    spec = {"anyOf": [{"$ref": "#/$defs/Foo"}, {"type": "null"}]}
    assert _common.compact_type_from_jsonschema(spec) == "Foo | None"


def test_compact_type_from_jsonschema_plain() -> None:
    assert _common.compact_type_from_jsonschema({"type": "string"}) == "string"
    assert _common.compact_type_from_jsonschema({"type": "null"}) == "None"
    assert _common.compact_type_from_jsonschema({}) == "unknown"


# ---------------------------------------------------------------------------
# make_envelope optional defs
# ---------------------------------------------------------------------------


def test_envelope_omits_empty_defs_keys() -> None:
    env = _common.make_envelope(
        engine="vllm",
        engine_version="x",
        engine_commit_sha=None,
        image_ref="x",
        base_image_ref="x",
        discovery_method="t",
        discovery_limitations=[],
        engine_params={},
        sampling_params={},
    )
    assert "engine_params_defs" not in env
    assert "sampling_params_defs" not in env


def test_envelope_includes_defs_when_present() -> None:
    env = _common.make_envelope(
        engine="vllm",
        engine_version="x",
        engine_commit_sha=None,
        image_ref="x",
        base_image_ref="x",
        discovery_method="t",
        discovery_limitations=[],
        engine_params={},
        sampling_params={},
        engine_params_defs={"Foo": {"type": "object"}},
    )
    assert env["engine_params_defs"] == {"Foo": {"type": "object"}}


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

    from scripts.engine_producers import transformers_introspector

    landmarks = transformers_introspector.LANDMARKS
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
