"""Tests for ``scripts.engine_producers`` - common helpers only. - pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
in :mod:`scripts.engine_producers._common` without requiring any engine
package.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
import uuid
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

# Make the top-level ``scripts`` package importable from tests.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.engine_producers import _common  # noqa: E402


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
