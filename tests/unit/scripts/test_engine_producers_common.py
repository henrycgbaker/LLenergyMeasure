"""Tests for ``scripts.engine_producers`` - common helpers only. - pure-Python helpers only.

Container-gated end-to-end discovery tests would use @pytest.mark.docker and
live in a separate test file if added later. This module tests the helpers
in :mod:`scripts.engine_producers._common` without requiring any engine
package.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

# Make the top-level ``scripts`` package importable from tests.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataclasses  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from scripts.engine_producers import _common  # noqa: E402

# ---------------------------------------------------------------------------
# Module-scope fixtures for the dataclass-into-pydantic recursion tests.
# Defined at module scope (not inside the test functions) so
# ``typing.get_type_hints`` can resolve the forward-referenced names.
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
        engine_version="1.2.1",
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
# dataclass_fields_to_specs - Pydantic-into-dataclass recursion ($defs)
# ---------------------------------------------------------------------------


def test_dataclass_specs_without_defs_flatten_pydantic_to_type_str() -> None:
    """Without a ``defs`` accumulator the walker keeps the legacy flat shape."""
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


def test_dataclass_specs_null_opaque_object_default_not_stringified() -> None:
    """A field whose default is an opaque object is recorded as None, never its
    repr string - else codegen would emit a bogus non-None default and forward
    that stringified blob to the engine on every unset run (the compilation_config
    regression). Pin the contract so the str() fallback can't return silently.
    """
    specs = _common.dataclass_fields_to_specs(_OpaqueDefaultArgs)
    assert specs["blob"]["default"] is None
    assert specs["plain"]["default"] == 0


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


def test_transformers_schema_introspector_landmarks_resolve() -> None:
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
# CPU-safety: the mining orchestration must import without torch
# ---------------------------------------------------------------------------


def test_mining_orchestration_import_is_torch_free() -> None:
    """Importing build_corpus + the static-miner shims must not pull torch.

    The mining orchestration runs on torchless CI runners (engine + torch deps
    live only inside each engine's Docker image); the real library import
    happens inside ``walk()``, in-container. A regression that imports torch at
    module load would crash the host-side dispatch before it can dispatch. Run
    in a fresh subprocess so the verdict is independent of what other tests have
    already imported - the old in-process ``post - pre`` check was order-flaky,
    only passing when a sibling test had already pre-imported torch.
    """
    probe = (
        "import sys\n"
        "import scripts.engine_producers.build_corpus\n"
        "import scripts.engine_producers.vllm_static_invariant_miner\n"
        "import scripts.engine_producers.tensorrt_static_invariant_miner\n"
        "import scripts.engine_producers.transformers_static_invariant_miner\n"
        "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "raise SystemExit('torch imported at module load: ' + repr(bad) if bad else 0)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
    }
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr).strip()
