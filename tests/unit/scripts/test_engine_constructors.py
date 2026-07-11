"""Host-side tests for scripts/_engine_constructors.py (the verification-ladder arbiter).

These run on a plain host with no engine installed. Every test exercises the
arbiter's own logic - class routing, constructor-signature acceptance,
construction so field validation fires, and required-arg scaffolding - against
small dataclass / pydantic / msgspec stand-in classes, never a real engine.
"""

from __future__ import annotations

import dataclasses
import sys
import typing
from pathlib import Path

import pydantic
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

import _engine_constructors as ec

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_section_of() -> None:
    assert ec.section_of("vllm.engine_params.all2all_backend") == "engine_params"
    assert ec.section_of("vllm") == ""


def test_leaf_of() -> None:
    assert ec.leaf_of("vllm.sampling_params.frequency_penalty") == "frequency_penalty"
    assert ec.leaf_of("bare") == "bare"


# ---------------------------------------------------------------------------
# accepts() - constructor-signature acceptance
# ---------------------------------------------------------------------------


class _HasFoo:
    def __init__(self, foo: int) -> None:
        self.foo = foo


class _HasKwargs:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _HasNothing:
    def __init__(self) -> None:
        pass


def test_accepts_named_param() -> None:
    assert ec.accepts(_HasFoo, "foo") is True


def test_accepts_var_keyword_catchall() -> None:
    # A **kwargs constructor accepts any leaf.
    assert ec.accepts(_HasKwargs, "anything") is True


def test_accepts_rejects_absent_param() -> None:
    assert ec.accepts(_HasNothing, "foo") is False


# ---------------------------------------------------------------------------
# construct() - build so the class's own validation fires
# ---------------------------------------------------------------------------


class _PydModel(pydantic.BaseModel):
    val: int


class _PydModelWithRequiredSibling(pydantic.BaseModel):
    val: int
    other: int  # required, not supplied by the probe kwargs


def test_construct_generic_pydantic() -> None:
    obj = ec.construct("vllm", _PydModel, {"val": 7}, validate=False)
    assert obj.val == 7


def test_construct_scaffolds_required_sibling() -> None:
    # `other` is required but absent from kwargs; the scaffold supplies a
    # neutral stand-in so construction reaches the field under test.
    obj = ec.construct("vllm", _PydModelWithRequiredSibling, {"val": 7}, validate=False)
    assert obj.val == 7
    assert obj.other == 1


class _FakeGenerationConfig:
    """Stand-in for transformers GenerationConfig (construction != validation)."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.validated_strict: bool | None = None

    def validate(self, strict: bool = False) -> None:
        self.validated_strict = strict


def test_construct_transformers_validate_true_calls_validate() -> None:
    obj = ec.construct("transformers", _FakeGenerationConfig, {"temperature": 0.1}, validate=True)
    assert obj.validated_strict is True


def test_construct_transformers_validate_false_skips_validate() -> None:
    obj = ec.construct("transformers", _FakeGenerationConfig, {"temperature": 0.1}, validate=False)
    assert obj.validated_strict is None


@dataclasses.dataclass
class _FakeTrtLlmArgs:
    model: str = ""
    max_beam_width: int = 1


def test_construct_tensorrt_injects_model_placeholder() -> None:
    obj = ec.construct("tensorrt", _FakeTrtLlmArgs, {"max_beam_width": 3}, validate=False)
    assert obj.model == ec._TRTLLM_MODEL_PLACEHOLDER
    assert obj.max_beam_width == 3


def test_construct_msgspec_struct() -> None:
    msgspec = pytest.importorskip("msgspec")

    # defstruct builds the Struct type at runtime, so this test exercises the
    # msgspec.convert construction path without a static subclass declaration.
    struct_cls = msgspec.defstruct("_ProbeStruct", [("val", int)])
    obj = ec.construct("vllm", struct_cls, {"val": 9}, validate=False)
    assert obj.val == 9


# ---------------------------------------------------------------------------
# resolved_value() - MISSING sentinel
# ---------------------------------------------------------------------------


def test_resolved_value_present() -> None:
    obj = _PydModel(val=5)
    assert ec.resolved_value(obj, "val") == 5


def test_resolved_value_absent_returns_missing() -> None:
    obj = _PydModel(val=5)
    assert ec.resolved_value(obj, "nope") is ec.MISSING


# ---------------------------------------------------------------------------
# candidate_classes() - routing / unprobeable classification
# ---------------------------------------------------------------------------


def test_candidate_classes_unknown_engine_section_raises() -> None:
    with pytest.raises(ec.ConstructorResolutionError):
        ec.candidate_classes("no-such-engine", "no-such-engine.section.leaf", ["leaf"])


# ---------------------------------------------------------------------------
# engine_importable() - infra fact, not a rule verdict
# ---------------------------------------------------------------------------


def test_engine_importable_unknown_engine_is_false() -> None:
    assert ec.engine_importable("does-not-exist") is False


def test_engine_importable_true_for_importable_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # Point a faux engine at a module that is always importable on the host.
    monkeypatch.setitem(ec._ROOT_MODULE, "faux", "json")
    assert ec.engine_importable("faux") is True


# ---------------------------------------------------------------------------
# _scaffold_required() / _scaffold_value() - required-arg stand-ins
# ---------------------------------------------------------------------------


def test_scaffold_required_pydantic_supplies_missing_required_only() -> None:
    class _M(pydantic.BaseModel):
        x: int  # required
        y: int = 5  # has a default

    scaffold = ec._scaffold_required(_M, {})
    assert scaffold == {"x": 1}


def test_scaffold_required_skips_kwargs_provided_field() -> None:
    class _M(pydantic.BaseModel):
        x: int

    assert ec._scaffold_required(_M, {"x": 99}) == {}


@dataclasses.dataclass
class _DC:
    a: int  # required
    b: int = 3  # default
    c: list[int] = dataclasses.field(default_factory=list)  # default_factory


def test_scaffold_required_dataclass_prefers_default_factory() -> None:
    scaffold = ec._scaffold_required(_DC, {})
    assert scaffold == {"a": 1, "c": []}


def test_scaffold_value_builtins_and_literal() -> None:
    assert ec._scaffold_value(int) == 1
    assert ec._scaffold_value(str) == "x"
    assert ec._scaffold_value(bool) is False
    assert ec._scaffold_value(float) == 1.0
    assert ec._scaffold_value(typing.Literal["a", "b"]) == "a"


def test_scaffold_value_unknown_type_falls_back_to_one() -> None:
    assert ec._scaffold_value(_HasNothing) == 1


def test_scaffold_value_initvar_unwraps() -> None:
    assert ec._scaffold_value(dataclasses.InitVar(int)) == 1
