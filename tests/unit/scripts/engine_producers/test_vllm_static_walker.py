"""Host unit tests for the vLLM static walker's pure-AST primitives.

These exercise the walker functions that operate on AST + type strings only, so
they run on a CPU host with no vLLM installed (the module imports without it;
only walk_vllm_static's _check_landmarks touches the live library).

Covers:
- call-following into a same-class validator helper (the V7 fix: vLLM relocated
  the max_num_batched_tokens < max_model_len raise out of __post_init__ into
  verify_max_model_len, and the bound is an InitVar parameter);
- bare-name -> @field-ref resolution for config fields / InitVars;
- the type-aware dormancy negative-probe synthesiser (_minimal_value_for_type).
"""

from __future__ import annotations

import ast

from engine_versions.vllm.v0_19_1.producers import static_invariant_miner as sm


def _expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


# ---------------------------------------------------------------------------
# call-following + bare-name field-ref (the V7 fix)
# ---------------------------------------------------------------------------


def test_self_method_name_detects_self_calls() -> None:
    assert sm._self_method_name(_expr("self.verify_max_model_len(x)")) == "verify_max_model_len"
    assert sm._self_method_name(_expr("other.method(x)")) is None
    assert sm._self_method_name(_expr("plain_func(x)")) is None
    assert sm._self_method_name(_expr("self.attr")) is None  # not a call


def test_rhs_value_resolves_bare_field_ref() -> None:
    name = _expr("max_model_len")
    # a bare name that is a known field / InitVar -> @field-ref
    assert sm._rhs_value(name, frozenset({"max_model_len"})) == (True, "@max_model_len")
    # a bare name not in the field set is unresolved (dropped)
    assert sm._rhs_value(name, frozenset()) == (False, None)
    # self.<attr> stays an @field-ref; a literal stays a literal
    assert sm._rhs_value(_expr("self.max_model_len"), frozenset()) == (True, "@max_model_len")
    assert sm._rhs_value(_expr("0"), frozenset()) == (True, 0)


def test_extract_compare_resolves_initvar_comparand() -> None:
    """self.<field> < <InitVar param> yields a cross-field @ref predicate."""
    cmp = _expr("self.max_num_batched_tokens < max_model_len")
    assert isinstance(cmp, ast.Compare)
    preds = sm._extract_compare(cmp, frozenset({"max_num_batched_tokens", "max_model_len"}))
    assert [(p.field, p.op, p.rhs) for p in preds] == [
        ("max_num_batched_tokens", "<", "@max_model_len")
    ]


def test_walk_function_follows_same_class_helper() -> None:
    """Walking __post_init__ recovers a raise relocated into a sibling helper.

    Mirrors vLLM 0.19's SchedulerConfig: __post_init__ delegates to
    verify_max_model_len, whose raise compares self.max_num_batched_tokens
    against the max_model_len InitVar parameter.
    """
    src = """
class SchedulerConfig:
    def __post_init__(self, max_model_len):
        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len):
        if self.max_num_batched_tokens < max_model_len:
            raise ValueError("max_num_batched_tokens is smaller than max_model_len")
"""
    cls = ast.parse(src).body[0]
    assert isinstance(cls, ast.ClassDef)
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    target = sm._ASTTarget(
        module_attr="config.scheduler.SchedulerConfig",
        method="__post_init__",
        namespace="vllm.config.scheduler",
        native_type="vllm.config.SchedulerConfig",
    )
    candidates = sm._walk_function(
        methods["__post_init__"],
        target=target,
        field_names=frozenset({"max_num_batched_tokens", "max_model_len"}),
        class_methods=methods,
        rel_source_path="config/scheduler.py",
        today="2026-06-23",
    )
    assert len(candidates) == 1
    match = candidates[0].match_fields
    assert match == {"vllm.config.scheduler.max_num_batched_tokens": {"<": "@max_model_len"}}


def test_walk_function_does_not_recurse_unlisted_or_external_calls() -> None:
    """Only same-class self.<helper> calls are followed, and only once each."""
    src = """
class C:
    def m(self):
        external.helper()
        self.not_a_method()
        self.helper()

    def helper(self):
        if self.x < y:
            raise ValueError("boom")
"""
    cls = ast.parse(src).body[0]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    target = sm._ASTTarget(
        module_attr="config.scheduler.C",
        method="m",
        namespace="vllm.config.scheduler",
        native_type="vllm.config.C",
    )
    candidates = sm._walk_function(
        methods["m"],
        target=target,
        field_names=frozenset({"x", "y"}),
        class_methods=methods,  # not_a_method is absent -> not followed
        rel_source_path="config/scheduler.py",
        today="2026-06-23",
    )
    # self.helper() is followed (x < y -> one rule); the external + missing calls are inert.
    assert len(candidates) == 1


# ---------------------------------------------------------------------------
# type-aware dormancy negative-probe synthesiser (BUMP-4)
# ---------------------------------------------------------------------------


def test_minimal_value_for_type_covers_scalars_containers_unions_and_opaque() -> None:
    mv = sm._minimal_value_for_type
    # scalars
    assert mv("bool") == (True, True)
    assert mv("int") == (True, 1)
    assert mv("float") == (True, 1.0)
    assert mv("str") == (True, "x")
    assert mv("bytes") == (True, b"x")
    # containers
    assert mv("list[str]") == (True, ["x"])
    assert mv("list[int]") == (True, [1])
    assert mv("dict[str, int]") == (True, {"x": 1})
    assert mv("list[list[int]]") == (True, [[1]])
    # Literal -> first member
    assert mv("Literal['a', 'b']") == (True, "a")
    # unions collapse to the non-None arm
    assert mv("int | None") == (True, 1)
    assert mv("list[int] | None") == (True, [1])
    assert mv("str | None") == (True, "x")
    # genuinely-opaque types fall back so the caller keeps the generic probe
    assert mv("SamplingType") == (False, None)
    assert mv("set[int]") == (False, None)
    assert mv("tuple[int, ...]") == (False, None)
    assert mv("Any") == (False, None)
