"""Tests for :mod:`scripts.engine_producers._base`.

Covers AST primitives, class/method finders, and structured error types -
all on synthetic AST fixtures so the tests never depend on a specific
library version.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Make the top-level ``scripts`` package importable from tests.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._base import (  # noqa: E402
    MinerLandmarkMissingError,
    call_func_path,
    extract_assign_target,
    extract_condition_fields,
    extract_loop_literal_iterable,
    find_class,
    find_method,
    first_string_arg,
    render_binop_concat_template,
    resolve_local_assign,
    self_attr_name,
    self_attr_path,
)


def _parse_if(src: str) -> ast.If:
    module = ast.parse(src.strip())
    stmt = module.body[0]
    assert isinstance(stmt, ast.If)
    return stmt


def _parse_expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


# ---------------------------------------------------------------------------
# AST primitives
# ---------------------------------------------------------------------------


def test_call_func_path_logger_warning() -> None:
    call = _parse_expr('logger.warning("msg")')
    assert isinstance(call, ast.Call)
    assert call_func_path(call) == ["logger", "warning"]


def test_call_func_path_warnings_warn() -> None:
    call = _parse_expr('warnings.warn("msg")')
    assert isinstance(call, ast.Call)
    assert call_func_path(call) == ["warnings", "warn"]


def test_call_func_path_self_method() -> None:
    call = _parse_expr("self.helper(x)")
    assert isinstance(call, ast.Call)
    assert call_func_path(call) == ["self", "helper"]


def test_call_func_path_opaque_returns_none() -> None:
    # double-call like foo()() is not a pure attr chain
    call = _parse_expr("foo()()")
    assert isinstance(call, ast.Call)
    assert call_func_path(call) is None


def test_self_attr_name_flat_only() -> None:
    assert self_attr_name(_parse_expr("self.max_batch_size")) == "max_batch_size"
    # Nested attribute is not a flat self.<name>, so the flat helper returns None.
    assert self_attr_name(_parse_expr("self.build_config.max_batch_size")) is None
    assert self_attr_name(_parse_expr("other.max_batch_size")) is None


def test_self_attr_path_flat_and_nested() -> None:
    assert self_attr_path(_parse_expr("self.max_batch_size")) == ("max_batch_size",)
    assert self_attr_path(_parse_expr("self.build_config.max_batch_size")) == (
        "build_config",
        "max_batch_size",
    )
    # Three-deep chain rooted at self.
    assert self_attr_path(_parse_expr("self.a.b.c")) == ("a", "b", "c")


def test_self_attr_path_rejects_non_self() -> None:
    assert self_attr_path(_parse_expr("other.build_config.max_batch_size")) is None
    assert self_attr_path(_parse_expr("max_batch_size")) is None
    assert self_attr_path(_parse_expr("foo().bar")) is None


def test_first_string_arg_constant() -> None:
    call = _parse_expr('logger.warning("hello")')
    assert isinstance(call, ast.Call)
    assert first_string_arg(call) == "hello"


def test_first_string_arg_fstring_self_attribute() -> None:
    """``self.X`` interpolations collapse to ``{X}`` (matches runtime
    substitution vocabulary for ``message_template``)."""
    call = _parse_expr('logger.warning(f"value must be > 0, got {self.temperature}")')
    assert isinstance(call, ast.Call)
    out = first_string_arg(call)
    assert out == "value must be > 0, got {temperature}"


def test_first_string_arg_fstring_local_variable() -> None:
    """Non-``self`` interpolations preserve the unparsed expression as the
    placeholder name."""
    call = _parse_expr('logger.warning(f"value is {x}")')
    assert isinstance(call, ast.Call)
    assert first_string_arg(call) == "value is {x}"


def test_first_string_arg_fstring_multiple_interpolations() -> None:
    call = _parse_expr(
        'logger.warning(f"max_batch_size [{self.max_batch_size}] '
        'exceeds [{self.build_config.max_batch_size}]")'
    )
    assert isinstance(call, ast.Call)
    out = first_string_arg(call)
    assert out == ("max_batch_size [{max_batch_size}] exceeds [{self.build_config.max_batch_size}]")


def test_first_string_arg_fstring_no_python_source_leak() -> None:
    """Output must not contain literal Python source artefacts (leading
    ``f"``, ``self.`` for self attributes)."""
    call = _parse_expr('ValueError(f"temperature={self.temperature}")')
    assert isinstance(call, ast.Call)
    out = first_string_arg(call)
    assert out is not None
    assert not out.startswith('f"')
    assert not out.startswith("f'")
    assert "self.temperature" not in out


def test_first_string_arg_format_call_literal_template() -> None:
    """``"literal {x}".format(...)`` - extract the LHS template literal
    rather than returning the unparsed call source."""
    call = _parse_expr('logger.warning("val={v}".format(v=1))')
    assert isinstance(call, ast.Call)
    assert first_string_arg(call) == "val={v}"


def test_first_string_arg_format_call_variable_template_returns_none() -> None:
    """Variable-resolved templates (e.g. ``msg_template.format(...)``) need
    scope resolution that's out of reach at this AST layer. Return ``None``
    rather than leaking the literal ``msg_template.format(...)`` source."""
    call = _parse_expr("logger.warning(msg_template.format(v=1))")
    assert isinstance(call, ast.Call)
    assert first_string_arg(call) is None


def test_render_binop_concat_template_simple() -> None:
    """``"prefix " + ", got " + str(self.x)`` → rendered concatenation."""
    expr = _parse_expr('"value > 0" + ", got " + str(self.temperature)')
    assert render_binop_concat_template(expr) == "value > 0, got {temperature}"


def test_render_binop_concat_template_self_attribute() -> None:
    """Bare ``self.X`` operands collapse to ``{X}``."""
    expr = _parse_expr('"got " + self.value')
    assert render_binop_concat_template(expr) == "got {value}"


def test_render_binop_concat_template_repr_call() -> None:
    """``repr(self.X)`` and ``str(self.X)`` both render as ``{X}``."""
    expr = _parse_expr('"got " + repr(self.config)')
    assert render_binop_concat_template(expr) == "got {config}"


def test_render_binop_concat_template_returns_none_on_unrenderable() -> None:
    """Operands without a clean placeholder mapping (e.g. external function
    calls, attribute chains beyond ``self.X``) cause the whole render to
    return ``None`` rather than leak literal source."""
    expr = _parse_expr('"got " + format_helper(x, y)')
    assert render_binop_concat_template(expr) is None


def test_extract_condition_fields_simple() -> None:
    expr = _parse_expr("self.temperature < 0.01")
    assert extract_condition_fields(expr) == {"temperature"}


def test_extract_condition_fields_multi() -> None:
    expr = _parse_expr("self.do_sample is False and self.temperature != 1.0")
    assert extract_condition_fields(expr) == {"do_sample", "temperature"}


def test_extract_assign_target_self_attr() -> None:
    stmt = ast.parse("self.temperature = 0.5").body[0]
    assert isinstance(stmt, ast.Assign)
    assert extract_assign_target(stmt) == "temperature"


def test_extract_assign_target_non_self_returns_none() -> None:
    stmt = ast.parse("other.temperature = 0.5").body[0]
    assert isinstance(stmt, ast.Assign)
    assert extract_assign_target(stmt) is None


def test_resolve_local_assign_finds_literal() -> None:
    src = """
def validate(self):
    greedy_msg = "Greedy wrong: {flag}"
    return greedy_msg.format(flag="temperature")
"""
    func = ast.parse(src.strip()).body[0]
    assert isinstance(func, ast.FunctionDef)
    assert resolve_local_assign(func, "greedy_msg") == "Greedy wrong: {flag}"


def test_resolve_local_assign_missing_returns_none() -> None:
    src = "def validate(self):\n    x = 1\n"
    func = ast.parse(src).body[0]
    assert isinstance(func, ast.FunctionDef)
    assert resolve_local_assign(func, "greedy_msg") is None


def test_extract_loop_literal_iterable_list() -> None:
    loop = ast.parse("for arg in ['a', 'b', 'c']: pass").body[0]
    assert isinstance(loop, ast.For)
    assert extract_loop_literal_iterable(loop) == ["a", "b", "c"]


def test_extract_loop_literal_iterable_tuple() -> None:
    loop = ast.parse("for arg in (1, 2, 3): pass").body[0]
    assert isinstance(loop, ast.For)
    assert extract_loop_literal_iterable(loop) == [1, 2, 3]


def test_extract_loop_literal_iterable_self_attr_returns_none() -> None:
    # Non-literal iterable (self.<field>) should downgrade detection.
    loop = ast.parse("for arg in self.allowed: pass").body[0]
    assert isinstance(loop, ast.For)
    assert extract_loop_literal_iterable(loop) is None


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


def test_walker_landmark_missing_error_carries_detail() -> None:
    exc = MinerLandmarkMissingError("GenerationConfig.validate", "method removed")
    assert exc.landmark == "GenerationConfig.validate"
    assert "method removed" in str(exc)


# ---------------------------------------------------------------------------
# Class/method finders
# ---------------------------------------------------------------------------


_HELPER_MODULE_SRC = """
class Thing:
    def entry(self):
        self.sub_check()

    def sub_check(self):
        if self.temperature < 0:
            raise ValueError("bad")

    def unrelated(self):
        return 1
"""


def test_find_class_and_method_helpers() -> None:
    module = ast.parse(_HELPER_MODULE_SRC)
    assert find_class(module, "NonExistent") is None
    cls = find_class(module, "Thing")
    assert cls is not None
    assert find_method(cls, "entry") is not None
    assert find_method(cls, "does_not_exist") is None
