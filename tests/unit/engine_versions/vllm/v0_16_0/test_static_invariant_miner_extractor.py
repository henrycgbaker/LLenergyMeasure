"""Unit tests for the vLLM 0.16.0 static-miner predicate extractor.

Covers the ``_extract_call`` recogniser extensions for vLLM hardware-compat
gates (``current_platform.is_cuda()`` etc., ``has_device_capability(N)``)
plus a regression case on the pre-existing ``isinstance`` shape.

Tests run on synthetic AST fixtures; no live vLLM import required.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MINER_PATH = (
    _PROJECT_ROOT
    / "engine_versions"
    / "vllm"
    / "v0_16_0"
    / "producers"
    / "static_invariant_miner.py"
)


def _load_miner_module() -> object:
    name = "_v0_16_0_static_invariant_miner_under_test"
    spec = importlib.util.spec_from_file_location(name, _MINER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so dataclass(frozen=True) can
    # resolve the defining module during class processing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def miner() -> object:
    return _load_miner_module()


def _parse_call(src: str) -> ast.Call:
    expr = ast.parse(src, mode="eval").body
    assert isinstance(expr, ast.Call)
    return expr


def test_extract_call_recognises_current_platform_is_cuda(miner: object) -> None:
    preds = miner._extract_call(_parse_call("current_platform.is_cuda()"))  # type: ignore[attr-defined]
    assert len(preds) == 1
    assert preds[0].field == "platform"
    assert preds[0].op == "=="
    assert preds[0].rhs == "cuda"


def test_extract_call_recognises_current_platform_is_rocm(miner: object) -> None:
    preds = miner._extract_call(_parse_call("current_platform.is_rocm()"))  # type: ignore[attr-defined]
    assert len(preds) == 1
    assert preds[0].field == "platform"
    assert preds[0].rhs == "rocm"


def test_extract_call_recognises_has_device_capability_with_literal(miner: object) -> None:
    preds = miner._extract_call(  # type: ignore[attr-defined]
        _parse_call("current_platform.has_device_capability(80)")
    )
    assert len(preds) == 1
    assert preds[0].field == "compute_capability"
    assert preds[0].op == ">="
    assert preds[0].rhs == 80


def test_extract_call_recognises_cls_has_device_capability(miner: object) -> None:
    # Inside vllm.platforms.cuda the classmethod dispatches on cls; the
    # extractor must recognise the same method on the cls receiver.
    preds = miner._extract_call(_parse_call("cls.has_device_capability(89)"))  # type: ignore[attr-defined]
    assert len(preds) == 1
    assert preds[0].field == "compute_capability"
    assert preds[0].op == ">="
    assert preds[0].rhs == 89


def test_extract_call_regression_isinstance_still_works(miner: object) -> None:
    # Pre-existing behaviour: isinstance(self.X, T) returns a type_is
    # predicate keyed on the self-attribute.
    preds = miner._extract_call(_parse_call("isinstance(self.top_p, float)"))  # type: ignore[attr-defined]
    assert len(preds) == 1
    assert preds[0].field == "top_p"
    assert preds[0].op == "type_is"
    assert preds[0].rhs == "float"


def test_extract_call_ignores_unknown_method(miner: object) -> None:
    # Methods not in the platform-shim allowlist must not synthesise a
    # predicate (false-positive guard).
    preds = miner._extract_call(  # type: ignore[attr-defined]
        _parse_call("current_platform.some_unknown_predicate()")
    )
    assert preds == []
