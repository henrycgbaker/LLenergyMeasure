"""Tests for scripts/engine_producers/_stub_factory.py (the cached producer resolver).

Every engine-producer shim funnels module-level attribute access through this
factory, which resolves the per-version archive from the SSOT's
``library.current_version`` and caches it process-lifetime. These tests mock
``load_producer`` / ``load_current`` so no vendored archive or engine is needed.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

# Make the top-level ``scripts`` package importable from tests.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import _stub_factory  # noqa: E402

_CURRENT = "scripts.engine_producers._current.load_current"
_LOAD = "engine_versions._dispatcher.load_producer"


@pytest.fixture(autouse=True)
def _clear_resolver_cache() -> Iterator[None]:
    _stub_factory._resolve_producer_cached.cache_clear()
    yield
    _stub_factory._resolve_producer_cached.cache_clear()


def _fake_producer(**attrs: object) -> ModuleType:
    module = ModuleType("fake_producer")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def _current(version: str) -> dict[str, object]:
    return {"library": {"current_version": version}}


def test_resolve_dispatches_using_current_version() -> None:
    produced = _fake_producer(LANDMARKS=("a.b",))
    with (
        patch(_CURRENT, return_value=_current("0.19.1")) as m_current,
        patch(_LOAD, return_value=produced) as m_load,
    ):
        got = _stub_factory._resolve_producer_cached("vllm", "static_invariant_miner")

    assert got is produced
    m_current.assert_called_once_with("vllm")
    m_load.assert_called_once_with(
        engine="vllm", version="0.19.1", producer="static_invariant_miner"
    )


def test_resolve_is_cached() -> None:
    produced = _fake_producer()
    with (
        patch(_CURRENT, return_value=_current("1.0.0")),
        patch(_LOAD, return_value=produced) as m_load,
    ):
        first = _stub_factory._resolve_producer_cached("tensorrt", "schema_introspector")
        second = _stub_factory._resolve_producer_cached("tensorrt", "schema_introspector")

    assert first is second
    assert m_load.call_count == 1


def test_resolve_missing_current_version_raises() -> None:
    with (
        patch(_CURRENT, return_value={"library": {}}),
        patch(_LOAD) as m_load,
        pytest.raises(ValueError, match="current_version"),
    ):
        _stub_factory._resolve_producer_cached("vllm", "static_invariant_miner")

    m_load.assert_not_called()


def test_make_schema_stub_getattr_dunder_raises_without_resolving() -> None:
    # Dunder access must short-circuit before any archive resolution.
    _resolver, getattr_fn, _discover = _stub_factory.make_schema_stub("vllm")
    with pytest.raises(AttributeError):
        getattr_fn("__wrapped__")


def test_make_schema_stub_getattr_unknown_attr_raises() -> None:
    produced = _fake_producer(LANDMARKS=("x.y",))
    with (
        patch(_CURRENT, return_value=_current("0.19.1")),
        patch(_LOAD, return_value=produced),
    ):
        _resolver, getattr_fn, _discover = _stub_factory.make_schema_stub("vllm")
        assert getattr_fn("LANDMARKS") == ("x.y",)
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr_fn("does_not_exist")


def test_make_schema_stub_discover_forwards_args(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_discover(repo_root: Path, image_ref: str | None) -> dict[str, object]:
        captured["args"] = (repo_root, image_ref)
        return {"schema": "ok"}

    produced = _fake_producer(discover=fake_discover)
    with (
        patch(_CURRENT, return_value=_current("0.19.1")),
        patch(_LOAD, return_value=produced) as m_load,
    ):
        _resolver, _getattr, discover = _stub_factory.make_schema_stub("vllm")
        result = discover(tmp_path, "img:tag")

    assert result == {"schema": "ok"}
    assert captured["args"] == (tmp_path, "img:tag")
    m_load.assert_called_once_with(engine="vllm", version="0.19.1", producer="schema_introspector")
