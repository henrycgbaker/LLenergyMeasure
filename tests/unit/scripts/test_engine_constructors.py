"""Tests for :mod:`scripts._engine_constructors` - per-engine native-type resolution.

The engines are not importable on the host, so these tests install synthetic
modules into ``sys.modules`` mirroring the post-refactor subpackage layouts the
resolver must survive (vllm's ``vllm.config.*`` move, tensorrt's deep
``llmapi.llm_args`` path). The resolver is pure name->class lookup, so synthetic
modules exercise every branch faithfully.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._engine_constructors import (  # noqa: E402
    NativeTypeResolutionError,
    resolve_native_type,
)


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> ModuleType:
    """Register a synthetic module under ``name`` with the given attributes.

    Parent packages are stubbed too so ``importlib.import_module`` can resolve
    the dotted path (``vllm.config.cache`` needs ``vllm`` and ``vllm.config``).
    """
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            monkeypatch.setitem(sys.modules, sub, ModuleType(sub))
    module = sys.modules[name]
    for key, value in attrs.items():
        monkeypatch.setattr(module, key, value, raising=False)
    return module


class _FakeClass:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class TestResolveVllm:
    def test_resolves_class_off_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SamplingParams stays on the vllm root.
        _install_module(monkeypatch, "vllm", SamplingParams=_FakeClass)
        assert resolve_native_type("vllm", "vllm.SamplingParams") is _FakeClass

    def test_module_probe_resolves_subpackage_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Post-0.19.1: CacheConfig moved into vllm.config.cache and is NOT on
        # the root - getattr(vllm, "CacheConfig") fails. The probe table must
        # find it. This is the P10 case (infra_error 100 -> 4).
        _install_module(monkeypatch, "vllm")
        _install_module(monkeypatch, "vllm.config.cache", CacheConfig=_FakeClass)
        assert resolve_native_type("vllm", "vllm.CacheConfig") is _FakeClass

    def test_deep_dotted_path_resolves_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The walker may emit the already-deep form vllm.config.SchedulerConfig.
        _install_module(monkeypatch, "vllm")
        _install_module(monkeypatch, "vllm.config", SchedulerConfig=_FakeClass)
        assert resolve_native_type("vllm", "vllm.config.SchedulerConfig") is _FakeClass

    def test_unresolvable_raises_resolution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_module(monkeypatch, "vllm")
        with pytest.raises(NativeTypeResolutionError):
            resolve_native_type("vllm", "vllm.DoesNotExist")


class TestResolveTensorrt:
    def test_override_maps_short_name_to_deep_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # BaseLlmArgs -> TrtLlmArgs deep in llmapi.llm_args (override table).
        _install_module(monkeypatch, "tensorrt_llm")
        _install_module(monkeypatch, "tensorrt_llm.llmapi.llm_args", TrtLlmArgs=_FakeClass)
        assert resolve_native_type("tensorrt", "tensorrt_llm.BaseLlmArgs") is _FakeClass

    def test_probe_resolves_plugin_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_module(monkeypatch, "tensorrt_llm")
        _install_module(monkeypatch, "tensorrt_llm.plugin.plugin", PluginConfig=_FakeClass)
        assert resolve_native_type("tensorrt", "tensorrt_llm.PluginConfig") is _FakeClass


class TestResolveTransformers:
    def test_monolithic_root_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # transformers is monolithic - classes live off the package root.
        _install_module(monkeypatch, "transformers", GenerationConfig=_FakeClass)
        assert resolve_native_type("transformers", "transformers.GenerationConfig") is _FakeClass


class TestResolveUnknownEngine:
    def test_unknown_engine_falls_back_to_dotted_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_module(monkeypatch, "someengine.sub", Thing=_FakeClass)
        assert resolve_native_type("someengine", "someengine.sub.Thing") is _FakeClass

    def test_unknown_engine_unresolvable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_module(monkeypatch, "someengine")
        with pytest.raises(NativeTypeResolutionError):
            resolve_native_type("someengine", "someengine.Missing")
