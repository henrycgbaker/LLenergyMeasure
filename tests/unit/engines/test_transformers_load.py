"""Model-load failure wrapping for TransformersEngine.load_model().

Parity with the vllm/tensorrt plugins: a from_pretrained() failure (missing or
gated model, OOM, bad dtype) surfaces as a single actionable EngineError naming
the model and engine, not a bare transformers traceback.

transformers is not installed on the host, so a fake module is injected into
sys.modules; torch IS required (load_model builds dtype kwargs before the
wrapped construction).
"""

from __future__ import annotations

import sys
import types

import pytest

from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.utils.exceptions import EngineError
from tests.conftest import make_config

pytest.importorskip("torch")


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer_raises: bool,
    model_raises: bool,
) -> None:
    """Inject a fake ``transformers`` module whose from_pretrained can raise."""
    fake = types.ModuleType("transformers")

    class _FakeTokenizer:
        pad_token = "<pad>"  # not None, so load_model skips the eos assignment
        eos_token = "<eos>"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            if tokenizer_raises:
                raise RuntimeError("tokenizer download failed")
            return _FakeTokenizer()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            if model_raises:
                raise RuntimeError("CUDA out of memory while loading weights")
            raise AssertionError("unexpected success in test")

    fake.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    fake.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_load_model_wraps_tokenizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tokenizer from_pretrained() failure becomes an EngineError."""
    _install_fake_transformers(monkeypatch, tokenizer_raises=True, model_raises=False)
    config = make_config(engine="transformers", model="fake/model-xyz")

    with pytest.raises(EngineError) as exc_info:
        TransformersEngine().load_model(config)

    msg = str(exc_info.value)
    assert "Transformers model loading failed" in msg
    assert "fake/model-xyz" in msg  # model name is in the context
    assert "tokenizer download failed" in msg  # original cause chained in


def test_load_model_wraps_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model-weights from_pretrained() failure becomes an EngineError."""
    _install_fake_transformers(monkeypatch, tokenizer_raises=False, model_raises=True)
    config = make_config(engine="transformers", model="fake/model-abc")

    with pytest.raises(EngineError) as exc_info:
        TransformersEngine().load_model(config)

    msg = str(exc_info.value)
    assert "Transformers model loading failed" in msg
    assert "fake/model-abc" in msg


def test_load_model_failure_chains_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """The EngineError preserves the original exception via __cause__."""
    _install_fake_transformers(monkeypatch, tokenizer_raises=True, model_raises=False)
    config = make_config(engine="transformers", model="fake/model-xyz")

    with pytest.raises(EngineError) as exc_info:
        TransformersEngine().load_model(config)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
