"""Tests for TransformersEngine._capture_observed_params (EN2 + D2 resolved dtype).

Host-safe: no GPU, no torch, no transformers import. The model's
``generation_config`` and resolved ``dtype`` are modelled with a fake object
that mirrors the HuggingFace ``GenerationConfig`` merge semantics
(``copy.deepcopy`` of the model's live config, then ``update(**kwargs)`` overlay)
and is dumpable via ``__dict__``, which is what the observed-params extractor
falls back to.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.transformers.plugin import TransformersEngine


class _FakeGenerationConfig:
    """Minimal stand-in for transformers.GenerationConfig.

    ``update(**kwargs)`` sets recognised attributes in place (mirroring HF, which
    overlays explicit kwargs on top of the model's merged defaults). Dumped via
    ``__dict__`` by :func:`extract_observed_params`.
    """

    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)

    def update(self, **kwargs: Any) -> dict[str, Any]:
        for k, v in kwargs.items():
            setattr(self, k, v)
        return {}


class _FakeHFModel:
    def __init__(self, generation_config: Any = None, dtype: Any = None) -> None:
        if generation_config is not None:
            self.generation_config = generation_config
        if dtype is not None:
            self.dtype = dtype


def _config() -> ExperimentConfig:
    return ExperimentConfig(task={"model": "gpt2"}, engine="transformers")


class TestEN2MergedGenerationConfig:
    """EN2: capture the EFFECTIVE merged config (model defaults + overrides)."""

    def test_model_default_survives_when_not_in_kwargs(self) -> None:
        # Model default repetition_penalty=1.3 is NOT in generate_kwargs.
        model = _FakeHFModel(
            generation_config=_FakeGenerationConfig(
                repetition_penalty=1.3, temperature=1.0, do_sample=False
            )
        )
        generate_kwargs = {"temperature": 0.7, "do_sample": True}

        observed = TransformersEngine._capture_observed_params(_config(), model, generate_kwargs)
        sampling = observed["sampling"]

        # Merged: model default preserved, requested override applied.
        assert sampling["repetition_penalty"] == 1.3, "model default must survive the merge"
        assert sampling["temperature"] == 0.7, "requested override must be applied"
        assert sampling["do_sample"] is True

    def test_override_does_not_get_dropped(self) -> None:
        # Even when the model already has the field, the override wins.
        model = _FakeHFModel(generation_config=_FakeGenerationConfig(temperature=1.0, top_p=0.9))
        generate_kwargs = {"temperature": 0.2}

        observed = TransformersEngine._capture_observed_params(_config(), model, generate_kwargs)
        sampling = observed["sampling"]

        assert sampling["temperature"] == 0.2
        assert sampling["top_p"] == 0.9  # untouched model default kept

    def test_falls_back_to_kwargs_when_no_model_config(self) -> None:
        # No live generation_config (e.g. a mock model without one) and no real
        # transformers installed: the GenerationConfig(**kwargs) fallback raises
        # on import, so sampling capture is empty but the call still succeeds.
        model = _FakeHFModel()  # no generation_config attribute
        observed = TransformersEngine._capture_observed_params(
            _config(), model, {"temperature": 0.5}
        )
        assert observed["sampling"] == {}
        assert observed["engine"] == {}


class TestD2ResolvedDtypeCapture:
    """D2: record the resolved dtype the model actually ran in."""

    def test_resolved_dtype_recorded(self) -> None:
        model = _FakeHFModel(
            generation_config=_FakeGenerationConfig(temperature=1.0),
            dtype="torch.bfloat16",
        )
        observed = TransformersEngine._capture_observed_params(_config(), model, {})
        assert observed["engine"]["dtype"] == "torch.bfloat16"

    def test_dtype_absent_when_model_has_none(self) -> None:
        model = _FakeHFModel(generation_config=_FakeGenerationConfig(temperature=1.0))
        observed = TransformersEngine._capture_observed_params(_config(), model, {})
        assert "dtype" not in observed["engine"]
