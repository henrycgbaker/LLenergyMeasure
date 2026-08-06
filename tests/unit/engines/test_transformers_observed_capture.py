"""Tests for TransformersEngine._capture_observed_params (resolved-dtype capture).

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
    return ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline")


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
        # No live generation_config (e.g. a mock model without one): the
        # extractor falls back to GenerationConfig(**kwargs). The captured
        # sampling shape then depends on whether transformers is importable in
        # this environment:
        #   - host (no transformers): the import raises, so capture is empty
        #   - engine container (transformers present): a full default config is
        #     built with the requested kwargs overlaid, so the kwarg survives
        # Either way the call must succeed and engine params stay empty.
        model = _FakeHFModel()  # no generation_config attribute
        observed = TransformersEngine._capture_observed_params(
            _config(), model, {"temperature": 0.5}
        )
        try:
            import transformers  # noqa: F401

            transformers_present = True
        except ImportError:
            transformers_present = False

        if transformers_present:
            assert observed["sampling"]["temperature"] == 0.5, (
                "the requested kwarg must survive the GenerationConfig(**kwargs) fallback"
            )
        else:
            assert observed["sampling"] == {}
        assert observed["engine"] == {}


class TestD2ResolvedDtypeCapture:
    """Record the resolved dtype the model actually ran in."""

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
