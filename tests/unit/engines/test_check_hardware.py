"""Unit tests for ``EnginePlugin.check_hardware`` across all three engines.

The ``check_hardware`` seam is the host-GPU-dependent counterpart to the
engine-invariants validator that runs at ``ExperimentConfig`` construction time.
Tests here cover:

- Static-method contract (``check_hardware`` callable without an instance).
- Transformers / vLLM return ``[]`` (behavioural stubs; invariants move to the
  engine-invariants corpus when their respective miners ship).
- TensorRT's SM floor, FP8 gate, FP8 KV-cache gate, and multi-error collection.
- Structural property: ``check_hardware`` and ``_build_llm_kwargs`` are
  independent code paths, so a T0 kwargs-build failure can no longer
  short-circuit hardware compat (the bug the new seam exists to preclude).
"""

from __future__ import annotations

import pytest

from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.engines.vllm import VLLMEngine
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Static-method contract (all three engines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine_cls",
    [TransformersEngine, VLLMEngine, TensorRTEngine],
    ids=["transformers", "vllm", "tensorrt"],
)
def test_check_hardware_is_static(engine_cls, monkeypatch):
    """``check_hardware`` is callable on the class (no instance required)."""
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: (8, 0),
    )
    # Keep hermetic: the tensorrt hook reads the HF quant_method; stub it so no
    # HF Hub call is attempted for the hub-style "test-model" id.
    monkeypatch.setattr(
        "llenergymeasure.engines.tensorrt.plugin._read_model_quant_method",
        lambda _: None,
    )
    engine_name = {
        TransformersEngine: "transformers",
        VLLMEngine: "vllm",
        TensorRTEngine: "tensorrt",
    }[engine_cls]
    config = make_config(model="test-model", engine=engine_name)
    result = engine_cls.check_hardware(config)
    assert isinstance(result, list)
    assert all(isinstance(e, str) for e in result)


# ---------------------------------------------------------------------------
# Transformers: no host-hardware invariants at MVP
# ---------------------------------------------------------------------------


class TestTransformersCheckHardware:
    def test_returns_empty_on_any_sm(self, monkeypatch):
        """Transformers has no host-hardware invariants; always returns ``[]``."""
        for sm in [(7, 0), (8, 0), (8, 9), (9, 0), None]:
            monkeypatch.setattr(
                "llenergymeasure.device.gpu_info.get_compute_capability",
                lambda gpu_index=0, _sm=sm: _sm,
            )
            config = make_config(model="test-model", engine="transformers")
            assert TransformersEngine.check_hardware(config) == []


# ---------------------------------------------------------------------------
# vLLM: no host-hardware invariants at MVP
# ---------------------------------------------------------------------------


class TestVLLMCheckHardware:
    def test_returns_empty_on_any_sm(self, monkeypatch):
        """vLLM has no host-hardware invariants at MVP; always returns ``[]``."""
        for sm in [(7, 0), (8, 0), (8, 9), (9, 0), None]:
            monkeypatch.setattr(
                "llenergymeasure.device.gpu_info.get_compute_capability",
                lambda gpu_index=0, _sm=sm: _sm,
            )
            config = make_config(model="test-model", engine="vllm")
            assert VLLMEngine.check_hardware(config) == []


# ---------------------------------------------------------------------------
# TensorRT: SM floor, FP8 gates, multi-error collection
# ---------------------------------------------------------------------------


_TRT_DEFAULTS = {"model": "test-model", "engine": "tensorrt"}


def _patch_sm(monkeypatch, sm: tuple[int, int] | None) -> None:
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: sm,
    )


class TestTensorRTCheckHardware:
    @pytest.fixture(autouse=True)
    def _hermetic_quant_reader(self, monkeypatch):
        """Stub the HF quant_method reader so the SM/FP8 tests never hit the Hub.

        check_hardware now routes the checkpoint-compat check too; these tests
        exercise the hardware gates only, so a None quant_method (not detected)
        keeps them focused and offline.
        """
        monkeypatch.setattr(
            "llenergymeasure.engines.tensorrt.plugin._read_model_quant_method",
            lambda _: None,
        )

    def test_sm_none_returns_empty(self, monkeypatch):
        """SM detection returns None (no GPU visible) -> no errors."""
        _patch_sm(monkeypatch, None)
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine.check_hardware(config) == []

    def test_sm_below_floor_errors(self, monkeypatch):
        """SM 7.0 (V100) fails the 7.5 floor."""
        _patch_sm(monkeypatch, (7, 0))
        config = make_config(**_TRT_DEFAULTS)
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "SM >= 7.5" in errors[0]

    def test_sm_at_floor_passes(self, monkeypatch):
        """SM 7.5 (Turing T4) passes exactly at the floor."""
        _patch_sm(monkeypatch, (7, 5))
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine.check_hardware(config) == []

    def test_fp8_on_a100_errors(self, monkeypatch):
        """FP8 quant on SM 8.0 (A100) is blocked."""
        _patch_sm(monkeypatch, (8, 0))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "quant_config": {"quant_algo": "FP8"}}},
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "FP8" in errors[0]
        assert "SM >= 8.9" in errors[0]

    def test_fp8_on_ada_passes(self, monkeypatch):
        """FP8 quant on SM 8.9 (Ada Lovelace) passes."""
        _patch_sm(monkeypatch, (8, 9))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "quant_config": {"quant_algo": "FP8"}}},
        )
        assert TensorRTEngine.check_hardware(config) == []

    def test_fp8_kv_cache_on_a100_errors(self, monkeypatch):
        """FP8 KV-cache quant on SM 8.0 is blocked."""
        _patch_sm(monkeypatch, (8, 0))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "backend": "trt",
                    "quant_config": {"kv_cache_quant_algo": "FP8"},
                }
            },
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "KV cache" in errors[0]

    def test_both_fp8_errors_collected(self, monkeypatch):
        """FP8 weight quant AND FP8 KV cache on SM 8.0 produces 2 errors."""
        _patch_sm(monkeypatch, (8, 0))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "backend": "trt",
                    "quant_config": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"},
                }
            },
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Short-circuit regression: pre-check_hardware, TensorRT's hardware check was
# only reachable downstream of ``_build_llm_kwargs``, so a kwargs-build failure
# silently skipped hardware compat. The fix is structural: ``check_hardware``
# is now a separate code path. This test exercises the same fixture against
# both: ``_build_llm_kwargs`` raises, yet ``check_hardware`` still returns the
# SM error.
# ---------------------------------------------------------------------------


class TestShortCircuitRegression:
    def test_kwargs_build_and_hardware_check_are_independent(self, monkeypatch, tmp_path):
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        _patch_sm(monkeypatch, (7, 0))  # below the 7.5 floor

        # engine_path pointing at a non-existent directory makes
        # _build_llm_kwargs raise ConfigError.
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "engine_path": str(tmp_path / "does-not-exist"),
                    "backend": "trt",
                }
            },
        )

        engine = TensorRTEngine()

        with pytest.raises(ConfigError):
            engine._build_llm_kwargs(config)

        errors = TensorRTEngine.check_hardware(config)
        assert any("SM >= 7.5" in e for e in errors), (
            f"expected SM-floor error from check_hardware; got {errors!r}"
        )


# ---------------------------------------------------------------------------
# TensorRT: HF pre-quantised checkpoint gate, now routed THROUGH check_hardware
# (was the standalone preflight Check 5; folded into the plugin so the harness
# preflight calls one uniform hook for every engine).
# ---------------------------------------------------------------------------


def _stub_quant_method(monkeypatch, value: str | None) -> None:
    monkeypatch.setattr(
        "llenergymeasure.engines.tensorrt.plugin._read_model_quant_method",
        lambda _: value,
    )


class TestTensorRTCheckpointCompat:
    """The AWQ/GPTQ checkpoint gate is reachable via TensorRTEngine.check_hardware.

    SM is pinned to None (no visible GPU) so only the GPU-independent
    checkpoint-compat error can appear - proving the checkpoint check runs even
    when the hardware gates short-circuit on a missing device.
    """

    def test_rejects_awq(self, monkeypatch):
        _patch_sm(monkeypatch, None)
        _stub_quant_method(monkeypatch, "awq")
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"tensor_parallel_size": 1}},
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        err = errors[0]
        assert "AWQ" in err
        assert "engine_path" in err
        assert "trtllm-build" in err
        # Names what was tried at 1.2.1: both backends fail, ModelOpt is the path.
        assert "1.2.1" in err
        assert "either backend" in err
        assert "ModelOpt" in err

    def test_rejects_gptq(self, monkeypatch):
        _patch_sm(monkeypatch, None)
        _stub_quant_method(monkeypatch, "gptq")
        config = make_config(
            engine="tensorrt",
            tensorrt={"engine_params": {"tensor_parallel_size": 1}},
            model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "GPTQ" in errors[0]

    def test_passes_non_quantised(self, monkeypatch):
        _patch_sm(monkeypatch, None)
        _stub_quant_method(monkeypatch, None)
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"tensor_parallel_size": 1}},
        )
        assert TensorRTEngine.check_hardware(config) == []

    def test_skips_when_engine_path_set(self, monkeypatch):
        """A prebuilt engine_path means the user pre-built the engine; do not block."""
        _patch_sm(monkeypatch, None)
        _stub_quant_method(monkeypatch, "awq")
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "tensor_parallel_size": 1,
                    "engine_path": "/some/built/engine",
                    "backend": "trt",
                }
            },
        )
        assert TensorRTEngine.check_hardware(config) == []

    def test_network_failure_does_not_block(self, monkeypatch):
        """Transient HF Hub failures (reader returns None) must not block."""
        _patch_sm(monkeypatch, None)
        _stub_quant_method(monkeypatch, None)
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"tensor_parallel_size": 1}},
        )
        assert TensorRTEngine.check_hardware(config) == []

    def test_checkpoint_and_sm_errors_collected_together(self, monkeypatch):
        """AWQ checkpoint AND a too-low SM both surface from the one hook."""
        _patch_sm(monkeypatch, (7, 0))  # below the 7.5 floor
        _stub_quant_method(monkeypatch, "awq")
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"tensor_parallel_size": 1}},
        )
        errors = TensorRTEngine.check_hardware(config)
        assert any("AWQ" in e for e in errors)
        assert any("SM >= 7.5" in e for e in errors)


class TestOtherEnginesNoCheckpointGate:
    """transformers / vLLM do not carry the TRT checkpoint gate (structural)."""

    @pytest.mark.parametrize(
        "engine_cls,engine_name",
        [(TransformersEngine, "transformers"), (VLLMEngine, "vllm")],
        ids=["transformers", "vllm"],
    )
    def test_awq_model_not_flagged(self, monkeypatch, engine_cls, engine_name):
        _patch_sm(monkeypatch, (8, 0))
        # Even if the tensorrt reader would say "awq", non-TRT hooks never call it.
        _stub_quant_method(monkeypatch, "awq")
        config = make_config(model="Qwen/Qwen2.5-7B-Instruct-AWQ", engine=engine_name)
        assert engine_cls.check_hardware(config) == []


# ---------------------------------------------------------------------------
# _read_model_quant_method: local-path parsing (moved from preflight to the
# tensorrt plugin). No stubbing here - these exercise the real reader.
# ---------------------------------------------------------------------------


def test_read_quant_method_local_path(tmp_path):
    """Local model directories with an AWQ config.json are detected."""
    import json

    from llenergymeasure.engines.tensorrt.plugin import _read_model_quant_method

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "AWQ"}})
    )
    assert _read_model_quant_method(str(tmp_path)) == "awq"


def test_read_quant_method_local_path_no_config(tmp_path):
    """Local directories without config.json return None (skip)."""
    from llenergymeasure.engines.tensorrt.plugin import _read_model_quant_method

    assert _read_model_quant_method(str(tmp_path)) is None


def test_read_quant_method_local_path_no_quant_block(tmp_path):
    """Local config.json without quantization_config returns None."""
    import json

    from llenergymeasure.engines.tensorrt.plugin import _read_model_quant_method

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert _read_model_quant_method(str(tmp_path)) is None
