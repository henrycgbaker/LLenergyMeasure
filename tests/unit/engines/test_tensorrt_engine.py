"""Unit tests for TensorRTEngine.

All tests run without GPU hardware and without tensorrt_llm installed.
TRT-LLM imports inside TensorRTEngine methods are lazy - the module is
importable on any host. Tests that exercise QuantConfig / BuildCacheConfig
construction inject mock classes via sys.modules so no real tensorrt_llm
import occurs.

Coverage:
  - Protocol compliance (BACK-01)
  - _build_llm_kwargs: field mapping + None omission
  - _build_sampling_params: defaults, greedy, TRT-specific overrides
  - _validate_engine_directory

SM/FP8 hardware gates are exercised via ``check_hardware`` in
``test_check_hardware.py``.
"""

from __future__ import annotations

import json
import sys
import types

from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.tensorrt.plugin import _validate_engine_directory
from tests.conftest import make_config

# =============================================================================
# Helpers
# =============================================================================

_TRT_DEFAULTS = {"model": "test-model", "engine": "tensorrt"}


class _MockQuantAlgo:
    """Mock for tensorrt_llm.llmapi.QuantAlgo enum."""

    INT8 = "INT8"
    FP8 = "FP8"
    W4A16_AWQ = "W4A16_AWQ"
    W4A16_GPTQ = "W4A16_GPTQ"
    W8A16 = "W8A16"

    def __class_getitem__(cls, item):
        return getattr(cls, item)


class _MockQuantConfig:
    """Mock for tensorrt_llm.llmapi.QuantConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockBuildCacheConfig:
    """Mock for tensorrt_llm.llmapi.BuildCacheConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockKvCacheConfig:
    """Mock for tensorrt_llm.llmapi.KvCacheConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockCapacitySchedulerPolicy:
    """Mock for tensorrt_llm.llmapi.CapacitySchedulerPolicy enum."""

    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    MAX_UTILIZATION = "MAX_UTILIZATION"
    STATIC_BATCH = "STATIC_BATCH"

    def __class_getitem__(cls, item):
        return getattr(cls, item)


class _MockSchedulerConfig:
    """Mock for tensorrt_llm.llmapi.SchedulerConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeSamplingParams:
    """Minimal stand-in for tensorrt_llm.SamplingParams - captures kwargs."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockBaseLLM:
    """Stand-in for tensorrt_llm.LLM (the pytorch backend, _TorchLLM-based)."""


class _MockTrtLLM:
    """Stand-in for tensorrt_llm._tensorrt_engine.LLM (the compiled-TRT backend)."""


def _make_fake_tensorrt_llm_module() -> types.ModuleType:
    """Build a minimal fake tensorrt_llm module for sys.modules injection.

    Includes the two LLM classes so the backend class-dispatch path
    (:meth:`TensorRTEngine._resolve_llm_class`) is exercisable without a real
    tensorrt_llm import. Callers that need dispatch also inject the
    ``tensorrt_llm._tensorrt_engine`` submodule (see :func:`_inject_trt`).
    """
    mock_trt = types.ModuleType("tensorrt_llm")
    mock_trt.__version__ = "1.2.1"  # type: ignore[attr-defined]
    mock_trt.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    mock_trt.LLM = _MockBaseLLM  # type: ignore[attr-defined]

    mock_engine = types.ModuleType("tensorrt_llm._tensorrt_engine")
    mock_engine.LLM = _MockTrtLLM  # type: ignore[attr-defined]
    mock_trt._tensorrt_engine = mock_engine  # type: ignore[attr-defined]

    mock_llmapi = types.ModuleType("tensorrt_llm.llmapi")
    mock_llmapi.QuantAlgo = _MockQuantAlgo  # type: ignore[attr-defined]
    mock_llmapi.QuantConfig = _MockQuantConfig  # type: ignore[attr-defined]
    mock_llmapi.BuildCacheConfig = _MockBuildCacheConfig  # type: ignore[attr-defined]
    mock_llmapi.KvCacheConfig = _MockKvCacheConfig  # type: ignore[attr-defined]
    mock_llmapi.SchedulerConfig = _MockSchedulerConfig  # type: ignore[attr-defined]
    mock_llmapi.CapacitySchedulerPolicy = _MockCapacitySchedulerPolicy  # type: ignore[attr-defined]

    mock_trt.llmapi = mock_llmapi  # type: ignore[attr-defined]
    return mock_trt


def _inject_trt(monkeypatch, mock_trt) -> None:
    """Register the fake tensorrt_llm module tree in sys.modules for dispatch."""
    monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._tensorrt_engine", mock_trt._tensorrt_engine)


# =============================================================================
# Test Group 1: Protocol compliance (BACK-01)
# =============================================================================


class TestProtocolCompliance:
    def test_tensorrt_engine_satisfies_plugin_protocol(self):
        """TensorRTEngine must satisfy the EnginePlugin Protocol."""
        from llenergymeasure.engines.protocol import EnginePlugin

        engine = TensorRTEngine()
        assert isinstance(engine, EnginePlugin)

    def test_tensorrt_engine_name(self):
        """TensorRTEngine.name returns 'tensorrt'."""
        assert TensorRTEngine().name == "tensorrt"

    def test_tensorrt_engine_has_all_protocol_methods(self):
        """TensorRTEngine implements all EnginePlugin methods."""
        engine = TensorRTEngine()
        assert hasattr(engine, "name")
        assert hasattr(engine, "load_model")
        assert hasattr(engine, "run_warmup_prompt")
        assert hasattr(engine, "run_inference")
        assert hasattr(engine, "cleanup")
        assert hasattr(engine, "check_hardware")


# =============================================================================
# Test Group 2: _build_llm_kwargs (BACK-01)
# =============================================================================


class TestBuildLlmKwargs:
    def test_build_llm_kwargs_minimal(self, monkeypatch):
        """No tensorrt config -> default (pytorch) backend: model only, no build cache.

        With no tensorrt section the default backend is pytorch, whose
        TorchLlmArgs has no ``enable_build_cache`` field, so the build cache is
        never applied even with LLEM_TRT_BUILD_CACHE_ENABLED=1.
        """
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["model"] == "test-model"
        assert "backend" not in kwargs
        assert "enable_build_cache" not in kwargs

    def test_build_llm_kwargs_backend_never_forwarded_trt(self):
        """backend='trt' selects the class; it is NOT forwarded as a kwarg."""
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "trt"}})
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "backend" not in kwargs

    def test_build_llm_kwargs_backend_never_forwarded_pytorch(self):
        """backend='pytorch' selects the class; it is NOT forwarded as a kwarg."""
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "pytorch"}})
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "backend" not in kwargs

    def test_build_llm_kwargs_tensor_parallel_size(self):
        """tensor_parallel_size=2 maps to kwargs tensor_parallel_size=2."""
        config = make_config(
            **_TRT_DEFAULTS, tensorrt={"engine_params": {"tensor_parallel_size": 2}}
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["tensor_parallel_size"] == 2

    def test_build_llm_kwargs_max_batch_size(self):
        """max_batch_size maps directly."""
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"max_batch_size": 16}})
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["max_batch_size"] == 16

    def test_build_llm_kwargs_dtype(self):
        """dtype='float16' maps directly."""
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"dtype": "float16"}})
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["dtype"] == "float16"

    def test_build_llm_kwargs_fast_build(self):
        """backend='trt' + fast_build=True maps directly (trt-build-only knob)."""
        config = make_config(
            **_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "trt", "fast_build": True}}
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["fast_build"] is True

    def test_fast_build_true_pytorch_rejected_at_config(self):
        """fast_build=True on the pytorch backend is rejected at config construction.

        The backend-applicability corpus rule fires at ExperimentConfig
        expansion - the config-grain enforcement, ahead of the plugin guard.
        """
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=r"fast_build.*requires the trt backend"):
            make_config(
                **_TRT_DEFAULTS,
                tensorrt={"engine_params": {"backend": "pytorch", "fast_build": True}},
            )

    def test_build_llm_kwargs_fast_build_true_pytorch_raises(self):
        """Plugin-grain guard (defense in depth): if fast_build=True + pytorch bypasses
        the corpus rule, _build_llm_kwargs still raises ConfigError (no TRT engine build)."""
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "pytorch"}},
        )
        # Force the conflicting value past config validation (pydantic does not
        # validate on assignment) to reach the plugin's own guard.
        config.tensorrt.engine_params.fast_build = True
        engine = TensorRTEngine()
        with pytest.raises(ConfigError, match="fast_build requires backend='trt'"):
            engine._build_llm_kwargs(config)

    def test_build_llm_kwargs_fast_build_false_pytorch_dropped(self):
        """The default fast_build=False on the pytorch backend is dropped, not forwarded.

        TorchLlmArgs has no ``fast_build`` field (extra='forbid'), so forwarding
        even the harmless default would crash construction.
        """
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "pytorch", "fast_build": False}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "fast_build" not in kwargs

    def test_build_llm_kwargs_none_values_not_included(self):
        """None fields from engine_params are NOT in kwargs.

        Only fields whose generated default is None are dropped by
        model_dump(exclude_none=True); fields with non-None generated defaults
        (tensor_parallel_size=1, fast_build=False, dtype='auto') forward verbatim
        and are not asserted here.
        """
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {}})
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "max_batch_size" not in kwargs
        assert "max_input_len" not in kwargs
        assert "max_seq_len" not in kwargs
        # max_num_tokens carries the upstream 1.2.1 default (8192) in the
        # generated config, so it forwards even when unset.
        assert kwargs["max_num_tokens"] == 8192
        assert "backend" not in kwargs

    def test_build_llm_kwargs_default_build_cache_when_no_build_cache_section(self, monkeypatch):
        """backend='trt' + LLEM_TRT_BUILD_CACHE_ENABLED=1 -> enable_build_cache=True."""
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs.get("enable_build_cache") is True

    def test_build_llm_kwargs_build_cache_skipped_on_pytorch(self, monkeypatch):
        """The build cache is a TRT-build knob: skipped on the pytorch backend even if enabled."""
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "pytorch", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "enable_build_cache" not in kwargs

    def test_build_llm_kwargs_quant_config(self, monkeypatch):
        """backend='trt' + quant.quant_algo='INT8' -> quant_config=QuantConfig(QuantAlgo.INT8).

        The native kwarg is ``quant_config`` (NOT the legacy ``quantization``,
        which TrtLlmArgs rejects under extra='forbid' at 1.2.1).
        """
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "quant_config": {"quant_algo": "INT8"}}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "quantization" not in kwargs
        assert "quant_config" in kwargs
        assert isinstance(kwargs["quant_config"], _MockQuantConfig)
        assert kwargs["quant_config"]._kwargs["quant_algo"] == "INT8"

    def test_quant_config_pytorch_rejected_at_config(self):
        """Declaring quant_config on the pytorch backend is rejected at config construction.

        The backend-applicability corpus rule fires at ExperimentConfig
        expansion - the config-grain enforcement, ahead of the plugin guard.
        """
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="quant_config requires the trt backend"):
            make_config(
                **_TRT_DEFAULTS,
                tensorrt={
                    "engine_params": {"backend": "pytorch", "quant_config": {"quant_algo": "INT8"}}
                },
            )

    def test_build_llm_kwargs_quant_config_pytorch_raises(self):
        """Plugin-grain guard (defense in depth): if quant_config + pytorch bypasses the
        corpus rule, _build_llm_kwargs still raises ConfigError (loud, not silent)."""
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "pytorch"}},
        )
        # Force the conflicting value past config validation (pydantic does not
        # validate on assignment) to reach the plugin's own guard.
        config.tensorrt.engine_params.quant_config = {"quant_algo": "INT8"}
        engine = TensorRTEngine()
        with pytest.raises(ConfigError, match="quant_config requires backend='trt'"):
            engine._build_llm_kwargs(config)

    def test_build_llm_kwargs_enable_build_cache_when_env_set(self, monkeypatch):
        """backend='trt' + LLEM_TRT_BUILD_CACHE_ENABLED=1 -> enable_build_cache=True."""
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "trt"}})
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs.get("enable_build_cache") is True

    def test_trt_build_cache_absent_when_env_unset(self, monkeypatch):
        """backend='trt', LLEM_TRT_BUILD_CACHE_ENABLED unset -> kwarg absent (TRT-LLM default)."""
        monkeypatch.delenv("LLEM_TRT_BUILD_CACHE_ENABLED", raising=False)
        monkeypatch.delenv("LLEM_TRT_BUILD_CACHE_PATH", raising=False)
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "enable_build_cache" not in kwargs

    def test_trt_build_cache_disabled_by_env_var(self, monkeypatch):
        """backend='trt', LLEM_TRT_BUILD_CACHE_ENABLED=0 -> kwarg absent."""
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "0")
        monkeypatch.delenv("LLEM_TRT_BUILD_CACHE_PATH", raising=False)
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "enable_build_cache" not in kwargs

        # Also covers the no-tensorrt-section branch (pytorch default).
        config_no_trt = make_config(**_TRT_DEFAULTS)
        kwargs_no_trt = engine._build_llm_kwargs(config_no_trt)
        assert "enable_build_cache" not in kwargs_no_trt

    def test_trt_build_cache_path_used_when_both_env_vars_set(self, monkeypatch):
        """backend='trt' + ENABLED=1 + PATH=... -> BuildCacheConfig(cache_root=path)."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_PATH", "/tmp/test")

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        from pathlib import Path as _Path

        assert "enable_build_cache" in kwargs
        assert isinstance(kwargs["enable_build_cache"], _MockBuildCacheConfig)
        assert kwargs["enable_build_cache"]._kwargs["cache_root"] == _Path("/tmp/test")

    def test_trt_build_cache_enabled_alone_is_bare_true(self, monkeypatch):
        """backend='trt', ENABLED=1, path unset -> enable_build_cache is bare True."""
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        monkeypatch.delenv("LLEM_TRT_BUILD_CACHE_PATH", raising=False)
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["enable_build_cache"] is True

    def test_trt_build_cache_path_requires_enabled(self, monkeypatch):
        """backend='trt', PATH set but ENABLED unset -> kwarg absent (passthrough invariant)."""
        monkeypatch.delenv("LLEM_TRT_BUILD_CACHE_ENABLED", raising=False)
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_PATH", "/tmp/test")
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "tensor_parallel_size": 1}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "enable_build_cache" not in kwargs

    def test_build_llm_kwargs_kv_cache_config(self, monkeypatch):
        """kv_cache section maps to KvCacheConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "kv_cache_config": {
                        "enable_block_reuse": True,
                        "free_gpu_memory_fraction": 0.8,
                    }
                }
            },
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "kv_cache_config" in kwargs
        assert isinstance(kwargs["kv_cache_config"], _MockKvCacheConfig)
        assert kwargs["kv_cache_config"]._kwargs["enable_block_reuse"] is True
        assert kwargs["kv_cache_config"]._kwargs["free_gpu_memory_fraction"] == 0.8

    def test_build_llm_kwargs_scheduler_config(self, monkeypatch):
        """scheduler section maps to SchedulerConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "scheduler_config": {
                        "capacity_scheduling_policy": "MAX_UTILIZATION",
                    }
                }
            },
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "scheduler_config" in kwargs
        assert isinstance(kwargs["scheduler_config"], _MockSchedulerConfig)
        assert (
            kwargs["scheduler_config"]._kwargs["capacity_scheduling_policy"]
            == _MockCapacitySchedulerPolicy.MAX_UTILIZATION
        )

    def test_build_llm_kwargs_model_always_present(self):
        """model key is always present regardless of tensorrt config."""
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)
        assert kwargs["model"] == "test-model"


# =============================================================================
# Test Group 3: _build_sampling_params (BACK-01)
# =============================================================================


class TestBuildSamplingParams:
    def test_build_sampling_params_defaults(self, monkeypatch):
        """Default config produces SamplingParams with max_tokens and no temperature."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert isinstance(params, _FakeSamplingParams)
        assert params._kwargs["max_tokens"] == config.task.max_output_tokens

    def test_build_sampling_params_passes_random_seed(self, monkeypatch):
        """random_seed from ExperimentConfig is forwarded to SamplingParams as `seed`."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(**_TRT_DEFAULTS, random_seed=123)
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert params._kwargs["seed"] == 123

    def test_build_sampling_params_default_no_temperature(self, monkeypatch):
        """Unset sampling -> no temperature in kwargs (TRT-LLM uses its own default)."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert isinstance(params, _FakeSamplingParams)
        # Sampling unset on engine section -> no temperature forwarded
        assert "temperature" not in params._kwargs

    def test_build_sampling_params_with_temperature(self, monkeypatch):
        """Explicit temperature on sampling_params is forwarded."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"sampling_params": {"temperature": 0.7}},
        )
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert params._kwargs.get("temperature") == 0.7

    def test_build_sampling_params_trt_overrides(self, monkeypatch):
        """tensorrt.sampling overrides (n, ignore_eos) take effect."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "sampling_params": {
                    "n": 3,
                    "ignore_eos": True,
                    "min_tokens": 5,
                }
            },
        )
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert params._kwargs.get("n") == 3
        assert params._kwargs.get("ignore_eos") is True
        assert params._kwargs.get("min_tokens") == 5


# =============================================================================
# Test Group 7: _validate_engine_directory
# =============================================================================


class TestValidateEngineDirectory:
    def test_valid_engine_dir(self, tmp_path):
        """Valid engine dir with config.json and rank0.engine passes."""
        config = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert errors == []

    def test_missing_dir(self, tmp_path):
        """Non-existent dir returns error."""
        errors = _validate_engine_directory(tmp_path / "nonexistent", tp_size=1)
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_missing_config_json(self, tmp_path):
        """Dir exists but no config.json returns error."""
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert len(errors) == 1
        assert "config.json" in errors[0]

    def test_tp_size_mismatch(self, tmp_path):
        """Engine tp_size=2 but requested tp_size=1 returns error."""
        config = {"pretrained_config": {"mapping": {"tp_size": 2}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert any("tp_size" in e for e in errors)

    def test_missing_rank_engine(self, tmp_path):
        """tp_size=2 but only rank0.engine exists returns error for rank1."""
        config = {"pretrained_config": {"mapping": {"tp_size": 2}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=2)
        assert any("rank1.engine" in e for e in errors)

    def test_corrupt_config_json(self, tmp_path):
        """Corrupt config.json returns parse error."""
        (tmp_path / "config.json").write_text("not json{{{")
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert any("config.json" in e for e in errors)

    def test_missing_tp_size_key_skips_check(self, tmp_path):
        """config.json without mapping.tp_size skips tp_size check (non-blocking)."""
        config: dict[str, object] = {"pretrained_config": {}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert errors == []


# =============================================================================
# Test Group 8: _build_llm_kwargs engine_path branches
# =============================================================================


class TestBuildLlmKwargsEnginePath:
    def test_build_llm_kwargs_engine_path(self, tmp_path):
        """engine_path set -> kwargs has model=engine_path as string; backend kwarg absent."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")

        config = make_config(
            **_TRT_DEFAULTS, tensorrt={"engine_params": {"engine_path": str(tmp_path)}}
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["model"] == str(tmp_path)
        assert "backend" not in kwargs

    def test_build_llm_kwargs_engine_path_with_typed_backend(self, tmp_path):
        """engine_path + typed backend -> only model in kwargs; backend selects the class."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"engine_path": str(tmp_path), "backend": "trt"}},
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["model"] == str(tmp_path)
        assert "backend" not in kwargs

    def test_build_llm_kwargs_engine_path_skips_compile_kwargs(self, tmp_path):
        """engine_path set -> no compile-time kwargs (tensor_parallel_size, max_batch_size, etc.)."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 2}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        (tmp_path / "rank1.engine").write_bytes(b"fake")

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "engine_path": str(tmp_path),
                    "tensor_parallel_size": 2,
                    "max_batch_size": 16,
                }
            },
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "tensor_parallel_size" not in kwargs
        assert "max_batch_size" not in kwargs
        assert "max_input_len" not in kwargs
        assert "max_seq_len" not in kwargs
        assert "fast_build" not in kwargs
        assert "dtype" not in kwargs

    def test_build_llm_kwargs_engine_path_no_build_cache(self, tmp_path):
        """engine_path set -> enable_build_cache not in kwargs."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")

        config = make_config(
            **_TRT_DEFAULTS, tensorrt={"engine_params": {"engine_path": str(tmp_path)}}
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "enable_build_cache" not in kwargs

    def test_build_llm_kwargs_engine_path_invalid_dir_raises(self, tmp_path):
        """engine_path pointing to non-existent dir raises ConfigError."""
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"engine_path": str(tmp_path / "nonexistent")}},
        )
        engine = TensorRTEngine()
        with pytest.raises(ConfigError, match="engine_path validation failed"):
            engine._build_llm_kwargs(config)


# =============================================================================
# Test Group 9: backend class dispatch
# =============================================================================


class TestResolveLlmClass:
    def test_backend_none_resolves_base_llm(self, monkeypatch):
        """backend unset -> tensorrt_llm.LLM (the pytorch backend)."""
        mock_trt = _make_fake_tensorrt_llm_module()
        _inject_trt(monkeypatch, mock_trt)
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {}})
        assert TensorRTEngine._resolve_llm_class(config) is _MockBaseLLM

    def test_backend_pytorch_resolves_base_llm(self, monkeypatch):
        """backend='pytorch' -> tensorrt_llm.LLM."""
        mock_trt = _make_fake_tensorrt_llm_module()
        _inject_trt(monkeypatch, mock_trt)
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "pytorch"}})
        assert TensorRTEngine._resolve_llm_class(config) is _MockBaseLLM

    def test_backend_no_section_resolves_base_llm(self, monkeypatch):
        """No tensorrt section -> default (pytorch) -> tensorrt_llm.LLM."""
        mock_trt = _make_fake_tensorrt_llm_module()
        _inject_trt(monkeypatch, mock_trt)
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine._resolve_llm_class(config) is _MockBaseLLM

    def test_backend_trt_resolves_tensorrt_engine_llm(self, monkeypatch):
        """backend='trt' -> tensorrt_llm._tensorrt_engine.LLM (compiled-TRT class)."""
        mock_trt = _make_fake_tensorrt_llm_module()
        _inject_trt(monkeypatch, mock_trt)
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "trt"}})
        assert TensorRTEngine._resolve_llm_class(config) is _MockTrtLLM

    def test_backend_autodeploy_rejected_at_config(self):
        """_autodeploy is not exposed: the backend Literal rejects it at config construction."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="pytorch"):
            make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "_autodeploy"}})

    def test_backend_unknown_rejected_at_config(self):
        """An arbitrary backend value is rejected at config construction by the Literal."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="pytorch"):
            make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "nonsense"}})

    def test_resolve_llm_class_guards_unexposed_backend(self, monkeypatch):
        """Plugin-grain guard (defense in depth): an unexposed backend that bypasses the
        Literal still raises ConfigError from _resolve_llm_class, naming {pytorch, trt}."""
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        mock_trt = _make_fake_tensorrt_llm_module()
        _inject_trt(monkeypatch, mock_trt)
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "pytorch"}})
        # Force a value the Literal would reject past config validation (pydantic
        # does not validate on assignment) to reach the plugin's own guard.
        config.tensorrt.engine_params.backend = "_autodeploy"
        with pytest.raises(ConfigError, match=r"\{pytorch, trt\}"):
            TensorRTEngine._resolve_llm_class(config)


# =============================================================================
# Test Group 10: loud sub-config import failures
# =============================================================================


class TestLoudSubConfigImportErrors:
    """A declared sub-config whose native class cannot be imported must raise
    EngineError, never silently drop the kwarg (a measurement instrument must
    not measure a different configuration than declared).

    These tests inject no fake tensorrt_llm, so the real (absent-on-host) import
    fails and the loud path fires.
    """

    def test_quant_config_import_failure_raises(self):
        import pytest

        from llenergymeasure.utils.exceptions import EngineError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"backend": "trt", "quant_config": {"quant_algo": "INT8"}}},
        )
        engine = TensorRTEngine()
        with pytest.raises(EngineError, match="quant_config was declared"):
            engine._build_llm_kwargs(config)

    def test_kv_cache_config_import_failure_raises(self):
        import pytest

        from llenergymeasure.utils.exceptions import EngineError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={"engine_params": {"kv_cache_config": {"free_gpu_memory_fraction": 0.8}}},
        )
        engine = TensorRTEngine()
        with pytest.raises(EngineError, match="kv_cache_config was declared"):
            engine._build_llm_kwargs(config)

    def test_scheduler_config_import_failure_raises(self):
        import pytest

        from llenergymeasure.utils.exceptions import EngineError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt={
                "engine_params": {
                    "scheduler_config": {"capacity_scheduling_policy": "MAX_UTILIZATION"}
                }
            },
        )
        engine = TensorRTEngine()
        with pytest.raises(EngineError, match="scheduler_config was declared"):
            engine._build_llm_kwargs(config)

    def test_build_cache_import_failure_raises(self, monkeypatch):
        import pytest

        from llenergymeasure.utils.exceptions import EngineError

        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_ENABLED", "1")
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_PATH", "/tmp/test")
        config = make_config(**_TRT_DEFAULTS, tensorrt={"engine_params": {"backend": "trt"}})
        engine = TensorRTEngine()
        with pytest.raises(EngineError, match="build cache path was configured"):
            engine._build_llm_kwargs(config)
