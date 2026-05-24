"""Unit tests for TensorRT-LLM config validation.

Tests cover:
- CFG-01: Compile-time params (tensor_parallel_size, pipeline_parallel_size, max_batch_size,
          max_input_len, max_seq_len, max_num_tokens, dtype, fast_build)
- CFG-02: Quantisation (QuantAlgo Literal type, kv_cache_quant_algo)
- CFG-03: (Removed) Calibration sub-config dropped - D3 build-only PTQ, project consumes
          pre-quantised checkpoints. Falls through extra="allow" if needed.
- CFG-04: KV cache (enable_block_reuse, free_gpu_memory_fraction, max_tokens, host_cache_size)
- CFG-05: Scheduler (capacity_scheduling_policy Literal)
- CFG-06: (Removed) Build cache sub-config dropped - D1 engine-cache plumbing.
          Falls through extra="allow" if needed.
- CFG-07: Sampling (min_tokens, n, ignore_eos; return_perf_metrics dropped D1)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.engine_configs import (
    TensorRTKvCacheConfig,
    TensorRTQuantConfig,
    TensorRTSamplingConfig,
    TensorRTSchedulerConfig,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.tensorrt.config import Config as TensorRTConfig

# ---------------------------------------------------------------------------
# CFG-01: Compile-time params
# ---------------------------------------------------------------------------


class TestCompileTimeParams:
    """Tests for TensorRT compile-time parameters."""

    def test_tensorrt_compile_params_accepted(self):
        """All compile-time params validate when set together."""
        config = TensorRTConfig(
            engine_params={
                "max_batch_size": 8,
                "max_input_len": 1024,
                "max_seq_len": 2048,
                "tensor_parallel_size": 2,
                "dtype": "float16",
                "fast_build": True,
            }
        )
        assert config.engine_params.max_batch_size == 8
        assert config.engine_params.max_input_len == 1024
        assert config.engine_params.max_seq_len == 2048
        assert config.engine_params.tensor_parallel_size == 2
        assert config.engine_params.dtype == "float16"
        assert config.engine_params.fast_build is True

    # test_tensorrt_dtype_literal_validation deleted: OLD Literal rejection no longer applies.

    def test_tensorrt_dtype_bfloat16_accepted(self):
        """dtype='bfloat16' is valid."""
        config = TensorRTConfig(engine_params={"dtype": "bfloat16"})
        assert config.engine_params.dtype == "bfloat16"

    def test_tensorrt_tensor_parallel_size_ge_1(self):
        """tensor_parallel_size=1 is accepted (positive value accepted)."""
        config = TensorRTConfig(engine_params={"tensor_parallel_size": 1})
        assert config.engine_params.tensor_parallel_size == 1

    def test_tensorrt_max_batch_size_ge_1(self):
        """max_batch_size=1 is accepted (positive value accepted)."""
        config = TensorRTConfig(engine_params={"max_batch_size": 1})
        assert config.engine_params.max_batch_size == 1

    def test_tensorrt_max_input_len_ge_1(self):
        """max_input_len=1 is accepted (positive value accepted)."""
        config = TensorRTConfig(engine_params={"max_input_len": 1})
        assert config.engine_params.max_input_len == 1

    def test_tensorrt_max_seq_len_ge_1(self):
        """max_seq_len=1 is accepted (positive value accepted)."""
        config = TensorRTConfig(engine_params={"max_seq_len": 1})
        assert config.engine_params.max_seq_len == 1


# ---------------------------------------------------------------------------
# CFG-02: Quantisation
# ---------------------------------------------------------------------------


_ALL_QUANT_ALGOS = [
    "INT8",
    "W4A16_AWQ",
    "W4A16_GPTQ",
    "W8A16",
    "W8A16_GPTQ",
    "W4A8_AWQ",
    "FP8",
    "NO_QUANT",
]


class TestQuantisation:
    """Tests for TensorRT quantisation config."""

    @pytest.mark.parametrize("algo", _ALL_QUANT_ALGOS)
    def test_valid_quant_algo_accepted(self, algo: str):
        """All 8 required QuantAlgo values are accepted."""
        config = TensorRTQuantConfig(quant_algo=algo)
        assert config.quant_algo == algo

    def test_invalid_quant_algo_rejected(self):
        """Misspelled value like 'fp8' (lowercase) raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTQuantConfig(quant_algo="fp8")

    @pytest.mark.parametrize("algo", ["FP8", "INT8"])
    def test_kv_cache_quant_algo_accepted(self, algo: str):
        """FP8 and INT8 accepted for kv_cache_quant_algo."""
        config = TensorRTQuantConfig(kv_cache_quant_algo=algo)
        assert config.kv_cache_quant_algo == algo

    def test_invalid_kv_cache_quant_algo_rejected(self):
        """Invalid kv_cache_quant_algo value raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTQuantConfig(kv_cache_quant_algo="INVALID")


# CFG-03: Calibration sub-config dropped (D3) - tests removed.
# calib fields remain settable via TensorRTConfig extra="allow" passthrough.


# ---------------------------------------------------------------------------
# CFG-04: KV Cache
# ---------------------------------------------------------------------------


class TestKvCache:
    """Tests for TensorRT KV cache config."""

    def test_kv_cache_config_accepted(self):
        """KV cache section with valid values validates."""
        config = TensorRTKvCacheConfig(
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.85,
            max_tokens=4096,
            host_cache_size=1073741824,
        )
        assert config.enable_block_reuse is True
        assert config.free_gpu_memory_fraction == 0.85
        assert config.max_tokens == 4096
        assert config.host_cache_size == 1073741824

    def test_kv_cache_free_gpu_memory_fraction_range(self):
        """free_gpu_memory_fraction must be 0.0-1.0."""
        # Valid boundary
        config = TensorRTKvCacheConfig(free_gpu_memory_fraction=0.0)
        assert config.free_gpu_memory_fraction == 0.0
        config = TensorRTKvCacheConfig(free_gpu_memory_fraction=1.0)
        assert config.free_gpu_memory_fraction == 1.0

        # Invalid: above 1.0
        with pytest.raises(ValidationError):
            TensorRTKvCacheConfig(free_gpu_memory_fraction=1.5)

        # Invalid: below 0.0
        with pytest.raises(ValidationError):
            TensorRTKvCacheConfig(free_gpu_memory_fraction=-0.1)


# ---------------------------------------------------------------------------
# CFG-05: Scheduler
# ---------------------------------------------------------------------------


_VALID_SCHEDULER_POLICIES = [
    "GUARANTEED_NO_EVICT",
    "MAX_UTILIZATION",
    "STATIC_BATCH",
]


class TestScheduler:
    """Tests for TensorRT scheduler config."""

    def test_scheduler_config_accepted(self):
        """Scheduler section with valid policy validates."""
        config = TensorRTSchedulerConfig(
            capacity_scheduling_policy="GUARANTEED_NO_EVICT",
        )
        assert config.capacity_scheduling_policy == "GUARANTEED_NO_EVICT"

    @pytest.mark.parametrize("policy", _VALID_SCHEDULER_POLICIES)
    def test_valid_scheduler_policies(self, policy: str):
        """All valid scheduler policies are accepted."""
        config = TensorRTSchedulerConfig(capacity_scheduling_policy=policy)
        assert config.capacity_scheduling_policy == policy

    def test_invalid_scheduler_policy_rejected(self):
        """Invalid policy raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTSchedulerConfig(capacity_scheduling_policy="INVALID_POLICY")


# CFG-06: Build cache sub-config dropped (D1) - tests removed.
# build_cache fields remain settable via TensorRTConfig extra="allow" passthrough.


# ---------------------------------------------------------------------------
# CFG-07: Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    """Tests for TensorRT sampling config."""

    def test_sampling_config_accepted(self):
        """Sampling section with valid values validates (return_perf_metrics dropped D1)."""
        config = TensorRTSamplingConfig(
            min_tokens=10,
            n=4,
            ignore_eos=True,
        )
        assert config.min_tokens == 10
        assert config.n == 4
        assert config.ignore_eos is True

    def test_sampling_return_perf_metrics_is_extra_allow(self):
        """return_perf_metrics still accepted via extra='allow' passthrough."""
        config = TensorRTSamplingConfig(return_perf_metrics=True)
        # No ValidationError - extra="allow"
        assert getattr(config, "return_perf_metrics", None) is True

    def test_sampling_n_ge_1(self):
        """n=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTSamplingConfig(n=0)


# ---------------------------------------------------------------------------
# Integration with ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfigIntegration:
    """Tests for TensorRTConfig integration with ExperimentConfig."""

    def test_experiment_config_with_full_tensorrt(self):
        """ExperimentConfig with engine='tensorrt' and full tensorrt section validates."""
        config = ExperimentConfig(
            task={"model": "gpt2"},
            engine="tensorrt",
            tensorrt={
                "engine_params": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 2,
                    "max_batch_size": 8,
                    "max_input_len": 1024,
                    "max_seq_len": 2048,
                    "max_num_tokens": 4096,
                    "dtype": "bfloat16",
                    "fast_build": True,
                    "quant_config": {"quant_algo": "W4A16_AWQ"},
                    "kv_cache_config": {
                        "enable_block_reuse": True,
                        "free_gpu_memory_fraction": 0.9,
                    },
                    "scheduler_config": {
                        "capacity_scheduling_policy": "MAX_UTILIZATION",
                    },
                },
                "sampling_params": {
                    "min_tokens": 5,
                    "n": 1,
                    "ignore_eos": False,
                },
            },
        )
        assert config.engine == "tensorrt"
        assert config.tensorrt is not None
        ep = config.tensorrt.engine_params
        assert ep is not None
        assert ep.tensor_parallel_size == 2
        assert ep.pipeline_parallel_size == 2
        assert ep.max_num_tokens == 4096
        assert ep.quant_config is not None
        # quant_config is a raw dict in the generated engine config
        assert ep.quant_config["quant_algo"] == "W4A16_AWQ"
        assert ep.kv_cache_config is not None
        assert ep.kv_cache_config["enable_block_reuse"] is True
        assert ep.scheduler_config is not None
        assert ep.scheduler_config["capacity_scheduling_policy"] == "MAX_UTILIZATION"
        sp = config.tensorrt.sampling_params
        assert sp is not None
        assert sp.n == 1

    def test_tensorrt_extra_allow_forwards_unknown(self):
        """Extra fields on TensorRTConfig and sub-configs are accepted (not rejected)."""
        config = TensorRTConfig(
            engine_params={"tensor_parallel_size": 1, "custom_future_field": "value"}
        )
        # Should not raise - extra="allow"
        assert config.engine_params.tensor_parallel_size == 1

        quant = TensorRTQuantConfig(
            quant_algo="INT8",
            custom_quant_field=42,
        )
        assert quant.quant_algo == "INT8"

    def test_tensorrt_none_defaults(self):
        """Generated class: optional sub-configs default to None; engine_params defaults present."""
        config = TensorRTConfig()
        # Top-level sections default to None when not provided
        assert config.engine_params is None
        assert config.sampling_params is None
        # Verify engine_params fields when section is explicitly provided
        ep = TensorRTConfig(engine_params={}).engine_params
        assert ep.max_batch_size is None
        assert ep.max_input_len is None
        assert ep.max_seq_len is None
        assert ep.max_num_tokens is None
        assert ep.quant_config is None
        assert ep.kv_cache_config is None
        assert ep.scheduler_config is None
        # Generated defaults differ from OLD None-as-default convention:
        # tensor_parallel_size=1, pipeline_parallel_size=1, fast_build=False, dtype='auto'
        assert ep.tensor_parallel_size == 1
        assert ep.pipeline_parallel_size == 1
        assert ep.fast_build is False
        assert ep.dtype == "auto"
