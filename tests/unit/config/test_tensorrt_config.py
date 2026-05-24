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
# CFG-02/04/05: Quant / KV cache / Scheduler sub-configs
# ---------------------------------------------------------------------------
# Per phase3_audit_llem_fields.md, sub-bundles (quant_config, kv_cache_config,
# scheduler_config) are currently typed `dict[str, Any]` on the generated
# `engines.tensorrt.config.EngineParams` - they only become validated nested
# Pydantic classes once the producer cells re-run with the $defs propagation
# commit (NEEDS_DEFS_PROPAGATION). Until then the old TensorRTQuantConfig /
# TensorRTKvCacheConfig / TensorRTSchedulerConfig classes have no NEW
# equivalent to migrate to; the tests that directly instantiated them have
# been removed. Sub-bundle structure still passes through via extra='allow'
# (see test_experiment_config_with_full_tensorrt and quant_config /
# kv_cache_config / scheduler_config dict assertions below).
# When $defs propagation lands these tests should be re-added against the
# regenerated nested Pydantic sub-classes.


# ---------------------------------------------------------------------------
# CFG-07: Sampling - migrated to use Config(sampling_params={...}) nested shape
# ---------------------------------------------------------------------------


class TestSampling:
    """Tests for TensorRT sampling params (now under sampling_params nested)."""

    def test_sampling_config_accepted(self):
        """Sampling section with valid values validates."""
        config = TensorRTConfig(
            sampling_params={"min_tokens": 10, "n": 4, "ignore_eos": True}
        )
        assert config.sampling_params.min_tokens == 10
        assert config.sampling_params.n == 4
        assert config.sampling_params.ignore_eos is True

    def test_sampling_extra_allow_forwards_unknown(self):
        """Unknown sampling fields accepted via extra='allow' passthrough."""
        config = TensorRTConfig(sampling_params={"return_perf_metrics": True})
        assert config.sampling_params is not None
        # extra="allow"; field surfaces via model_dump
        assert config.sampling_params.model_dump().get("return_perf_metrics") is True

    def test_sampling_n_accepts_positive(self):
        """n=1 accepted (generated class has no ge=1 constraint today)."""
        config = TensorRTConfig(sampling_params={"n": 1})
        assert config.sampling_params.n == 1


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
        """Extra fields on engine_params accepted (not rejected) via extra='allow'."""
        config = TensorRTConfig(
            engine_params={"tensor_parallel_size": 1, "custom_future_field": "value"}
        )
        assert config.engine_params.tensor_parallel_size == 1
        # Custom field surfaces via model_dump
        assert config.engine_params.model_dump().get("custom_future_field") == "value"

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
