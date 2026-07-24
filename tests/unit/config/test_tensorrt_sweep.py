"""Unit tests for TensorRT sweep expansion (CFG-08).

Tests that engine is a sweepable axis producing distinct experiment configs,
and that dotted nested sweep keys expand correctly into TensorRT sub-configs.
"""

from __future__ import annotations

from typing import cast

from llenergymeasure.config.grid import expand_grid


class TestEngineSweepAxis:
    """Tests for engine as a sweepable axis."""

    def test_engine_sweep_axis_produces_distinct_configs(self):
        """sweep over engine: [pytorch, tensorrt] produces 2 ExperimentConfig objects."""
        raw_study = {
            "task": {"model": "gpt2"},
            "serving_mode": "offline",
            "sweep": {
                "engine": ["transformers", "tensorrt"],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 2
        assert {c.engine for c in valid} == {"transformers", "tensorrt"}

    def test_engine_sweep_with_trt_scoped_params(self):
        """engine: [transformers, tensorrt] with tensorrt.tensor_parallel_size: [1, 2] produces 3 configs."""
        raw_study = {
            "task": {"model": "gpt2"},
            "engine": ["transformers", "tensorrt"],
            "serving_mode": "offline",
            "sweep": {
                "tensorrt.engine_params.tensor_parallel_size": [1, 2],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        # transformers: 1 config (no scoped dims), tensorrt: 2 configs (tensor_parallel_size=[1,2])
        assert len(valid) == 3
        pytorch_configs = [c for c in valid if c.engine == "transformers"]
        tensorrt_configs = [c for c in valid if c.engine == "tensorrt"]
        assert len(pytorch_configs) == 1
        assert len(tensorrt_configs) == 2
        tp_sizes = {c.tensorrt.engine_params.tensor_parallel_size for c in tensorrt_configs}
        assert tp_sizes == {1, 2}


class TestDottedNestedSweep:
    """Tests for dotted sweep keys expanding into nested TensorRT config."""

    def test_dotted_quant_algo_sweep(self):
        """tensorrt.quant_config.quant_algo: [INT8, FP8, W4A16_AWQ] produces 3 configs."""
        raw_study = {
            "task": {"model": "gpt2"},
            "engine": "tensorrt",
            "serving_mode": "offline",
            # quant_config exists only on the trt backend's TrtLlmArgs, so pin it.
            "tensorrt": {"engine_params": {"backend": "trt"}},
            "sweep": {
                "tensorrt.engine_params.quant_config.quant_algo": ["INT8", "FP8", "W4A16_AWQ"],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 3
        algos = [cast(dict, c.tensorrt.engine_params.quant_config)["quant_algo"] for c in valid]
        assert set(algos) == {"INT8", "FP8", "W4A16_AWQ"}

    def test_dotted_nested_max_num_tokens_sweep(self):
        """tensorrt.max_num_tokens: [2048, 4096] produces 2 configs."""
        raw_study = {
            "task": {"model": "gpt2"},
            "engine": "tensorrt",
            "serving_mode": "offline",
            "sweep": {
                "tensorrt.engine_params.max_num_tokens": [2048, 4096],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 2
        token_values = {c.tensorrt.engine_params.max_num_tokens for c in valid}
        assert token_values == {2048, 4096}

    def test_full_tensorrt_study_yaml_parses(self):
        """Comprehensive study YAML with all sub-sections and quant sweep round-trips."""
        raw_study = {
            "task": {"model": "gpt2"},
            "engine": "tensorrt",
            "serving_mode": "offline",
            "tensorrt": {
                "engine_params": {
                    "backend": "trt",
                    "tensor_parallel_size": 2,
                    "max_batch_size": 8,
                    "max_input_len": 1024,
                    "max_seq_len": 2048,
                    "dtype": "bfloat16",
                    "kv_cache_config": {
                        "enable_block_reuse": True,
                        "free_gpu_memory_fraction": 0.9,
                    },
                    "scheduler_config": {
                        "capacity_scheduling_policy": "MAX_UTILIZATION",
                    },
                },
                "sampling_params": {
                    "n": 1,
                },
            },
            "sweep": {
                "tensorrt.engine_params.quant_config.quant_algo": ["INT8", "W4A16_AWQ"],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 2
        for config in valid:
            assert config.engine == "tensorrt"
            assert config.tensorrt is not None
            assert config.tensorrt.engine_params.tensor_parallel_size == 2
            assert config.tensorrt.engine_params.max_batch_size == 8
            assert config.tensorrt.engine_params.dtype == "bfloat16"
            assert config.tensorrt.engine_params.kv_cache_config is not None
            assert config.tensorrt.engine_params.kv_cache_config["enable_block_reuse"] is True
            assert config.tensorrt.engine_params.scheduler_config is not None
            assert config.tensorrt.sampling_params is not None
            assert config.tensorrt.sampling_params.n == 1
        algos = {cast(dict, c.tensorrt.engine_params.quant_config)["quant_algo"] for c in valid}
        assert algos == {"INT8", "W4A16_AWQ"}

    def test_quant_algo_sweep_values_pass_through(self):
        """Unknown quant algo is not skipped: quant_config is an Any passthrough.

        The generated tensorrt engine_params.quant_config is an untyped (Any)
        passthrough - the mined schema did not surface TRT-LLM's quant_algo enum,
        so the Literal validation that once skipped invalid values is gone
        (discovery debt, mirroring the dtype passthrough on the other engines).
        Both sweep values parse cleanly; TRT-LLM validates the algo at load.
        """
        raw_study = {
            "task": {"model": "gpt2"},
            "engine": "tensorrt",
            "serving_mode": "offline",
            # quant_config exists only on the trt backend's TrtLlmArgs, so pin it.
            "tensorrt": {"engine_params": {"backend": "trt"}},
            "sweep": {
                "tensorrt.engine_params.quant_config.quant_algo": ["INT8", "INVALID_VALUE"],
            },
        }
        valid, skipped = expand_grid(raw_study)
        assert len(valid) == 2
        assert len(skipped) == 0
        algos = {cast(dict, c.tensorrt.engine_params.quant_config)["quant_algo"] for c in valid}
        assert algos == {"INT8", "INVALID_VALUE"}
