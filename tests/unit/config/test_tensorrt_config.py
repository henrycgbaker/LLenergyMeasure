"""Unit tests for the generated TensorRT-LLM config (nested engine_params shape).

The tensorrt section is the generated
``llenergymeasure.engines.tensorrt.config.Config`` (engine_params /
sampling_params). Engine fields live on engine_params; quant_config /
kv_cache_config / scheduler_config round-trip as Any-typed dicts (not pydantic
sub-models). The fields are permissive (Any-typed, no Literal / bound
validators), so this file verifies the nested shape PARSES and round-trips,
not the deleted hand-written validators.

Coverage:
- Compile-time params (tensor_parallel_size, max_batch_size, dtype, fast_build, ...)
- Quantisation dict (quant_algo, kv_cache_quant_algo)
- KV cache dict (enable_block_reuse, free_gpu_memory_fraction, ...)
- Scheduler dict (capacity_scheduling_policy)
- Sampling params (min_tokens, n, ignore_eos)
- extra="allow" passthrough for unknown fields
"""

from __future__ import annotations

from llenergymeasure.config.models import ExperimentConfig

_TRT_DEFAULTS = {"task": {"model": "gpt2"}, "engine": "tensorrt"}


def _make_trt(**engine_params) -> ExperimentConfig:
    """Build an ExperimentConfig with the given tensorrt engine_params."""
    return ExperimentConfig(**_TRT_DEFAULTS, tensorrt={"engine_params": engine_params})


# ---------------------------------------------------------------------------
# Compile-time params
# ---------------------------------------------------------------------------


class TestCompileTimeParams:
    """Tests for TensorRT compile-time parameters on engine_params."""

    def test_tensorrt_compile_params_accepted(self):
        """All compile-time params validate when set together."""
        config = _make_trt(
            max_batch_size=8,
            max_input_len=1024,
            max_seq_len=2048,
            tensor_parallel_size=2,
            dtype="float16",
            fast_build=True,
        )
        ep = config.tensorrt.engine_params
        assert ep.max_batch_size == 8
        assert ep.max_input_len == 1024
        assert ep.max_seq_len == 2048
        assert ep.tensor_parallel_size == 2
        assert ep.dtype == "float16"
        assert ep.fast_build is True

    def test_tensorrt_dtype_bfloat16_accepted(self):
        """dtype='bfloat16' is valid (dtype is an Any passthrough, not a Literal)."""
        config = _make_trt(dtype="bfloat16")
        assert config.tensorrt.engine_params.dtype == "bfloat16"


# ---------------------------------------------------------------------------
# Quantisation (round-trips as an Any-typed dict)
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
    """Tests for the TensorRT quant_config dict."""

    def test_quant_config_round_trips(self):
        """quant_config keys round-trip verbatim as a dict on engine_params."""
        config = _make_trt(quant_config={"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"})
        quant = config.tensorrt.engine_params.quant_config
        assert quant == {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}

    def test_all_quant_algos_accepted(self):
        """All quant_algo values round-trip (no Literal restriction on the dict)."""
        for algo in _ALL_QUANT_ALGOS:
            config = _make_trt(quant_config={"quant_algo": algo})
            assert config.tensorrt.engine_params.quant_config["quant_algo"] == algo


# ---------------------------------------------------------------------------
# KV Cache (round-trips as an Any-typed dict)
# ---------------------------------------------------------------------------


class TestKvCache:
    """Tests for the TensorRT kv_cache_config dict."""

    def test_kv_cache_config_round_trips(self):
        """KV cache dict round-trips verbatim on engine_params."""
        config = _make_trt(
            kv_cache_config={
                "enable_block_reuse": True,
                "free_gpu_memory_fraction": 0.85,
                "max_tokens": 4096,
                "host_cache_size": 1073741824,
            }
        )
        kv = config.tensorrt.engine_params.kv_cache_config
        assert kv["enable_block_reuse"] is True
        assert kv["free_gpu_memory_fraction"] == 0.85
        assert kv["max_tokens"] == 4096
        assert kv["host_cache_size"] == 1073741824


# ---------------------------------------------------------------------------
# Scheduler (round-trips as an Any-typed dict)
# ---------------------------------------------------------------------------


_VALID_SCHEDULER_POLICIES = [
    "GUARANTEED_NO_EVICT",
    "MAX_UTILIZATION",
    "STATIC_BATCH",
]


class TestScheduler:
    """Tests for the TensorRT scheduler_config dict."""

    def test_scheduler_policies_round_trip(self):
        """All scheduler policies round-trip on the scheduler_config dict."""
        for policy in _VALID_SCHEDULER_POLICIES:
            config = _make_trt(scheduler_config={"capacity_scheduling_policy": policy})
            sched = config.tensorrt.engine_params.scheduler_config
            assert sched["capacity_scheduling_policy"] == policy


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    """Tests for the TensorRT sampling_params block."""

    def test_sampling_params_accepted(self):
        """sampling_params with valid values validates."""
        config = ExperimentConfig(
            **_TRT_DEFAULTS,
            tensorrt={"sampling_params": {"min_tokens": 10, "n": 4, "ignore_eos": True}},
        )
        sp = config.tensorrt.sampling_params
        assert sp.min_tokens == 10
        assert sp.n == 4
        assert sp.ignore_eos is True

    def test_sampling_return_perf_metrics_is_extra_allow(self):
        """Unknown sampling fields are accepted via extra='allow' passthrough."""
        config = ExperimentConfig(
            **_TRT_DEFAULTS,
            tensorrt={"sampling_params": {"return_perf_metrics": True}},
        )
        assert getattr(config.tensorrt.sampling_params, "return_perf_metrics", None) is True


# ---------------------------------------------------------------------------
# Integration with ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfigIntegration:
    """Tests for the generated tensorrt Config integration with ExperimentConfig."""

    def test_experiment_config_with_full_tensorrt(self):
        """ExperimentConfig with engine='tensorrt' and a full nested section validates."""
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
        assert ep.tensor_parallel_size == 2
        assert ep.pipeline_parallel_size == 2
        assert ep.max_num_tokens == 4096
        assert ep.quant_config == {"quant_algo": "W4A16_AWQ"}
        assert ep.kv_cache_config["enable_block_reuse"] is True
        assert ep.scheduler_config["capacity_scheduling_policy"] == "MAX_UTILIZATION"
        assert config.tensorrt.sampling_params is not None
        assert config.tensorrt.sampling_params.n == 1

    def test_tensorrt_extra_allow_forwards_unknown(self):
        """Extra fields on engine_params are accepted (not rejected)."""
        config = _make_trt(tensor_parallel_size=1, custom_future_field="value")
        ep = config.tensorrt.engine_params
        assert ep.tensor_parallel_size == 1
        assert getattr(ep, "custom_future_field", None) == "value"

    def test_tensorrt_sub_configs_default_none(self):
        """The Any-typed sub-config dicts default to None when not specified."""
        config = _make_trt()
        ep = config.tensorrt.engine_params
        assert ep.max_batch_size is None
        assert ep.max_input_len is None
        assert ep.max_seq_len is None
        assert ep.max_num_tokens is None
        assert ep.quant_config is None
        assert ep.kv_cache_config is None
        assert ep.scheduler_config is None
