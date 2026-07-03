"""Unit tests for the generated TensorRT-LLM config (nested engine_params shape).

The tensorrt section is the generated
``llenergymeasure.engines.tensorrt.config.Config`` (engine_params /
sampling_params). Engine fields live on engine_params; quant_config /
kv_cache_config / scheduler_config round-trip as Any-typed dicts (not pydantic
sub-models). The fields are permissive (Any-typed or plain str, no Literal /
Pydantic-bound validators), so this file verifies the nested shape PARSES and
round-trips, and moves bounds that were re-homed to the mined rules corpus
into ExperimentConfig-level assertions.

Coverage:
- CFG-01: Compile-time params (tensor_parallel_size, pipeline_parallel_size,
          max_batch_size, max_input_len, max_seq_len, max_num_tokens, dtype,
          fast_build)
- CFG-02: Quantisation dict (quant_algo, kv_cache_quant_algo)
- CFG-03: (Removed) Calibration sub-config dropped - D3 build-only PTQ.
- CFG-04: KV cache dict (enable_block_reuse, free_gpu_memory_fraction, ...)
- CFG-05: Scheduler dict (capacity_scheduling_policy)
- CFG-06: (Removed) Build cache sub-config dropped - D1 engine-cache plumbing.
- CFG-07: Sampling (min_tokens, n, ignore_eos; return_perf_metrics dropped D1)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig

_TRT_DEFAULTS = {"task": {"model": "gpt2"}, "engine": "tensorrt"}


def _make_trt(**engine_params) -> ExperimentConfig:
    """Build an ExperimentConfig with the given tensorrt engine_params."""
    return ExperimentConfig(**_TRT_DEFAULTS, tensorrt={"engine_params": engine_params})


# ---------------------------------------------------------------------------
# CFG-01: Compile-time params
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
        """dtype='bfloat16' is valid (dtype is a plain str passthrough, not a Literal)."""
        config = _make_trt(dtype="bfloat16")
        assert config.tensorrt.engine_params.dtype == "bfloat16"

    def test_tensorrt_dtype_float32_accepted(self):
        """dtype='float32' is accepted at parse time (plain str, no Literal enforcement).

        The dtype field is now an un-narrowed str (discovery debt: the mined schema
        did not surface TRT-LLM's internal dtype enum). TRT-LLM itself rejects fp32
        at runtime; the static constraint will be restored when a container re-mine
        surfaces the real enum.
        """
        config = _make_trt(dtype="float32")
        assert config.tensorrt.engine_params.dtype == "float32"

    def test_tensorrt_tensor_parallel_size_ge_1(self):
        """tensor_parallel_size=0 raises ValidationError (re-homed rule fires at ExperimentConfig)."""
        with pytest.raises(ValidationError):
            _make_trt(tensor_parallel_size=0)

    def test_tensorrt_max_batch_size_nonneg(self):
        """max_batch_size=0 is accepted (rule fires at < 0, not < 1).

        The mined rule `tensorrt_raises_max_batch_size_lt_0_cuda_graph_max_batch_size`
        uses `< 0` (TRT-LLM's ``cuda_graph_config.max_batch_size must be non-negative``),
        so 0 is valid at parse time. Negative values are rejected.
        """
        config = _make_trt(max_batch_size=0)
        assert config.tensorrt.engine_params.max_batch_size == 0

    def test_tensorrt_max_batch_size_negative_rejected(self):
        """max_batch_size=-1 raises ValidationError (mined rule: must be non-negative)."""
        with pytest.raises(ValidationError):
            _make_trt(max_batch_size=-1)

    def test_tensorrt_max_input_len_ge_1(self):
        """max_input_len=0 raises ValidationError (re-homed rule fires at ExperimentConfig)."""
        with pytest.raises(ValidationError):
            _make_trt(max_input_len=0)

    def test_tensorrt_max_seq_len_ge_1(self):
        """max_seq_len=0 raises ValidationError (re-homed rule fires at ExperimentConfig)."""
        with pytest.raises(ValidationError):
            _make_trt(max_seq_len=0)


# ---------------------------------------------------------------------------
# CFG-02: Quantisation (round-trips as an Any-typed dict)
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
    """Tests for the TensorRT quant_config dict.

    quant_config is now an Any-typed field: it round-trips as a plain dict
    without any Pydantic validation on the inner keys. Literal enforcement
    (quant_algo, kv_cache_quant_algo) was on the hand-written TensorRTQuantConfig
    which no longer exists.
    """

    @pytest.mark.parametrize("algo", _ALL_QUANT_ALGOS)
    def test_valid_quant_algo_accepted(self, algo: str):
        """All 8 required QuantAlgo values are accepted (dict round-trip)."""
        config = _make_trt(quant_config={"quant_algo": algo})
        assert config.tensorrt.engine_params.quant_config == {"quant_algo": algo}

    def test_quant_config_round_trips_verbatim(self):
        """quant_config keys round-trip verbatim as a dict on engine_params."""
        config = _make_trt(quant_config={"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"})
        quant = config.tensorrt.engine_params.quant_config
        assert quant == {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}

    def test_arbitrary_quant_algo_accepted(self):
        """Arbitrary quant_algo values round-trip (no Literal restriction on the dict).

        The hand-written TensorRTQuantConfig rejected unknown values via Literal;
        the generated config accepts any dict without Pydantic validation on inner
        fields (discovery debt for the quant_config sub-schema).
        """
        config = _make_trt(quant_config={"quant_algo": "fp8"})  # lowercase
        quant_config = config.tensorrt.engine_params.quant_config
        assert quant_config is not None
        assert quant_config["quant_algo"] == "fp8"


# CFG-03: Calibration sub-config dropped (D3) - tests removed.
# calib fields remain settable via engine_params extra="allow" passthrough.


# ---------------------------------------------------------------------------
# CFG-04: KV Cache (round-trips as an Any-typed dict)
# ---------------------------------------------------------------------------


class TestKvCache:
    """Tests for the TensorRT kv_cache_config dict.

    kv_cache_config is now an Any-typed field: it round-trips as a plain dict.
    Pydantic bounds (free_gpu_memory_fraction range) that were on the
    hand-written TensorRTKvCacheConfig are no longer enforced at parse time.
    """

    def test_kv_cache_config_accepted(self):
        """KV cache dict with valid values round-trips."""
        config = _make_trt(
            kv_cache_config={
                "enable_block_reuse": True,
                "free_gpu_memory_fraction": 0.85,
                "max_tokens": 4096,
                "host_cache_size": 1073741824,
            }
        )
        kv = config.tensorrt.engine_params.kv_cache_config
        assert kv is not None
        assert kv["enable_block_reuse"] is True
        assert kv["free_gpu_memory_fraction"] == 0.85
        assert kv["max_tokens"] == 4096
        assert kv["host_cache_size"] == 1073741824

    def test_kv_cache_free_gpu_memory_fraction_round_trips(self):
        """free_gpu_memory_fraction values round-trip (no Pydantic range check on dict)."""
        # Valid boundaries accepted
        config = _make_trt(kv_cache_config={"free_gpu_memory_fraction": 0.0})
        kv = config.tensorrt.engine_params.kv_cache_config
        assert kv is not None
        assert kv["free_gpu_memory_fraction"] == 0.0
        config = _make_trt(kv_cache_config={"free_gpu_memory_fraction": 1.0})
        kv = config.tensorrt.engine_params.kv_cache_config
        assert kv is not None
        assert kv["free_gpu_memory_fraction"] == 1.0


# ---------------------------------------------------------------------------
# CFG-05: Scheduler (round-trips as an Any-typed dict)
# ---------------------------------------------------------------------------


_VALID_SCHEDULER_POLICIES = [
    "GUARANTEED_NO_EVICT",
    "MAX_UTILIZATION",
    "STATIC_BATCH",
]


class TestScheduler:
    """Tests for the TensorRT scheduler_config dict.

    scheduler_config is now an Any-typed field: it round-trips as a plain dict.
    Literal enforcement on capacity_scheduling_policy was on the hand-written
    TensorRTSchedulerConfig which no longer exists.
    """

    def test_scheduler_config_accepted(self):
        """Scheduler dict with valid policy round-trips."""
        config = _make_trt(scheduler_config={"capacity_scheduling_policy": "GUARANTEED_NO_EVICT"})
        sched = config.tensorrt.engine_params.scheduler_config
        assert sched is not None
        assert sched["capacity_scheduling_policy"] == "GUARANTEED_NO_EVICT"

    @pytest.mark.parametrize("policy", _VALID_SCHEDULER_POLICIES)
    def test_valid_scheduler_policies(self, policy: str):
        """All valid scheduler policies round-trip on the dict."""
        config = _make_trt(scheduler_config={"capacity_scheduling_policy": policy})
        sched = config.tensorrt.engine_params.scheduler_config
        assert sched is not None
        assert sched["capacity_scheduling_policy"] == policy

    def test_arbitrary_scheduler_policy_accepted(self):
        """Arbitrary scheduler policy values round-trip (no Literal restriction on dict)."""
        config = _make_trt(scheduler_config={"capacity_scheduling_policy": "INVALID_POLICY"})
        sched = config.tensorrt.engine_params.scheduler_config
        assert sched is not None
        assert sched["capacity_scheduling_policy"] == "INVALID_POLICY"


# CFG-06: Build cache sub-config dropped (D1) - tests removed.
# build_cache fields remain settable via engine_params extra="allow" passthrough.


# ---------------------------------------------------------------------------
# CFG-07: Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    """Tests for TensorRT sampling_params block."""

    def test_sampling_config_accepted(self):
        """Sampling section with valid values validates (return_perf_metrics dropped D1)."""
        config = ExperimentConfig(
            **_TRT_DEFAULTS,
            tensorrt={"sampling_params": {"min_tokens": 10, "n": 4, "ignore_eos": True}},
        )
        sp = config.tensorrt.sampling_params
        assert sp.min_tokens == 10
        assert sp.n == 4
        assert sp.ignore_eos is True

    def test_sampling_return_perf_metrics_is_extra_allow(self):
        """return_perf_metrics still accepted via extra='allow' passthrough."""
        config = ExperimentConfig(
            **_TRT_DEFAULTS,
            tensorrt={"sampling_params": {"return_perf_metrics": True}},
        )
        # No ValidationError - extra="allow"
        assert getattr(config.tensorrt.sampling_params, "return_perf_metrics", None) is True

    def test_sampling_n_ge_1(self):
        """n=0 raises ValidationError (re-homed rule fires at ExperimentConfig)."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                **_TRT_DEFAULTS,
                tensorrt={"sampling_params": {"n": 0}},
            )


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
        assert ep.kv_cache_config is not None
        assert ep.kv_cache_config["enable_block_reuse"] is True
        assert ep.scheduler_config is not None
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
        # tensor_parallel_size and pipeline_parallel_size default to 1 (not None)
        # in the generated config (EngineParams defaults match the engine's own defaults)
        assert ep.tensor_parallel_size == 1
        assert ep.pipeline_parallel_size == 1
