"""Golden characterisation of the engine kwarg-builder / nested-config accessor contract.

These tests freeze the exact dicts produced by the pure kwarg builders on every
engine plugin. The frozen literals were captured from the current main branch and
are asserted verbatim: they are a tripwire, not a re-derivation. Refactors that
rename an accessor, drop a field, or change the shape of the generated nested
config will break an exact-equality assertion here rather than silently drifting
the measured behaviour.

What is locked:

  - TransformersEngine._build_generate_kwargs: sampling + generation-control
    forwarding and the greedy-strip rule (temperature==0 or do_sample==False
    removes temperature/top_k/top_p/min_p and forces do_sample=False).
  - VLLMEngine._build_llm_kwargs: model_dump(exclude_none=True) plus the
    attention-backend flattening and offload_params set-coercion.
  - VLLMEngine._build_sampling_kwargs: the effective SamplingParams kwargs, and
    the beam-search preemption contract (returns {} when a beam_search block is
    present so the caller dispatches to the beam path).
  - VLLMEngine._build_beam_search_params: the BeamSearchParams kwargs assembled
    from the Any-typed beam_search sub-dict.
  - TensorRTEngine._build_llm_kwargs: scalar forwarding, backend selecting the
    constructor class (never forwarded as a kwarg), the TRT-build-only knobs
    (fast_build / build cache) staying off the pytorch backend, and a declared
    sub-config raising loudly on a plain host rather than silently vanishing.
  - TensorRTEngine._build_sampling_kwargs: the effective SamplingParams kwargs
    including the injected seed and max_tokens.

These builders import no engine library, so they run on a plain host. The
transformers generate path (_run_batch), the vLLM/TensorRT observed-capture
paths, and the native SamplingParams/BeamSearchParams/QuantConfig object
construction all require the real engine libraries and are exercised by the
GPU/container suites; the measurement fixes those paths carry (see PR #730) are
locked by test_transformers_token_counting.py, test_extended_capture.py,
test_transformers_observed_capture.py, and test_engine_protocol.py.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.engines.vllm import VLLMEngine
from tests.conftest import make_config


@pytest.fixture(autouse=True)
def _deterministic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the opinionated env-var defaults the builders read.

    _build_llm_kwargs reads trust_remote_code (vLLM) and the TRT build-cache
    toggle; leaving ambient env in play would make the golden dicts host
    dependent. Removing the vars pins them to their library defaults.
    """
    for var in (
        "LLEM_TRUST_REMOTE_CODE",
        "LLEM_TRT_BUILD_CACHE_ENABLED",
        "LLEM_TRT_BUILD_CACHE_PATH",
    ):
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# Transformers: _build_generate_kwargs
# =============================================================================


class TestTransformersGenerateKwargs:
    def test_greedy_strips_sampling_and_sets_do_sample_false(self) -> None:
        """temperature==0 removes top_k/top_p/min_p, forces do_sample=False."""
        config = make_config(
            model="test-model",
            engine="transformers",
            transformers={
                "sampling_params": {
                    "temperature": 0.0,
                    "top_k": 50,
                    "top_p": 0.9,
                    "min_p": 0.1,
                },
                "engine_params": {"use_cache": True},
            },
        )
        assert TransformersEngine()._build_generate_kwargs(config) == {
            "use_cache": True,
            "do_sample": False,
        }

    def test_do_sample_false_strips_sampling(self) -> None:
        """do_sample=False (with non-zero temperature) also strips sampling knobs."""
        config = make_config(
            model="test-model",
            engine="transformers",
            transformers={"sampling_params": {"temperature": 0.7, "top_k": 50, "do_sample": False}},
        )
        assert TransformersEngine()._build_generate_kwargs(config) == {"do_sample": False}

    def test_sampling_and_generation_control_forwarded(self) -> None:
        """Sampling knobs plus every engine_params generation-control field forward."""
        config = make_config(
            model="test-model",
            engine="transformers",
            transformers={
                "sampling_params": {
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.95,
                    "min_p": 0.05,
                },
                "engine_params": {
                    "use_cache": True,
                    "cache_implementation": "static",
                    "num_beams": 4,
                    "length_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                    "prompt_lookup_num_tokens": 5,
                },
            },
        )
        assert TransformersEngine()._build_generate_kwargs(config) == {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "use_cache": True,
            "cache_implementation": "static",
            "num_beams": 4,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "prompt_lookup_num_tokens": 5,
        }


# =============================================================================
# vLLM: _build_llm_kwargs / _build_sampling_kwargs / _build_beam_search_params
# =============================================================================


class TestVLLMKwargBuilders:
    def test_build_llm_kwargs_flattens_attention_and_offload(self) -> None:
        """Attention backend is hoisted, offload_params becomes a set, defaults dump."""
        config = make_config(
            model="test-model",
            engine="vllm",
            random_seed=1234,
            vllm={
                "engine_params": {
                    "tensor_parallel_size": 2,
                    "max_model_len": 4096,
                    "attention": {"backend": "FLASHINFER", "block_size": 16},
                    "offload_params": ["cpu", "disk"],
                }
            },
        )
        assert VLLMEngine()._build_llm_kwargs(config) == {
            "model": "test-model",
            "trust_remote_code": False,
            "seed": 1234,
            "dtype": "auto",
            "gpu_memory_utilization": 0.9,
            "cpu_offload_gb": 0,
            "kv_cache_dtype": "auto",
            "enforce_eager": False,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "max_model_len": 4096,
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 1,
            "disable_custom_all_reduce": False,
            "offload_params": {"cpu", "disk"},
            "attention_backend": "FLASHINFER",
            "block_size": 16,
        }

    def test_build_sampling_kwargs_non_beam(self) -> None:
        """Non-beam path dumps the sampling defaults and injects max_tokens."""
        config = make_config(
            model="test-model",
            engine="vllm",
            max_output_tokens=64,
            vllm={"sampling_params": {"temperature": 0.7, "top_p": 0.9, "top_k": 40}},
        )
        assert VLLMEngine._build_sampling_kwargs(config) == {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.0,
            "min_tokens": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "ignore_eos": False,
            "n": 1,
            "max_tokens": 64,
        }

    def test_build_sampling_kwargs_beam_active_returns_empty(self) -> None:
        """A beam_search block preempts the sampling path: returns {}."""
        config = make_config(
            model="test-model",
            engine="vllm",
            max_output_tokens=64,
            vllm={
                "engine_params": {
                    "beam_search": {"beam_width": 4, "length_penalty": 1.0, "early_stopping": True}
                },
                "sampling_params": {"temperature": 0.7},
            },
        )
        assert VLLMEngine._build_sampling_kwargs(config) == {}

    def test_build_beam_search_params_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Beam kwargs come from the Any-typed beam_search dict plus max_tokens.

        vllm is not importable on host, so a capturing stand-in stands in for
        vllm.BeamSearchParams; the frozen dict is the kwargs it receives.
        """
        captured: dict[str, Any] = {}

        class _FakeBeamSearchParams:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.BeamSearchParams = _FakeBeamSearchParams  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        config = make_config(
            model="test-model",
            engine="vllm",
            max_output_tokens=64,
            vllm={
                "engine_params": {
                    "beam_search": {"beam_width": 4, "length_penalty": 1.0, "early_stopping": True}
                }
            },
        )
        beam_cfg = config.engine_sub_dict("beam_search")
        VLLMEngine._build_beam_search_params(config, beam_cfg)
        assert captured == {
            "beam_width": 4,
            "length_penalty": 1.0,
            "early_stopping": True,
            "max_tokens": 64,
        }


# =============================================================================
# TensorRT: _build_llm_kwargs / _build_sampling_kwargs
# =============================================================================


class TestTensorRTKwargBuilders:
    def test_build_llm_kwargs_scalar_forwarding(self) -> None:
        """Scalars forward on the pytorch backend; backend + TRT-build knobs stay off.

        ``backend`` selects the constructor class (never a kwarg) and the
        TRT-build-only ``fast_build`` (absent from the pytorch TorchLlmArgs) is
        dropped rather than forwarded.
        """
        config = make_config(
            model="test-model",
            engine="tensorrt",
            tensorrt={
                "engine_params": {
                    "tensor_parallel_size": 2,
                    "max_batch_size": 8,
                    "dtype": "bfloat16",
                    "backend": "pytorch",
                }
            },
        )
        assert TensorRTEngine()._build_llm_kwargs(config) == {
            "model": "test-model",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "max_batch_size": 8,
            "max_num_tokens": 8192,
            "dtype": "bfloat16",
        }

    def test_build_llm_kwargs_sub_config_declared_raises_loud_on_host(self) -> None:
        """A declared sub-config raises loudly on a plain host, never vanishes silently.

        The Any-typed sub-config dicts are popped from the scalar dump; building
        the native KvCacheConfig object needs tensorrt_llm, so on a plain host the
        import fails and _build_llm_kwargs raises EngineError rather than silently
        measuring a different configuration than the user declared.
        """
        from llenergymeasure.utils.exceptions import EngineError

        config = make_config(
            model="test-model",
            engine="tensorrt",
            tensorrt={
                "engine_params": {
                    "tensor_parallel_size": 1,
                    "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
                }
            },
        )
        with pytest.raises(EngineError, match="kv_cache_config was declared"):
            TensorRTEngine()._build_llm_kwargs(config)

    def test_build_llm_kwargs_scalars_only_no_sub_config_keys(self) -> None:
        """With no sub-configs declared, only scalars forward (no raw dict leakage)."""
        config = make_config(
            model="test-model",
            engine="tensorrt",
            tensorrt={"engine_params": {"tensor_parallel_size": 1}},
        )
        built = TensorRTEngine()._build_llm_kwargs(config)
        assert built == {
            "model": "test-model",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_num_tokens": 8192,
            "dtype": "auto",
        }
        for key in ("quant_config", "kv_cache_config", "scheduler_config", "backend", "fast_build"):
            assert key not in built

    def test_build_sampling_kwargs(self) -> None:
        """Sampling dump plus the injected seed and max_tokens."""
        config = make_config(
            model="test-model",
            engine="tensorrt",
            random_seed=99,
            max_output_tokens=32,
            tensorrt={"sampling_params": {"temperature": 0.8, "top_k": 20, "top_p": 0.9}},
        )
        assert TensorRTEngine()._build_sampling_kwargs(config) == {
            "temperature": 0.8,
            "top_k": 20,
            "top_p": 0.9,
            "n": 1,
            "ignore_eos": False,
            "seed": 99,
            "max_tokens": 32,
        }
