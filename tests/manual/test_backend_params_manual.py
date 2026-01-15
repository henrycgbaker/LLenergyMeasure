#!/usr/bin/env python3
"""Manual testing script for backend-specific parameters.

This script verifies that all backend parameters:
1. Parse correctly from dict/YAML-style input
2. Validate constraints (bounds, types, enums)
3. Pass through to backend correctly via _build_*_kwargs methods

Run with: poetry run python tests/manual/test_backend_params_manual.py
"""

from __future__ import annotations

import sys
from typing import Any

from pydantic import ValidationError

# Import all config models
from llm_energy_measure.config.backend_configs import (
    PyTorchAssistedGenerationConfig,
    PyTorchConfig,
    VLLMAttentionConfig,
    VLLMConfig,
    VLLMLoRAConfig,
    VLLMSpeculativeConfig,
)
from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.constants import PRESETS


def header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def test_pass(name: str) -> None:
    """Print a passing test."""
    print(f"  ✓ {name}")


def test_fail(name: str, error: str) -> None:
    """Print a failing test."""
    print(f"  ✗ {name}: {error}")


def expect_valid(name: str, factory: callable) -> Any:
    """Expect a config to be valid."""
    try:
        result = factory()
        test_pass(name)
        return result
    except Exception as e:
        test_fail(name, str(e))
        return None


def expect_invalid(name: str, factory: callable, match: str = "") -> bool:
    """Expect a config to be invalid."""
    try:
        factory()
        test_fail(name, "Expected ValidationError but config was valid")
        return False
    except ValidationError as e:
        if match and match.lower() not in str(e).lower():
            test_fail(name, f"Expected error containing '{match}', got: {e}")
            return False
        test_pass(name)
        return True
    except Exception as e:
        test_fail(name, f"Unexpected error: {e}")
        return False


# =============================================================================
# vLLM Config Tests
# =============================================================================


def test_vllm_attention_config() -> None:
    """Test VLLMAttentionConfig parameters."""
    subheader("VLLMAttentionConfig")

    # Valid backends
    for backend in ["auto", "FLASH_ATTN", "FLASHINFER", "TORCH_SDPA"]:
        expect_valid(
            f"backend={backend}",
            lambda b=backend: VLLMAttentionConfig(backend=b),
        )

    # Invalid backend
    expect_invalid(
        "backend=invalid (should fail)",
        lambda: VLLMAttentionConfig(backend="invalid"),
    )

    # Flash versions
    expect_valid("flash_version=2", lambda: VLLMAttentionConfig(flash_version=2))
    expect_valid("flash_version=3", lambda: VLLMAttentionConfig(flash_version=3))
    expect_invalid(
        "flash_version=1 (should fail)",
        lambda: VLLMAttentionConfig(flash_version=1),
    )

    # disable_sliding_window
    expect_valid(
        "disable_sliding_window=True",
        lambda: VLLMAttentionConfig(disable_sliding_window=True),
    )


def test_vllm_speculative_config() -> None:
    """Test VLLMSpeculativeConfig parameters."""
    subheader("VLLMSpeculativeConfig")

    # Valid methods
    for method in ["ngram", "eagle", "eagle3", "medusa", "mlp", "lookahead"]:
        expect_valid(
            f"method={method}",
            lambda m=method: VLLMSpeculativeConfig(method=m),
        )

    expect_invalid(
        "method=invalid (should fail)",
        lambda: VLLMSpeculativeConfig(method="invalid"),
    )

    # num_tokens bounds (1-10)
    expect_valid("num_tokens=1", lambda: VLLMSpeculativeConfig(num_tokens=1))
    expect_valid("num_tokens=10", lambda: VLLMSpeculativeConfig(num_tokens=10))
    expect_invalid(
        "num_tokens=0 (should fail)",
        lambda: VLLMSpeculativeConfig(num_tokens=0),
    )
    expect_invalid(
        "num_tokens=11 (should fail)",
        lambda: VLLMSpeculativeConfig(num_tokens=11),
    )

    # ngram_min (>= 1)
    expect_valid("ngram_min=1", lambda: VLLMSpeculativeConfig(ngram_min=1))
    expect_valid("ngram_min=5", lambda: VLLMSpeculativeConfig(ngram_min=5))
    expect_invalid(
        "ngram_min=0 (should fail)",
        lambda: VLLMSpeculativeConfig(ngram_min=0),
    )

    # ngram_max (optional)
    expect_valid("ngram_max=None", lambda: VLLMSpeculativeConfig(ngram_max=None))
    expect_valid("ngram_max=4", lambda: VLLMSpeculativeConfig(ngram_max=4))

    # draft_tp_size (>= 1)
    expect_valid("draft_tp_size=1", lambda: VLLMSpeculativeConfig(draft_tp_size=1))
    expect_valid("draft_tp_size=4", lambda: VLLMSpeculativeConfig(draft_tp_size=4))
    expect_invalid(
        "draft_tp_size=0 (should fail)",
        lambda: VLLMSpeculativeConfig(draft_tp_size=0),
    )

    # model
    expect_valid(
        "model=TinyLlama",
        lambda: VLLMSpeculativeConfig(model="TinyLlama/TinyLlama-1.1B"),
    )


def test_vllm_lora_config() -> None:
    """Test VLLMLoRAConfig parameters."""
    subheader("VLLMLoRAConfig")

    expect_valid("enabled=True", lambda: VLLMLoRAConfig(enabled=True))

    # max_loras (>= 1)
    expect_valid("max_loras=1", lambda: VLLMLoRAConfig(max_loras=1))
    expect_valid("max_loras=8", lambda: VLLMLoRAConfig(max_loras=8))
    expect_invalid(
        "max_loras=0 (should fail)",
        lambda: VLLMLoRAConfig(max_loras=0),
    )

    # max_rank (>= 1)
    expect_valid("max_rank=16", lambda: VLLMLoRAConfig(max_rank=16))
    expect_valid("max_rank=64", lambda: VLLMLoRAConfig(max_rank=64))
    expect_invalid(
        "max_rank=0 (should fail)",
        lambda: VLLMLoRAConfig(max_rank=0),
    )

    # extra_vocab_size (>= 0)
    expect_valid("extra_vocab_size=0", lambda: VLLMLoRAConfig(extra_vocab_size=0))
    expect_valid("extra_vocab_size=512", lambda: VLLMLoRAConfig(extra_vocab_size=512))
    expect_invalid(
        "extra_vocab_size=-1 (should fail)",
        lambda: VLLMLoRAConfig(extra_vocab_size=-1),
    )


def test_vllm_config() -> None:
    """Test VLLMConfig parameters."""
    subheader("VLLMConfig - Memory & Batching")

    # max_num_seqs (1-1024)
    expect_valid("max_num_seqs=1", lambda: VLLMConfig(max_num_seqs=1))
    expect_valid("max_num_seqs=256", lambda: VLLMConfig(max_num_seqs=256))
    expect_valid("max_num_seqs=1024", lambda: VLLMConfig(max_num_seqs=1024))
    expect_invalid(
        "max_num_seqs=0 (should fail)",
        lambda: VLLMConfig(max_num_seqs=0),
    )
    expect_invalid(
        "max_num_seqs=1025 (should fail)",
        lambda: VLLMConfig(max_num_seqs=1025),
    )

    # max_num_batched_tokens (optional)
    expect_valid(
        "max_num_batched_tokens=None",
        lambda: VLLMConfig(max_num_batched_tokens=None),
    )
    expect_valid(
        "max_num_batched_tokens=8192",
        lambda: VLLMConfig(max_num_batched_tokens=8192),
    )

    # gpu_memory_utilization (0.5-0.99)
    expect_valid(
        "gpu_memory_utilization=0.5",
        lambda: VLLMConfig(gpu_memory_utilization=0.5),
    )
    expect_valid(
        "gpu_memory_utilization=0.9",
        lambda: VLLMConfig(gpu_memory_utilization=0.9),
    )
    expect_valid(
        "gpu_memory_utilization=0.99",
        lambda: VLLMConfig(gpu_memory_utilization=0.99),
    )
    expect_invalid(
        "gpu_memory_utilization=0.49 (should fail)",
        lambda: VLLMConfig(gpu_memory_utilization=0.49),
    )
    expect_invalid(
        "gpu_memory_utilization=1.0 (should fail)",
        lambda: VLLMConfig(gpu_memory_utilization=1.0),
    )

    # swap_space (>= 0)
    expect_valid("swap_space=0.0", lambda: VLLMConfig(swap_space=0.0))
    expect_valid("swap_space=8.0", lambda: VLLMConfig(swap_space=8.0))
    expect_invalid(
        "swap_space=-1.0 (should fail)",
        lambda: VLLMConfig(swap_space=-1.0),
    )

    # cpu_offload_gb (>= 0)
    expect_valid("cpu_offload_gb=0.0", lambda: VLLMConfig(cpu_offload_gb=0.0))
    expect_valid("cpu_offload_gb=4.0", lambda: VLLMConfig(cpu_offload_gb=4.0))
    expect_invalid(
        "cpu_offload_gb=-1.0 (should fail)",
        lambda: VLLMConfig(cpu_offload_gb=-1.0),
    )

    subheader("VLLMConfig - KV Cache")

    # enable_prefix_caching
    expect_valid(
        "enable_prefix_caching=True",
        lambda: VLLMConfig(enable_prefix_caching=True),
    )

    # enable_chunked_prefill
    expect_valid(
        "enable_chunked_prefill=True",
        lambda: VLLMConfig(enable_chunked_prefill=True),
    )

    # kv_cache_dtype
    for dtype in ["auto", "float16", "bfloat16", "fp8"]:
        expect_valid(
            f"kv_cache_dtype={dtype}",
            lambda d=dtype: VLLMConfig(kv_cache_dtype=d),
        )
    expect_invalid(
        "kv_cache_dtype=float32 (should fail)",
        lambda: VLLMConfig(kv_cache_dtype="float32"),
    )

    # block_size
    for size in [8, 16, 32]:
        expect_valid(f"block_size={size}", lambda s=size: VLLMConfig(block_size=s))
    expect_invalid(
        "block_size=64 (should fail)",
        lambda: VLLMConfig(block_size=64),
    )

    subheader("VLLMConfig - Context & Execution")

    # max_model_len
    expect_valid("max_model_len=None", lambda: VLLMConfig(max_model_len=None))
    expect_valid("max_model_len=8192", lambda: VLLMConfig(max_model_len=8192))

    # max_seq_len_to_capture
    expect_valid(
        "max_seq_len_to_capture=None",
        lambda: VLLMConfig(max_seq_len_to_capture=None),
    )
    expect_valid(
        "max_seq_len_to_capture=8192",
        lambda: VLLMConfig(max_seq_len_to_capture=8192),
    )

    # enforce_eager
    expect_valid("enforce_eager=True", lambda: VLLMConfig(enforce_eager=True))

    subheader("VLLMConfig - Parallelism")

    # distributed_backend
    for backend in ["mp", "ray"]:
        expect_valid(
            f"distributed_backend={backend}",
            lambda b=backend: VLLMConfig(distributed_backend=b),
        )
    expect_invalid(
        "distributed_backend=invalid (should fail)",
        lambda: VLLMConfig(distributed_backend="invalid"),
    )

    # disable_custom_all_reduce
    expect_valid(
        "disable_custom_all_reduce=True",
        lambda: VLLMConfig(disable_custom_all_reduce=True),
    )

    subheader("VLLMConfig - Quantization")

    # quantization_method
    expect_valid(
        "quantization_method=awq",
        lambda: VLLMConfig(quantization_method="awq"),
    )
    expect_valid(
        "quantization_method=gptq",
        lambda: VLLMConfig(quantization_method="gptq"),
    )
    expect_valid(
        "quantization_method=fp8",
        lambda: VLLMConfig(quantization_method="fp8"),
    )

    # load_format
    for fmt in ["auto", "pt", "safetensors", "gguf"]:
        expect_valid(f"load_format={fmt}", lambda f=fmt: VLLMConfig(load_format=f))
    expect_invalid(
        "load_format=invalid (should fail)",
        lambda: VLLMConfig(load_format="invalid"),
    )

    subheader("VLLMConfig - Advanced Sampling")

    # best_of (>= 1)
    expect_valid("best_of=None", lambda: VLLMConfig(best_of=None))
    expect_valid("best_of=1", lambda: VLLMConfig(best_of=1))
    expect_valid("best_of=5", lambda: VLLMConfig(best_of=5))
    expect_invalid("best_of=0 (should fail)", lambda: VLLMConfig(best_of=0))

    # use_beam_search
    expect_valid("use_beam_search=True", lambda: VLLMConfig(use_beam_search=True))

    # length_penalty
    expect_valid("length_penalty=0.5", lambda: VLLMConfig(length_penalty=0.5))
    expect_valid("length_penalty=2.0", lambda: VLLMConfig(length_penalty=2.0))

    # logprobs (1-20)
    expect_valid("logprobs=None", lambda: VLLMConfig(logprobs=None))
    expect_valid("logprobs=1", lambda: VLLMConfig(logprobs=1))
    expect_valid("logprobs=20", lambda: VLLMConfig(logprobs=20))
    expect_invalid("logprobs=0 (should fail)", lambda: VLLMConfig(logprobs=0))
    expect_invalid("logprobs=21 (should fail)", lambda: VLLMConfig(logprobs=21))

    # logit_bias
    expect_valid("logit_bias=None", lambda: VLLMConfig(logit_bias=None))
    expect_valid(
        "logit_bias={123: -100}",
        lambda: VLLMConfig(logit_bias={123: -100.0}),
    )

    subheader("VLLMConfig - Nested Configs")

    # attention
    expect_valid(
        "attention (nested)",
        lambda: VLLMConfig(attention=VLLMAttentionConfig(backend="FLASH_ATTN")),
    )

    # speculative
    expect_valid(
        "speculative (nested)",
        lambda: VLLMConfig(speculative=VLLMSpeculativeConfig(method="ngram", num_tokens=5)),
    )

    # lora
    expect_valid(
        "lora (nested)",
        lambda: VLLMConfig(lora=VLLMLoRAConfig(enabled=True, max_loras=4)),
    )

    # extra (escape hatch)
    expect_valid(
        "extra (escape hatch)",
        lambda: VLLMConfig(extra={"custom_param": "value", "another": 123}),
    )


# =============================================================================
# PyTorch Config Tests
# =============================================================================


def test_pytorch_assisted_generation_config() -> None:
    """Test PyTorchAssistedGenerationConfig parameters."""
    subheader("PyTorchAssistedGenerationConfig")

    # model
    expect_valid(
        "model=TinyLlama",
        lambda: PyTorchAssistedGenerationConfig(model="TinyLlama/TinyLlama-1.1B"),
    )

    # num_tokens (1-10)
    expect_valid("num_tokens=1", lambda: PyTorchAssistedGenerationConfig(num_tokens=1))
    expect_valid("num_tokens=10", lambda: PyTorchAssistedGenerationConfig(num_tokens=10))
    expect_invalid(
        "num_tokens=0 (should fail)",
        lambda: PyTorchAssistedGenerationConfig(num_tokens=0),
    )
    expect_invalid(
        "num_tokens=11 (should fail)",
        lambda: PyTorchAssistedGenerationConfig(num_tokens=11),
    )


def test_pytorch_config() -> None:
    """Test PyTorchConfig parameters."""
    subheader("PyTorchConfig - Attention")

    # attn_implementation
    for impl in ["sdpa", "flash_attention_2", "eager"]:
        expect_valid(
            f"attn_implementation={impl}",
            lambda i=impl: PyTorchConfig(attn_implementation=i),
        )
    expect_invalid(
        "attn_implementation=invalid (should fail)",
        lambda: PyTorchConfig(attn_implementation="invalid"),
    )

    subheader("PyTorchConfig - Compilation")

    # torch_compile (bool or string)
    expect_valid("torch_compile=False", lambda: PyTorchConfig(torch_compile=False))
    expect_valid("torch_compile=True", lambda: PyTorchConfig(torch_compile=True))
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        expect_valid(
            f"torch_compile={mode}",
            lambda m=mode: PyTorchConfig(torch_compile=m),
        )

    subheader("PyTorchConfig - Legacy")

    # use_bettertransformer
    expect_valid(
        "use_bettertransformer=True",
        lambda: PyTorchConfig(use_bettertransformer=True),
    )

    subheader("PyTorchConfig - KV Caching")

    # use_cache
    expect_valid("use_cache=False", lambda: PyTorchConfig(use_cache=False))
    expect_valid("use_cache=True", lambda: PyTorchConfig(use_cache=True))

    subheader("PyTorchConfig - Memory")

    # low_cpu_mem_usage
    expect_valid(
        "low_cpu_mem_usage=False",
        lambda: PyTorchConfig(low_cpu_mem_usage=False),
    )

    # max_memory
    expect_valid("max_memory=None", lambda: PyTorchConfig(max_memory=None))
    expect_valid(
        "max_memory={'0': '20GiB'}",
        lambda: PyTorchConfig(max_memory={"0": "20GiB", "cpu": "30GiB"}),
    )

    subheader("PyTorchConfig - Generation")

    # num_beams (>= 1)
    expect_valid("num_beams=1", lambda: PyTorchConfig(num_beams=1))
    expect_valid("num_beams=5", lambda: PyTorchConfig(num_beams=5))
    expect_invalid("num_beams=0 (should fail)", lambda: PyTorchConfig(num_beams=0))

    # early_stopping
    expect_valid("early_stopping=True", lambda: PyTorchConfig(early_stopping=True))

    # length_penalty
    expect_valid("length_penalty=0.5", lambda: PyTorchConfig(length_penalty=0.5))

    # output_scores
    expect_valid("output_scores=True", lambda: PyTorchConfig(output_scores=True))

    # return_dict_in_generate
    expect_valid(
        "return_dict_in_generate=True",
        lambda: PyTorchConfig(return_dict_in_generate=True),
    )

    subheader("PyTorchConfig - Nested Configs")

    # assisted_generation
    expect_valid(
        "assisted_generation (nested)",
        lambda: PyTorchConfig(assisted_generation=PyTorchAssistedGenerationConfig(num_tokens=5)),
    )

    # extra (escape hatch)
    expect_valid(
        "extra (escape hatch)",
        lambda: PyTorchConfig(extra={"custom_param": "value"}),
    )


# =============================================================================
# ExperimentConfig Integration Tests
# =============================================================================


def test_experiment_config_integration() -> None:
    """Test backend config integration in ExperimentConfig."""
    header("ExperimentConfig Integration")

    base = {"config_name": "test", "model_name": "test-model"}

    subheader("Backend/Config Matching")

    # vllm config with vllm backend
    expect_valid(
        "vllm config with backend=vllm",
        lambda: ExperimentConfig(
            **base,
            backend="vllm",
            vllm=VLLMConfig(max_num_seqs=512),
        ),
    )

    # pytorch config with pytorch backend
    expect_valid(
        "pytorch config with backend=pytorch",
        lambda: ExperimentConfig(
            **base,
            backend="pytorch",
            pytorch=PyTorchConfig(attn_implementation="flash_attention_2"),
        ),
    )

    # vllm config with pytorch backend (should fail)
    expect_invalid(
        "vllm config with backend=pytorch (should fail)",
        lambda: ExperimentConfig(
            **base,
            backend="pytorch",
            vllm=VLLMConfig(max_num_seqs=512),
        ),
        match="vllm",
    )

    # pytorch config with vllm backend (should fail)
    expect_invalid(
        "pytorch config with backend=vllm (should fail)",
        lambda: ExperimentConfig(
            **base,
            backend="vllm",
            pytorch=PyTorchConfig(attn_implementation="sdpa"),
        ),
        match="pytorch",
    )

    subheader("Dict-style Config (YAML simulation)")

    # vLLM from dict
    expect_valid(
        "vllm config from dict",
        lambda: ExperimentConfig(
            **base,
            backend="vllm",
            vllm={
                "max_num_seqs": 256,
                "enable_prefix_caching": True,
                "kv_cache_dtype": "fp8",
            },
        ),
    )

    # PyTorch from dict
    expect_valid(
        "pytorch config from dict",
        lambda: ExperimentConfig(
            **base,
            backend="pytorch",
            pytorch={
                "attn_implementation": "flash_attention_2",
                "torch_compile": "reduce-overhead",
            },
        ),
    )

    # Nested vLLM speculative from dict
    expect_valid(
        "nested vllm speculative from dict",
        lambda: ExperimentConfig(
            **base,
            backend="vllm",
            vllm={
                "speculative": {
                    "model": "TinyLlama/TinyLlama-1.1B",
                    "method": "ngram",
                    "num_tokens": 5,
                }
            },
        ),
    )

    # Nested pytorch assisted_generation from dict
    expect_valid(
        "nested pytorch assisted_generation from dict",
        lambda: ExperimentConfig(
            **base,
            backend="pytorch",
            pytorch={
                "assisted_generation": {
                    "model": "TinyLlama/TinyLlama-1.1B",
                    "num_tokens": 5,
                }
            },
        ),
    )

    subheader("Full vLLM Config (all params)")

    # Full vLLM config
    full_vllm = expect_valid(
        "full vllm config",
        lambda: ExperimentConfig(
            **base,
            backend="vllm",
            vllm={
                "max_num_seqs": 512,
                "max_num_batched_tokens": 8192,
                "gpu_memory_utilization": 0.95,
                "swap_space": 4.0,
                "cpu_offload_gb": 2.0,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "kv_cache_dtype": "fp8",
                "block_size": 16,
                "max_model_len": 8192,
                "enforce_eager": False,
                "distributed_backend": "mp",
                "attention": {
                    "backend": "FLASH_ATTN",
                    "flash_version": 2,
                },
                "speculative": {
                    "method": "ngram",
                    "num_tokens": 5,
                    "ngram_max": 4,
                },
                "lora": {
                    "enabled": True,
                    "max_loras": 4,
                    "max_rank": 32,
                },
                "quantization_method": "awq",
                "load_format": "safetensors",
                "best_of": 3,
                "logprobs": 5,
            },
        ),
    )
    if full_vllm:
        assert full_vllm.vllm.max_num_seqs == 512
        assert full_vllm.vllm.kv_cache_dtype == "fp8"
        assert full_vllm.vllm.speculative.method == "ngram"
        assert full_vllm.vllm.lora.max_loras == 4
        test_pass("full vllm config values correct")

    subheader("Full PyTorch Config (all params)")

    # Full PyTorch config
    full_pytorch = expect_valid(
        "full pytorch config",
        lambda: ExperimentConfig(
            **base,
            backend="pytorch",
            pytorch={
                "attn_implementation": "flash_attention_2",
                "torch_compile": "reduce-overhead",
                "use_bettertransformer": False,
                "use_cache": True,
                "low_cpu_mem_usage": True,
                "max_memory": {"0": "20GiB", "cpu": "30GiB"},
                "assisted_generation": {
                    "model": "TinyLlama/TinyLlama-1.1B",
                    "num_tokens": 5,
                },
                "num_beams": 1,
                "early_stopping": False,
                "length_penalty": 1.0,
                "output_scores": True,
                "return_dict_in_generate": True,
            },
        ),
    )
    if full_pytorch:
        assert full_pytorch.pytorch.attn_implementation == "flash_attention_2"
        assert full_pytorch.pytorch.torch_compile == "reduce-overhead"
        assert full_pytorch.pytorch.assisted_generation.num_tokens == 5
        test_pass("full pytorch config values correct")


# =============================================================================
# Preset Tests
# =============================================================================


def test_presets() -> None:
    """Test all backend-specific presets load correctly."""
    header("Preset Loading")

    base = {"config_name": "preset_test", "model_name": "test-model"}

    # General presets
    subheader("General Presets")
    for name in ["quick-test", "benchmark", "throughput"]:
        expect_valid(
            f"preset: {name}",
            lambda n=name: ExperimentConfig(**base, **PRESETS[n]),
        )

    # vLLM presets
    subheader("vLLM Presets")
    for name in [
        "vllm-throughput",
        "vllm-speculative",
        "vllm-memory-efficient",
        "vllm-low-latency",
    ]:
        config = expect_valid(
            f"preset: {name}",
            lambda n=name: ExperimentConfig(**base, **PRESETS[n]),
        )
        if config:
            assert config.backend == "vllm", f"{name} should have backend=vllm"
            assert config.vllm is not None, f"{name} should have vllm config"

    # PyTorch presets
    subheader("PyTorch Presets")
    for name in ["pytorch-optimized", "pytorch-speculative", "pytorch-compatible"]:
        config = expect_valid(
            f"preset: {name}",
            lambda n=name: ExperimentConfig(**base, **PRESETS[n]),
        )
        if config:
            assert config.backend == "pytorch", f"{name} should have backend=pytorch"
            assert config.pytorch is not None, f"{name} should have pytorch config"

    # Verify specific preset values
    subheader("Preset Value Verification")

    # vllm-throughput
    config = ExperimentConfig(**base, **PRESETS["vllm-throughput"])
    assert config.vllm.max_num_seqs == 512
    assert config.vllm.enable_chunked_prefill is True
    assert config.vllm.enable_prefix_caching is True
    test_pass("vllm-throughput values correct")

    # vllm-speculative
    config = ExperimentConfig(**base, **PRESETS["vllm-speculative"])
    assert config.vllm.speculative is not None
    assert config.vllm.speculative.method == "ngram"
    test_pass("vllm-speculative values correct")

    # vllm-memory-efficient
    config = ExperimentConfig(**base, **PRESETS["vllm-memory-efficient"])
    assert config.vllm.kv_cache_dtype == "fp8"
    assert config.vllm.gpu_memory_utilization == 0.95
    test_pass("vllm-memory-efficient values correct")

    # vllm-low-latency
    config = ExperimentConfig(**base, **PRESETS["vllm-low-latency"])
    assert config.vllm.max_num_seqs == 32
    assert config.vllm.enforce_eager is True
    test_pass("vllm-low-latency values correct")

    # pytorch-optimized
    config = ExperimentConfig(**base, **PRESETS["pytorch-optimized"])
    assert config.pytorch.attn_implementation == "flash_attention_2"
    assert config.pytorch.torch_compile == "reduce-overhead"
    test_pass("pytorch-optimized values correct")

    # pytorch-speculative
    config = ExperimentConfig(**base, **PRESETS["pytorch-speculative"])
    assert config.pytorch.assisted_generation is not None
    assert config.pytorch.assisted_generation.num_tokens == 5
    test_pass("pytorch-speculative values correct")

    # pytorch-compatible
    config = ExperimentConfig(**base, **PRESETS["pytorch-compatible"])
    assert config.pytorch.attn_implementation == "eager"
    assert config.pytorch.torch_compile is False
    test_pass("pytorch-compatible values correct")


# =============================================================================
# Serialization Tests
# =============================================================================


def test_serialization() -> None:
    """Test config serialization/deserialization."""
    header("Serialization Roundtrip")

    base = {"config_name": "serial_test", "model_name": "test-model"}

    subheader("vLLM Config Roundtrip")

    original = ExperimentConfig(
        **base,
        backend="vllm",
        vllm=VLLMConfig(
            max_num_seqs=512,
            enable_prefix_caching=True,
            speculative=VLLMSpeculativeConfig(method="ngram", num_tokens=5),
        ),
    )
    json_str = original.model_dump_json()
    restored = ExperimentConfig.model_validate_json(json_str)

    assert restored.vllm.max_num_seqs == 512
    assert restored.vllm.enable_prefix_caching is True
    assert restored.vllm.speculative.method == "ngram"
    test_pass("vllm config roundtrip")

    subheader("PyTorch Config Roundtrip")

    original = ExperimentConfig(
        **base,
        backend="pytorch",
        pytorch=PyTorchConfig(
            attn_implementation="flash_attention_2",
            torch_compile="reduce-overhead",
            assisted_generation=PyTorchAssistedGenerationConfig(num_tokens=5),
        ),
    )
    json_str = original.model_dump_json()
    restored = ExperimentConfig.model_validate_json(json_str)

    assert restored.pytorch.attn_implementation == "flash_attention_2"
    assert restored.pytorch.torch_compile == "reduce-overhead"
    assert restored.pytorch.assisted_generation.num_tokens == 5
    test_pass("pytorch config roundtrip")


# =============================================================================
# Backend Build Methods Tests
# =============================================================================


def test_vllm_build_methods() -> None:
    """Test vLLM backend _build_engine_kwargs method."""
    header("vLLM Backend Build Methods")

    # We need to import and test the backend's build methods
    from llm_energy_measure.core.inference_backends.vllm import VLLMBackend

    backend = VLLMBackend()
    base = {"config_name": "build_test", "model_name": "test-model"}

    subheader("_build_engine_kwargs")

    # Note: Backend only passes non-default values to keep vLLM args clean.
    # Test with non-default values to verify mappings work correctly.
    config = ExperimentConfig(
        **base,
        backend="vllm",
        fp_precision="float16",
        vllm=VLLMConfig(
            max_num_seqs=512,  # non-default (default=256)
            max_num_batched_tokens=8192,  # non-default (default=None)
            gpu_memory_utilization=0.95,  # non-default (default=0.9)
            swap_space=8.0,  # non-default (default=4.0)
            cpu_offload_gb=2.0,  # non-default (default=0.0)
            enable_prefix_caching=True,  # non-default (default=False)
            enable_chunked_prefill=True,  # non-default (default=False)
            kv_cache_dtype="fp8",  # non-default (default="auto")
            block_size=32,  # non-default (default=16)
            max_model_len=8192,  # non-default (default=None)
            max_seq_len_to_capture=4096,  # non-default (default=None)
            enforce_eager=True,  # non-default (default=False)
            distributed_backend="ray",  # non-default (default="mp")
            disable_custom_all_reduce=True,  # non-default (default=False)
            quantization_method="awq",
            load_format="safetensors",  # non-default (default="auto")
        ),
    )

    kwargs = backend._build_engine_kwargs(config)

    # Verify all non-default params are mapped
    assert kwargs["max_num_seqs"] == 512, "max_num_seqs not mapped"
    test_pass("max_num_seqs mapped")

    assert kwargs["max_num_batched_tokens"] == 8192, "max_num_batched_tokens not mapped"
    test_pass("max_num_batched_tokens mapped")

    assert kwargs["gpu_memory_utilization"] == 0.95, "gpu_memory_utilization not mapped"
    test_pass("gpu_memory_utilization mapped")

    assert kwargs["swap_space"] == 8.0, "swap_space not mapped"
    test_pass("swap_space mapped")

    assert kwargs["cpu_offload_gb"] == 2.0, "cpu_offload_gb not mapped"
    test_pass("cpu_offload_gb mapped")

    assert kwargs["enable_prefix_caching"] is True, "enable_prefix_caching not mapped"
    test_pass("enable_prefix_caching mapped")

    assert kwargs["enable_chunked_prefill"] is True, "enable_chunked_prefill not mapped"
    test_pass("enable_chunked_prefill mapped")

    assert kwargs["kv_cache_dtype"] == "fp8", "kv_cache_dtype not mapped"
    test_pass("kv_cache_dtype mapped")

    assert kwargs["block_size"] == 32, "block_size not mapped"
    test_pass("block_size mapped")

    assert kwargs["max_model_len"] == 8192, "max_model_len not mapped"
    test_pass("max_model_len mapped")

    assert kwargs["max_seq_len_to_capture"] == 4096, "max_seq_len_to_capture not mapped"
    test_pass("max_seq_len_to_capture mapped")

    assert kwargs["enforce_eager"] is True, "enforce_eager not mapped"
    test_pass("enforce_eager mapped")

    assert kwargs["distributed_executor_backend"] == "ray", "distributed_backend not mapped"
    test_pass("distributed_backend mapped")

    assert kwargs["disable_custom_all_reduce"] is True, "disable_custom_all_reduce not mapped"
    test_pass("disable_custom_all_reduce mapped")

    assert kwargs["quantization"] == "awq", "quantization_method not mapped"
    test_pass("quantization_method mapped")

    assert kwargs["load_format"] == "safetensors", "load_format not mapped"
    test_pass("load_format mapped")

    # Test default values are NOT passed (clean kwargs)
    subheader("_build_engine_kwargs - Default Skipping")

    config_defaults = ExperimentConfig(
        **base,
        backend="vllm",
        vllm=VLLMConfig(),  # All defaults
    )
    kwargs_defaults = backend._build_engine_kwargs(config_defaults)

    # These should NOT be in kwargs since they're defaults
    assert "max_num_seqs" not in kwargs_defaults, "default max_num_seqs should not be passed"
    test_pass("default max_num_seqs not passed")
    assert (
        "gpu_memory_utilization" not in kwargs_defaults
    ), "default gpu_memory_utilization should not be passed"
    test_pass("default gpu_memory_utilization not passed")
    assert "swap_space" not in kwargs_defaults, "default swap_space should not be passed"
    test_pass("default swap_space not passed")
    assert "block_size" not in kwargs_defaults, "default block_size should not be passed"
    test_pass("default block_size not passed")

    subheader("_build_engine_kwargs - Speculative Config")

    config = ExperimentConfig(
        **base,
        backend="vllm",
        vllm=VLLMConfig(
            speculative=VLLMSpeculativeConfig(
                model="TinyLlama/TinyLlama-1.1B",
                method="ngram",
                num_tokens=5,
                ngram_max=4,
                draft_tp_size=2,
            )
        ),
    )

    kwargs = backend._build_engine_kwargs(config)
    spec_config = kwargs.get("speculative_config", {})

    assert spec_config.get("model") == "TinyLlama/TinyLlama-1.1B", "spec model not mapped"
    test_pass("speculative.model mapped")

    assert spec_config.get("num_speculative_tokens") == 5, "spec num_tokens not mapped"
    test_pass("speculative.num_tokens mapped")

    assert spec_config.get("ngram_prompt_lookup_max") == 4, "spec ngram_max not mapped"
    test_pass("speculative.ngram_max mapped")

    assert spec_config.get("draft_tensor_parallel_size") == 2, "spec draft_tp_size not mapped"
    test_pass("speculative.draft_tp_size mapped")

    subheader("_build_engine_kwargs - LoRA Config")

    config = ExperimentConfig(
        **base,
        backend="vllm",
        vllm=VLLMConfig(
            lora=VLLMLoRAConfig(
                enabled=True,
                max_loras=4,
                max_rank=32,
                extra_vocab_size=512,
            )
        ),
    )

    kwargs = backend._build_engine_kwargs(config)

    assert kwargs.get("enable_lora") is True, "lora.enabled not mapped"
    test_pass("lora.enabled mapped")

    assert kwargs.get("max_loras") == 4, "lora.max_loras not mapped"
    test_pass("lora.max_loras mapped")

    assert kwargs.get("max_lora_rank") == 32, "lora.max_rank not mapped"
    test_pass("lora.max_rank mapped")

    assert kwargs.get("lora_extra_vocab_size") == 512, "lora.extra_vocab_size not mapped"
    test_pass("lora.extra_vocab_size mapped")

    subheader("_build_sampling_kwargs")

    config = ExperimentConfig(
        **base,
        backend="vllm",
        vllm=VLLMConfig(
            best_of=3,
            use_beam_search=True,
            length_penalty=0.8,
            logprobs=5,
            logit_bias={123: -100.0, 456: 50.0},
        ),
    )

    kwargs = backend._build_sampling_kwargs(config)

    assert kwargs.get("best_of") == 3, "best_of not mapped"
    test_pass("best_of mapped")

    assert kwargs.get("use_beam_search") is True, "use_beam_search not mapped"
    test_pass("use_beam_search mapped")

    assert kwargs.get("length_penalty") == 0.8, "length_penalty not mapped"
    test_pass("length_penalty mapped")

    assert kwargs.get("logprobs") == 5, "logprobs not mapped"
    test_pass("logprobs mapped")

    assert kwargs.get("logit_bias") == {123: -100.0, 456: 50.0}, "logit_bias not mapped"
    test_pass("logit_bias mapped")


def test_pytorch_build_methods() -> None:
    """Test PyTorch backend _build_*_kwargs methods."""
    header("PyTorch Backend Build Methods")

    from llm_energy_measure.core.inference_backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    base = {"config_name": "build_test", "model_name": "test-model"}

    subheader("_build_model_kwargs")

    # Note: Backend only passes non-default values.
    # attn_implementation="sdpa" is default so test with flash_attention_2
    # low_cpu_mem_usage=True is default so it won't appear in kwargs
    config = ExperimentConfig(
        **base,
        backend="pytorch",
        pytorch=PyTorchConfig(
            attn_implementation="flash_attention_2",  # non-default (default=sdpa)
            low_cpu_mem_usage=True,  # This is actually the default
            max_memory={"0": "20GiB", "cpu": "30GiB"},  # non-default (default=None)
            extra={"trust_remote_code": True},
        ),
    )

    kwargs = backend._build_model_kwargs(config)

    assert (
        kwargs.get("attn_implementation") == "flash_attention_2"
    ), "attn_implementation not mapped"
    test_pass("attn_implementation mapped")

    # low_cpu_mem_usage=True is the default but PyTorch backend passes it when True
    # (different from vLLM's skip-defaults pattern)
    assert kwargs.get("low_cpu_mem_usage") is True, "low_cpu_mem_usage not mapped"
    test_pass("low_cpu_mem_usage mapped")

    assert kwargs.get("max_memory") == {"0": "20GiB", "cpu": "30GiB"}, "max_memory not mapped"
    test_pass("max_memory mapped")

    assert kwargs.get("trust_remote_code") is True, "extra params not mapped"
    test_pass("extra params mapped")

    # Test that sdpa (default) is NOT passed
    subheader("_build_model_kwargs - Default Skipping")

    config_defaults = ExperimentConfig(
        **base,
        backend="pytorch",
        pytorch=PyTorchConfig(),  # All defaults
    )
    kwargs_defaults = backend._build_model_kwargs(config_defaults)

    # attn_implementation=sdpa is default and should NOT be passed
    assert (
        "attn_implementation" not in kwargs_defaults
    ), "default attn_implementation should not be passed"
    test_pass("default attn_implementation not passed")

    subheader("_build_generation_kwargs")

    # Note: Backend only passes non-default values.
    # num_beams default=1, only passed when >1
    # early_stopping/length_penalty only passed with beam search
    config = ExperimentConfig(
        **base,
        backend="pytorch",
        pytorch=PyTorchConfig(
            use_cache=False,  # non-default (default=True)
            num_beams=4,  # non-default (default=1)
            early_stopping=True,  # non-default, only applies with beam search
            length_penalty=0.8,  # non-default (default=1.0)
            output_scores=True,  # non-default (default=False)
            return_dict_in_generate=True,  # non-default (default=False)
        ),
    )

    kwargs = backend._build_generation_kwargs(config)

    assert kwargs.get("use_cache") is False, "use_cache not mapped"
    test_pass("use_cache mapped")

    assert kwargs.get("num_beams") == 4, "num_beams not mapped"
    test_pass("num_beams mapped")

    assert kwargs.get("early_stopping") is True, "early_stopping not mapped"
    test_pass("early_stopping mapped")

    assert kwargs.get("length_penalty") == 0.8, "length_penalty not mapped"
    test_pass("length_penalty mapped")

    assert kwargs.get("output_scores") is True, "output_scores not mapped"
    test_pass("output_scores mapped")

    assert kwargs.get("return_dict_in_generate") is True, "return_dict_in_generate not mapped"
    test_pass("return_dict_in_generate mapped")

    # Test default values are NOT passed
    subheader("_build_generation_kwargs - Default Skipping")

    config_defaults = ExperimentConfig(
        **base,
        backend="pytorch",
        pytorch=PyTorchConfig(),  # All defaults
    )
    kwargs_defaults = backend._build_generation_kwargs(config_defaults)

    # These should NOT be in kwargs since they're defaults
    assert "use_cache" not in kwargs_defaults, "default use_cache should not be passed"
    test_pass("default use_cache not passed")
    assert "num_beams" not in kwargs_defaults, "default num_beams should not be passed"
    test_pass("default num_beams not passed")
    assert "output_scores" not in kwargs_defaults, "default output_scores should not be passed"
    test_pass("default output_scores not passed")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run all manual tests."""
    print("\n" + "=" * 60)
    print("  BACKEND-SPECIFIC PARAMETERS MANUAL TEST SUITE")
    print("=" * 60)

    # vLLM config tests
    header("vLLM Configuration Models")
    test_vllm_attention_config()
    test_vllm_speculative_config()
    test_vllm_lora_config()
    test_vllm_config()

    # PyTorch config tests
    header("PyTorch Configuration Models")
    test_pytorch_assisted_generation_config()
    test_pytorch_config()

    # Integration tests
    test_experiment_config_integration()

    # Preset tests
    test_presets()

    # Serialization tests
    test_serialization()

    # Backend build method tests
    test_vllm_build_methods()
    test_pytorch_build_methods()

    print("\n" + "=" * 60)
    print("  ALL MANUAL TESTS COMPLETED")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
