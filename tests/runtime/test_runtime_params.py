"""Runtime parameter tests using pytest.

These tests verify that configuration parameters are actually applied at inference time.
They require CUDA GPU access and run real inference with a small model.

The test_all_params.py script in scripts/ provides a standalone CLI interface to the same
functionality. This module provides pytest integration for CI and development workflows.

Test Strategy:
- For params with defined option sets (Literal types): test ALL values
- For numeric params: test min/max/sensible values

Run with:
    pytest tests/runtime/ -v                          # All tests
    pytest tests/runtime/ -v -k pytorch               # PyTorch only
    pytest tests/runtime/ -v --backend vllm           # vLLM only
    pytest tests/runtime/ -v --quick                  # Quick subset
"""

from __future__ import annotations

import json
import os
import subprocess

# Import from local test_all_params module (canonical location in tests/runtime/)
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from .test_all_params import (
    PYTORCH_PARAMS,
    QUICK_PYTORCH_PARAMS,
    QUICK_SHARED_PARAMS,
    QUICK_TENSORRT_PARAMS,
    QUICK_VLLM_PARAMS,
    SHARED_PARAMS,
    TENSORRT_PARAMS,
    TEST_MAX_OUTPUT,
    TEST_MODEL,
    TEST_SAMPLE_SIZE,
    TEST_TIMEOUT_SECONDS,
    VLLM_PARAMS,
    verify_inference_results,
)

# =============================================================================
# Parameter Value Extraction Helpers
# =============================================================================


def get_param_values(param_dict: dict[str, list[Any]], param_key: str) -> list[Any]:
    """Get test values for a parameter from param dict.

    Falls back to reasonable defaults if param not defined.
    """
    return param_dict.get(param_key, [])


PROJECT_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# Test Configuration
# =============================================================================


def get_backend_params(backend: str, quick: bool) -> dict[str, list[Any]]:
    """Get parameters to test for a backend."""
    if quick:
        shared = QUICK_SHARED_PARAMS
        backend_specific = {
            "pytorch": QUICK_PYTORCH_PARAMS,
            "vllm": QUICK_VLLM_PARAMS,
            "tensorrt": QUICK_TENSORRT_PARAMS,
        }.get(backend, {})
    else:
        shared = SHARED_PARAMS
        backend_specific = {
            "pytorch": PYTORCH_PARAMS,
            "vllm": VLLM_PARAMS,
            "tensorrt": TENSORRT_PARAMS,
        }.get(backend, {})

    return {**shared, **backend_specific}


def create_test_config(
    backend: str,
    param: str,
    value: Any,
    output_dir: Path,
) -> Path:
    """Create a test configuration file.

    Uses backend-native architecture where batching/quantization/parallelism
    are in backend-specific sections.
    """
    config = {
        "config_name": f"{backend}-test-{param.replace('.', '_')}-{value}".replace("/", "_"),
        "model_name": TEST_MODEL,
        "backend": backend,
        "gpus": [0],
        "max_input_tokens": 64,
        "max_output_tokens": TEST_MAX_OUTPUT,
        "num_input_prompts": TEST_SAMPLE_SIZE,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "dataset": {"name": "ai_energy_score", "sample_size": TEST_SAMPLE_SIZE},
    }

    # Add backend defaults (including batching)
    if backend == "pytorch":
        config["pytorch"] = {
            "batch_size": 1,
            "batching_strategy": "static",
            "attn_implementation": "sdpa",
        }
    elif backend == "vllm":
        config["vllm"] = {
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 512,
        }
    elif backend == "tensorrt":
        config["tensorrt"] = {
            "max_batch_size": 4,
            "builder_opt_level": 3,
            "force_rebuild": True,
        }

    # Apply the parameter variation
    parts = param.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_value = str(value).replace(".", "_").replace("/", "_")
    config_path = output_dir / f"{backend}_{parts[-1]}_{safe_value}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def run_experiment_subprocess(config_path: Path) -> tuple[int, str, str]:
    """Run experiment via subprocess and return (exit_code, stdout, stderr)."""
    cmd = ["lem", "experiment", str(config_path), "--yes"]

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src") + ":" + os.environ.get("PYTHONPATH", ""),
    }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {TEST_TIMEOUT_SECONDS}s"


# =============================================================================
# PyTorch Tests
# =============================================================================


class TestPyTorchRuntime:
    """Runtime tests for PyTorch backend.

    Tests ALL values for params with defined option sets (Literal types).
    Tests min/max/sensible values for numeric params.
    """

    @pytest.mark.requires_gpu
    @pytest.mark.slow
    def test_pytorch_baseline(
        self,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch baseline configuration runs successfully."""
        config_path = create_test_config(
            backend="pytorch",
            param="fp_precision",
            value="float16",
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)

        assert exit_code == 0, f"Baseline failed: {stderr}"

        # Verify actual inference happened
        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"
        assert metrics.get("throughput_tokens_per_second", 0) > 0

    # -------------------------------------------------------------------------
    # Precision (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("precision", SHARED_PARAMS["fp_precision"])
    def test_pytorch_precision(
        self,
        precision: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL precision settings."""
        config_path = create_test_config(
            backend="pytorch",
            param="fp_precision",
            value=precision,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Precision {precision} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {precision}: {error}"

    # -------------------------------------------------------------------------
    # Batch Size (Numeric - test min/max/mid)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", PYTORCH_PARAMS["pytorch.batch_size"])
    def test_pytorch_batch_size(
        self,
        batch_size: int,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with different batch sizes (backend-native: pytorch.batch_size)."""
        config_path = create_test_config(
            backend="pytorch",
            param="pytorch.batch_size",
            value=batch_size,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Batch size {batch_size} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for batch_size={batch_size}: {error}"

    # -------------------------------------------------------------------------
    # Batching Strategy (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("strategy", PYTORCH_PARAMS["pytorch.batching_strategy"])
    def test_pytorch_batching_strategy(
        self,
        strategy: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL batching strategies."""
        config_path = create_test_config(
            backend="pytorch",
            param="pytorch.batching_strategy",
            value=strategy,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Batching strategy {strategy} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {strategy}: {error}"

    # -------------------------------------------------------------------------
    # Attention Implementation (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("attn_impl", PYTORCH_PARAMS["pytorch.attn_implementation"])
    def test_pytorch_attention(
        self,
        attn_impl: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL attention implementations."""
        config_path = create_test_config(
            backend="pytorch",
            param="pytorch.attn_implementation",
            value=attn_impl,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Attention {attn_impl} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {attn_impl}: {error}"

    # -------------------------------------------------------------------------
    # Torch Compile Mode (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("compile_mode", PYTORCH_PARAMS["pytorch.torch_compile"])
    def test_pytorch_torch_compile(
        self,
        compile_mode: bool | str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL torch.compile modes."""
        config_path = create_test_config(
            backend="pytorch",
            param="pytorch.torch_compile",
            value=compile_mode,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"torch_compile={compile_mode} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for torch_compile={compile_mode}: {error}"

    # -------------------------------------------------------------------------
    # Cache Implementation (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("cache_impl", PYTORCH_PARAMS["pytorch.cache_implementation"])
    def test_pytorch_cache_implementation(
        self,
        cache_impl: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL cache implementations."""
        config_path = create_test_config(
            backend="pytorch",
            param="pytorch.cache_implementation",
            value=cache_impl,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"cache_implementation={cache_impl} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert (
            success
        ), f"Inference validation failed for cache_implementation={cache_impl}: {error}"

    # -------------------------------------------------------------------------
    # Quantization (Boolean - test both values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("load_4bit", PYTORCH_PARAMS["pytorch.load_in_4bit"])
    def test_pytorch_4bit_quantization(
        self,
        load_4bit: bool,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with 4-bit quantization."""
        config_path = create_test_config(
            backend="pytorch",
            param="pytorch.load_in_4bit",
            value=load_4bit,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"load_in_4bit={load_4bit} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for load_in_4bit={load_4bit}: {error}"

    # -------------------------------------------------------------------------
    # BnB 4-bit Quant Type (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("quant_type", PYTORCH_PARAMS["pytorch.bnb_4bit_quant_type"])
    def test_pytorch_bnb_4bit_quant_type(
        self,
        quant_type: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL BnB 4-bit quant types (requires load_in_4bit=True)."""
        # Create config with both load_in_4bit and the quant type
        config = {
            "config_name": f"pytorch-test-bnb_4bit_quant_type-{quant_type}",
            "model_name": TEST_MODEL,
            "backend": "pytorch",
            "gpus": [0],
            "max_input_tokens": 64,
            "max_output_tokens": TEST_MAX_OUTPUT,
            "num_input_prompts": TEST_SAMPLE_SIZE,
            "fp_precision": "float16",
            "decoder": {"preset": "deterministic"},
            "dataset": {"name": "ai_energy_score", "sample_size": TEST_SAMPLE_SIZE},
            "pytorch": {
                "batch_size": 1,
                "batching_strategy": "static",
                "attn_implementation": "sdpa",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": quant_type,
            },
        }

        temp_config_dir.mkdir(parents=True, exist_ok=True)
        config_path = temp_config_dir / f"pytorch_bnb_4bit_quant_type_{quant_type}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"bnb_4bit_quant_type={quant_type} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for bnb_4bit_quant_type={quant_type}: {error}"

    # -------------------------------------------------------------------------
    # Decoder Preset (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("preset", SHARED_PARAMS["decoder.preset"])
    def test_pytorch_decoder_preset(
        self,
        preset: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch with ALL decoder presets."""
        config_path = create_test_config(
            backend="pytorch",
            param="decoder.preset",
            value=preset,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"decoder.preset={preset} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for preset={preset}: {error}"


# =============================================================================
# vLLM Tests
# =============================================================================


class TestVLLMRuntime:
    """Runtime tests for vLLM backend.

    Tests ALL values for params with defined option sets (Literal types).
    Tests min/max/sensible values for numeric params.
    """

    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    def test_vllm_baseline(
        self,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM baseline configuration runs successfully."""
        config_path = create_test_config(
            backend="vllm",
            param="fp_precision",
            value="float16",
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Baseline failed: {stderr}"

        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"

    # -------------------------------------------------------------------------
    # Max Num Seqs (Numeric - test range)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("max_num_seqs", VLLM_PARAMS["vllm.max_num_seqs"])
    def test_vllm_max_seqs(
        self,
        max_num_seqs: int,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with different max_num_seqs settings."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.max_num_seqs",
            value=max_num_seqs,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"max_num_seqs={max_num_seqs} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"

    # -------------------------------------------------------------------------
    # Enforce Eager (Boolean - test both values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("enforce_eager", VLLM_PARAMS["vllm.enforce_eager"])
    def test_vllm_enforce_eager(
        self,
        enforce_eager: bool,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM eager mode."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.enforce_eager",
            value=enforce_eager,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"enforce_eager={enforce_eager} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"

    # -------------------------------------------------------------------------
    # KV Cache Dtype (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("kv_cache_dtype", VLLM_PARAMS["vllm.kv_cache_dtype"])
    def test_vllm_kv_cache_dtype(
        self,
        kv_cache_dtype: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with ALL KV cache dtypes."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.kv_cache_dtype",
            value=kv_cache_dtype,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"kv_cache_dtype={kv_cache_dtype} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for kv_cache_dtype={kv_cache_dtype}: {error}"

    # -------------------------------------------------------------------------
    # Block Size (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("block_size", VLLM_PARAMS["vllm.block_size"])
    def test_vllm_block_size(
        self,
        block_size: int,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with ALL block sizes."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.block_size",
            value=block_size,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"block_size={block_size} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for block_size={block_size}: {error}"

    # -------------------------------------------------------------------------
    # Attention Backend (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("attention_backend", VLLM_PARAMS["vllm.attention_backend"])
    def test_vllm_attention_backend(
        self,
        attention_backend: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with ALL attention backends."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.attention_backend",
            value=attention_backend,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"attention_backend={attention_backend} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert (
            success
        ), f"Inference validation failed for attention_backend={attention_backend}: {error}"

    # -------------------------------------------------------------------------
    # Distributed Backend (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("distributed_backend", VLLM_PARAMS["vllm.distributed_backend"])
    def test_vllm_distributed_backend(
        self,
        distributed_backend: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with ALL distributed backends."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.distributed_backend",
            value=distributed_backend,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"distributed_backend={distributed_backend} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert (
            success
        ), f"Inference validation failed for distributed_backend={distributed_backend}: {error}"

    # -------------------------------------------------------------------------
    # Load Format (Literal - test ALL values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("load_format", VLLM_PARAMS["vllm.load_format"])
    def test_vllm_load_format(
        self,
        load_format: str,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with ALL load formats."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.load_format",
            value=load_format,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"load_format={load_format} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for load_format={load_format}: {error}"

    # -------------------------------------------------------------------------
    # Prefix Caching (Boolean - test both values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("enable_prefix_caching", VLLM_PARAMS["vllm.enable_prefix_caching"])
    def test_vllm_prefix_caching(
        self,
        enable_prefix_caching: bool,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with prefix caching enabled/disabled."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.enable_prefix_caching",
            value=enable_prefix_caching,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"enable_prefix_caching={enable_prefix_caching} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert (
            success
        ), f"Inference validation failed for enable_prefix_caching={enable_prefix_caching}: {error}"

    # -------------------------------------------------------------------------
    # Chunked Prefill (Boolean - test both values)
    # -------------------------------------------------------------------------
    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("enable_chunked_prefill", VLLM_PARAMS["vllm.enable_chunked_prefill"])
    def test_vllm_chunked_prefill(
        self,
        enable_chunked_prefill: bool,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM with chunked prefill enabled/disabled."""
        config_path = create_test_config(
            backend="vllm",
            param="vllm.enable_chunked_prefill",
            value=enable_chunked_prefill,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"enable_chunked_prefill={enable_chunked_prefill} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for enable_chunked_prefill={enable_chunked_prefill}: {error}"


# =============================================================================
# TensorRT Tests
# =============================================================================


class TestTensorRTRuntime:
    """Runtime tests for TensorRT-LLM backend."""

    @pytest.mark.requires_gpu
    @pytest.mark.requires_tensorrt
    @pytest.mark.slow
    def test_tensorrt_baseline(
        self,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test TensorRT baseline configuration runs successfully."""
        config_path = create_test_config(
            backend="tensorrt",
            param="fp_precision",
            value="float16",
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Baseline failed: {stderr}"

        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_tensorrt
    @pytest.mark.slow
    @pytest.mark.parametrize("max_batch_size", [1, 4, 8])
    def test_tensorrt_max_batch_size(
        self,
        max_batch_size: int,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test TensorRT with different max_batch_size settings."""
        config_path = create_test_config(
            backend="tensorrt",
            param="tensorrt.max_batch_size",
            value=max_batch_size,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"max_batch_size={max_batch_size} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"


# =============================================================================
# Comprehensive Parameter Sweep (via test_all_params.py)
# =============================================================================


class TestFullParameterSweep:
    """Run comprehensive parameter sweep using test_all_params.py script.

    This is a convenience test that runs the full parameter sweep script.
    For more granular testing, use the individual test classes above.
    """

    @pytest.mark.requires_gpu
    @pytest.mark.slow
    def test_pytorch_full_sweep(self, request: pytest.FixtureRequest) -> None:
        """Run full PyTorch parameter sweep."""
        quick = request.config.getoption("--quick")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "test_all_params.py"),
            "--backend",
            "pytorch",
            "--output",
            str(PROJECT_ROOT / "results" / "test_results_pytorch.json"),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour for full sweep
            cwd=PROJECT_ROOT,
        )

        # Check results file
        results_path = PROJECT_ROOT / "results" / "test_results_pytorch.json"
        if results_path.exists():
            with open(results_path) as f:
                report = json.load(f)
            failed = report.get("summary", {}).get("failed", 0)
            total = report.get("summary", {}).get("total", 0)
            assert failed == 0, f"{failed}/{total} tests failed. See {results_path}"
        else:
            assert result.returncode == 0, f"Sweep failed: {result.stderr}"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    def test_vllm_full_sweep(self, request: pytest.FixtureRequest) -> None:
        """Run full vLLM parameter sweep."""
        quick = request.config.getoption("--quick")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "test_all_params.py"),
            "--backend",
            "vllm",
            "--output",
            str(PROJECT_ROOT / "results" / "test_results_vllm.json"),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=PROJECT_ROOT,
        )

        results_path = PROJECT_ROOT / "results" / "test_results_vllm.json"
        if results_path.exists():
            with open(results_path) as f:
                report = json.load(f)
            failed = report.get("summary", {}).get("failed", 0)
            total = report.get("summary", {}).get("total", 0)
            assert failed == 0, f"{failed}/{total} tests failed. See {results_path}"
        else:
            assert result.returncode == 0, f"Sweep failed: {result.stderr}"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_tensorrt
    @pytest.mark.slow
    def test_tensorrt_full_sweep(self, request: pytest.FixtureRequest) -> None:
        """Run full TensorRT parameter sweep."""
        quick = request.config.getoption("--quick")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "test_all_params.py"),
            "--backend",
            "tensorrt",
            "--output",
            str(PROJECT_ROOT / "results" / "test_results_tensorrt.json"),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=PROJECT_ROOT,
        )

        results_path = PROJECT_ROOT / "results" / "test_results_tensorrt.json"
        if results_path.exists():
            with open(results_path) as f:
                report = json.load(f)
            failed = report.get("summary", {}).get("failed", 0)
            total = report.get("summary", {}).get("total", 0)
            assert failed == 0, f"{failed}/{total} tests failed. See {results_path}"
        else:
            assert result.returncode == 0, f"Sweep failed: {result.stderr}"
