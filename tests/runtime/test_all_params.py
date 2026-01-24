#!/usr/bin/env python3
"""Comprehensive backend parameter testing.

This is the CANONICAL location for parameter testing. The Pydantic models in
src/.../config/backend_configs.py are the single source of truth for parameters.

This module provides:
- Auto-discovery from Pydantic models (--discover flag)
- Manual parameter definitions (fallback/overrides)
- Test helper functions (verify_inference_results, run_experiment, etc.)
- Data classes (TestResult, ValidationResult, TestReport)

Usage:
    # Run from project root
    python -m tests.runtime.test_all_params --backend pytorch
    python -m tests.runtime.test_all_params --discover --backend vllm
    python -m tests.runtime.test_all_params --list-params --discover

    # Or via pytest
    pytest tests/runtime/ -v --backend pytorch
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path (tests/runtime/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =============================================================================
# Configuration
# =============================================================================

# Small, fast model for testing (no HF_TOKEN required)
TEST_MODEL = "Qwen/Qwen2.5-0.5B"
TEST_SAMPLE_SIZE = 5
TEST_MAX_OUTPUT = 32
TEST_TIMEOUT_SECONDS = 300  # 5 minutes per test

# Quantized test models for AWQ/GPTQ parameters
# These are required because quantization params only work with pre-quantized models
QUANTIZED_TEST_MODELS = {
    "awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    "gptq": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
}

# Directories to clean before testing
RESULTS_DIRS = ["results/raw", "results/aggregated"]
STATE_DIRS = [".state"]
# Test configs directory (configs/ is now writable in production containers)
TEST_CONFIG_DIR = "configs/test_grid"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationResult:
    """Result of validating parameter application."""

    param_applied: bool  # Was param visible in runtime logs?
    param_evidence: str | None  # Log snippet proving application
    behavior_changed: bool  # Did metrics change vs baseline?
    metric_delta: dict[str, tuple[float, float, float]]  # {metric: (baseline, actual, delta%)}
    validation_status: str  # VERIFIED | UNVERIFIED | NO_EFFECT


@dataclass
class TestResult:
    """Result of a single parameter test."""

    config_name: str
    config_path: str
    parameter_varied: str
    parameter_value: Any
    status: str  # passed | failed | skipped
    exit_code: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    error_summary: str | None = None
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None
    validation: ValidationResult | None = None


@dataclass
class TestReport:
    """Complete test report."""

    run_id: str
    timestamp: str
    backend: str | None
    summary: dict[str, int]
    results: list[TestResult]
    failed_tests: list[dict[str, Any]]
    warnings_by_type: dict[str, int]
    baseline_metrics: dict[str, float] | None = None


# =============================================================================
# Parameter Discovery from Pydantic Models
# =============================================================================


def get_discovered_params(backend: str) -> dict[str, list[Any]]:
    """Get auto-discovered params for a backend using SSOT introspection.

    Uses llenergymeasure.config.introspection as the Single Source of Truth.
    Falls back to manual definitions only if introspection import fails.
    """
    try:
        from llenergymeasure.config.introspection import get_backend_params

        # Convert introspection format to test values format
        introspected = get_backend_params(backend)
        params: dict[str, list[Any]] = {}
        for param_path, meta in introspected.items():
            test_values = meta.get("test_values", [])
            if test_values:
                params[param_path] = test_values
        return params
    except ImportError:
        pass

    # Fallback to manual definitions (for environments without full install)
    return {
        "pytorch": PYTORCH_PARAMS,
        "vllm": VLLM_PARAMS,
        "tensorrt": TENSORRT_PARAMS,
    }.get(backend, {})


# =============================================================================
# Parameter Variations (Manual Fallback + Overrides)
# =============================================================================
# Backend-native architecture: batching, quantization, parallelism are backend-specific

# Shared parameters that apply to all backends (Tier 1: Universal)
SHARED_PARAMS: dict[str, list[Any]] = {
    # Precision
    "fp_precision": ["float32", "float16", "bfloat16"],
    # Token limits (affects memory/compute)
    # Note: max_input_tokens must leave room for output within model's context window
    # Test model (Qwen2.5-0.5B) has max_model_len=512, so we use conservative values
    "max_input_tokens": [128, 256],
    "max_output_tokens": [32, 128],
    # Decoder sampling parameters (universal)
    "decoder.temperature": [0.0, 0.7, 1.0],
    "decoder.top_p": [0.9, 0.95, 1.0],
    "decoder.top_k": [0, 10, 50],
    # Decoder presets
    "decoder.preset": ["deterministic", "standard", "creative", "factual"],
    # Streaming
    "streaming": [False, True],
    # Traffic simulation
    "traffic_simulation.enabled": [False, True],
    "traffic_simulation.mode": ["constant", "poisson"],
}

# Parallelism (2 GPUs) - backend-native names
PARALLELISM_PYTORCH: dict[str, list[Any]] = {
    "pytorch.parallelism_strategy": ["none", "tensor_parallel", "data_parallel"],
    "pytorch.parallelism_degree": [1, 2],
}

PARALLELISM_VLLM: dict[str, list[Any]] = {
    "vllm.tensor_parallel_size": [1, 2],
    "vllm.pipeline_parallel_size": [1, 2],
}

PARALLELISM_TENSORRT: dict[str, list[Any]] = {
    "tensorrt.tp_size": [1, 2],
    "tensorrt.pp_size": [1, 2],
}

# PyTorch-specific parameters (Tier 2: Backend-native)
PYTORCH_PARAMS: dict[str, list[Any]] = {
    # Batching (PyTorch-specific)
    "pytorch.batch_size": [1, 4, 8],
    "pytorch.batching_strategy": ["static", "dynamic", "sorted_static", "sorted_dynamic"],
    # BitsAndBytes quantization (PyTorch only)
    "pytorch.load_in_4bit": [False, True],
    "pytorch.load_in_8bit": [False, True],
    "pytorch.bnb_4bit_quant_type": ["nf4", "fp4"],
    "pytorch.bnb_4bit_use_double_quant": [False, True],
    # Attention and compilation
    "pytorch.attn_implementation": ["sdpa", "flash_attention_2", "eager"],
    "pytorch.torch_compile": [False, "default", "reduce-overhead", "max-autotune"],
    "pytorch.torch_compile_backend": ["inductor", "cudagraphs"],
    "pytorch.use_cache": [True, False],
    "pytorch.cache_implementation": ["dynamic", "static", "hybrid", "sliding_window"],
    "pytorch.low_cpu_mem_usage": [True, False],
    "pytorch.use_bettertransformer": [False, True],
    "pytorch.output_scores": [False, True],
    "pytorch.return_dict_in_generate": [False, True],
    # Decoder extensions (PyTorch Tier 2)
    "pytorch.min_p": [0.0, 0.05, 0.1],
    "pytorch.no_repeat_ngram_size": [0, 2, 3],
}

# vLLM-specific parameters (Tier 2: Backend-native)
VLLM_PARAMS: dict[str, list[Any]] = {
    "vllm.max_num_seqs": [32, 64, 128, 256],
    "vllm.max_num_batched_tokens": [None, 2048, 4096],
    "vllm.gpu_memory_utilization": [0.5, 0.6, 0.7],
    "vllm.swap_space": [0.0, 4.0],
    "vllm.cpu_offload_gb": [0.0, 2.0],
    "vllm.enable_prefix_caching": [False, True],
    "vllm.enable_chunked_prefill": [False, True],
    "vllm.kv_cache_dtype": ["auto", "float16", "bfloat16", "fp8"],
    "vllm.block_size": [8, 16, 32],
    "vllm.enforce_eager": [False, True],
    "vllm.distributed_backend": ["mp", "ray"],
    # Note: attention.backend uses nested path (vLLM attention is a nested config)
    "vllm.attention.backend": ["auto", "FLASH_ATTN", "FLASHINFER"],
    # Note: best_of was removed in vLLM v1
    "vllm.logprobs": [None, 5],
    "vllm.quantization": [None, "fp8", "awq", "gptq"],
    "vllm.load_format": ["auto", "pt", "safetensors"],
    # Decoder extensions (vLLM Tier 2)
    "vllm.min_p": [0.0, 0.05, 0.1],
}

# TensorRT-specific parameters (Tier 2: Backend-native)
TENSORRT_PARAMS: dict[str, list[Any]] = {
    "tensorrt.max_batch_size": [1, 4, 8],
    "tensorrt.builder_opt_level": [2, 3, 4, 5],
    "tensorrt.strongly_typed": [False, True],
    "tensorrt.multiple_profiles": [False, True],
    "tensorrt.quantization": ["none", "fp8", "int8_sq", "int8_weight_only"],
    "tensorrt.kv_cache_type": ["paged", "continuous"],
    "tensorrt.enable_chunked_context": [False, True],
    "tensorrt.gpu_memory_utilization": [0.7, 0.9],
    "tensorrt.enable_kv_cache_reuse": [False, True],
    "tensorrt.force_rebuild": [True],
}

# Quick mode - fewer variations (backend-native architecture)
QUICK_SHARED_PARAMS: dict[str, list[Any]] = {
    "fp_precision": ["float16", "bfloat16"],
    "decoder.preset": ["deterministic"],
    "streaming": [False, True],
}

QUICK_PYTORCH_PARAMS: dict[str, list[Any]] = {
    "pytorch.batch_size": [1, 4],
    "pytorch.batching_strategy": ["static", "dynamic"],
    "pytorch.attn_implementation": ["sdpa", "eager"],
    "pytorch.torch_compile": [False, "default"],
}

QUICK_VLLM_PARAMS: dict[str, list[Any]] = {
    "vllm.max_num_seqs": [64, 128],
    "vllm.gpu_memory_utilization": [0.7],
    "vllm.enforce_eager": [False, True],
}

QUICK_TENSORRT_PARAMS: dict[str, list[Any]] = {
    "tensorrt.max_batch_size": [1, 4],
    "tensorrt.builder_opt_level": [3],
}

# =============================================================================
# Skip Conditions (Hardware/Model Limitations)
# =============================================================================
# Some parameter values require specific hardware, pre-quantized models, or
# have known compatibility issues. These are documented here for clarity.
#
# Skip types:
#   - "hardware": Requires specific GPU (Ampere+, Hopper, etc.)
#   - "model": Requires a specific pre-quantized model checkpoint
#   - "dependency": Requires optional dependency not installed
#   - "known_issue": Known compatibility issue, not a bug in our code

SKIP_CONDITIONS: dict[tuple[str, Any], dict[str, str]] = {
    # vLLM KV cache dtype limitations
    # float16/bfloat16 KV cache requires matching attention backend
    # fp8 KV cache requires Hopper (H100) or newer GPU
    ("vllm.kv_cache_dtype", "float16"): {
        "type": "hardware",
        "reason": "float16 KV cache requires compatible attention backend (model-dependent)",
    },
    ("vllm.kv_cache_dtype", "bfloat16"): {
        "type": "hardware",
        "reason": "bfloat16 KV cache requires compatible attention backend (model-dependent)",
    },
    ("vllm.kv_cache_dtype", "fp8"): {
        "type": "hardware",
        "reason": "FP8 KV cache requires Hopper (H100) or newer GPU with FlashInfer support",
    },
    # vLLM block_size=8 not supported by PagedAttention for many models
    ("vllm.block_size", 8): {
        "type": "known_issue",
        "reason": "block_size=8 not compatible with PagedAttention for this model architecture",
    },
    # vLLM FLASHINFER requires JIT compilation (slow first-run, may fail)
    ("vllm.attention.backend", "FLASHINFER"): {
        "type": "dependency",
        "reason": "FlashInfer backend requires JIT compilation; may fail in Docker without cache",
    },
    # vLLM quantization methods requiring pre-quantized models
    # These require model checkpoints that were quantized with AWQ/GPTQ
    ("vllm.quantization", "awq"): {
        "type": "model",
        "reason": "AWQ quantization requires pre-quantized model (e.g., TheBloke/*-AWQ)",
        "model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",  # Suggested model for testing
    },
    ("vllm.quantization", "gptq"): {
        "type": "model",
        "reason": "GPTQ quantization requires pre-quantized model (e.g., TheBloke/*-GPTQ)",
        "model": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",  # Suggested model for testing
    },
    # vLLM load_format=pt requires .bin files (not safetensors)
    ("vllm.load_format", "pt"): {
        "type": "model",
        "reason": "load_format='pt' requires model with .bin files (most modern models use safetensors)",
    },
    # TensorRT quantization requiring calibration or pre-quantized models
    ("tensorrt.quantization", "int8_sq"): {
        "type": "dependency",
        "reason": "INT8 SmoothQuant requires calibration dataset configuration",
    },
    ("tensorrt.quantization", "int4_awq"): {
        "type": "model",
        "reason": "INT4 AWQ requires pre-quantized model checkpoint",
    },
    # PyTorch flash_attention_2 requires flash-attn package
    ("pytorch.attn_implementation", "flash_attention_2"): {
        "type": "dependency",
        "reason": "Flash Attention 2 requires flash-attn package (not installed by default)",
    },
}


def should_skip_param(
    param: str, value: Any, skip_known_issues: bool = True
) -> tuple[bool, str | None]:
    """Check if a parameter value should be skipped.

    Args:
        param: Parameter path (e.g., "vllm.kv_cache_dtype")
        value: Parameter value to test
        skip_known_issues: Whether to skip known compatibility issues

    Returns:
        Tuple of (should_skip, skip_reason)
    """
    key = (param, value)
    if key in SKIP_CONDITIONS:
        condition = SKIP_CONDITIONS[key]
        skip_type = condition.get("type", "unknown")

        # Always skip hardware/model issues unless we have the right setup
        if skip_type in ("hardware", "model", "dependency"):
            return True, f"[SKIP:{skip_type.upper()}] {condition['reason']}"

        # Optionally skip known issues
        if skip_type == "known_issue" and skip_known_issues:
            return True, f"[SKIP:KNOWN_ISSUE] {condition['reason']}"

    return False, None


# =============================================================================
# Helper Functions
# =============================================================================


def reset_environment(keep_results: bool = False) -> None:
    """Clean slate for testing - removes all results and state."""
    if keep_results:
        print("  Keeping existing results (--no-cleanup)")
        return

    print("  Resetting environment...")

    # Delete result directories (ignore errors for permission issues in Docker)
    for dir_path in RESULTS_DIRS + STATE_DIRS:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            shutil.rmtree(full_path, ignore_errors=True)
            print(f"    Deleted: {dir_path}")

    # Delete test config directory
    test_config_path = PROJECT_ROOT / TEST_CONFIG_DIR
    if test_config_path.exists():
        shutil.rmtree(test_config_path, ignore_errors=True)
        print(f"    Deleted: {TEST_CONFIG_DIR}")

    # Recreate necessary directories
    for dir_path in RESULTS_DIRS + STATE_DIRS:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

    print("  Environment reset complete")


def create_base_config(backend: str, output_dir: Path) -> Path:
    """Create a minimal base config for testing.

    Uses backend-native architecture where batching/quantization/parallelism
    are in backend-specific sections.
    """
    import yaml

    config = {
        "config_name": f"{backend}-test-base",
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

    # Backend-specific defaults (including batching)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{backend}_base.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def create_parallelism_config(backend: str, strategy: str, output_dir: Path) -> Path:
    """Create config for parallelism testing (2 GPUs).

    Uses backend-native parallelism parameters:
    - PyTorch: parallelism_strategy, parallelism_degree
    - vLLM: tensor_parallel_size, pipeline_parallel_size
    - TensorRT: tp_size, pp_size
    """
    import yaml

    config = {
        "config_name": f"{backend}-parallel-{strategy}",
        "model_name": TEST_MODEL,
        "backend": backend,
        "gpus": [0, 1],
        "max_input_tokens": 64,
        "max_output_tokens": TEST_MAX_OUTPUT,
        "num_input_prompts": TEST_SAMPLE_SIZE,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "dataset": {"name": "ai_energy_score", "sample_size": TEST_SAMPLE_SIZE},
    }

    # Backend-specific parallelism (backend-native architecture)
    if backend == "pytorch":
        config["pytorch"] = {
            "batch_size": 1,
            "batching_strategy": "static",
            "attn_implementation": "sdpa",
            "parallelism_strategy": strategy,
            "parallelism_degree": 2,
        }
    elif backend == "vllm":
        config["vllm"] = {
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 512,
        }
        if strategy == "tensor_parallel":
            config["vllm"]["tensor_parallel_size"] = 2
        elif strategy == "pipeline_parallel":
            config["vllm"]["pipeline_parallel_size"] = 2
    elif backend == "tensorrt":
        config["tensorrt"] = {
            "max_batch_size": 4,
            "builder_opt_level": 3,
        }
        if strategy == "tensor_parallel":
            config["tensorrt"]["tp_size"] = 2
        elif strategy == "pipeline_parallel":
            config["tensorrt"]["pp_size"] = 2

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{backend}_parallel_{strategy}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def generate_config_grid(
    base_config: Path,
    param: str,
    values: list[Any],
    output_dir: Path,
) -> list[Path]:
    """Generate config variations using the CLI generate-grid command."""
    # Format values for CLI
    formatted_values = ",".join(str(v) if v is not None else "null" for v in values)

    # Use the short alias
    cmd = [
        "lem",
        "config",
        "generate-grid",
        str(base_config),
        "--vary",
        f"{param}={formatted_values}",
        "--output-dir",
        str(output_dir),
    ]

    # Set PYTHONPATH to include src directory
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src") + ":" + os.environ.get("PYTHONPATH", ""),
    }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT,
            env=env,
        )

        if result.returncode != 0:
            print(f"    Warning: Grid generation failed for {param}")
            print(f"      stderr: {result.stderr[:200]}")
            return []

    except subprocess.TimeoutExpired:
        print(f"    Warning: Grid generation timed out for {param}")
        return []

    # Find generated configs
    generated = list(output_dir.glob(f"{base_config.stem}_*.yaml"))
    return generated


def create_single_variation_config(
    base_config: Path,
    param: str,
    value: Any,
    output_dir: Path,
) -> Path | None:
    """Create a single config with one parameter varied.

    Uses direct YAML manipulation instead of CLI for nested params.
    """
    import yaml

    with open(base_config) as f:
        config = yaml.safe_load(f)

    # Apply the variation
    parts = param.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]

    # Handle None/null values
    if value is None:
        target[parts[-1]] = None
    else:
        target[parts[-1]] = value

    # Update config name
    safe_value = str(value).replace(".", "_").replace("/", "_")
    config["config_name"] = f"{base_config.stem}_{parts[-1]}_{safe_value}"

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{config['config_name']}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def verify_inference_results(config_name: str) -> tuple[bool, str | None, dict[str, Any]]:
    """Verify that actual inference ran and produced results.

    Returns:
        (success, error_message, metrics_dict)
    """
    results_dir = PROJECT_ROOT / "results" / "aggregated"
    raw_dir = PROJECT_ROOT / "results" / "raw"

    # Results are stored by experiment_id, not config_name
    # We need to search inside JSON files for matching config_name
    result_data = None

    # Check aggregated results first (faster - single file per experiment)
    if results_dir.exists():
        for agg_file in sorted(results_dir.glob("*.json"), reverse=True):
            try:
                with open(agg_file) as f:
                    data = json.load(f)
                if data.get("config_name") == config_name:
                    result_data = data
                    break
            except (json.JSONDecodeError, OSError):
                continue

    # Fall back to raw results (check most recent experiment directories first)
    if result_data is None and raw_dir.exists():
        for exp_dir in sorted(raw_dir.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue
            for raw_file in exp_dir.glob("process_*.json"):
                try:
                    with open(raw_file) as f:
                        data = json.load(f)
                    if data.get("config_name") == config_name:
                        result_data = data
                        break
                except (json.JSONDecodeError, OSError):
                    continue
            if result_data:
                break

    if not result_data:
        return False, "No result files found - inference may not have run", {}

    # Extract metrics - check for actual values
    # Raw results have inference_metrics, aggregated have metrics
    metrics = result_data.get("metrics", {})
    if not metrics:
        metrics = result_data.get("aggregated_metrics", {})
    if not metrics:
        # For raw results, extract from inference_metrics
        inf_metrics = result_data.get("inference_metrics", {})
        metrics = {
            "total_output_tokens": inf_metrics.get("output_tokens", 0),
            "throughput_tokens_per_second": inf_metrics.get("tokens_per_second", 0),
            "mean_latency_ms": inf_metrics.get("latency_per_token_ms", 0),
            "total_energy_joules": result_data.get("energy_metrics", {}).get("total_joules", -1),
        }

    # Critical checks
    checks_failed = []

    # Check if this is a streaming result
    # Streaming results have latency_measurements nested in inference_metrics
    inf_metrics = result_data.get("inference_metrics", {})
    latency_measurements = inf_metrics.get("latency_measurements") or {}
    is_streaming = latency_measurements.get("streaming_mode", False) or result_data.get(
        "config", {}
    ).get("streaming", False)

    # 1. Check tokens were generated
    total_tokens = metrics.get("total_output_tokens", 0)
    # For streaming, tokens may be in latency_measurements
    if total_tokens == 0 and is_streaming:
        total_tokens = latency_measurements.get("total_output_tokens", 0)
    if total_tokens == 0:
        checks_failed.append("No output tokens generated")

    # 2. Check throughput is reasonable (not zero, not negative)
    # For streaming mode, throughput may be calculated differently or not present
    throughput = metrics.get("throughput_tokens_per_second", 0)
    if throughput <= 0 and not is_streaming:
        checks_failed.append(f"Invalid throughput: {throughput}")
    elif throughput <= 0 and is_streaming and total_tokens > 0:
        # Streaming with tokens but no throughput is acceptable
        # (throughput calculated at aggregation time)
        pass

    # 3. Check energy was measured (if available)
    energy = metrics.get("total_energy_joules", metrics.get("gpu_energy_joules", -1))
    if energy == 0:
        checks_failed.append("Zero energy recorded")

    # 4. Check latency is reasonable
    # For streaming: check TTFT samples exist
    latency = metrics.get("mean_latency_ms", metrics.get("latency_ms", 0))
    if latency <= 0 and not is_streaming:
        checks_failed.append(f"Invalid latency: {latency}")
    elif is_streaming:
        # For streaming, check TTFT samples instead
        ttft_samples = latency_measurements.get("ttft_ms", [])
        if not ttft_samples:
            checks_failed.append("No TTFT samples in streaming result")

    if checks_failed:
        return False, "; ".join(checks_failed), metrics

    return True, None, metrics


def run_experiment(config_path: Path) -> TestResult:
    """Run an experiment and capture all output."""
    import time

    config_name = config_path.stem
    start_time = time.time()

    # Use the short alias (works in Docker where package is installed)
    # --yes bypasses confirmation prompts for config warnings
    cmd = [
        "lem",
        "experiment",
        str(config_path),
        "--yes",
    ]

    # Set PYTHONPATH to include src directory
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

        elapsed = time.time() - start_time

        # Extract warnings from output
        warnings = extract_warnings(result.stdout + result.stderr)
        error_summary = None
        traceback_str = None

        # STRICT VALIDATION: exit code 0 is necessary but not sufficient
        if result.returncode != 0:
            status = "failed"
            error_summary = extract_error_summary(result.stderr)
            traceback_str = extract_traceback(result.stderr)
        else:
            # Exit code 0 - now verify actual inference happened
            inference_ok, inference_error, inference_metrics = verify_inference_results(config_name)

            if inference_ok:
                status = "passed"
            else:
                status = "failed"
                error_summary = f"Inference validation failed: {inference_error}"
                # Store metrics for debugging even on failure
                if inference_metrics:
                    warnings.append(f"Partial metrics: {inference_metrics}")

        return TestResult(
            config_name=config_name,
            config_path=str(config_path),
            parameter_varied="",  # Will be filled in by caller
            parameter_value=None,  # Will be filled in by caller
            status=status,
            exit_code=result.returncode,
            elapsed_seconds=elapsed,
            warnings=warnings,
            error_summary=error_summary,
            stdout=result.stdout,
            stderr=result.stderr,
            traceback=traceback_str,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return TestResult(
            config_name=config_name,
            config_path=str(config_path),
            parameter_varied="",
            parameter_value=None,
            status="failed",
            exit_code=-1,
            elapsed_seconds=elapsed,
            error_summary=f"Timeout after {TEST_TIMEOUT_SECONDS}s",
            stdout="",
            stderr="",
        )


def extract_warnings(text: str) -> list[str]:
    """Extract warning messages from output."""
    warnings = []
    patterns = [
        r"Warning:.*",
        r"WARNING:.*",
        r"\[warning\].*",
        r"DeprecationWarning:.*",
        r"FutureWarning:.*",
        r"UserWarning:.*",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        warnings.extend(matches)

    return warnings


def extract_error_summary(text: str) -> str | None:
    """Extract a brief error summary from stderr."""
    # Look for common error patterns
    patterns = [
        r"Error: (.+)",
        r"RuntimeError: (.+)",
        r"ValueError: (.+)",
        r"TypeError: (.+)",
        r"KeyError: (.+)",
        r"AttributeError: (.+)",
        r"ImportError: (.+)",
        r"ModuleNotFoundError: (.+)",
        r"FileNotFoundError: (.+)",
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"torch\.cuda\.OutOfMemoryError",
        r"Traceback \(most recent call last\)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)[:200]

    # Filter out warnings when falling back to last line
    warning_patterns = [
        r"warning",
        r"FutureWarning",
        r"DeprecationWarning",
        r"UserWarning",
        r"import pynvml",  # Common false positive
    ]

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Filter out warning lines
    error_lines = [
        line
        for line in lines
        if not any(re.search(p, line, re.IGNORECASE) for p in warning_patterns)
    ]

    if error_lines:
        return error_lines[-1][:200]

    return None


def extract_traceback(text: str) -> str | None:
    """Extract Python traceback from stderr."""
    # Find traceback block
    match = re.search(r"Traceback \(most recent call last\):.*", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def validate_param_application(
    result: TestResult,
    param: str,
    value: Any,
    baseline_metrics: dict[str, float] | None,
) -> ValidationResult:
    """Validate that a parameter was applied and affected behaviour."""
    combined_output = result.stdout + result.stderr

    # Check if param was logged/echoed at runtime
    param_applied = False
    param_evidence = None

    # Look for evidence in logs
    param_patterns = [
        f"{param}.*{value}",
        f"{param.split('.')[-1]}.*{value}",
        f"Using.*{param.split('.')[-1]}.*{value}",
        f"{value}",  # Direct value match
    ]

    for pattern in param_patterns:
        match = re.search(pattern, combined_output, re.IGNORECASE)
        if match:
            param_applied = True
            # Extract context around match
            start = max(0, match.start() - 50)
            end = min(len(combined_output), match.end() + 50)
            param_evidence = combined_output[start:end].strip()
            break

    # Check for behaviour change via metrics
    behavior_changed = False
    metric_delta: dict[str, tuple[float, float, float]] = {}

    if baseline_metrics and result.status == "passed":
        # Extract metrics from this run
        current_metrics = extract_metrics_from_output(combined_output)

        for metric, baseline_val in baseline_metrics.items():
            if metric in current_metrics:
                current_val = current_metrics[metric]
                if baseline_val != 0:
                    delta_pct = ((current_val - baseline_val) / baseline_val) * 100
                else:
                    delta_pct = 100 if current_val != 0 else 0

                metric_delta[metric] = (baseline_val, current_val, delta_pct)

                # Consider >5% change as meaningful
                if abs(delta_pct) > 5:
                    behavior_changed = True

    # Determine validation status
    if param_applied and behavior_changed:
        validation_status = "VERIFIED"
    elif param_applied:
        validation_status = "NO_EFFECT"
    else:
        validation_status = "UNVERIFIED"

    return ValidationResult(
        param_applied=param_applied,
        param_evidence=param_evidence,
        behavior_changed=behavior_changed,
        metric_delta=metric_delta,
        validation_status=validation_status,
    )


def extract_metrics_from_output(text: str) -> dict[str, float]:
    """Extract metrics from experiment output."""
    metrics = {}

    patterns = {
        "throughput": r"throughput.*?(\d+\.?\d*)",
        "latency": r"latency.*?(\d+\.?\d*)",
        "energy": r"energy.*?(\d+\.?\d*)",
        "memory": r"memory.*?(\d+\.?\d*)",
        "tokens_per_second": r"tokens?/s.*?(\d+\.?\d*)",
    }

    for metric, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            with contextlib.suppress(ValueError):
                metrics[metric] = float(match.group(1))

    return metrics


def generate_report(
    results: list[TestResult],
    backend: str | None,
    baseline_metrics: dict[str, float] | None,
) -> TestReport:
    """Generate comprehensive test report."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()

    # Summary counts
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")

    # Count warnings by type
    warnings_by_type: dict[str, int] = {}
    for result in results:
        for warning in result.warnings:
            # Categorise warning
            if "deprecat" in warning.lower():
                category = "deprecation"
            elif "memory" in warning.lower():
                category = "memory"
            elif "config" in warning.lower():
                category = "config"
            else:
                category = "other"
            warnings_by_type[category] = warnings_by_type.get(category, 0) + 1

    # Collect failed test details with FULL output
    failed_tests = []
    for result in results:
        if result.status == "failed":
            failed_tests.append(
                {
                    "config": result.config_name,
                    "parameter_varied": result.parameter_varied,
                    "parameter_value": result.parameter_value,
                    "exit_code": result.exit_code,
                    "error_summary": result.error_summary,
                    "stdout_full": result.stdout,  # COMPLETE output for debugging
                    "stderr_full": result.stderr,  # COMPLETE output for debugging
                    "traceback": result.traceback,
                }
            )

    return TestReport(
        run_id=run_id,
        timestamp=timestamp,
        backend=backend,
        summary={
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        results=results,
        failed_tests=failed_tests,
        warnings_by_type=warnings_by_type,
        baseline_metrics=baseline_metrics,
    )


def save_report(report: TestReport, output_path: Path) -> None:
    """Save report as JSON."""

    def serialize(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(serialize(report), f, indent=2)


def run_baseline(base_config: Path) -> dict[str, float] | None:
    """Run baseline experiment to get reference metrics."""
    print("    Running baseline...")
    result = run_experiment(base_config)

    if result.status != "passed":
        print(f"    Warning: Baseline failed - {result.error_summary}")
        return None

    metrics = extract_metrics_from_output(result.stdout + result.stderr)
    if metrics:
        print(f"    Baseline metrics: {metrics}")
    else:
        print("    Warning: Could not extract baseline metrics")

    return metrics


def test_backend(
    backend: str,
    quick_mode: bool,
    output_dir: Path,
    discover_mode: bool = False,
) -> list[TestResult]:
    """Test all parameters for a specific backend.

    Args:
        backend: Backend to test (pytorch, vllm, tensorrt)
        quick_mode: Use fewer parameter variations
        output_dir: Directory for generated configs
        discover_mode: Auto-discover params from Pydantic models
    """
    results: list[TestResult] = []

    print(f"\n{'=' * 60}")
    print(f"Testing backend: {backend}")
    print(f"  Mode: {'discover' if discover_mode else 'manual'}")
    print("=" * 60)

    # Create base config
    base_config = create_base_config(backend, output_dir)
    print(f"  Base config: {base_config}")

    # Run baseline for metric comparison
    baseline_metrics = run_baseline(base_config)

    # Select parameter sets based on mode
    if discover_mode:
        # Auto-discover from Pydantic models
        print("  Discovering parameters from Pydantic models...")
        backend_params = get_discovered_params(backend)
        shared = SHARED_PARAMS  # Still use shared for non-backend params
        parallelism_params = {}  # Parallelism handled separately

        if quick_mode:
            # Limit to first 2 values per param in quick mode
            backend_params = {k: v[:2] for k, v in backend_params.items()}
            shared = QUICK_SHARED_PARAMS
    elif quick_mode:
        shared = QUICK_SHARED_PARAMS
        if backend == "pytorch":
            backend_params = QUICK_PYTORCH_PARAMS
            parallelism_params = {}  # Skip in quick mode
        elif backend == "vllm":
            backend_params = QUICK_VLLM_PARAMS
            parallelism_params = {}
        else:
            backend_params = QUICK_TENSORRT_PARAMS
            parallelism_params = {}
    else:
        shared = SHARED_PARAMS
        if backend == "pytorch":
            backend_params = PYTORCH_PARAMS
            parallelism_params = PARALLELISM_PYTORCH
        elif backend == "vllm":
            backend_params = VLLM_PARAMS
            parallelism_params = PARALLELISM_VLLM
        else:
            backend_params = TENSORRT_PARAMS
            parallelism_params = PARALLELISM_TENSORRT

    all_params = {**shared, **backend_params}

    # Test each parameter variation
    print(f"\n  Testing {len(all_params)} parameters...")

    for param, values in all_params.items():
        print(f"\n  Parameter: {param}")
        print(f"    Values: {values}")

        for value in values:
            # Check if this param/value should be skipped (hardware/model limitations)
            should_skip, skip_reason = should_skip_param(param, value)
            if should_skip:
                print(f"    Testing {param}={value}... {skip_reason}")
                # Record skipped test in results
                skipped_result = TestResult(
                    config_name=f"{backend}_base_{param.replace('.', '_')}_{value}",
                    config_path="",  # No config file for skipped tests
                    status="skipped",
                    exit_code=0,  # Not a failure
                    elapsed_seconds=0.0,
                    error_summary=skip_reason,
                    parameter_varied=param,
                    parameter_value=value,
                )
                results.append(skipped_result)
                continue

            config_path = create_single_variation_config(
                base_config,
                param,
                value,
                output_dir / "variations",
            )

            if config_path is None:
                print(f"    [SKIP] Could not create config for {value}")
                continue

            print(f"    Testing {param}={value}...", end=" ", flush=True)

            result = run_experiment(config_path)
            result.parameter_varied = param
            result.parameter_value = value

            # Validate param application
            result.validation = validate_param_application(
                result,
                param,
                value,
                baseline_metrics,
            )

            status_icon = "✓" if result.status == "passed" else "✗"
            validation_status = result.validation.validation_status if result.validation else "N/A"
            print(
                f"[{status_icon}] {result.status} ({result.elapsed_seconds:.1f}s) [{validation_status}]"
            )

            if result.status == "failed":
                print(f"      Error: {result.error_summary}")

            results.append(result)

    # Test parallelism (requires 2 GPUs)
    if parallelism_params:
        print("\n  Testing parallelism strategies (2 GPUs)...")

        for strategy in parallelism_params.get("parallelism.strategy", []):
            if strategy == "none":
                continue  # Already tested in baseline

            print(f"    Strategy: {strategy}...", end=" ", flush=True)

            config_path = create_parallelism_config(
                backend,
                strategy,
                output_dir / "parallelism",
            )

            result = run_experiment(config_path)
            result.parameter_varied = "parallelism.strategy"
            result.parameter_value = strategy

            result.validation = validate_param_application(
                result,
                "parallelism.strategy",
                strategy,
                baseline_metrics,
            )

            status_icon = "✓" if result.status == "passed" else "✗"
            print(f"[{status_icon}] {result.status} ({result.elapsed_seconds:.1f}s)")

            if result.status == "failed":
                print(f"      Error: {result.error_summary}")

            results.append(result)

    return results


def print_summary(report: TestReport) -> None:
    """Print summary of test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    summary = report.summary
    print(f"  Total:   {summary['total']}")
    print(
        f"  Passed:  {summary['passed']} ({100 * summary['passed'] / max(1, summary['total']):.1f}%)"
    )
    print(f"  Failed:  {summary['failed']}")
    print(f"  Skipped: {summary['skipped']}")

    if report.warnings_by_type:
        print("\n  Warnings by type:")
        for category, count in sorted(report.warnings_by_type.items()):
            print(f"    {category}: {count}")

    if report.failed_tests:
        print("\n  Failed tests:")
        for test in report.failed_tests[:10]:
            print(f"    - {test['config']}: {test['error_summary']}")
        if len(report.failed_tests) > 10:
            print(f"    ... and {len(report.failed_tests) - 10} more")

    # Print validation summary
    verified = sum(
        1 for r in report.results if r.validation and r.validation.validation_status == "VERIFIED"
    )
    unverified = sum(
        1 for r in report.results if r.validation and r.validation.validation_status == "UNVERIFIED"
    )
    no_effect = sum(
        1 for r in report.results if r.validation and r.validation.validation_status == "NO_EFFECT"
    )

    print("\n  Parameter validation:")
    print(f"    VERIFIED:   {verified} (param applied + behavior changed)")
    print(f"    NO_EFFECT:  {no_effect} (param applied but no metric change)")
    print(f"    UNVERIFIED: {unverified} (no evidence param was applied)")


def list_discovered_params(backend: str | None = None) -> None:
    """List all auto-discovered parameters for debugging."""
    backends = [backend] if backend else ["pytorch", "vllm", "tensorrt"]

    for b in backends:
        print(f"\n{'=' * 60}")
        print(f"Discovered parameters for: {b}")
        print("=" * 60)

        params = get_discovered_params(b)
        if not params:
            print("  (no params discovered - import may have failed)")
            continue

        for param, values in sorted(params.items()):
            print(f"  {param}:")
            for v in values:
                print(f"    - {v}")

        print(
            f"\n  Total: {len(params)} parameters, {sum(len(v) for v in params.values())} test values"
        )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test all backend parameters")
    parser.add_argument(
        "--backend",
        choices=["pytorch", "vllm", "tensorrt"],
        help="Test specific backend (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer parameter variations",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep test results and state",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/test_results.json"),  # Write to mounted volume
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Auto-discover parameters from Pydantic models (single source of truth)",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List discovered parameters and exit (no tests run)",
    )

    args = parser.parse_args()

    # Handle --list-params (info mode, no tests)
    if args.list_params:
        list_discovered_params(args.backend)
        return

    print("=" * 60)
    print("LLenergyMeasure - Parameter Testing")
    print("=" * 60)
    print(f"  Model: {TEST_MODEL}")
    print(f"  Sample size: {TEST_SAMPLE_SIZE}")
    print(f"  Max output tokens: {TEST_MAX_OUTPUT}")
    print(f"  Timeout: {TEST_TIMEOUT_SECONDS}s per test")
    print(f"  Quick mode: {args.quick}")
    print(f"  Discover mode: {args.discover}")

    # Reset environment
    reset_environment(keep_results=args.no_cleanup)

    # Create output directory
    output_dir = PROJECT_ROOT / TEST_CONFIG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which backends to test
    backends = [args.backend] if args.backend else ["pytorch", "vllm", "tensorrt"]

    # Run tests
    all_results: list[TestResult] = []
    for backend in backends:
        results = test_backend(
            backend,
            args.quick,
            output_dir / backend,
            discover_mode=args.discover,
        )
        all_results.extend(results)

    # Generate report
    report = generate_report(
        all_results,
        backend=args.backend,
        baseline_metrics=None,  # Could aggregate from all backends
    )

    # Save report
    report_path = PROJECT_ROOT / args.output
    save_report(report, report_path)
    print(f"\n  Report saved: {report_path}")

    # Print summary
    print_summary(report)

    # Exit with appropriate code
    if report.summary["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
