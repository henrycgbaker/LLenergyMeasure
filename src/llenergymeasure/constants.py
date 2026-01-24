"""Constants for LLM Bench framework."""

import os
from pathlib import Path
from typing import Any

# Results directories
# Precedence: CLI --results-dir > LLM_ENERGY_RESULTS_DIR env var > "results"
DEFAULT_RESULTS_DIR = Path(os.environ.get("LLM_ENERGY_RESULTS_DIR", "results"))
RAW_RESULTS_SUBDIR = "raw"
AGGREGATED_RESULTS_SUBDIR = "aggregated"

# Experiment defaults
DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLING_INTERVAL_SEC = 1.0
DEFAULT_ACCELERATE_PORT = 29500

# Inference defaults
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0

# Streaming latency measurement
DEFAULT_STREAMING_WARMUP_REQUESTS = 5

# Schema version for result files
SCHEMA_VERSION = "2.0.0"

# State management
# Precedence: LLM_ENERGY_STATE_DIR env var > ".state"
DEFAULT_STATE_DIR = Path(os.environ.get("LLM_ENERGY_STATE_DIR", ".state"))
COMPLETION_MARKER_PREFIX = ".completed_"

# Timeouts
GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 2
DEFAULT_BARRIER_TIMEOUT_SEC = 600  # 10 minutes for distributed sync
DEFAULT_FLOPS_TIMEOUT_SEC = 30
DEFAULT_GPU_INFO_TIMEOUT_SEC = 10
DEFAULT_SIGKILL_WAIT_SEC = 2

# Built-in presets for quick experiment configuration (SSOT for CLI + docs)
# Presets provide convenience defaults but NOT model (model is always required)
# All presets use deterministic sampling for reproducible measurements
#
# Each preset includes _meta for CLI display and documentation:
#   - description: Short description shown in `lem list presets`
#   - use_case: When to use this preset
PRESETS: dict[str, dict[str, Any]] = {
    # ==========================================================================
    # General presets (backend-agnostic)
    # ==========================================================================
    "quick-test": {
        "_meta": {
            "description": "Fast validation runs",
            "use_case": "Quick sanity checks, CI testing",
        },
        "max_input_tokens": 64,
        "max_output_tokens": 32,
        "num_processes": 1,
        "gpus": [0],
        "batching": {"batch_size": 1},
        "decoder": {"preset": "deterministic"},  # Greedy for speed
    },
    "benchmark": {
        "_meta": {
            "description": "Formal benchmark measurements",
            "use_case": "Reproducible benchmarks, paper results",
        },
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "batching": {"batch_size": 1},
        "decoder": {"preset": "deterministic"},  # Greedy for reproducibility
    },
    "throughput": {
        "_meta": {
            "description": "Throughput-optimised testing",
            "use_case": "Maximum tokens/second measurement",
        },
        "max_input_tokens": 512,
        "max_output_tokens": 256,
        "fp_precision": "float16",
        "batching": {"batch_size": 8, "dynamic_batching": True},
        "decoder": {"preset": "deterministic"},  # Greedy for consistent throughput
    },
    # ==========================================================================
    # vLLM-specific presets
    # ==========================================================================
    "vllm-throughput": {
        "_meta": {
            "description": "vLLM high-throughput serving",
            "use_case": "Production serving, max tokens/second",
        },
        "backend": "vllm",
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "vllm": {
            "max_num_seqs": 512,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
        },
    },
    "vllm-speculative": {
        "_meta": {
            "description": "vLLM with speculative decoding",
            "use_case": "Lower latency via n-gram speculation",
        },
        "backend": "vllm",
        "max_input_tokens": 2048,
        "max_output_tokens": 256,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "vllm": {
            "speculative": {
                "method": "ngram",
                "num_tokens": 5,
                "ngram_max": 4,
            },
        },
    },
    "vllm-memory-efficient": {
        "_meta": {
            "description": "vLLM with FP8 KV cache",
            "use_case": "Large context, memory-constrained GPUs",
        },
        "backend": "vllm",
        "max_input_tokens": 4096,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "vllm": {
            "kv_cache_dtype": "fp8",
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
        },
    },
    "vllm-low-latency": {
        "_meta": {
            "description": "vLLM optimised for TTFT",
            "use_case": "Interactive chat, low first-token latency",
        },
        "backend": "vllm",
        "max_input_tokens": 512,
        "max_output_tokens": 128,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "vllm": {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 2048,
            "enforce_eager": True,  # Disable CUDA graphs for lower first-token latency
        },
    },
    # ==========================================================================
    # PyTorch-specific presets
    # ==========================================================================
    "pytorch-optimized": {
        "_meta": {
            "description": "PyTorch with Flash Attention + compile",
            "use_case": "Best PyTorch performance (Ampere+ GPU)",
        },
        "backend": "pytorch",
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "pytorch": {
            "attn_implementation": "flash_attention_2",
            "torch_compile": "reduce-overhead",
        },
    },
    "pytorch-speculative": {
        "_meta": {
            "description": "PyTorch with assisted generation",
            "use_case": "Speculative decoding for lower latency",
        },
        "backend": "pytorch",
        "max_input_tokens": 2048,
        "max_output_tokens": 256,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "pytorch": {
            "attn_implementation": "sdpa",
            "assisted_generation": {
                "num_tokens": 5,
            },
        },
    },
    "pytorch-compatible": {
        "_meta": {
            "description": "PyTorch maximum compatibility",
            "use_case": "Older GPUs, debugging, model issues",
        },
        "backend": "pytorch",
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "pytorch": {
            "attn_implementation": "eager",
            "torch_compile": False,
        },
    },
}


def get_preset_metadata(preset_name: str) -> dict[str, str] | None:
    """Get metadata for a preset (SSOT accessor).

    Args:
        preset_name: Name of the preset.

    Returns:
        Dict with description and use_case, or None if preset not found.
    """
    preset = PRESETS.get(preset_name)
    if preset:
        return preset.get("_meta")
    return None


def get_preset_config(preset_name: str) -> dict[str, Any] | None:
    """Get preset config without metadata (for applying to experiments).

    Args:
        preset_name: Name of the preset.

    Returns:
        Config dict (excluding _meta), or None if preset not found.
    """
    preset = PRESETS.get(preset_name)
    if preset:
        return {k: v for k, v in preset.items() if k != "_meta"}
    return None


# Alias for backward compatibility
EXPERIMENT_PRESETS = PRESETS


# =============================================================================
# DEPRECATED CLI FLAGS
# =============================================================================
#
# CLI flags that should be set in config files instead.
# Philosophy: CLI = workflow/meta params, YAML = testable experiment params
#
# These flags are deprecated but still work with a warning.
# Use --allow-deprecated to suppress the warning during migration.

DEPRECATED_CLI_FLAGS: dict[str, dict[str, str]] = {
    "--batch-size": {
        "canonical": "batching.batch_size",
        "removal_version": "2.0",
        "migration": "Set batching.batch_size in config YAML",
    },
    "-b": {
        "canonical": "batching.batch_size",
        "removal_version": "2.0",
        "migration": "Set batching.batch_size in config YAML",
    },
    "--temperature": {
        "canonical": "decoder.temperature",
        "removal_version": "2.0",
        "migration": "Set decoder.temperature in config YAML (or use decoder.preset)",
    },
    "--precision": {
        "canonical": "fp_precision",
        "removal_version": "2.0",
        "migration": "Set fp_precision in config YAML",
    },
    "--num-processes": {
        "canonical": "num_processes",
        "removal_version": "2.0",
        "migration": "Set num_processes in config YAML (or use parallelism settings)",
    },
    "--gpu-list": {
        "canonical": "gpus",
        "removal_version": "2.0",
        "migration": "Set gpus in config YAML",
    },
    "--quantization": {
        "canonical": "quantization.load_in_4bit",
        "removal_version": "2.0",
        "migration": "Set quantization settings in config YAML",
    },
}


def is_cli_flag_deprecated(flag: str) -> bool:
    """Check if a CLI flag is deprecated.

    Args:
        flag: CLI flag including dashes (e.g., "--batch-size")

    Returns:
        True if deprecated, False otherwise.
    """
    return flag in DEPRECATED_CLI_FLAGS


def get_deprecation_info(flag: str) -> dict[str, str] | None:
    """Get deprecation information for a CLI flag.

    Args:
        flag: CLI flag including dashes

    Returns:
        Deprecation info dict or None if not deprecated.
    """
    return DEPRECATED_CLI_FLAGS.get(flag)
