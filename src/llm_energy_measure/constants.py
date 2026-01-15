"""Constants for LLM Bench framework."""

from pathlib import Path
from typing import Any

# Results directories
DEFAULT_RESULTS_DIR = Path("results")
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

# Schema version for result files
SCHEMA_VERSION = "2.0.0"

# State management
DEFAULT_STATE_DIR = ".llm_energy_measure_state"
COMPLETION_MARKER_PREFIX = ".completed_"
GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 2

# Built-in presets for quick experiment configuration
# Presets provide convenience defaults but NOT model (model is always required)
# All presets use deterministic sampling for reproducible measurements
PRESETS: dict[str, dict[str, Any]] = {
    # ==========================================================================
    # General presets (backend-agnostic)
    # ==========================================================================
    "quick-test": {
        "max_input_tokens": 64,
        "max_output_tokens": 32,
        "num_processes": 1,
        "gpus": [0],
        "batching": {"batch_size": 1},
        "decoder": {"preset": "deterministic"},  # Greedy for speed
    },
    "benchmark": {
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "batching": {"batch_size": 1},
        "decoder": {"preset": "deterministic"},  # Greedy for reproducibility
    },
    "throughput": {
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
