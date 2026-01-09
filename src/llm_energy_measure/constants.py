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
    "quick-test": {
        "max_input_tokens": 64,
        "max_output_tokens": 32,
        "num_processes": 1,
        "gpu_list": [0],
        "batching_options": {"batch_size": 1},
        "decoder_config": {"preset": "deterministic"},  # Greedy for speed
    },
    "benchmark": {
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "fp_precision": "float16",
        "batching_options": {"batch_size": 1},
        "decoder_config": {"preset": "deterministic"},  # Greedy for reproducibility
    },
    "throughput": {
        "max_input_tokens": 512,
        "max_output_tokens": 256,
        "fp_precision": "float16",
        "batching_options": {"batch_size": 8, "dynamic_batching": True},
        "decoder_config": {"preset": "deterministic"},  # Greedy for consistent throughput
    },
}
