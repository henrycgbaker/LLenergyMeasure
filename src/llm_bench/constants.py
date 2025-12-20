"""Constants for LLM Bench framework."""

from pathlib import Path

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
