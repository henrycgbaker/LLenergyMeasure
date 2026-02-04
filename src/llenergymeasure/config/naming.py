"""Parameter naming consistency registry.

This module provides a canonical naming registry for configuration parameters,
ensuring consistent naming across CLI, YAML configs, and internal code.

All code uses canonical names only (no legacy aliases):
- YAML: batching, decoder, quantization, sharding, gpus
- CLI: Deprecated flags map to canonical config paths
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# PARAMETER ALIAS REGISTRY
# =============================================================================
#
# Maps canonical parameter names to their aliases across different contexts.
# Structure:
#   "canonical.path": {
#       "cli": ["--flag-name", "-s"],      # CLI flags
#       "yaml_legacy": ["old_name"],        # Legacy YAML field names
#       "canonical": "canonical.path",      # Canonical name (for reference)
#       "description": "Human-readable description",
#   }
#
# The canonical name is the "source of truth" used throughout internal code.

PARAMETER_ALIASES: dict[str, dict[str, Any]] = {
    # =========================================================================
    # BATCHING PARAMETERS
    # =========================================================================
    "batching.batch_size": {
        "cli": ["--batch-size", "-b"],
        "yaml_legacy": ["batching_options.batch_size", "batch_size"],
        "canonical": "batching.batch_size",
        "description": "Number of prompts per batch",
        "deprecated_cli": True,  # Should be set in config
    },
    "batching.strategy": {
        "cli": [],
        "yaml_legacy": ["batching_options.strategy"],
        "canonical": "batching.strategy",
        "description": "Batching strategy (static, dynamic, sorted_static, sorted_dynamic)",
    },
    "batching.max_tokens_per_batch": {
        "cli": [],
        "yaml_legacy": ["batching_options.max_tokens_per_batch"],
        "canonical": "batching.max_tokens_per_batch",
        "description": "Maximum tokens per batch for dynamic batching",
    },
    # =========================================================================
    # DECODER/SAMPLING PARAMETERS
    # =========================================================================
    "decoder.temperature": {
        "cli": ["--temperature"],
        "yaml_legacy": ["decoder_config.temperature", "temperature"],
        "canonical": "decoder.temperature",
        "description": "Sampling temperature (0=greedy)",
        "deprecated_cli": True,
    },
    "decoder.top_p": {
        "cli": ["--top-p"],
        "yaml_legacy": ["decoder_config.top_p", "top_p"],
        "canonical": "decoder.top_p",
        "description": "Top-p nucleus sampling threshold",
    },
    "decoder.top_k": {
        "cli": ["--top-k"],
        "yaml_legacy": ["decoder_config.top_k", "top_k"],
        "canonical": "decoder.top_k",
        "description": "Top-k sampling limit",
    },
    "decoder.do_sample": {
        "cli": [],
        "yaml_legacy": ["decoder_config.do_sample"],
        "canonical": "decoder.do_sample",
        "description": "Whether to use sampling vs greedy",
    },
    "decoder.repetition_penalty": {
        "cli": [],
        "yaml_legacy": ["decoder_config.repetition_penalty"],
        "canonical": "decoder.repetition_penalty",
        "description": "Repetition penalty factor",
    },
    "decoder.preset": {
        "cli": [],
        "yaml_legacy": ["decoder_config.preset"],
        "canonical": "decoder.preset",
        "description": "Decoder preset (deterministic, standard, creative, factual)",
    },
    # =========================================================================
    # PRECISION/QUANTIZATION PARAMETERS
    # =========================================================================
    "fp_precision": {
        "cli": ["--precision"],
        "yaml_legacy": ["precision"],
        "canonical": "fp_precision",
        "description": "Floating point precision (float16, bfloat16, float32)",
        "deprecated_cli": True,
    },
    "quantization.load_in_4bit": {
        "cli": ["--quantization"],  # Legacy shortcut
        "yaml_legacy": ["quantization_config.load_in_4bit"],
        "canonical": "quantization.load_in_4bit",
        "description": "Enable 4-bit quantization",
        "deprecated_cli": True,
    },
    "quantization.load_in_8bit": {
        "cli": [],
        "yaml_legacy": ["quantization_config.load_in_8bit"],
        "canonical": "quantization.load_in_8bit",
        "description": "Enable 8-bit quantization",
    },
    # =========================================================================
    # PARALLELISM/SHARDING PARAMETERS
    # =========================================================================
    "parallelism.strategy": {
        "cli": [],
        "yaml_legacy": ["sharding.strategy", "sharding_config.strategy"],
        "canonical": "parallelism.strategy",
        "description": "Parallelism strategy (none, tensor_parallel, pipeline_parallel, data_parallel)",
    },
    "parallelism.degree": {
        "cli": [],
        "yaml_legacy": ["sharding.num_shards", "sharding_config.num_shards", "num_processes"],
        "canonical": "parallelism.degree",
        "description": "Parallelism degree (number of GPUs/processes)",
    },
    "num_processes": {
        "cli": ["--num-processes"],
        "yaml_legacy": [],
        "canonical": "num_processes",
        "description": "[Deprecated] Number of processes - use parallelism.degree",
        "deprecated_cli": True,
    },
    "gpus": {
        "cli": ["--gpu-list"],
        "yaml_legacy": ["gpu_list"],
        "canonical": "gpus",
        "description": "List of GPU indices to use",
        "deprecated_cli": True,
    },
    # =========================================================================
    # TOKEN/INPUT PARAMETERS
    # =========================================================================
    "max_input_tokens": {
        "cli": [],
        "yaml_legacy": [],
        "canonical": "max_input_tokens",
        "description": "Maximum input tokens per prompt",
    },
    "max_output_tokens": {
        "cli": ["--max-tokens", "-t"],
        "yaml_legacy": [],
        "canonical": "max_output_tokens",
        "description": "Maximum output tokens to generate",
        # Not deprecated - this is a workflow param
    },
    "min_output_tokens": {
        "cli": [],
        "yaml_legacy": [],
        "canonical": "min_output_tokens",
        "description": "Minimum output tokens to generate",
    },
    "num_input_prompts": {
        "cli": ["-n", "--num-prompts"],
        "yaml_legacy": [],
        "canonical": "num_input_prompts",
        "description": "Number of prompts to process",
        # Not deprecated - this is a workflow param
    },
    # =========================================================================
    # BACKEND PARAMETERS
    # =========================================================================
    "backend": {
        "cli": ["--backend"],
        "yaml_legacy": [],
        "canonical": "backend",
        "description": "Inference backend (pytorch, vllm, tensorrt)",
        # Not deprecated - this is a workflow param
    },
    # =========================================================================
    # STREAMING PARAMETERS
    # =========================================================================
    "streaming": {
        "cli": ["--streaming"],
        "yaml_legacy": [],
        "canonical": "streaming",
        "description": "Enable streaming mode for TTFT/ITL measurement",
        # Not deprecated - this is a workflow param
    },
    "streaming_warmup_requests": {
        "cli": ["--streaming-warmup"],
        "yaml_legacy": [],
        "canonical": "streaming_warmup_requests",
        "description": "Warmup requests before streaming measurement",
        # Not deprecated - this is a workflow param
    },
    # =========================================================================
    # REPRODUCIBILITY PARAMETERS
    # =========================================================================
    "random_seed": {
        "cli": ["--seed"],
        "yaml_legacy": ["seed"],
        "canonical": "random_seed",
        "description": "Random seed for reproducibility",
        # Not deprecated - this is a workflow param
    },
}


def get_canonical_name(alias: str) -> str:
    """Get the canonical parameter name for an alias.

    Args:
        alias: Parameter alias (CLI flag, legacy YAML name, etc.)

    Returns:
        Canonical parameter name, or the input if no mapping found.
    """
    # Check if already canonical
    if alias in PARAMETER_ALIASES:
        return alias

    # Search through aliases
    for canonical, info in PARAMETER_ALIASES.items():
        # Check CLI aliases (strip leading dashes)
        cli_aliases = info.get("cli", [])
        alias_stripped = alias.lstrip("-")
        for cli_alias in cli_aliases:
            if cli_alias.lstrip("-").replace("-", "_") == alias_stripped.replace("-", "_"):
                return canonical

        # Check YAML legacy aliases
        yaml_aliases = info.get("yaml_legacy", [])
        if alias in yaml_aliases:
            return canonical

    # Not found - return as-is
    return alias


def get_cli_flag_for_param(canonical: str) -> str | None:
    """Get the primary CLI flag for a canonical parameter.

    Args:
        canonical: Canonical parameter name.

    Returns:
        Primary CLI flag (e.g., "--batch-size") or None if no CLI flag.
    """
    if canonical in PARAMETER_ALIASES:
        cli_flags: list[str] = PARAMETER_ALIASES[canonical].get("cli", [])
        if cli_flags:
            return str(cli_flags[0])
    return None


def is_deprecated_cli_flag(flag: str) -> bool:
    """Check if a CLI flag is deprecated.

    Args:
        flag: CLI flag to check (e.g., "--batch-size").

    Returns:
        True if the flag is deprecated, False otherwise.
    """
    flag_stripped = flag.lstrip("-").replace("-", "_")

    for info in PARAMETER_ALIASES.values():
        cli_aliases: list[str] = info.get("cli", [])
        for cli_alias in cli_aliases:
            if cli_alias.lstrip("-").replace("-", "_") == flag_stripped:
                return bool(info.get("deprecated_cli", False))

    return False


def get_all_deprecated_cli_flags() -> dict[str, dict[str, str]]:
    """Get all deprecated CLI flags with migration info.

    Returns:
        Dictionary mapping deprecated flags to their migration info:
        {
            "--batch-size": {
                "canonical": "batching.batch_size",
                "migration": "Set batching.batch_size in config YAML",
            }
        }
    """
    deprecated: dict[str, dict[str, str]] = {}

    for canonical, info in PARAMETER_ALIASES.items():
        if info.get("deprecated_cli", False):
            for flag in info.get("cli", []):
                deprecated[flag] = {
                    "canonical": canonical,
                    "migration": f"Set {canonical} in config YAML",
                    "description": info.get("description", ""),
                }

    return deprecated
