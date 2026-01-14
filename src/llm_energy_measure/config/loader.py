"""Configuration loading with inheritance support."""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError


@dataclass
class ConfigWarning:
    """A configuration warning with severity level.

    Severity levels:
    - error: Impossible/invalid config combination (blocks execution without --force)
    - warning: Problematic config that may cause unexpected behaviour
    - info: Suboptimal but valid configuration
    """

    field: str
    message: str
    severity: Literal["error", "warning", "info"] = "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"

    def to_result_string(self) -> str:
        """Format for embedding in results."""
        return f"{self.severity}: {self.field} - {self.message}"


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with overlay taking precedence.

    Args:
        base: Base dictionary.
        overlay: Dictionary to overlay on base.

    Returns:
        Merged dictionary.
    """
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config_dict(path: Path) -> dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If file cannot be loaded.
    """
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            result = yaml.safe_load(content)
            return result if isinstance(result, dict) else {}
        elif path.suffix == ".json":
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ConfigurationError(f"Config must be a JSON object: {path}")
            return result
        else:
            raise ConfigurationError(f"Unsupported config format: {path.suffix}")
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Failed to parse config {path}: {e}") from e


def resolve_inheritance(
    config_dict: dict[str, Any],
    config_path: Path,
    seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Resolve config inheritance via _extends field.

    Supports single inheritance where a config can extend a base config.
    The _extends path is relative to the current config file.

    Args:
        config_dict: Configuration dictionary.
        config_path: Path to the configuration file.
        seen: Set of already processed paths (for cycle detection).

    Returns:
        Resolved configuration dictionary with inheritance applied.

    Raises:
        ConfigurationError: If circular inheritance detected.
    """
    if seen is None:
        seen = set()

    resolved_path = config_path.resolve()
    if resolved_path in seen:
        raise ConfigurationError(f"Circular config inheritance detected: {resolved_path}")
    seen.add(resolved_path)

    if "_extends" not in config_dict:
        return config_dict

    extends_path = config_dict.pop("_extends")
    base_path = config_path.parent / extends_path

    base_dict = load_config_dict(base_path)
    base_resolved = resolve_inheritance(base_dict, base_path, seen)

    return deep_merge(base_resolved, config_dict)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate experiment configuration.

    Supports YAML and JSON formats, and config inheritance via _extends.

    Example with inheritance:
        ```yaml
        # configs/base.yaml
        max_input_tokens: 512
        warmup_runs: 3

        # configs/llama2-7b.yaml
        _extends: base.yaml
        config_name: llama2-7b
        model_name: meta-llama/Llama-2-7b-hf
        ```

    Args:
        path: Path to configuration file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        ConfigurationError: If config is invalid.
    """
    path = Path(path)
    config_dict = load_config_dict(path)
    resolved = resolve_inheritance(config_dict, path)

    try:
        return ExperimentConfig(**resolved)
    except Exception as e:
        raise ConfigurationError(f"Invalid config {path}: {e}") from e


def validate_config(config: ExperimentConfig) -> list[ConfigWarning]:
    """Validate configuration and return any warnings.

    Checks for potential issues in configuration. Returns warnings at three levels:
    - error: Impossible/conflicting config (blocks without --force)
    - warning: Problematic config that may cause unexpected behaviour
    - info: Suboptimal but valid configuration

    Args:
        config: Configuration to validate.

    Returns:
        List of ConfigWarning objects (empty if no warnings).
    """
    warnings: list[ConfigWarning] = []

    # =========================================================================
    # DISTRIBUTED CONFIG
    # =========================================================================

    if config.num_processes > 1 and len(config.gpu_list) == 1:
        warnings.append(
            ConfigWarning(
                field="num_processes",
                message="Multiple processes with single GPU may not provide parallelism benefits",
                severity="info",
            )
        )

    # =========================================================================
    # TOKEN CONFIG
    # =========================================================================

    if config.max_output_tokens > 2048:
        warnings.append(
            ConfigWarning(
                field="max_output_tokens",
                message=f"max_output_tokens={config.max_output_tokens} is very high, may cause memory issues",
                severity="warning",
            )
        )

    # =========================================================================
    # QUANTIZATION CONFIG
    # =========================================================================

    quant = config.quantization_config
    if quant.quantization and config.fp_precision == "float32":
        warnings.append(
            ConfigWarning(
                field="quantization_config",
                message="Quantization enabled with float32 precision - quantization typically uses float16 compute",
                severity="warning",
            )
        )

    if quant.quantization and not quant.load_in_4bit and not quant.load_in_8bit:
        warnings.append(
            ConfigWarning(
                field="quantization_config",
                message="quantization=True but neither load_in_4bit nor load_in_8bit specified",
                severity="error",
            )
        )

    # =========================================================================
    # BATCHING CONFIG
    # =========================================================================

    batch = config.batching_options
    if batch.strategy in ("dynamic", "sorted_dynamic") and batch.max_tokens_per_batch is None:
        warnings.append(
            ConfigWarning(
                field="batching_options",
                message=f"Dynamic batching strategy '{batch.strategy}' without max_tokens_per_batch - will use max_input_tokens as budget",
                severity="info",
            )
        )

    if batch.strategy in ("sorted_static", "sorted_dynamic") and batch.batch_size == 1:
        warnings.append(
            ConfigWarning(
                field="batching_options",
                message=f"Sorted strategy '{batch.strategy}' with batch_size=1 provides no benefit (sorting is pointless)",
                severity="info",
            )
        )

    # =========================================================================
    # SHARDING CONFIG
    # =========================================================================

    shard = config.sharding_config
    if shard.strategy != "none":
        if shard.num_shards > len(config.gpu_list):
            warnings.append(
                ConfigWarning(
                    field="sharding_config",
                    message=f"num_shards={shard.num_shards} exceeds available GPUs ({len(config.gpu_list)})",
                    severity="error",
                )
            )
        if len(config.gpu_list) == 1:
            warnings.append(
                ConfigWarning(
                    field="sharding_config",
                    message=f"Sharding strategy '{shard.strategy}' with single GPU provides no benefit",
                    severity="info",
                )
            )

    # Tensor parallelism specific validation
    if shard.strategy == "tensor_parallel":
        # Check model support for native TP
        from llm_energy_measure.core.parallelism import is_model_tp_compatible

        if not is_model_tp_compatible(config.model_name):
            warnings.append(
                ConfigWarning(
                    field="sharding_config",
                    message=(
                        f"Model '{config.model_name}' may not support HuggingFace native tensor parallelism. "
                        f"Supported architectures: Llama, Mistral, Mixtral, Qwen, Phi, Gemma, Falcon, MPT, BLOOM, OPT"
                    ),
                    severity="warning",
                )
            )

        # Quantization + TP warning
        if quant.quantization:
            warnings.append(
                ConfigWarning(
                    field="sharding_config",
                    message="Quantization with tensor parallelism is experimental and may not work correctly",
                    severity="warning",
                )
            )

    # =========================================================================
    # DECODER/SAMPLING CONFIG
    # =========================================================================

    decoder = config.decoder_config

    # Sampling params have no effect in greedy/deterministic mode
    if decoder.is_deterministic:
        ignored_params = []
        if decoder.top_k != 50:  # Non-default
            ignored_params.append(f"top_k={decoder.top_k}")
        if decoder.top_p != 1.0:  # Non-default
            ignored_params.append(f"top_p={decoder.top_p}")
        if decoder.min_p != 0.0:  # Non-default
            ignored_params.append(f"min_p={decoder.min_p}")
        if decoder.repetition_penalty != 1.0:  # Non-default
            ignored_params.append(f"repetition_penalty={decoder.repetition_penalty}")

        if ignored_params:
            warnings.append(
                ConfigWarning(
                    field="decoder_config",
                    message=f"Sampling params [{', '.join(ignored_params)}] have no effect in deterministic mode (temp=0 or do_sample=False)",
                    severity="error",
                )
            )

    # do_sample=True has no effect when temperature=0
    if decoder.do_sample and decoder.temperature == 0.0:
        warnings.append(
            ConfigWarning(
                field="decoder_config.do_sample",
                message="do_sample=True has no effect when temperature=0 (greedy decoding)",
                severity="info",
            )
        )

    # Not recommended to alter both temperature and top_p
    if decoder.temperature != 1.0 and decoder.temperature != 0.0 and decoder.top_p != 1.0:
        warnings.append(
            ConfigWarning(
                field="decoder_config",
                message="Both temperature and top_p modified - not recommended, alter one or the other",
                severity="warning",
            )
        )

    # =========================================================================
    # TRAFFIC SIMULATION CONFIG
    # =========================================================================

    traffic = config.latency_simulation
    if traffic.enabled and traffic.target_qps > 100:
        warnings.append(
            ConfigWarning(
                field="latency_simulation",
                message=f"target_qps={traffic.target_qps} is very high - may not achieve target rate",
                severity="warning",
            )
        )

    return warnings


def has_blocking_warnings(warnings: list[ConfigWarning]) -> bool:
    """Check if any warnings are blocking (error severity)."""
    return any(w.severity == "error" for w in warnings)
