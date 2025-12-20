"""Configuration loading with inheritance support."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError


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


def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate configuration and return any warnings.

    Args:
        config: Configuration to validate.

    Returns:
        List of warning messages (empty if no warnings).
    """
    warnings = []

    # Check for potential issues (not errors, just warnings)
    if config.num_processes > 1 and len(config.gpu_list) == 1:
        warnings.append("Multiple processes with single GPU may not provide parallelism benefits")

    if config.max_output_tokens > 2048:
        warnings.append(
            f"max_output_tokens={config.max_output_tokens} is very high, " "may cause memory issues"
        )

    if config.quantization_config.quantization and config.fp_precision == "float32":
        warnings.append(
            "Quantization enabled with float32 precision - "
            "quantization typically uses float16 compute"
        )

    return warnings
