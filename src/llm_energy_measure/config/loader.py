"""Configuration loading with inheritance support and provenance tracking."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError

if TYPE_CHECKING:
    from llm_energy_measure.config.provenance import ResolvedConfig


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

    if config.num_processes > 1 and len(config.gpus) == 1:
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

    quant = config.quantization
    if quant.quantization and config.fp_precision == "float32":
        warnings.append(
            ConfigWarning(
                field="quantization",
                message="Quantization enabled with float32 precision - quantization typically uses float16 compute",
                severity="warning",
            )
        )

    if quant.quantization and not quant.load_in_4bit and not quant.load_in_8bit:
        warnings.append(
            ConfigWarning(
                field="quantization",
                message="quantization=True but neither load_in_4bit nor load_in_8bit specified",
                severity="error",
            )
        )

    # =========================================================================
    # BATCHING CONFIG
    # =========================================================================

    batch = config.batching
    if batch.strategy in ("dynamic", "sorted_dynamic") and batch.max_tokens_per_batch is None:
        warnings.append(
            ConfigWarning(
                field="batching",
                message=f"Dynamic batching strategy '{batch.strategy}' without max_tokens_per_batch - will use max_input_tokens as budget",
                severity="info",
            )
        )

    # batch_size is ignored for dynamic strategies
    if batch.strategy in ("dynamic", "sorted_dynamic") and batch.batch_size != 1:
        warnings.append(
            ConfigWarning(
                field="batching.batch_size",
                message=f"batch_size={batch.batch_size} is ignored with '{batch.strategy}' strategy. Dynamic strategies use max_tokens_per_batch instead.",
                severity="warning",
            )
        )

    if batch.strategy in ("sorted_static", "sorted_dynamic") and batch.batch_size == 1:
        warnings.append(
            ConfigWarning(
                field="batching",
                message=f"Sorted strategy '{batch.strategy}' with batch_size=1 provides no benefit (sorting is pointless)",
                severity="info",
            )
        )

    # =========================================================================
    # SHARDING CONFIG
    # =========================================================================

    shard = config.sharding
    if shard.strategy != "none":
        if shard.num_shards > len(config.gpus):
            warnings.append(
                ConfigWarning(
                    field="sharding",
                    message=f"num_shards={shard.num_shards} exceeds available GPUs ({len(config.gpus)})",
                    severity="error",
                )
            )
        if len(config.gpus) == 1:
            warnings.append(
                ConfigWarning(
                    field="sharding",
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
                    field="sharding",
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
                    field="sharding",
                    message="Quantization with tensor parallelism is experimental and may not work correctly",
                    severity="warning",
                )
            )

    # =========================================================================
    # PARALLELISM CONFIG (new unified config)
    # =========================================================================

    para = config.parallelism
    if para.strategy == "data_parallel" and config.backend == "vllm":
        warnings.append(
            ConfigWarning(
                field="parallelism.strategy",
                message=(
                    "data_parallel is not fully supported for vLLM. "
                    "vLLM manages tensor/pipeline parallelism internally. "
                    "For data parallelism, use multiple separate experiments or the PyTorch backend."
                ),
                severity="error",
            )
        )

    if para.strategy != "none" and para.degree > len(config.gpus):
        warnings.append(
            ConfigWarning(
                field="parallelism.degree",
                message=f"degree={para.degree} exceeds available GPUs ({len(config.gpus)})",
                severity="error",
            )
        )

    # =========================================================================
    # DECODER/SAMPLING CONFIG
    # =========================================================================

    decoder = config.decoder

    # Warn about preset + individual params (mutual exclusivity guidance)
    if decoder.preset is not None:
        warnings.append(
            ConfigWarning(
                field="decoder.preset",
                message=(
                    f"Using preset '{decoder.preset}'. Any individual params (temperature, top_p, etc.) "
                    "set in config will override the preset defaults. "
                    "Recommendation: use EITHER preset OR individual params, not both."
                ),
                severity="info",
            )
        )

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
                    field="decoder",
                    message=f"Sampling params [{', '.join(ignored_params)}] have no effect in deterministic mode (temp=0 or do_sample=False)",
                    severity="error",
                )
            )

    # do_sample=True has no effect when temperature=0
    if decoder.do_sample and decoder.temperature == 0.0:
        warnings.append(
            ConfigWarning(
                field="decoder.do_sample",
                message="do_sample=True has no effect when temperature=0 (greedy decoding)",
                severity="info",
            )
        )

    # Not recommended to alter both temperature and top_p
    if decoder.temperature != 1.0 and decoder.temperature != 0.0 and decoder.top_p != 1.0:
        warnings.append(
            ConfigWarning(
                field="decoder",
                message="Both temperature and top_p modified - not recommended, alter one or the other",
                severity="warning",
            )
        )

    # =========================================================================
    # TRAFFIC SIMULATION CONFIG
    # =========================================================================

    traffic = config.traffic_simulation
    if traffic.enabled and traffic.target_qps > 100:
        warnings.append(
            ConfigWarning(
                field="latency_simulation",
                message=f"target_qps={traffic.target_qps} is very high - may not achieve target rate",
                severity="warning",
            )
        )

    # =========================================================================
    # BACKEND-SPECIFIC CONFIG VALIDATION
    # =========================================================================

    # Validate config against the selected backend (if not default pytorch)
    backend_name = getattr(config, "backend", "pytorch")
    try:
        from llm_energy_measure.core.inference_backends import get_backend

        backend = get_backend(backend_name)
        backend_warnings = backend.validate_config(config)

        # Convert backend ConfigWarnings to config ConfigWarnings
        for bw in backend_warnings:
            severity = bw.severity if bw.severity in ("error", "warning", "info") else "warning"
            warnings.append(
                ConfigWarning(
                    field=f"backend.{bw.param}",
                    message=bw.message,
                    severity=severity,  # type: ignore[arg-type]
                )
            )
    except Exception as e:
        # Backend not available or validation failed - add warning
        if backend_name != "pytorch":
            warnings.append(
                ConfigWarning(
                    field="backend",
                    message=f"Could not validate config for backend '{backend_name}': {e}",
                    severity="warning",
                )
            )

    return warnings


def has_blocking_warnings(warnings: list[ConfigWarning]) -> bool:
    """Check if any warnings are blocking (error severity)."""
    return any(w.severity == "error" for w in warnings)


def get_pydantic_defaults() -> dict[str, Any]:
    """Extract default values from ExperimentConfig Pydantic model.

    Returns:
        Flattened dictionary of all default values.
    """
    from llm_energy_measure.config.provenance import flatten_dict

    # Create a minimal config to get defaults
    minimal: dict[str, Any] = {"config_name": "__defaults__", "model_name": "__defaults__"}
    defaults_config = ExperimentConfig(**minimal)
    defaults_dict = defaults_config.model_dump()

    # Remove the placeholder values
    defaults_dict.pop("config_name", None)
    defaults_dict.pop("model_name", None)

    return flatten_dict(defaults_dict)


def load_config_with_provenance(
    path: Path | str | None = None,
    preset_name: str | None = None,
    preset_dict: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Load configuration with full provenance tracking.

    Builds the final configuration by layering sources in order of precedence:
    1. Pydantic defaults (lowest precedence)
    2. Preset values (if preset_name or preset_dict provided)
    3. Config file values (if path provided)
    4. CLI overrides (highest precedence)

    Each parameter's source is tracked for debugging and reproducibility.

    Args:
        path: Path to configuration file (optional).
        preset_name: Name of preset to apply (optional).
        preset_dict: Preset dictionary to apply (alternative to preset_name).
        cli_overrides: CLI override dictionary (optional).

    Returns:
        ResolvedConfig with config and provenance information.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    from llm_energy_measure.config.provenance import (
        ParameterProvenance,
        ParameterSource,
        ResolvedConfig,
        compare_dicts,
        flatten_dict,
        unflatten_dict,
    )
    from llm_energy_measure.constants import EXPERIMENT_PRESETS

    provenance: dict[str, ParameterProvenance] = {}
    preset_chain: list[str] = []

    # =================================================================
    # Layer 1: Pydantic defaults
    # =================================================================
    pydantic_defaults = get_pydantic_defaults()

    # Mark all defaults as PYDANTIC_DEFAULT
    for path_key, value in pydantic_defaults.items():
        provenance[path_key] = ParameterProvenance(
            path=path_key,
            value=value,
            source=ParameterSource.PYDANTIC_DEFAULT,
        )

    # Start with empty config dict (will build from required fields)
    config_dict: dict[str, Any] = {}

    # =================================================================
    # Layer 2: Preset values
    # =================================================================
    if preset_name:
        if preset_name not in EXPERIMENT_PRESETS:
            raise ConfigurationError(f"Unknown preset: {preset_name}")
        preset_dict = EXPERIMENT_PRESETS[preset_name]
        preset_chain.append(preset_name)

    if preset_dict:
        flat_preset = flatten_dict(preset_dict)
        changed, _ = compare_dicts(pydantic_defaults, flat_preset)

        for path_key, value in changed.items():
            provenance[path_key] = ParameterProvenance(
                path=path_key,
                value=value,
                source=ParameterSource.PRESET,
                source_detail=preset_name,
            )

        config_dict = deep_merge(config_dict, preset_dict)

    # =================================================================
    # Layer 3: Config file values
    # =================================================================
    config_file_path: str | None = None
    if path:
        path = Path(path)
        config_file_path = str(path)
        file_dict = load_config_dict(path)
        resolved_file_dict = resolve_inheritance(file_dict, path)

        flat_file = flatten_dict(resolved_file_dict)
        # Get current state for comparison
        current_flat = flatten_dict(config_dict) if config_dict else pydantic_defaults
        changed, _ = compare_dicts(current_flat, flat_file)

        for path_key, value in changed.items():
            provenance[path_key] = ParameterProvenance(
                path=path_key,
                value=value,
                source=ParameterSource.CONFIG_FILE,
                source_detail=str(path),
            )

        config_dict = deep_merge(config_dict, resolved_file_dict)

    # =================================================================
    # Layer 4: CLI overrides
    # =================================================================
    if cli_overrides:
        flat_cli = flatten_dict(cli_overrides)

        for path_key, value in flat_cli.items():
            if value is not None:
                provenance[path_key] = ParameterProvenance(
                    path=path_key,
                    value=value,
                    source=ParameterSource.CLI,
                    source_detail=path_key.replace(".", "_"),  # CLI flag approximation
                )

        # Apply CLI overrides (using deep merge with flattened then unflattened)
        cli_nested = unflatten_dict({k: v for k, v in flat_cli.items() if v is not None})
        config_dict = deep_merge(config_dict, cli_nested)

    # =================================================================
    # Build and validate final config
    # =================================================================
    try:
        final_config = ExperimentConfig(**config_dict)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e

    return ResolvedConfig(
        config=final_config,
        provenance=provenance,
        preset_chain=preset_chain,
        config_file_path=config_file_path,
    )
