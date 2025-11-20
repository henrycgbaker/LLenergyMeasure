"""Input validation utilities for configuration and parameters."""

from pathlib import Path
from typing import Any, List, Optional, Union

from llm_efficiency.utils.exceptions import ConfigurationError


def validate_positive_int(value: int, name: str, allow_zero: bool = False) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Allow zero as valid value

    Raises:
        ConfigurationError: If validation fails
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer, got {type(value).__name__}")

    if allow_zero and value < 0:
        raise ConfigurationError(f"{name} must be non-negative, got {value}")
    elif not allow_zero and value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def validate_positive_float(value: float, name: str, allow_zero: bool = False) -> None:
    """Validate that a value is a positive float."""
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number, got {type(value).__name__}")

    if allow_zero and value < 0:
        raise ConfigurationError(f"{name} must be non-negative, got {value}")
    elif not allow_zero and value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def validate_in_range(
    value: Union[int, float],
    name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> None:
    """Validate that a value is within a range."""
    if min_value is not None and value < min_value:
        raise ConfigurationError(f"{name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ConfigurationError(f"{name} must be <= {max_value}, got {value}")


def validate_choice(value: Any, name: str, choices: List[Any]) -> None:
    """Validate that a value is one of the allowed choices."""
    if value not in choices:
        raise ConfigurationError(
            f"{name} must be one of {choices}, got '{value}'"
        )


def validate_path_exists(path: Path, name: str, must_be_file: bool = False, must_be_dir: bool = False) -> None:
    """Validate that a path exists."""
    if not path.exists():
        raise ConfigurationError(f"{name} does not exist: {path}")

    if must_be_file and not path.is_file():
        raise ConfigurationError(f"{name} must be a file: {path}")

    if must_be_dir and not path.is_dir():
        raise ConfigurationError(f"{name} must be a directory: {path}")


def validate_model_name(model_name: str) -> None:
    """Validate model name format."""
    if not model_name or not isinstance(model_name, str):
        raise ConfigurationError("model_name must be a non-empty string")

    if len(model_name) > 500:
        raise ConfigurationError("model_name is too long (max 500 characters)")


def validate_precision(precision: str) -> None:
    """Validate precision type."""
    valid_precisions = ["float32", "float16", "bfloat16", "float8"]
    validate_choice(precision, "precision", valid_precisions)


def validate_batch_config(batch_size: int, num_batches: int) -> None:
    """Validate batch configuration."""
    validate_positive_int(batch_size, "batch_size")
    validate_positive_int(num_batches, "num_batches")

    if batch_size > 1024:
        raise ConfigurationError(
            f"batch_size={batch_size} is very large, consider using a smaller value"
        )


def validate_quantization_config(
    load_in_4bit: bool,
    load_in_8bit: bool,
    quant_type: Optional[str],
) -> None:
    """Validate quantization configuration."""
    if load_in_4bit and load_in_8bit:
        raise ConfigurationError("Cannot enable both 4-bit and 8-bit quantization")

    if load_in_4bit and quant_type not in ["nf4", "fp4", None]:
        raise ConfigurationError(f"Invalid quant_type for 4-bit: {quant_type}")
