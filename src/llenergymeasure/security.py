"""Security utilities for LLM Bench framework."""

import os
from pathlib import Path

from llenergymeasure.exceptions import ConfigurationError


def validate_path(path: Path, must_exist: bool = False, allow_relative: bool = True) -> Path:
    """Validate and resolve a file path safely.

    Args:
        path: Path to validate.
        must_exist: If True, raise error if path doesn't exist.
        allow_relative: If False, require absolute paths only.

    Returns:
        Resolved absolute path.

    Raises:
        ConfigurationError: If path validation fails.
    """
    if not allow_relative and not path.is_absolute():
        raise ConfigurationError(f"Absolute path required, got: {path}")

    resolved = path.resolve()

    if must_exist and not resolved.exists():
        raise ConfigurationError(f"Path does not exist: {resolved}")

    return resolved


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check if target_path is within base_dir (prevent path traversal).

    Uses Path.is_relative_to() for robust checking - this correctly handles
    edge cases like /foo/bar vs /foo/bar_malicious that string prefix
    checks would fail on.

    Args:
        base_dir: The allowed base directory.
        target_path: The path to check.

    Returns:
        True if target_path is within base_dir.
    """
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        return target_resolved.is_relative_to(base_resolved)
    except (OSError, ValueError):
        return False


def sanitize_experiment_id(experiment_id: str) -> str:
    """Sanitize experiment ID for use in file paths.

    Args:
        experiment_id: Raw experiment identifier.

    Returns:
        Sanitized string safe for filesystem use.

    Raises:
        ConfigurationError: If experiment_id is invalid.
    """
    if not experiment_id:
        raise ConfigurationError("Experiment ID cannot be empty")

    # Allow alphanumeric, underscore, hyphen, dot
    sanitized = "".join(c if c.isalnum() or c in "_-." else "_" for c in experiment_id)

    if sanitized != experiment_id:
        # Log warning about sanitization (import avoided to prevent circular deps)
        pass

    return sanitized


def check_env_for_secrets(env_vars: list[str]) -> dict[str, bool]:
    """Check which environment variables are set (for validation, not exposure).

    Args:
        env_vars: List of environment variable names to check.

    Returns:
        Dict mapping var names to whether they are set.
    """
    return {var: var in os.environ for var in env_vars}
