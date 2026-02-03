"""User preferences configuration loading.

Loads optional user preferences from .lem-config.yaml in project root.
This is a minimal implementation for Phase 2.2 (thermal gaps, docker strategy).
Full user preferences system (lem init wizard) is Phase 2.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ThermalGapConfig(BaseModel):
    """Thermal gap configuration."""

    between_experiments: float = Field(
        default=60.0,
        ge=0,
        description="Seconds between experiments within a cycle",
    )
    between_cycles: float = Field(
        default=300.0,
        ge=0,
        description="Seconds between cycles",
    )


class DockerConfig(BaseModel):
    """Docker execution configuration."""

    strategy: Literal["ephemeral", "persistent"] = Field(
        default="ephemeral",
        description="Container strategy: ephemeral (run --rm) or persistent (up + exec)",
    )
    warmup_delay: float = Field(
        default=0.0,
        ge=0,
        description="Seconds to wait after container start before first experiment",
    )
    auto_teardown: bool = Field(
        default=True,
        description="Auto-teardown containers after campaign (persistent mode only)",
    )


class UserConfig(BaseModel):
    """User preferences loaded from .lem-config.yaml.

    All fields are optional with sensible defaults. Missing file
    or fields gracefully fall back to defaults.
    """

    thermal_gaps: ThermalGapConfig = Field(
        default_factory=ThermalGapConfig,
        description="Thermal gap settings",
    )
    docker: DockerConfig = Field(
        default_factory=DockerConfig,
        description="Docker execution settings",
    )
    default_backend: str = Field(
        default="pytorch",
        description="Default backend for single-backend campaigns",
    )
    results_dir: str = Field(
        default="results",
        description="Default results directory",
    )


def load_user_config(config_path: Path | None = None) -> UserConfig:
    """Load user configuration from .lem-config.yaml.

    Args:
        config_path: Optional explicit path. Defaults to .lem-config.yaml in cwd.

    Returns:
        UserConfig with values from file or defaults if file missing.
    """
    if config_path is None:
        config_path = Path(".lem-config.yaml")

    if not config_path.exists():
        # No config file - return defaults
        return UserConfig()

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return UserConfig.model_validate(data)
    except Exception:
        # Invalid config - return defaults (don't crash)
        return UserConfig()


__all__ = [
    "DockerConfig",
    "ThermalGapConfig",
    "UserConfig",
    "load_user_config",
]
