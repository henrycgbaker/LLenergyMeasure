"""User preferences configuration loading.

Loads optional user preferences from ~/.config/llenergymeasure/config.yaml
(XDG-compliant path via platformdirs). Missing file silently applies all
defaults. Invalid YAML or schema raises ConfigError.

This module only LOADS the file. Where its values rank against the study file,
env vars and call-site overrides is the precedence chain's business
(:mod:`llenergymeasure.config.precedence`): user config sits above the built-in
defaults and below everything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from llenergymeasure.config.models import EnergySamplerName, ServerWarmupConfig
from llenergymeasure.config.ssot import (
    ALL_ENGINES,
    DEFAULT_CYCLE_GAP_SECONDS,
    DEFAULT_EXPERIMENT_GAP_SECONDS,
    DEFAULT_RESULTS_DIR,
    legacy_runner_migration_message,
    legacy_runner_replacement,
)


class UserOutputConfig(BaseModel):
    """Output path preferences."""

    model_config = {"extra": "forbid"}

    results_dir: str = Field(
        default=DEFAULT_RESULTS_DIR, description="Default results output location"
    )


class UserRunnersConfig(BaseModel):
    """Runner selection per engine."""

    model_config = {"extra": "forbid"}

    transformers: str = Field(
        default="auto",
        description="Transformers runner: 'auto', 'process', 'container' (built-in image), or "
        "'container:<image>'",
    )
    vllm: str = Field(
        default="auto",
        description="vLLM runner: 'auto', 'process', 'container' (built-in image), or "
        "'container:<image>'",
    )
    tensorrt: str = Field(
        default="auto",
        description="TensorRT runner: 'auto', 'process', 'container' (built-in image), or "
        "'container:<image>'",
    )

    @model_validator(mode="after")
    def validate_runner_format(self) -> UserRunnersConfig:
        # The runner vocabulary was renamed in v0.7 (local->process, docker->container,
        # docker:<image>->container:<image>). The old values are a clean break: reject
        # them here with the shared migration hint rather than silently accepting.
        for field_name in ALL_ENGINES:
            value = getattr(self, field_name)
            replacement = legacy_runner_replacement(value)
            if replacement is not None:
                raise ValueError(
                    legacy_runner_migration_message(
                        value, replacement, context=f"runners.{field_name}"
                    )
                )
            if value.startswith("singularity:"):
                raise ValueError(
                    f"Singularity runner not yet supported (runners.{field_name}='{value}'). "
                    "Use 'auto', 'process', 'container', or 'container:<image>'."
                )
            if value not in {"auto", "process", "container"} and not value.startswith("container:"):
                raise ValueError(
                    f"runners.{field_name}: expected 'auto', 'process', 'container', or "
                    f"'container:<image>', got '{value}'"
                )
        return self


class UserMeasurementConfig(BaseModel):
    """Energy measurement preferences."""

    model_config = {"extra": "forbid"}

    energy_sampler: EnergySamplerName = Field(
        default="auto", description="Energy sampler: auto=best available (Zeus>NVML>CodeCarbon)"
    )


class UserUIConfig(BaseModel):
    """User interface preferences."""

    model_config = {"extra": "forbid"}

    progress_mode: Literal["auto", "plain", "quiet"] = Field(
        default="auto",
        description="Progress output mode: auto=Rich Live TTY, plain=sequential print, quiet=silent",
    )


class UserExecutionConfig(BaseModel):
    """Execution preferences: machine-local thermal defaults and GPU scoping."""

    model_config = {"extra": "forbid"}

    experiment_gap_seconds: float = Field(
        default=DEFAULT_EXPERIMENT_GAP_SECONDS,
        ge=0.0,
        description="Thermal gap between experiments",
    )
    cycle_gap_seconds: float = Field(
        default=DEFAULT_CYCLE_GAP_SECONDS, ge=0.0, description="Thermal gap between cycles"
    )
    gpu_indices: list[int] | None = Field(
        default=None,
        description=(
            "HOST GPU indices (as `nvidia-smi` shows) llem is allowed to use on this "
            "machine. When set, llem only ever uses these physical devices - for "
            "compute placement AND for energy measurement - on both the container and "
            "the process runner path. null (the default) = every visible GPU, the "
            "historical behaviour. A study that omits `study_execution.gpu_indices` "
            "inherits this set; a study that declares its own indices must stay inside "
            "it or resolution fails. Placement metadata only: never part of any config "
            "or study-design hash, so restricting llem to a subset of the host's GPUs "
            "never changes dedup grouping or study identity. `LLEM_DOCKER_GPUS` is the "
            "per-invocation escape hatch and still wins at `docker run` time."
        ),
    )

    @model_validator(mode="after")
    def _validate_gpu_indices(self) -> UserExecutionConfig:
        """Reject empty, negative, or duplicate GPU indices (fail loudly).

        Absence is expressed as ``None`` (every visible GPU); an empty list is a
        mistake, not "all". Negative indices and duplicates cannot name real
        distinct host devices. Deliberately no hardware check here: the config
        must load on a machine with a remote Docker daemon or no NVIDIA driver at
        all. Comparing the allowlist against the host's actual device count is a
        fail-soft preflight warning instead.
        """
        if self.gpu_indices is None:
            return self
        if not self.gpu_indices:
            raise ValueError(
                "execution.gpu_indices must not be empty; omit it (null) to allow all GPUs."
            )
        if any(i < 0 for i in self.gpu_indices):
            raise ValueError(
                f"execution.gpu_indices must be non-negative host device indices, "
                f"got {self.gpu_indices}."
            )
        if len(set(self.gpu_indices)) != len(self.gpu_indices):
            raise ValueError(
                f"execution.gpu_indices must not contain duplicates, got {self.gpu_indices}."
            )
        return self


class UserServerConfig(BaseModel):
    """Server-mode preferences: a tool-wide warmup protocol default overlay.

    Mirrors the per-mode warmup grammar - a ``server:`` namespace whose ``warmup`` block
    supplies machine-local defaults for the server warmup protocol. The block is
    ``ServerWarmupConfig``-shaped and overlaid PER FIELD: only the warmup fields the
    user actually writes take effect, and a study YAML that sets a field always wins
    (study YAML > user config > built-in default). The overlay lands in the RESOLVED
    config hash, never the declared one, so sharing a study file reproduces the
    declared identity while each machine's warmup default still shapes the realised
    protocol (and dedup treats runs under different defaults as distinct).
    """

    model_config = {"extra": "forbid"}

    warmup: ServerWarmupConfig = Field(
        default_factory=ServerWarmupConfig,
        description=(
            "Tool-wide server warmup protocol defaults, overlaid beneath study YAML "
            "(server mode only). Only the fields you set are overlaid."
        ),
    )


class UserConfig(BaseModel):
    """User preferences loaded from ~/.config/llenergymeasure/config.yaml.

    All fields are optional - missing file or missing fields fall back to
    built-in defaults. Invalid values raise ConfigError via load_user_config().
    """

    model_config = {"extra": "forbid"}

    output: UserOutputConfig = Field(default_factory=UserOutputConfig)
    runners: UserRunnersConfig = Field(default_factory=UserRunnersConfig)
    server: UserServerConfig | None = Field(
        default=None,
        description=(
            "Server-mode preferences: a tool-wide warmup protocol default overlaid "
            "beneath study YAML (server mode only). None = no tool-wide warmup default."
        ),
    )
    images: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-engine Docker image overrides (orthogonal to runners). "
            "Keys are engine names, values are image references. "
            "Empty dict = use smart default (local build → registry fallback)."
        ),
    )
    measurement: UserMeasurementConfig = Field(default_factory=UserMeasurementConfig)
    ui: UserUIConfig = Field(default_factory=UserUIConfig)
    execution: UserExecutionConfig = Field(default_factory=UserExecutionConfig)


def get_user_config_path() -> Path:
    """Return the XDG-compliant user config path.

    Linux:   ~/.config/llenergymeasure/config.yaml
    macOS:   ~/Library/Application Support/llenergymeasure/config.yaml
    Windows: %APPDATA%\\llenergymeasure\\config.yaml
    """
    from platformdirs import user_config_dir

    return Path(user_config_dir("llenergymeasure")) / "config.yaml"


def load_user_config(config_path: Path | None = None) -> UserConfig:
    """Load user configuration from ~/.config/llenergymeasure/config.yaml.

    Missing file: silently applies all defaults - no error.
    Invalid YAML: raises ConfigError with parse error detail.
    Invalid schema: raises ConfigError with field path context.

    Args:
        config_path: Explicit path override (for testing). None = XDG default.

    Returns:
        UserConfig with file values merged over defaults.
    """
    from llenergymeasure.utils.exceptions import ConfigError

    path = config_path or get_user_config_path()

    if not path.exists():
        # Missing file - zero-config, all defaults
        return UserConfig()

    try:
        content = path.read_text()
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"User config must be a YAML mapping: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in user config {path}: {e}") from e

    try:
        return UserConfig.model_validate(data)
    except ValidationError as e:
        # Format Pydantic errors as ConfigError with field paths for researcher clarity
        errors = [f"  {_format_user_config_error(err)}" for err in e.errors()]
        raise ConfigError(f"Invalid user config {path}:\n" + "\n".join(errors)) from e


#: Fields deleted from the user-config schema, with what replaced each. A config
#: file still carrying one gets the removal reason, not a bare "extra input" error.
_REMOVED_FIELDS: dict[str, str] = {
    "output.model_cache_dir": (
        "model caching follows the standard HuggingFace cache environment variables"
        " (HF_HOME etc.); for container runs, LLEM_DOCKER_HF_CACHE sets the mounted cache"
    ),
    "measurement.carbon_intensity_gco2_kwh": (
        "CO2 estimation belongs to the CodeCarbon sampler, which sources grid intensity itself"
    ),
    "measurement.datacenter_pue": (
        "CO2 estimation belongs to the CodeCarbon sampler, which sources PUE itself"
    ),
    "ui.log_level": "logging is controlled by the -v flag or the LLEM_LOG_LEVEL env var",
    "ui.prompt": "the tool has no interactive prompts to disable",
}


def _format_user_config_error(err: ErrorDetails) -> str:
    """One pydantic error as a researcher-facing line, naming removed fields."""
    dotted = ".".join(str(part) for part in err["loc"])
    if err["type"] == "extra_forbidden" and dotted in _REMOVED_FIELDS:
        return f"{dotted}: this field was removed - {_REMOVED_FIELDS[dotted]}"
    return f"{dotted}: {err['msg']}"


__all__ = ["UserConfig", "get_user_config_path", "load_user_config"]
