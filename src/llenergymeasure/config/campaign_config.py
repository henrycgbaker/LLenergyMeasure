"""Campaign configuration models for multi-config comparison experiments.

A campaign runs multiple experiment configs across multiple cycles for
statistical robustness and fair comparison. Key features:

- Multiple configs run in configurable order (interleaved, shuffled, grouped)
- Per-config warmup using actual dataset prompts
- Thermal cooldown gaps between configs and cycles
- Optional scheduling for automated overnight runs
- Grid-based experiment generation (shared + backend-specific params)
- Explicit experiment lists alongside grid definitions
- Cold start benchmarking and container health monitoring
- Daemon mode for long-running automated campaigns
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VALID_BACKENDS = {"pytorch", "vllm", "tensorrt"}


class CampaignExecutionConfig(BaseModel):
    """Execution parameters for campaign runs.

    Controls how experiments are ordered, warmed up, and spaced.
    """

    cycles: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of complete cycles through all configs",
    )
    structure: Literal["interleaved", "shuffled", "grouped"] = Field(
        default="interleaved",
        description=(
            "Execution order: "
            "interleaved=A->B->C,A->B->C (fixed order per cycle), "
            "shuffled=random order within each cycle, "
            "grouped=Axn,Bxn,Cxn (all cycles of one config before next)"
        ),
    )

    # Warmup: dual-criteria (whichever comes first)
    warmup_prompts: int = Field(
        default=5,
        ge=0,
        description="Minimum warmup prompts before measurement (per config)",
    )
    warmup_timeout_seconds: float = Field(
        default=30.0,
        ge=0,
        description="Maximum warmup time in seconds (per config)",
    )

    # Thermal management gaps
    config_gap_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Gap between configs within a cycle (GPU thermal recovery)",
    )
    cycle_gap_seconds: float = Field(
        default=300.0,
        ge=0,
        description="Gap between cycles (full thermal reset)",
    )


class CampaignScheduleConfig(BaseModel):
    """Optional scheduling for automated campaign execution.

    When schedule is provided, it overrides cycle_gap_seconds timing.
    """

    at: str | None = Field(
        default=None,
        description="Time of day to run (e.g., '03:00', '14:30')",
    )
    days: list[str] | None = Field(
        default=None,
        description="Days to run on (e.g., ['mon', 'wed', 'fri'] or ['weekdays'])",
    )

    @field_validator("at")
    @classmethod
    def validate_time_format(cls, v: str | None) -> str | None:
        """Validate time is in HH:MM format."""
        if v is None:
            return v

        if not re.match(r"^\d{2}:\d{2}$", v):
            msg = f"Time must be in HH:MM format, got: {v}"
            raise ValueError(msg)
        hours, minutes = map(int, v.split(":"))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            msg = f"Invalid time: {v}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Phase 2: Grid, explicit experiments, health check, cold start, daemon, IO
# ---------------------------------------------------------------------------


def _parse_duration(value: str, field_name: str) -> str:
    """Validate and return a duration string in Nh or Nm format.

    Args:
        value: Duration string like "6h", "30m", "48h".
        field_name: Field name for error messages.

    Returns:
        The validated string (unchanged).

    Raises:
        ValueError: If format is invalid.
    """
    if not re.match(r"^\d+[hm]$", value):
        msg = f"{field_name} must be in Nh or Nm format (e.g., '6h', '30m'), got: {value}"
        raise ValueError(msg)
    return value


def _duration_to_seconds(value: str) -> int:
    """Convert a duration string (Nh or Nm) to seconds."""
    num = int(value[:-1])
    unit = value[-1]
    if unit == "h":
        return num * 3600
    return num * 60  # "m"


class CampaignGridConfig(BaseModel):
    """Two-level grid definition for campaign experiment generation.

    Defines a grid sweep with shared parameters applied to all backends
    and optional per-backend overrides.
    """

    backends: list[str] = Field(
        ...,
        min_length=1,
        description="Backends to sweep (e.g., ['pytorch', 'vllm'])",
    )
    models: list[str] | None = Field(
        default=None,
        description="Model names to sweep across all backends",
    )
    shared: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Shared grid axes applied to all backends (e.g., {'fp_precision': ['float16', 'bfloat16']})",
    )
    backend_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-backend param overrides keyed by backend name",
    )

    @field_validator("backends")
    @classmethod
    def validate_backends(cls, v: list[str]) -> list[str]:
        """Validate that all backends are known."""
        invalid = set(v) - VALID_BACKENDS
        if invalid:
            msg = f"Unknown backends: {invalid}. Valid backends: {sorted(VALID_BACKENDS)}"
            raise ValueError(msg)
        return v


class CampaignExplicitExperiment(BaseModel):
    """Single explicit experiment entry for a campaign.

    Allows specifying a config file path with optional inline overrides
    applied on top.
    """

    config: str = Field(
        ...,
        description="Path to experiment config YAML file",
    )
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional inline overrides applied on top of config",
    )


class CampaignHealthCheckConfig(BaseModel):
    """Container health monitoring configuration.

    Controls GPU memory monitoring and automatic container restarts
    when health thresholds are exceeded.
    """

    enabled: bool = Field(
        default=True,
        description="Enable container health checks",
    )
    interval_experiments: int = Field(
        default=0,
        ge=0,
        description="Check every N experiments (0 = disabled, check per-cycle only)",
    )
    gpu_memory_threshold_pct: float = Field(
        default=90.0,
        ge=50.0,
        le=100.0,
        description="GPU memory percentage threshold for unhealthy status",
    )
    restart_on_failure: bool = Field(
        default=True,
        description="Automatically restart container on health check failure",
    )
    max_restarts: int = Field(
        default=3,
        ge=0,
        description="Maximum container restarts per campaign",
    )


class CampaignColdStartConfig(BaseModel):
    """Cold start benchmarking configuration.

    Controls whether models are unloaded between experiments to measure
    cold start performance, and whether full container restarts are used
    for guaranteed clean state.
    """

    force_cold_start: bool = Field(
        default=False,
        description="Unload model between experiments for cold start measurement",
    )
    restart_container: bool = Field(
        default=False,
        description="Full container restart for guaranteed clean state (slower but complete)",
    )


class CampaignDaemonConfig(BaseModel):
    """Daemon scheduling configuration for long-running campaigns.

    Extends the basic scheduling with interval-based repetition and
    maximum duration controls for automated overnight/multi-day runs.
    """

    enabled: bool = Field(
        default=False,
        description="Enable daemon mode for automated long-running campaigns",
    )
    at: str | None = Field(
        default=None,
        description="Start time in HH:MM format (e.g., '03:00')",
    )
    interval: str | None = Field(
        default=None,
        description="Interval between cycles (e.g., '6h', '30m')",
    )
    total_duration: str | None = Field(
        default=None,
        description="Maximum campaign duration (e.g., '48h', '120m')",
    )
    quiet: bool = Field(
        default=True,
        description="Suppress interactive output in daemon mode",
    )

    @field_validator("at")
    @classmethod
    def validate_time_format(cls, v: str | None) -> str | None:
        """Validate time is in HH:MM format (reuses same logic as CampaignScheduleConfig)."""
        if v is None:
            return v
        if not re.match(r"^\d{2}:\d{2}$", v):
            msg = f"Time must be in HH:MM format, got: {v}"
            raise ValueError(msg)
        hours, minutes = map(int, v.split(":"))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            msg = f"Invalid time: {v}"
            raise ValueError(msg)
        return v

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str | None) -> str | None:
        """Validate interval is in Nh or Nm format."""
        if v is None:
            return v
        return _parse_duration(v, "interval")

    @field_validator("total_duration")
    @classmethod
    def validate_total_duration(cls, v: str | None) -> str | None:
        """Validate total_duration is in Nh or Nm format."""
        if v is None:
            return v
        return _parse_duration(v, "total_duration")

    @property
    def interval_seconds(self) -> int | None:
        """Return interval as seconds, or None if not set."""
        if self.interval is None:
            return None
        return _duration_to_seconds(self.interval)

    @property
    def total_duration_seconds(self) -> int | None:
        """Return total_duration as seconds, or None if not set."""
        if self.total_duration is None:
            return None
        return _duration_to_seconds(self.total_duration)


class CampaignIOConfig(BaseModel):
    """IO path configuration for campaign directories.

    Controls where campaign results, configs, state, and manifest files
    are stored.
    """

    results_dir: str = Field(
        default="results",
        description="Results output directory",
    )
    configs_dir: str = Field(
        default="configs",
        description="Config files directory",
    )
    state_dir: str = Field(
        default=".state",
        description="State and manifest directory",
    )
    manifest_filename: str = Field(
        default="campaign_manifest.json",
        description="Manifest file name within state_dir",
    )

    @property
    def manifest_path(self) -> Path:
        """Return full path to the manifest file."""
        return Path(self.state_dir) / self.manifest_filename


class CampaignConfig(BaseModel):
    """Configuration for a multi-config comparison campaign.

    A campaign runs multiple experiment configurations across multiple cycles
    to enable fair comparison with statistical robustness. Experiments can be
    specified via config file paths, grid definitions, or explicit experiment
    lists (at least one source must be provided).

    Example YAML:
        campaign_name: "pytorch-vs-vllm-comparison"
        dataset: alpaca
        num_samples: 100
        configs:
          - configs/pytorch_base.yaml
          - configs/vllm_base.yaml
        execution:
          cycles: 5
          structure: shuffled
          warmup_prompts: 5
          warmup_timeout_seconds: 30
          config_gap_seconds: 60
          cycle_gap_seconds: 300
    """

    campaign_name: str = Field(
        ...,  # Required
        min_length=1,
        description="Descriptive name for this campaign (used in results)",
    )

    # Default model (shared across all grid/explicit experiments)
    model: str | None = Field(
        default=None,
        description="Default model name for all experiments (required for grid without grid.models)",
    )

    # Prompt source (shared across all configs)
    dataset: str | None = Field(
        default=None,
        description="HuggingFace dataset or alias (overrides config datasets)",
    )
    num_samples: int | None = Field(
        default=None,
        ge=1,
        description="Number of prompts to use (overrides config sample sizes)",
    )

    # Config paths (now optional — grid or experiments can replace)
    configs: list[str] = Field(
        default_factory=list,
        description="Paths to experiment config YAML files",
    )

    # Grid-based experiment generation
    grid: CampaignGridConfig | None = Field(
        default=None,
        description="Grid definition for automated experiment generation",
    )

    # Explicit experiment list
    experiments: list[CampaignExplicitExperiment] = Field(
        default_factory=list,
        description="Explicit experiment entries with optional overrides",
    )

    # Execution parameters
    execution: CampaignExecutionConfig = Field(
        default_factory=CampaignExecutionConfig,
        description="Execution parameters (cycles, structure, warmup, gaps)",
    )

    # Optional scheduling (legacy)
    schedule: CampaignScheduleConfig | None = Field(
        default=None,
        description="Optional scheduling for automated runs",
    )

    # Phase 2 additions
    health_check: CampaignHealthCheckConfig = Field(
        default_factory=CampaignHealthCheckConfig,
        description="Container health monitoring configuration",
    )
    cold_start: CampaignColdStartConfig = Field(
        default_factory=CampaignColdStartConfig,
        description="Cold start benchmarking configuration",
    )
    daemon: CampaignDaemonConfig | None = Field(
        default=None,
        description="Daemon scheduling for long-running automated campaigns",
    )
    io: CampaignIOConfig = Field(
        default_factory=CampaignIOConfig,
        description="IO path configuration for campaign directories",
    )

    @field_validator("configs")
    @classmethod
    def validate_config_paths(cls, v: list[str]) -> list[str]:
        """Validate that config paths look reasonable (not checking existence yet)."""
        for path in v:
            if not path.endswith((".yaml", ".yml")):
                msg = f"Config path should be YAML file: {path}"
                raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_experiment_sources(self) -> CampaignConfig:
        """Validate that at least one experiment source is provided and configs exist."""
        # At least one source must be provided
        has_configs = len(self.configs) > 0
        has_grid = self.grid is not None
        has_experiments = len(self.experiments) > 0

        if not (has_configs or has_grid or has_experiments):
            msg = "At least one of 'configs', 'grid', or 'experiments' must be provided"
            raise ValueError(msg)

        # Only validate config file existence when configs are provided
        if has_configs:
            missing = [p for p in self.configs if not Path(p).exists()]
            if missing:
                msg = f"Config files not found: {', '.join(missing)}"
                raise ValueError(msg)

        return self

    @property
    def campaign_id(self) -> str:
        """Generate deterministic campaign ID from campaign_name.

        Returns first 8 characters of MD5 hash for brevity while
        maintaining uniqueness for practical purposes.
        """
        return hashlib.md5(self.campaign_name.encode()).hexdigest()[:8]

    def get_config_paths(self) -> list[Path]:
        """Return config paths as Path objects."""
        return [Path(p) for p in self.configs]

    def get_config_names(self) -> list[str]:
        """Extract config names from paths (stem without extension)."""
        return [Path(p).stem for p in self.configs]


def generate_campaign_id(name: str) -> str:
    """Generate a deterministic campaign ID from name.

    Args:
        name: Campaign name string.

    Returns:
        8-character hex string derived from MD5 hash.
    """
    return hashlib.md5(name.encode()).hexdigest()[:8]


__all__ = [
    "CampaignColdStartConfig",
    "CampaignConfig",
    "CampaignDaemonConfig",
    "CampaignExecutionConfig",
    "CampaignExplicitExperiment",
    "CampaignGridConfig",
    "CampaignHealthCheckConfig",
    "CampaignIOConfig",
    "CampaignScheduleConfig",
    "generate_campaign_id",
]
