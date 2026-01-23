"""Campaign configuration models for multi-config comparison experiments.

A campaign runs multiple experiment configs across multiple cycles for
statistical robustness and fair comparison. Key features:

- Multiple configs run in configurable order (interleaved, shuffled, grouped)
- Per-config warmup using actual dataset prompts
- Thermal cooldown gaps between configs and cycles
- Optional scheduling for automated overnight runs
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
        import re

        if not re.match(r"^\d{2}:\d{2}$", v):
            msg = f"Time must be in HH:MM format, got: {v}"
            raise ValueError(msg)
        hours, minutes = map(int, v.split(":"))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            msg = f"Invalid time: {v}"
            raise ValueError(msg)
        return v


class CampaignConfig(BaseModel):
    """Configuration for a multi-config comparison campaign.

    A campaign runs multiple experiment configurations across multiple cycles
    to enable fair comparison with statistical robustness.

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

    # Config paths
    configs: list[str] = Field(
        ...,  # Required
        min_length=1,
        description="Paths to experiment config YAML files",
    )

    # Execution parameters
    execution: CampaignExecutionConfig = Field(
        default_factory=CampaignExecutionConfig,
        description="Execution parameters (cycles, structure, warmup, gaps)",
    )

    # Optional scheduling
    schedule: CampaignScheduleConfig | None = Field(
        default=None,
        description="Optional scheduling for automated runs",
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
    def validate_configs_exist(self) -> CampaignConfig:
        """Validate that all config files exist."""
        missing = []
        for config_path in self.configs:
            if not Path(config_path).exists():
                missing.append(config_path)
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
    "CampaignConfig",
    "CampaignExecutionConfig",
    "CampaignScheduleConfig",
    "generate_campaign_id",
]
