"""Campaign manifest persistence for tracking experiment status and enabling resume.

The manifest is the state backbone of campaign orchestration. It tracks every
experiment's lifecycle (pending -> running -> completed/failed/skipped) and
persists atomically so campaigns can be safely resumed after interruption.

Follows the StateManager atomic write pattern: write to temp file, then rename.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

__all__ = [
    "CampaignManifest",
    "CampaignManifestEntry",
    "ManifestManager",
]


class CampaignManifestEntry(BaseModel):
    """Single experiment entry within a campaign manifest.

    Tracks the full lifecycle of one experiment: its config, backend,
    execution status, timestamps, and result location.
    """

    exp_id: str = Field(..., description="Unique experiment identifier")
    config_name: str = Field(..., description="Stem name of the config file")
    config_path: str = Field(..., description="Path to experiment config (relative or absolute)")
    config_hash: str = Field(..., description="MD5 hash of config content for change detection")
    backend: str = Field(..., description="Backend name (pytorch, vllm, tensorrt)")
    container: str = Field(..., description="Docker service name")
    cycle_index: int = Field(..., description="Which cycle this experiment belongs to")
    status: Literal["pending", "running", "completed", "failed", "skipped"] = Field(
        default="pending", description="Current experiment status"
    )
    result_path: str | None = Field(default=None, description="Path to results directory")
    started_at: datetime | None = Field(default=None, description="When experiment started")
    completed_at: datetime | None = Field(default=None, description="When experiment completed")
    error: str | None = Field(default=None, description="Error message if failed")
    retry_count: int = Field(
        default=0, description="Number of times this experiment has been retried"
    )


class CampaignManifest(BaseModel):
    """Full campaign manifest tracking all experiments.

    The manifest is the persistent state of a campaign run. It records
    every experiment's status and enables safe resume after interruption.
    """

    campaign_id: str = Field(..., description="Unique campaign identifier")
    campaign_name: str = Field(..., description="Human-readable campaign name")
    created_at: datetime = Field(..., description="When the manifest was created")
    updated_at: datetime = Field(..., description="Last modification timestamp")
    config_hash: str = Field(
        ..., description="Hash of the campaign config file for change detection on resume"
    )
    total_experiments: int = Field(..., description="Total number of experiments in the campaign")
    experiments: list[CampaignManifestEntry] = Field(
        default_factory=list, description="All experiment entries"
    )

    @property
    def completed_count(self) -> int:
        """Count of experiments with status 'completed'."""
        return sum(1 for e in self.experiments if e.status == "completed")

    @property
    def failed_count(self) -> int:
        """Count of experiments with status 'failed'."""
        return sum(1 for e in self.experiments if e.status == "failed")

    @property
    def pending_count(self) -> int:
        """Count of experiments with status 'pending'."""
        return sum(1 for e in self.experiments if e.status == "pending")

    @property
    def is_complete(self) -> bool:
        """True if all experiments are completed or skipped."""
        return all(e.status in ("completed", "skipped") for e in self.experiments)

    @property
    def progress_fraction(self) -> float:
        """Fraction of experiments completed (0.0 to 1.0)."""
        if self.total_experiments == 0:
            return 0.0
        return self.completed_count / self.total_experiments

    def get_remaining(self) -> list[CampaignManifestEntry]:
        """Return experiments that still need to run (pending + failed)."""
        return [e for e in self.experiments if e.status in ("pending", "failed")]

    def get_by_status(self, status: str) -> list[CampaignManifestEntry]:
        """Filter experiments by status."""
        return [e for e in self.experiments if e.status == status]

    def update_entry(self, exp_id: str, **kwargs: object) -> None:
        """Update fields on a specific experiment entry.

        Args:
            exp_id: Experiment identifier to update.
            **kwargs: Fields to update on the entry.

        Raises:
            KeyError: If exp_id is not found in the manifest.
        """
        for entry in self.experiments:
            if entry.exp_id == exp_id:
                for key, value in kwargs.items():
                    setattr(entry, key, value)
                return
        msg = f"Experiment entry not found: {exp_id}"
        raise KeyError(msg)


class ManifestManager:
    """Persists and loads campaign manifests with atomic writes.

    Follows StateManager pattern: write to temp file, then atomic rename.
    This ensures no partial writes on crash — the manifest is either
    fully written or not updated at all.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = Path(manifest_path)

    def save(self, manifest: CampaignManifest) -> None:
        """Save manifest atomically (temp file then rename).

        Creates parent directories if needed. Updates the manifest's
        updated_at timestamp before writing.
        """
        manifest.updated_at = datetime.now()
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self._manifest_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(manifest.model_dump_json(indent=2))
            tmp_path.rename(self._manifest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def load(self) -> CampaignManifest | None:
        """Load manifest from disk.

        Returns None if the file doesn't exist or is corrupted.
        """
        if not self._manifest_path.exists():
            return None

        try:
            content = self._manifest_path.read_text()
            return CampaignManifest.model_validate_json(content)
        except json.JSONDecodeError:
            logger.warning("Corrupt manifest file at {}, returning None", self._manifest_path)
            return None

    def exists(self) -> bool:
        """Check if the manifest file exists on disk."""
        return self._manifest_path.exists()

    def create_manifest(
        self,
        campaign_id: str,
        campaign_name: str,
        config_hash: str,
        experiments: list[CampaignManifestEntry],
    ) -> CampaignManifest:
        """Factory method to create a new manifest with timestamps."""
        now = datetime.now()
        return CampaignManifest(
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            created_at=now,
            updated_at=now,
            config_hash=config_hash,
            total_experiments=len(experiments),
            experiments=experiments,
        )

    def check_config_changed(self, manifest: CampaignManifest, current_config_hash: str) -> bool:
        """Return True if the campaign config has changed since manifest creation."""
        return manifest.config_hash != current_config_hash
