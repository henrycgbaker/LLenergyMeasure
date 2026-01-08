"""Experiment state management for LLM Bench."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_energy_measure.constants import DEFAULT_STATE_DIR
from llm_energy_measure.exceptions import ConfigurationError
from llm_energy_measure.security import is_safe_path, sanitize_experiment_id


class ExperimentStatus(str, Enum):
    """Status of an experiment lifecycle."""

    INITIALISED = "initialised"
    RUNNING = "running"
    COMPLETED = "completed"
    AGGREGATED = "aggregated"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ProcessStatus(str, Enum):
    """Status of a single process within an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessProgress(BaseModel):
    """Per-process progress tracking."""

    process_index: int = Field(..., description="Process index (0-based)")
    status: ProcessStatus = Field(default=ProcessStatus.PENDING, description="Process status")
    gpu_id: int | None = Field(default=None, description="GPU device index")
    started_at: datetime | None = Field(default=None, description="When process started")
    completed_at: datetime | None = Field(default=None, description="When process completed")
    error_message: str | None = Field(default=None, description="Error if failed")


class ExperimentState(BaseModel):
    """Persistent experiment state with atomic operations.

    Tracks experiment progress across runs, enabling resume capability
    and preventing duplicate work.
    """

    experiment_id: str = Field(..., description="Unique experiment identifier")
    status: ExperimentStatus = Field(
        default=ExperimentStatus.INITIALISED,
        description="Current experiment status",
    )
    cycle_id: int = Field(default=0, description="Current experiment cycle")

    # Config tracking
    config_path: str | None = Field(default=None, description="Original config file path")
    config_hash: str | None = Field(default=None, description="Hash of config for matching")
    model_name: str | None = Field(default=None, description="Model name for display")
    prompt_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Prompt source args (--dataset, -n, etc.)",
    )

    # Process tracking
    num_processes: int = Field(default=1, description="Total number of processes")
    subprocess_pid: int | None = Field(default=None, description="PID for stale detection")
    process_progress: dict[int, ProcessProgress] = Field(
        default_factory=dict,
        description="Per-process completion status",
    )

    # Timestamps
    started_at: datetime | None = Field(default=None, description="When experiment started")
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last state update timestamp",
    )

    # Error info
    error_message: str | None = Field(default=None, description="Error if failed")

    # Legacy fields for backward compatibility
    completed_runs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of config_name -> result file path",
    )
    failed_runs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of config_name -> error message",
    )
    total_runs: int = Field(default=0, description="Total runs planned")

    model_config = {"extra": "forbid"}

    @property
    def completed_count(self) -> int:
        """Number of completed runs."""
        return len(self.completed_runs)

    @property
    def failed_count(self) -> int:
        """Number of failed runs."""
        return len(self.failed_runs)

    @property
    def pending_count(self) -> int:
        """Number of pending runs."""
        done = self.completed_count + self.failed_count
        return max(0, self.total_runs - done)

    @property
    def is_complete(self) -> bool:
        """Check if all runs are done (completed or failed)."""
        return self.pending_count == 0 and self.total_runs > 0

    @property
    def processes_completed(self) -> int:
        """Count of completed processes."""
        return sum(1 for p in self.process_progress.values() if p.status == ProcessStatus.COMPLETED)

    @property
    def processes_failed(self) -> int:
        """Count of failed processes."""
        return sum(1 for p in self.process_progress.values() if p.status == ProcessStatus.FAILED)

    def can_aggregate(self) -> bool:
        """Check if all processes completed successfully."""
        if not self.process_progress:
            # No process tracking - assume single process completed if status is COMPLETED
            return self.status == ExperimentStatus.COMPLETED
        return len(self.process_progress) == self.num_processes and all(
            p.status == ProcessStatus.COMPLETED for p in self.process_progress.values()
        )

    def is_subprocess_running(self) -> bool:
        """Check if subprocess is still running (for stale state detection)."""
        if self.subprocess_pid is None:
            return False
        try:
            os.kill(self.subprocess_pid, 0)  # Signal 0 = check existence
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def mark_completed(self, config_name: str, result_path: str) -> None:
        """Mark a run as completed.

        Args:
            config_name: Configuration name that completed.
            result_path: Path to the result file.
        """
        self.completed_runs[config_name] = result_path
        self.failed_runs.pop(config_name, None)  # Remove from failed if retried
        self.last_updated = datetime.now()

    def mark_failed(self, config_name: str, error: str) -> None:
        """Mark a run as failed.

        Args:
            config_name: Configuration name that failed.
            error: Error message.
        """
        self.failed_runs[config_name] = error
        self.last_updated = datetime.now()

    def is_pending(self, config_name: str) -> bool:
        """Check if a config is pending (not completed or failed).

        Args:
            config_name: Configuration name to check.

        Returns:
            True if the config hasn't been run yet.
        """
        return config_name not in self.completed_runs and config_name not in self.failed_runs


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute hash of config for matching incomplete experiments.

    Excludes volatile fields like timestamps and experiment_id.
    """
    # Create a copy and remove volatile fields
    stable = {k: v for k, v in config_dict.items() if k not in ("experiment_id", "_metadata")}
    # Sort keys for deterministic hashing
    import json

    content = json.dumps(stable, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class StateManager:
    """Manages persistent experiment state with atomic file operations."""

    def __init__(self, state_dir: Path | None = None):
        """Initialize state manager.

        Args:
            state_dir: Directory to store state files. Defaults to DEFAULT_STATE_DIR.
        """
        self._state_dir = Path(state_dir) if state_dir else Path(DEFAULT_STATE_DIR)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, experiment_id: str) -> Path:
        """Get path to state file for an experiment."""
        safe_id = sanitize_experiment_id(experiment_id)
        return self._state_dir / f"{safe_id}.json"

    def load(self, experiment_id: str) -> ExperimentState | None:
        """Load experiment state if it exists.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            ExperimentState if found, None otherwise.
        """
        path = self._state_path(experiment_id)
        if not path.exists():
            return None

        if not is_safe_path(self._state_dir, path):
            raise ConfigurationError(f"Invalid state path: {path}")

        try:
            content = path.read_text()
            return ExperimentState.model_validate_json(content)
        except Exception as e:
            raise ConfigurationError(f"Failed to load state: {e}") from e

    def save(self, state: ExperimentState) -> Path:
        """Save experiment state atomically.

        Uses write-to-temp-then-rename pattern for atomicity.

        Args:
            state: State to save.

        Returns:
            Path to the saved state file.
        """
        path = self._state_path(state.experiment_id)
        temp_path = path.with_suffix(".tmp")

        try:
            state.last_updated = datetime.now()
            temp_path.write_text(state.model_dump_json(indent=2))
            temp_path.rename(path)
            return path
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise ConfigurationError(f"Failed to save state: {e}") from e

    def create(
        self,
        experiment_id: str,
        total_runs: int = 0,
        cycle_id: int = 0,
    ) -> ExperimentState:
        """Create a new experiment state.

        Args:
            experiment_id: Unique experiment identifier.
            total_runs: Total number of runs planned.
            cycle_id: Experiment cycle number.

        Returns:
            New ExperimentState instance.
        """
        state = ExperimentState(
            experiment_id=experiment_id,
            cycle_id=cycle_id,
            total_runs=total_runs,
        )
        self.save(state)
        return state

    def delete(self, experiment_id: str) -> bool:
        """Delete experiment state.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            True if state was deleted, False if it didn't exist.
        """
        path = self._state_path(experiment_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_experiments(self) -> list[str]:
        """List all experiment IDs with saved state.

        Returns:
            List of experiment IDs.
        """
        return [p.stem for p in self._state_dir.glob("*.json")]

    def find_incomplete(self) -> list[ExperimentState]:
        """Find all incomplete experiments (not AGGREGATED status).

        Returns:
            List of incomplete ExperimentState objects.
        """
        incomplete = []
        for exp_id in self.list_experiments():
            try:
                state = self.load(exp_id)
                if state and state.status != ExperimentStatus.AGGREGATED:
                    incomplete.append(state)
            except ConfigurationError:
                # Skip corrupted state files
                continue
        return incomplete

    def find_by_config_hash(self, config_hash: str) -> ExperimentState | None:
        """Find incomplete experiment matching config hash.

        Args:
            config_hash: Hash of config to match.

        Returns:
            Matching ExperimentState if found, None otherwise.
        """
        for state in self.find_incomplete():
            if state.config_hash == config_hash:
                return state
        return None

    def cleanup_stale(self) -> list[str]:
        """Clean up stale RUNNING states where subprocess is no longer running.

        Returns:
            List of experiment IDs that were marked as INTERRUPTED.
        """
        cleaned = []
        for state in self.find_incomplete():
            if state.status == ExperimentStatus.RUNNING and not state.is_subprocess_running():
                state.status = ExperimentStatus.INTERRUPTED
                state.error_message = "Subprocess no longer running (stale state detected)"
                self.save(state)
                cleaned.append(state.experiment_id)
        return cleaned
