"""Experiment state management for LLM Bench."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from llm_energy_measure.exceptions import ConfigurationError
from llm_energy_measure.security import is_safe_path, sanitize_experiment_id


class ExperimentState(BaseModel):
    """Persistent experiment state with atomic operations.

    Tracks experiment progress across runs, enabling resume capability
    and preventing duplicate work.
    """

    experiment_id: str = Field(..., description="Unique experiment identifier")
    cycle_id: int = Field(default=0, description="Current experiment cycle")
    completed_runs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of config_name -> result file path",
    )
    failed_runs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of config_name -> error message",
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last state update timestamp",
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


class StateManager:
    """Manages persistent experiment state with atomic file operations."""

    def __init__(self, state_dir: Path = Path(".llm_energy_measure_state")):
        """Initialize state manager.

        Args:
            state_dir: Directory to store state files.
        """
        self._state_dir = state_dir
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
