"""Results repository for persistent storage of experiment results."""

from pathlib import Path

from llenergymeasure.constants import (
    AGGREGATED_RESULTS_SUBDIR,
    DEFAULT_RESULTS_DIR,
    RAW_RESULTS_SUBDIR,
)
from llenergymeasure.domain.experiment import AggregatedResult, RawProcessResult
from llenergymeasure.exceptions import ConfigurationError
from llenergymeasure.security import is_safe_path, sanitize_experiment_id


class FileSystemRepository:
    """File system based results repository.

    Implements the late aggregation pattern where raw per-process results
    are saved separately from aggregated results.

    Directory structure:
        results/
        ├── raw/
        │   └── exp_123/
        │       ├── process_0.json
        │       ├── process_1.json
        │       └── ...
        └── aggregated/
            └── exp_123.json
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize repository.

        Args:
            base_path: Base directory for results. Defaults to 'results/'.
        """
        self._base = base_path or DEFAULT_RESULTS_DIR
        self._raw_dir = self._base / RAW_RESULTS_SUBDIR
        self._aggregated_dir = self._base / AGGREGATED_RESULTS_SUBDIR

    def _ensure_dirs(self) -> None:
        """Ensure result directories exist."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._aggregated_dir.mkdir(parents=True, exist_ok=True)

    def _experiment_raw_dir(self, experiment_id: str) -> Path:
        """Get raw results directory for an experiment."""
        safe_id = sanitize_experiment_id(experiment_id)
        return self._raw_dir / safe_id

    def save_raw(self, experiment_id: str, result: RawProcessResult) -> Path:
        """Save raw per-process result.

        Args:
            experiment_id: Unique experiment identifier.
            result: Raw result from a single process.

        Returns:
            Path to the saved result file.
        """
        self._ensure_dirs()
        exp_dir = self._experiment_raw_dir(experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        filename = f"process_{result.process_index}.json"
        path = exp_dir / filename

        if not is_safe_path(self._base, path):
            raise ConfigurationError(f"Invalid result path: {path}")

        path.write_text(result.model_dump_json(indent=2))
        return path

    def list_raw(self, experiment_id: str) -> list[Path]:
        """List all raw result files for an experiment.

        Args:
            experiment_id: Unique experiment identifier.

        Returns:
            List of paths to raw result files, sorted by process index.
        """
        exp_dir = self._experiment_raw_dir(experiment_id)
        if not exp_dir.exists():
            return []

        paths = list(exp_dir.glob("process_*.json"))
        # Sort by process index
        return sorted(paths, key=lambda p: int(p.stem.split("_")[1]))

    def load_raw(self, path: Path) -> RawProcessResult:
        """Load a raw result from file.

        Args:
            path: Path to the result file.

        Returns:
            Loaded RawProcessResult.

        Raises:
            ConfigurationError: If file cannot be loaded.
        """
        if not path.exists():
            raise ConfigurationError(f"Result file not found: {path}")

        try:
            content = path.read_text()
            return RawProcessResult.model_validate_json(content)
        except Exception as e:
            raise ConfigurationError(f"Failed to load result: {e}") from e

    def load_all_raw(self, experiment_id: str) -> list[RawProcessResult]:
        """Load all raw results for an experiment.

        Args:
            experiment_id: Unique experiment identifier.

        Returns:
            List of RawProcessResult, sorted by process index.
        """
        paths = self.list_raw(experiment_id)
        return [self.load_raw(p) for p in paths]

    def save_aggregated(self, result: AggregatedResult) -> Path:
        """Save aggregated experiment result.

        Args:
            result: Aggregated result to save.

        Returns:
            Path to the saved result file.
        """
        self._ensure_dirs()
        safe_id = sanitize_experiment_id(result.experiment_id)
        path = self._aggregated_dir / f"{safe_id}.json"

        if not is_safe_path(self._base, path):
            raise ConfigurationError(f"Invalid result path: {path}")

        path.write_text(result.model_dump_json(indent=2))
        return path

    def load_aggregated(self, experiment_id: str) -> AggregatedResult | None:
        """Load aggregated result for an experiment.

        Args:
            experiment_id: Unique experiment identifier.

        Returns:
            AggregatedResult if found, None otherwise.
        """
        safe_id = sanitize_experiment_id(experiment_id)
        path = self._aggregated_dir / f"{safe_id}.json"

        if not path.exists():
            return None

        try:
            content = path.read_text()
            return AggregatedResult.model_validate_json(content)
        except Exception as e:
            raise ConfigurationError(f"Failed to load aggregated result: {e}") from e

    def list_experiments(self) -> list[str]:
        """List all experiment IDs with raw results.

        Returns:
            List of experiment IDs.
        """
        if not self._raw_dir.exists():
            return []
        return [d.name for d in self._raw_dir.iterdir() if d.is_dir()]

    def list_aggregated(self) -> list[str]:
        """List all experiment IDs with aggregated results.

        Returns:
            List of experiment IDs.
        """
        if not self._aggregated_dir.exists():
            return []
        return [p.stem for p in self._aggregated_dir.glob("*.json")]

    def has_raw(self, experiment_id: str) -> bool:
        """Check if raw results exist for an experiment."""
        return len(self.list_raw(experiment_id)) > 0

    def has_aggregated(self, experiment_id: str) -> bool:
        """Check if aggregated result exists for an experiment."""
        safe_id = sanitize_experiment_id(experiment_id)
        return (self._aggregated_dir / f"{safe_id}.json").exists()

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete all results for an experiment.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            True if anything was deleted.
        """
        deleted = False

        # Delete raw results
        exp_dir = self._experiment_raw_dir(experiment_id)
        if exp_dir.exists():
            for f in exp_dir.glob("*.json"):
                f.unlink()
            exp_dir.rmdir()
            deleted = True

        # Delete aggregated result
        safe_id = sanitize_experiment_id(experiment_id)
        agg_path = self._aggregated_dir / f"{safe_id}.json"
        if agg_path.exists():
            agg_path.unlink()
            deleted = True

        return deleted
