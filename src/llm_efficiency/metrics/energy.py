"""
Energy consumption tracking using CodeCarbon.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

from codecarbon import EmissionsTracker

logger = logging.getLogger(__name__)

# Suppress CodeCarbon's verbose logging
logging.getLogger("codecarbon").setLevel(logging.ERROR)


class EnergyTracker:
    """
    Wrapper around CodeCarbon for energy consumption tracking.

    Tracks CPU, GPU, and RAM energy consumption during inference,
    plus carbon emissions based on regional grid data.
    """

    def __init__(
        self,
        experiment_id: str,
        output_dir: Path = Path("results/energy"),
        measure_power_secs: int = 1,
        save_to_file: bool = True,
    ):
        """
        Initialize energy tracker.

        Args:
            experiment_id: Unique experiment identifier
            output_dir: Directory for saving energy data
            measure_power_secs: Measurement interval in seconds
            save_to_file: Whether to save results to file
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_dir / f"emissions_{experiment_id}.csv"

        logger.info(f"Initializing energy tracker for experiment {experiment_id}")
        logger.debug(f"Output file: {self.output_file}")

        self.tracker = EmissionsTracker(
            project_name=f"llm-efficiency-{experiment_id}",
            measure_power_secs=measure_power_secs,
            save_to_file=save_to_file,
            output_dir=str(self.output_dir),
            output_file=f"emissions_{experiment_id}.csv",
            log_level="error",  # Suppress verbose output
        )

        self._started = False
        self._stopped = False

    def start(self) -> None:
        """Start energy tracking."""
        if self._started:
            logger.warning("Energy tracker already started")
            return

        logger.info("Starting energy tracking...")
        self.tracker.start()
        self._started = True
        logger.info("Energy tracking started")

    def stop(self) -> Optional[float]:
        """
        Stop energy tracking and return total emissions.

        Returns:
            Total emissions in kg CO2, or None if tracking failed
        """
        if not self._started:
            logger.warning("Energy tracker was never started")
            return None

        if self._stopped:
            logger.warning("Energy tracker already stopped")
            return None

        logger.info("Stopping energy tracking...")

        try:
            emissions = self.tracker.stop()
            self._stopped = True
            logger.info(f"Energy tracking stopped. Emissions: {emissions:.6f} kg CO2")
            return emissions
        except Exception as e:
            logger.error(f"Failed to stop energy tracker: {e}")
            return None

    def get_results(self) -> Dict[str, float]:
        """
        Get energy consumption results.

        Returns:
            Dictionary with energy metrics
        """
        if not self._stopped:
            logger.warning("Tracker not stopped, results may be incomplete")

        # Read emissions file
        try:
            import csv

            with open(self.output_file) as f:
                reader = csv.DictReader(f)
                # Get last row (most recent measurement)
                for row in reader:
                    last_row = row

            results = {
                "duration_seconds": float(last_row.get("duration", 0)),
                "emissions_kg_co2": float(last_row.get("emissions", 0)),
                "energy_consumed_kwh": float(last_row.get("energy_consumed", 0)),
                "cpu_energy_kwh": float(last_row.get("cpu_energy", 0)),
                "gpu_energy_kwh": float(last_row.get("gpu_energy", 0)),
                "ram_energy_kwh": float(last_row.get("ram_energy", 0)),
                "cpu_power_w": float(last_row.get("cpu_power", 0)),
                "gpu_power_w": float(last_row.get("gpu_power", 0)),
                "ram_power_w": float(last_row.get("ram_power", 0)),
            }

            logger.debug(f"Energy results: {results}")
            return results

        except Exception as e:
            logger.error(f"Failed to read energy results: {e}")
            return {}

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def track_energy(experiment_id: str, output_dir: Path = Path("results/energy")) -> EnergyTracker:
    """
    Convenience function to create energy tracker.

    Args:
        experiment_id: Unique experiment identifier
        output_dir: Directory for saving energy data

    Returns:
        EnergyTracker instance

    Example:
        >>> with track_energy("0001") as tracker:
        ...     # Run inference
        ...     pass
        >>> results = tracker.get_results()
    """
    return EnergyTracker(experiment_id, output_dir)
