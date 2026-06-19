"""CodeCarbon energy tracking backend."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from llenergymeasure.energy.nvml import EnergyMeasurement

logger = logging.getLogger(__name__)

# Suppress FutureWarning from codecarbon's pandas usage
# (DataFrame concatenation with empty columns deprecation)
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries",
    category=FutureWarning,
    module="codecarbon",
)


@dataclass
class CodeCarbonData:
    """Structured CodeCarbon emissions data."""

    cpu_power: float | None
    gpu_power: float | None
    ram_power: float | None
    cpu_energy: float | None
    gpu_energy: float | None
    ram_energy: float | None
    total_energy_kwh: float
    emissions_kg: float | None
    duration_sec: float


class CodeCarbonSampler:
    """Energy tracking sampler using CodeCarbon.

    CodeCarbon tracks energy consumption at the process level,
    measuring CPU, GPU, and RAM power usage.

    Implements the EnergySampler protocol.
    """

    def __init__(
        self,
        measure_power_secs: int = 1,
        tracking_mode: str = "process",
    ) -> None:
        """Initialize CodeCarbon backend.

        Args:
            measure_power_secs: Interval between power measurements.
            tracking_mode: Either 'process' or 'machine'.
        """
        self._measure_power_secs = measure_power_secs
        self._tracking_mode = tracking_mode

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return "codecarbon"

    def is_available(self) -> bool:
        """Check if CodeCarbon is available on this system.

        Returns:
            True if codecarbon package is installed.
        """
        try:
            import codecarbon  # noqa: F401

            return True
        except ImportError:
            return False

    def start_tracking(self) -> Any:
        """Start energy tracking.

        Returns:
            EmissionsTracker instance for stop_tracking, or None if unavailable.
        """
        from codecarbon import EmissionsTracker

        # Suppress codecarbon's verbose logging
        logging.getLogger("codecarbon").setLevel(logging.ERROR)

        try:
            tracker = EmissionsTracker(
                measure_power_secs=self._measure_power_secs,
                allow_multiple_runs=True,
                tracking_mode=self._tracking_mode,
                log_level=logging.ERROR,
                save_to_file=False,
            )
            tracker.start()
            logger.debug("CodeCarbon tracking started")
            return tracker
        except Exception as e:
            # Handle NVML permission errors in containers, etc.
            logger.warning("Energy tracking unavailable: %s", e)
            logger.warning("Continuing without energy metrics (common in containers)")
            return None

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        """Stop tracking and return energy measurement.

        Mirrors the NVML/Zeus backends: returns an :class:`EnergyMeasurement`
        with ``total_j`` and ``duration_sec``. CodeCarbon tracks at the process
        level and does not expose a per-GPU breakdown, so ``per_gpu_j`` is None.

        Args:
            tracker: EmissionsTracker from start_tracking.

        Returns:
            EnergyMeasurement with collected energy data.
        """
        if tracker is None:
            logger.debug("Energy tracker was unavailable, returning empty measurement")
            return self._empty_measurement()

        try:
            tracker.stop()
            data = self._extract_data(tracker)
            return self._convert_to_measurement(data)
        except Exception as e:
            logger.warning("Failed to stop energy tracking: %s", e)
            return self._empty_measurement()

    def _empty_measurement(self) -> EnergyMeasurement:
        """Return an empty measurement for error cases."""
        return EnergyMeasurement(total_j=0.0, duration_sec=0.0, samples=[], per_gpu_j=None)

    def _extract_data(self, tracker: Any) -> CodeCarbonData:
        """Extract data from the tracker."""
        try:
            emissions_data = tracker._prepare_emissions_data()
        except AttributeError:
            logger.warning("Could not extract emissions data")
            return CodeCarbonData(
                cpu_power=None,
                gpu_power=None,
                ram_power=None,
                cpu_energy=None,
                gpu_energy=None,
                ram_energy=None,
                total_energy_kwh=0.0,
                emissions_kg=None,
                duration_sec=0.0,
            )

        energy_kwh = getattr(emissions_data, "energy_consumed", 0.0) or 0.0

        return CodeCarbonData(
            cpu_power=getattr(emissions_data, "cpu_power", None),
            gpu_power=getattr(emissions_data, "gpu_power", None),
            ram_power=getattr(emissions_data, "ram_power", None),
            cpu_energy=getattr(emissions_data, "cpu_energy", None),
            gpu_energy=getattr(emissions_data, "gpu_energy", None),
            ram_energy=getattr(emissions_data, "ram_energy", None),
            total_energy_kwh=energy_kwh,
            emissions_kg=getattr(emissions_data, "emissions", None),
            duration_sec=getattr(emissions_data, "duration", 0.0) or 0.0,
        )

    def _convert_to_measurement(self, data: CodeCarbonData) -> EnergyMeasurement:
        """Convert CodeCarbon data to the shared EnergyMeasurement shape."""
        # Convert kWh to Joules (1 kWh = 3.6e6 J)
        total_j = data.total_energy_kwh * 3.6e6

        return EnergyMeasurement(
            total_j=total_j,
            duration_sec=data.duration_sec,
            samples=[],
            per_gpu_j=None,
        )
