"""Zeus energy measurement backend.

Wraps ZeusMonitor to measure per-GPU energy consumption during inference.
Zeus provides more accurate GPU energy readings than NVML power polling
by integrating with the GPU's hardware energy counters.

All zeus imports are deferred - this module is safe to import without zeus installed.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.energy.nvml import EnergyMeasurement


class ZeusSampler:
    """Energy sampler using the Zeus GPU energy monitor.

    Wraps ``zeus.monitor.ZeusMonitor`` to track per-GPU energy over a named
    measurement window.  Zeus is the preferred sampler when available, as it
    reads hardware energy registers directly.

    All zeus imports are deferred - safe to import without zeus installed.

    Args:
        gpu_indices: PHYSICAL GPU indices to monitor. Defaults to ``[0]`` when
            None. Translated into the CUDA-visible space at window start, because
            ``ZeusMonitor`` indexes the visible set rather than physical devices.
    """

    WINDOW_NAME = "llem_measurement"

    def __init__(self, gpu_indices: list[int] | None = None) -> None:
        self._gpu_indices = gpu_indices if gpu_indices is not None else [0]
        # (physical, logical) pairs for the open window; set by start_tracking.
        self._pairs: list[tuple[int, int]] = []

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "zeus"

    def is_available(self) -> bool:
        """Return True if the zeus package is installed and importable."""
        try:
            from zeus.monitor import ZeusMonitor  # noqa: F401

            return True
        except ImportError:
            return False

    def _monitored_pairs(self) -> list[tuple[int, int]]:
        """Return ``(physical, logical)`` pairs for the devices this window covers.

        ``ZeusMonitor`` indexes the CUDA-VISIBLE set, not physical devices, so a
        restricting ``CUDA_VISIBLE_DEVICES`` (which the process runner sets when
        llem is scoped to a subset of the host's GPUs) makes the two spaces
        differ. Handing Zeus a physical index there would monitor the wrong
        device or raise. Identity when nothing restricts visibility.

        Raises:
            ExperimentError: A monitored device is not CUDA-visible. Zeus cannot
                see it, so dropping it would silently corrupt the measurement:
                an all-invisible set records a plausible-looking 0.0 J, and a
                partially-invisible one under-reports. Both must be impossible
                without a loud failure.
        """
        from llenergymeasure.device.gpu_info import cuda_visible_physical_order
        from llenergymeasure.utils.exceptions import ExperimentError

        visible = cuda_visible_physical_order()
        if visible is None:
            return [(i, i) for i in self._gpu_indices]
        invisible = [i for i in self._gpu_indices if i not in visible]
        if invisible:
            raise ExperimentError(
                f"Zeus cannot monitor physical GPU(s) {invisible}: CUDA_VISIBLE_DEVICES "
                f"exposes only {visible} to this process. Measuring the remaining devices "
                f"would silently under-report energy (monitored set: {self._gpu_indices}), "
                "so the measurement is refused. Align the monitored devices with "
                "CUDA_VISIBLE_DEVICES, or widen it to cover them."
            )
        return [(i, visible.index(i)) for i in self._gpu_indices]

    def start_tracking(self) -> Any:
        """Begin a Zeus measurement window.

        Returns:
            A ZeusMonitor instance with an active window.
        """
        from zeus.monitor import ZeusMonitor

        self._pairs = self._monitored_pairs()
        monitor = ZeusMonitor(gpu_indices=[logical for _, logical in self._pairs], cpu_indices=[])
        monitor.begin_window(self.WINDOW_NAME)
        return monitor

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        """Close the measurement window and return energy totals.

        The per-GPU breakdown is re-keyed to PHYSICAL device indices so it lands
        in the same index space as the NVML sampler's, whatever
        ``CUDA_VISIBLE_DEVICES`` says.

        Args:
            tracker: ZeusMonitor returned by start_tracking().

        Returns:
            EnergyMeasurement with total_j and per-GPU breakdown.
        """
        measurement = tracker.end_window(self.WINDOW_NAME)

        # measurement.energy is dict[int, float]: Zeus's own (logical) index -> joules
        physical_of: dict[int, int] = {logical: physical for physical, logical in self._pairs}
        per_gpu_j: dict[int, float] = {}
        for logical, joules in dict(measurement.energy).items():
            index = int(logical)
            per_gpu_j[physical_of.get(index, index)] = float(joules)
        total_j: float = sum(per_gpu_j.values())
        duration_sec: float = float(measurement.time)

        return EnergyMeasurement(
            total_j=total_j,
            duration_sec=duration_sec,
            samples=[],  # Zeus does not expose raw samples
            per_gpu_j=per_gpu_j if per_gpu_j else None,
        )
