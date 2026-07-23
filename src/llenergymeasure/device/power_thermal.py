"""Power and thermal sampling during inference.

Provides background sampling of GPU power, memory, temperature, and thermal
throttle state using NVML via the nvidia-ml-py package (imports as pynvml).

Gracefully handles unavailability - returns empty samples and default
ThrottleInfo if NVML is not available (e.g., no GPU, CUDA context
conflicts with vLLM).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from llenergymeasure.device.gpu_info import nvml_context
from llenergymeasure.domain.metrics import ThrottleAxis, ThrottleInfo
from llenergymeasure.utils.formatting import bytes_to_mb

logger = logging.getLogger(__name__)


def _throttle_bit(pynvml: object, new_name: str, old_name: str) -> int:
    """Resolve an NVML clocks-reason bit by its new (NVML 12+) or legacy name.

    Returns 0 when neither constant is present, so the bit is simply never set.
    """
    return getattr(pynvml, new_name, getattr(pynvml, old_name, 0))


@dataclass
class PowerThermalSample:
    """Single power/thermal sample from GPU."""

    timestamp: float
    power_w: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None
    sm_utilisation: float | None = None
    memory_bandwidth_utilisation: float | None = None
    """Percent of time the memory controller was active during the sample
    interval (NVML ``utilization.memory`` proxy). This is NOT true achieved
    memory bandwidth - it is the fraction of time any read/write was issued."""
    thermal_throttle: bool = False
    throttle_reasons: int = 0
    gpu_index: int = 0


class PowerThermalSampler:
    """Background sampler for GPU power, memory, temperature, and throttle state.

    Uses pynvml to sample GPU metrics during inference. Thread-safe context
    manager pattern. Gracefully handles pynvml unavailability.

    Supports monitoring multiple GPUs concurrently. Each sample tick produces
    one PowerThermalSample per monitored GPU, tagged with its gpu_index.

    Usage:
        with PowerThermalSampler(gpu_indices=[0, 1]) as sampler:
            # ... run inference ...
            pass
        samples = sampler.get_samples()
        throttle_info = sampler.get_throttle_info()
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        sample_interval_ms: int = 100,
    ) -> None:
        """Initialise power/thermal sampler.

        Args:
            gpu_indices: CUDA device indices to monitor. Defaults to [0] when None.
            sample_interval_ms: Interval between samples in milliseconds.
        """
        self._gpu_indices = gpu_indices if gpu_indices is not None else [0]

        self._sample_interval = sample_interval_ms / 1000.0
        self._sample_interval_ms = sample_interval_ms
        self._samples: list[PowerThermalSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._pynvml_available = False

    def __enter__(self) -> PowerThermalSampler:
        """Start sampling on context entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop sampling on context exit."""
        self.stop()

    def start(self) -> None:
        """Start background sampling thread."""
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for thread to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _sample_loop(self) -> None:
        """Background sampling loop using pynvml - polls all gpu_indices per tick."""
        try:
            import pynvml
        except ImportError:
            logger.debug("Power/thermal: pynvml not available")
            return

        with nvml_context():
            self._pynvml_available = True

            # Get handles for all monitored GPUs
            handles: list[tuple[int, object]] = []
            for gpu_idx in self._gpu_indices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                    handles.append((gpu_idx, handle))
                except Exception as e:
                    logger.debug("Power/thermal: failed to get handle for GPU %d: %s", gpu_idx, e)

            if not handles:
                logger.debug("Power/thermal: no GPU handles obtained")
                return

            # Resolve NVML throttle reason constants (prefer non-deprecated names)
            thermal_bits = 0
            for attr_new, attr_old in [
                (
                    "nvmlClocksEventReasonSwThermalSlowdown",
                    "nvmlClocksThrottleReasonSwThermalSlowdown",
                ),
                (
                    "nvmlClocksEventReasonHwThermalSlowdown",
                    "nvmlClocksThrottleReasonHwThermalSlowdown",
                ),
            ]:
                thermal_bits |= _throttle_bit(pynvml, attr_new, attr_old)

            # Prefer non-deprecated clock reasons query (NVML 12+)
            _get_clocks_reasons = getattr(
                pynvml,
                "nvmlDeviceGetCurrentClocksEventReasons",
                getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None),
            )

            try:
                while self._running:
                    for gpu_idx, handle in handles:
                        try:
                            sample = PowerThermalSample(
                                timestamp=time.perf_counter(),
                                gpu_index=gpu_idx,
                            )

                            # Power (milliwatts -> watts)
                            try:
                                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                                sample.power_w = power_mw / 1000.0
                            except pynvml.NVMLError:
                                pass

                            # Memory
                            try:
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                sample.memory_used_mb = bytes_to_mb(mem_info.used)
                                sample.memory_total_mb = bytes_to_mb(mem_info.total)
                            except pynvml.NVMLError:
                                pass

                            # Temperature
                            try:
                                temp = pynvml.nvmlDeviceGetTemperature(
                                    handle, pynvml.NVML_TEMPERATURE_GPU
                                )
                                sample.temperature_c = float(temp)
                            except pynvml.NVMLError:
                                pass

                            # Utilisation (SM + memory-controller activity proxy)
                            try:
                                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                sample.sm_utilisation = float(util.gpu)
                                sample.memory_bandwidth_utilisation = float(util.memory)
                            except pynvml.NVMLError:
                                pass

                            # Throttle reasons
                            try:
                                if _get_clocks_reasons is not None:
                                    reasons = _get_clocks_reasons(handle)
                                    sample.throttle_reasons = reasons
                                    sample.thermal_throttle = bool(reasons & thermal_bits)
                            except pynvml.NVMLError:
                                pass

                            self._samples.append(sample)

                        except pynvml.NVMLError:
                            # Entire sample for this GPU failed, skip
                            pass

                    time.sleep(self._sample_interval)
            except Exception as e:
                logger.debug("Power/thermal sampling failed: %s", e)

    def get_samples(self) -> list[PowerThermalSample]:
        """Get all collected samples."""
        return list(self._samples)

    def get_throttle_info(self) -> ThrottleInfo:
        """Summarise thermal throttle state from collected samples (all GPUs).

        Returns:
            ThrottleInfo with aggregated throttle data across all GPUs.
        """
        if not self._samples:
            return ThrottleInfo()

        throttled_timestamps = [s.timestamp for s in self._samples if s.thermal_throttle]
        throttle_duration = len(throttled_timestamps) * (self._sample_interval_ms / 1000.0)

        temperatures = [s.temperature_c for s in self._samples if s.temperature_c is not None]
        max_temp = max(temperatures) if temperatures else None

        # Check individual throttle reason bits across all samples
        # NVML throttle reason bitmask constants
        try:
            import pynvml

            combined_reasons = 0
            for s in self._samples:
                combined_reasons |= s.throttle_reasons

            hw_thermal_bit = _throttle_bit(
                pynvml,
                "nvmlClocksEventReasonHwThermalSlowdown",
                "nvmlClocksThrottleReasonHwThermalSlowdown",
            )
            sw_thermal_bit = _throttle_bit(
                pynvml,
                "nvmlClocksEventReasonSwThermalSlowdown",
                "nvmlClocksThrottleReasonSwThermalSlowdown",
            )
            sw_power_bit = _throttle_bit(
                pynvml,
                "nvmlClocksEventReasonSwPowerCap",
                "nvmlClocksThrottleReasonSwPowerCap",
            )
            hw_power_bit = _throttle_bit(
                pynvml,
                "nvmlClocksEventReasonHwPowerBrakeSlowdown",
                "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
            )
            # Combined per-axis "any" bits: True if either hw or sw slowdown occurred.
            thermal_bit = hw_thermal_bit | sw_thermal_bit
            power_bit = hw_power_bit | sw_power_bit

            return ThrottleInfo(
                thermal=ThrottleAxis(
                    any=bool(combined_reasons & thermal_bit),
                    hw=bool(combined_reasons & hw_thermal_bit),
                    sw=bool(combined_reasons & sw_thermal_bit),
                ),
                power=ThrottleAxis(
                    any=bool(combined_reasons & power_bit),
                    hw=bool(combined_reasons & hw_power_bit),
                    sw=bool(combined_reasons & sw_power_bit),
                ),
                throttle_duration_sec=throttle_duration,
                max_temperature_c=max_temp,
                throttle_timestamps=throttled_timestamps,
            )
        except ImportError:
            # pynvml not available - return basic info from sample flags
            return ThrottleInfo(
                throttle_duration_sec=throttle_duration,
                max_temperature_c=max_temp,
                throttle_timestamps=throttled_timestamps,
            )

    @property
    def sample_count(self) -> int:
        """Number of samples collected."""
        return len(self._samples)

    @property
    def is_available(self) -> bool:
        """Whether pynvml sampling is available."""
        return self._pynvml_available
