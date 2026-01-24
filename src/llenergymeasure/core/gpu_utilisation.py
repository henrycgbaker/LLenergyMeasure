"""GPU utilisation sampling during inference.

Provides background sampling of GPU metrics using NVML via the nvidia-ml-py package
(imports as pynvml). Note: the old 'pynvml' PyPI package is deprecated - this uses
the official 'nvidia-ml-py' package which provides the same pynvml API.

Gracefully handles unavailability - returns empty samples if NVML not available
(e.g., CUDA context conflicts with vLLM).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class UtilisationSample:
    """Single GPU utilisation sample."""

    timestamp: float
    sm_utilisation: float | None = None
    memory_bandwidth: float | None = None


class GPUUtilisationSampler:
    """Background sampler for GPU utilisation metrics.

    Uses pynvml to sample GPU metrics during inference. Thread-safe context
    manager pattern. Gracefully handles pynvml unavailability.

    Usage:
        with GPUUtilisationSampler(device_index=0) as sampler:
            # ... run inference ...
            pass
        samples = sampler.get_samples()
        mean_util = sampler.get_mean_utilisation()
    """

    def __init__(
        self,
        device_index: int = 0,
        sample_interval_ms: int = 100,
    ) -> None:
        """Initialise GPU utilisation sampler.

        Args:
            device_index: CUDA device index to monitor.
            sample_interval_ms: Interval between samples in milliseconds.
        """
        self._device_index = device_index
        self._sample_interval = sample_interval_ms / 1000.0
        self._samples: list[UtilisationSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._pynvml_available = False

    def __enter__(self) -> GPUUtilisationSampler:
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
        """Background sampling loop using pynvml."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml_available = True

            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            except pynvml.NVMLError as e:
                logger.debug(f"GPU utilisation: failed to get device handle: {e}")
                pynvml.nvmlShutdown()
                return

            while self._running:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._samples.append(
                        UtilisationSample(
                            timestamp=time.perf_counter(),
                            sm_utilisation=float(util.gpu),
                            memory_bandwidth=float(util.memory),
                        )
                    )
                except pynvml.NVMLError:
                    # Sampling failed for this interval, skip
                    pass
                time.sleep(self._sample_interval)

            pynvml.nvmlShutdown()

        except ImportError:
            logger.debug("GPU utilisation: pynvml not available")
        except Exception as e:
            logger.debug(f"GPU utilisation sampling failed: {e}")

    def get_samples(self) -> list[float]:
        """Get SM utilisation samples.

        Returns:
            List of SM utilisation percentages (0-100).
        """
        return [s.sm_utilisation for s in self._samples if s.sm_utilisation is not None]

    def get_memory_bandwidth_samples(self) -> list[float]:
        """Get memory bandwidth utilisation samples.

        Returns:
            List of memory bandwidth utilisation percentages (0-100).
        """
        return [s.memory_bandwidth for s in self._samples if s.memory_bandwidth is not None]

    def get_mean_utilisation(self) -> float | None:
        """Get mean SM utilisation.

        Returns:
            Mean SM utilisation percentage, or None if no samples.
        """
        samples = self.get_samples()
        return sum(samples) / len(samples) if samples else None

    def get_mean_memory_bandwidth(self) -> float | None:
        """Get mean memory bandwidth utilisation.

        Returns:
            Mean memory bandwidth utilisation percentage, or None if no samples.
        """
        samples = self.get_memory_bandwidth_samples()
        return sum(samples) / len(samples) if samples else None

    @property
    def sample_count(self) -> int:
        """Number of samples collected."""
        return len(self._samples)

    @property
    def is_available(self) -> bool:
        """Whether pynvml sampling is available."""
        return self._pynvml_available


@dataclass
class GPUSamplerResult:
    """Result from GPU utilisation sampling."""

    sm_samples: list[float] = field(default_factory=list)
    memory_bandwidth_samples: list[float] = field(default_factory=list)
    sample_count: int = 0
    available: bool = False

    @classmethod
    def from_sampler(cls, sampler: GPUUtilisationSampler) -> GPUSamplerResult:
        """Create result from sampler."""
        return cls(
            sm_samples=sampler.get_samples(),
            memory_bandwidth_samples=sampler.get_memory_bandwidth_samples(),
            sample_count=sampler.sample_count,
            available=sampler.is_available,
        )
