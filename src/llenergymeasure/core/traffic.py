"""Traffic simulation for MLPerf-style load testing.

This module provides realistic traffic pattern generation following
MLPerf inference benchmark methodology, using Poisson arrivals to
model real-world API request patterns.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from llenergymeasure.config.models import TrafficSimulation


class TrafficGenerator:
    """Generate inter-arrival times for realistic load simulation.

    Implements MLPerf-style traffic patterns:
    - constant: Fixed inter-arrival time (1/target_qps seconds)
    - poisson: Exponential inter-arrival times for Poisson process

    The Poisson process is the industry standard for modeling
    request arrivals in production API services.

    Example:
        >>> from llenergymeasure.config.models import TrafficSimulation
        >>> config = TrafficSimulation(enabled=True, mode="poisson", target_qps=10.0)
        >>> generator = TrafficGenerator(config)
        >>> for i in range(5):
        ...     delay = generator.get_inter_arrival_time()
        ...     print(f"Wait {delay:.3f}s before request {i+1}")
    """

    def __init__(self, config: TrafficSimulation, seed: int | None = None) -> None:
        """Initialize the traffic generator.

        Args:
            config: Traffic simulation configuration.
            seed: Random seed for reproducibility. If None, uses config.seed.
        """
        self.config = config
        self._seed = seed if seed is not None else config.seed
        self.rng = np.random.default_rng(self._seed)
        self._request_count = 0

        if config.enabled:
            logger.info(
                f"TrafficGenerator initialized: mode={config.mode}, "
                f"target_qps={config.target_qps}, seed={self._seed}"
            )

    def get_inter_arrival_time(self) -> float:
        """Get the time to wait before the next request.

        Returns:
            Seconds to wait. Returns 0.0 if traffic simulation is disabled.
        """
        if not self.config.enabled:
            return 0.0

        self._request_count += 1

        if self.config.mode == "poisson":
            # Exponential distribution for Poisson process
            # Mean inter-arrival time = 1/λ where λ = target_qps
            delay = float(self.rng.exponential(1.0 / self.config.target_qps))
        else:  # constant
            delay = 1.0 / self.config.target_qps

        return delay

    def wait_for_next_request(self) -> float:
        """Wait the appropriate time before the next request.

        This is a convenience method that combines getting the delay
        and sleeping.

        Returns:
            The actual delay applied in seconds.
        """
        delay = self.get_inter_arrival_time()
        if delay > 0:
            time.sleep(delay)
        return delay

    @property
    def request_count(self) -> int:
        """Number of requests generated so far."""
        return self._request_count

    def reset(self, seed: int | None = None) -> None:
        """Reset the generator state.

        Args:
            seed: New random seed. If None, uses original seed.
        """
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self._request_count = 0
        logger.debug(f"TrafficGenerator reset with seed={self._seed}")


def apply_traffic_delay(
    config: TrafficSimulation,
    batch_idx: int,
    generator: TrafficGenerator | None = None,
) -> float:
    """Apply traffic simulation delay before processing a batch.

    This is the main entry point for traffic simulation in the inference loop.

    Args:
        config: Traffic simulation configuration.
        batch_idx: Current batch index (0-indexed).
        generator: Optional pre-initialized generator. If None, creates one.

    Returns:
        The delay applied in seconds.
    """
    if not config.enabled:
        return 0.0

    # Skip delay for first batch (start immediately)
    if batch_idx == 0:
        logger.debug("Skipping traffic delay for first batch")
        return 0.0

    if generator is None:
        generator = TrafficGenerator(config)

    delay = generator.wait_for_next_request()

    if delay > 0.1:  # Only log significant delays
        logger.debug(f"Traffic delay: {delay:.3f}s before batch {batch_idx + 1}")

    return delay
