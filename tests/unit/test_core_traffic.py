"""Tests for traffic simulation module."""

import time

import numpy as np
import pytest

from llm_energy_measure.config.models import TrafficSimulation
from llm_energy_measure.core.traffic import TrafficGenerator, apply_traffic_delay


class TestTrafficGenerator:
    """Tests for TrafficGenerator."""

    def test_disabled_returns_zero(self):
        """When disabled, inter-arrival time is 0."""
        config = TrafficSimulation(enabled=False)
        generator = TrafficGenerator(config)
        assert generator.get_inter_arrival_time() == 0.0

    def test_constant_mode_fixed_delay(self):
        """Constant mode returns fixed inter-arrival time."""
        config = TrafficSimulation(enabled=True, mode="constant", target_qps=10.0)
        generator = TrafficGenerator(config)

        # Should return 1/10 = 0.1 seconds every time
        for _ in range(5):
            delay = generator.get_inter_arrival_time()
            assert delay == pytest.approx(0.1, rel=1e-9)

    def test_poisson_mode_exponential_distribution(self):
        """Poisson mode returns exponentially distributed delays."""
        config = TrafficSimulation(enabled=True, mode="poisson", target_qps=10.0, seed=42)
        generator = TrafficGenerator(config)

        # Collect many samples
        delays = [generator.get_inter_arrival_time() for _ in range(1000)]

        # Mean should be close to 1/lambda = 0.1
        mean_delay = np.mean(delays)
        assert mean_delay == pytest.approx(0.1, rel=0.1)  # 10% tolerance

        # All delays should be positive
        assert all(d > 0 for d in delays)

    def test_seed_reproducibility(self):
        """Same seed produces same sequence."""
        config = TrafficSimulation(enabled=True, mode="poisson", target_qps=5.0, seed=123)

        gen1 = TrafficGenerator(config)
        delays1 = [gen1.get_inter_arrival_time() for _ in range(10)]

        gen2 = TrafficGenerator(config)
        delays2 = [gen2.get_inter_arrival_time() for _ in range(10)]

        assert delays1 == delays2

    def test_request_count_tracking(self):
        """Request count increments correctly."""
        config = TrafficSimulation(enabled=True, mode="constant", target_qps=1.0)
        generator = TrafficGenerator(config)

        assert generator.request_count == 0
        generator.get_inter_arrival_time()
        assert generator.request_count == 1
        generator.get_inter_arrival_time()
        assert generator.request_count == 2

    def test_reset_clears_state(self):
        """Reset clears request count and optionally updates seed."""
        config = TrafficSimulation(enabled=True, mode="poisson", target_qps=1.0, seed=42)
        generator = TrafficGenerator(config)

        generator.get_inter_arrival_time()
        generator.get_inter_arrival_time()
        assert generator.request_count == 2

        generator.reset()
        assert generator.request_count == 0

    def test_reset_with_new_seed(self):
        """Reset with new seed changes sequence."""
        config = TrafficSimulation(enabled=True, mode="poisson", target_qps=5.0, seed=42)
        generator = TrafficGenerator(config)

        delays_seed42 = [generator.get_inter_arrival_time() for _ in range(5)]

        generator.reset(seed=123)
        delays_seed123 = [generator.get_inter_arrival_time() for _ in range(5)]

        assert delays_seed42 != delays_seed123


class TestApplyTrafficDelay:
    """Tests for apply_traffic_delay function."""

    def test_disabled_returns_zero(self):
        """When disabled, no delay is applied."""
        config = TrafficSimulation(enabled=False)
        delay = apply_traffic_delay(config, batch_idx=1)
        assert delay == 0.0

    def test_first_batch_no_delay(self):
        """First batch (idx=0) gets no delay."""
        config = TrafficSimulation(enabled=True, mode="constant", target_qps=1.0)
        delay = apply_traffic_delay(config, batch_idx=0)
        assert delay == 0.0

    def test_subsequent_batches_get_delay(self):
        """Batches after first get appropriate delay."""
        config = TrafficSimulation(enabled=True, mode="constant", target_qps=100.0)  # 10ms delays

        start = time.perf_counter()
        delay = apply_traffic_delay(config, batch_idx=1)
        elapsed = time.perf_counter() - start

        # Should have slept for ~10ms
        assert delay == pytest.approx(0.01, rel=0.01)
        assert elapsed >= 0.005  # At least 5ms passed
