"""Tests for domain metrics models."""

import pytest

from llenergymeasure.domain.metrics import (
    CombinedMetrics,
    ComputeMetrics,
    EnergyMetrics,
    InferenceMetrics,
)


class TestInferenceMetrics:
    """Tests for InferenceMetrics."""

    def test_create_valid(self):
        metrics = InferenceMetrics(
            total_tokens=100,
            input_tokens=20,
            output_tokens=80,
            inference_time_sec=2.0,
            tokens_per_second=50.0,
            latency_per_token_ms=20.0,
        )
        assert metrics.total_tokens == 100
        assert metrics.throughput == 50.0

    def test_throughput_property(self):
        metrics = InferenceMetrics(
            total_tokens=100,
            input_tokens=20,
            output_tokens=80,
            inference_time_sec=2.0,
            tokens_per_second=40.0,
            latency_per_token_ms=25.0,
        )
        assert metrics.throughput == metrics.tokens_per_second

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError):
            InferenceMetrics(
                total_tokens=100,
                # Missing other required fields
            )


class TestEnergyMetrics:
    """Tests for EnergyMetrics."""

    def test_create_with_defaults(self):
        metrics = EnergyMetrics(
            total_energy_j=100.0,
            duration_sec=10.0,
        )
        assert metrics.total_energy_j == 100.0
        assert metrics.gpu_energy_j == 0.0
        assert metrics.cpu_energy_j == 0.0

    def test_total_power_property(self):
        metrics = EnergyMetrics(
            total_energy_j=100.0,
            gpu_power_w=150.0,
            cpu_power_w=50.0,
            duration_sec=10.0,
        )
        assert metrics.total_power_w == 200.0

    def test_full_metrics(self):
        metrics = EnergyMetrics(
            total_energy_j=500.0,
            gpu_energy_j=400.0,
            cpu_energy_j=80.0,
            ram_energy_j=20.0,
            gpu_power_w=200.0,
            cpu_power_w=40.0,
            duration_sec=2.5,
            emissions_kg_co2=0.05,
            energy_per_token_j=0.5,
        )
        assert metrics.total_energy_j == 500.0
        assert metrics.emissions_kg_co2 == 0.05


class TestComputeMetrics:
    """Tests for ComputeMetrics."""

    def test_create_with_defaults(self):
        metrics = ComputeMetrics(flops_total=1e12)
        assert metrics.flops_total == 1e12
        assert metrics.flops_method == "unknown"
        assert metrics.compute_precision == "fp16"

    def test_full_metrics(self):
        metrics = ComputeMetrics(
            flops_total=1e12,
            flops_per_token=1e9,
            flops_per_second=5e11,
            peak_memory_mb=8000.0,
            model_memory_mb=7000.0,
            flops_method="calflops",
            flops_confidence="high",
            compute_precision="fp16",
        )
        assert metrics.flops_method == "calflops"
        assert metrics.flops_confidence == "high"


class TestCombinedMetrics:
    """Tests for CombinedMetrics."""

    @pytest.fixture
    def sample_metrics(self):
        return CombinedMetrics(
            inference=InferenceMetrics(
                total_tokens=1000,
                input_tokens=100,
                output_tokens=900,
                inference_time_sec=10.0,
                tokens_per_second=100.0,
                latency_per_token_ms=10.0,
            ),
            energy=EnergyMetrics(
                total_energy_j=500.0,
                gpu_power_w=100.0,
                cpu_power_w=50.0,
                duration_sec=10.0,
            ),
            compute=ComputeMetrics(
                flops_total=1e12,
                flops_per_second=1e11,
            ),
        )

    def test_efficiency_tokens_per_joule(self, sample_metrics):
        # 1000 tokens / 500 J = 2 tokens/J
        assert sample_metrics.efficiency_tokens_per_joule == 2.0

    def test_efficiency_flops_per_watt(self, sample_metrics):
        # 1e11 FLOPS / 150W = 6.67e8 FLOPS/W
        expected = 1e11 / 150.0
        assert sample_metrics.efficiency_flops_per_watt == pytest.approx(expected)

    def test_zero_energy_returns_zero_efficiency(self):
        metrics = CombinedMetrics(
            inference=InferenceMetrics(
                total_tokens=100,
                input_tokens=10,
                output_tokens=90,
                inference_time_sec=1.0,
                tokens_per_second=100.0,
                latency_per_token_ms=10.0,
            ),
            energy=EnergyMetrics(
                total_energy_j=0.0,
                duration_sec=1.0,
            ),
            compute=ComputeMetrics(flops_total=1e12),
        )
        assert metrics.efficiency_tokens_per_joule == 0.0
        assert metrics.efficiency_flops_per_watt == 0.0
