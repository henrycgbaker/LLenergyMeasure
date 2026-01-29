"""Tests for schema v3 domain models.

Tests the new domain models introduced in schema v3:
- EnergyBreakdown: baseline-adjusted energy attribution
- ThermalThrottleInfo: GPU thermal/power throttling detection
- WarmupResult: convergence detection result
- EnvironmentMetadata: hardware/software environment capture
- Backwards compatibility: v2-style RawProcessResult still works
"""

from datetime import datetime

import pytest

from llenergymeasure.domain.environment import (
    ContainerEnvironment,
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    GPUEnvironment,
    ThermalEnvironment,
)
from llenergymeasure.domain.experiment import RawProcessResult, Timestamps
from llenergymeasure.domain.metrics import (
    ComputeMetrics,
    EnergyBreakdown,
    EnergyMetrics,
    InferenceMetrics,
    ThermalThrottleInfo,
    WarmupResult,
)


class TestEnergyBreakdown:
    """Tests for EnergyBreakdown model."""

    def test_construction_with_defaults(self):
        eb = EnergyBreakdown(raw_j=100.0)
        assert eb.raw_j == 100.0
        assert eb.adjusted_j is None
        assert eb.baseline_power_w is None
        assert eb.baseline_method is None
        assert eb.baseline_timestamp is None
        assert eb.baseline_cache_age_sec is None

    def test_construction_with_baseline(self):
        ts = datetime(2024, 1, 1, 12, 0, 0)
        eb = EnergyBreakdown(
            raw_j=100.0,
            adjusted_j=50.0,
            baseline_power_w=10.0,
            baseline_method="fresh",
            baseline_timestamp=ts,
            baseline_cache_age_sec=0.5,
        )
        assert eb.raw_j == 100.0
        assert eb.adjusted_j == 50.0
        assert eb.baseline_power_w == 10.0
        assert eb.baseline_method == "fresh"
        assert eb.baseline_timestamp == ts
        assert eb.baseline_cache_age_sec == pytest.approx(0.5)

    def test_unavailable_baseline(self):
        eb = EnergyBreakdown(
            raw_j=75.0,
            adjusted_j=None,
            baseline_power_w=None,
            baseline_method="unavailable",
        )
        assert eb.raw_j == 75.0
        assert eb.adjusted_j is None
        assert eb.baseline_method == "unavailable"


class TestThermalThrottleInfo:
    """Tests for ThermalThrottleInfo model."""

    def test_defaults_all_false(self):
        tt = ThermalThrottleInfo()
        assert tt.detected is False
        assert tt.thermal is False
        assert tt.power is False
        assert tt.sw_thermal is False
        assert tt.hw_thermal is False
        assert tt.hw_power is False
        assert tt.throttle_duration_sec == 0.0
        assert tt.max_temperature_c is None
        assert tt.throttle_timestamps == []

    def test_with_throttle_detected(self):
        tt = ThermalThrottleInfo(
            detected=True,
            thermal=True,
            sw_thermal=True,
            throttle_duration_sec=5.0,
            max_temperature_c=95.0,
            throttle_timestamps=[1.0, 2.0, 3.0],
        )
        assert tt.detected is True
        assert tt.thermal is True
        assert tt.sw_thermal is True
        assert tt.hw_thermal is False  # not set
        assert tt.throttle_duration_sec == 5.0
        assert tt.max_temperature_c == 95.0
        assert len(tt.throttle_timestamps) == 3


class TestWarmupResult:
    """Tests for WarmupResult model."""

    def test_converged_state(self):
        wr = WarmupResult(
            converged=True,
            final_cv=0.02,
            iterations_completed=15,
            target_cv=0.05,
            max_prompts=50,
            latencies_ms=[100.0, 101.0, 99.5],
        )
        assert wr.converged is True
        assert wr.final_cv == pytest.approx(0.02)
        assert wr.iterations_completed == 15
        assert wr.target_cv == pytest.approx(0.05)
        assert wr.max_prompts == 50
        assert len(wr.latencies_ms) == 3

    def test_not_converged_state(self):
        wr = WarmupResult(
            converged=False,
            final_cv=0.15,
            iterations_completed=50,
            target_cv=0.05,
            max_prompts=50,
        )
        assert wr.converged is False
        assert wr.final_cv == pytest.approx(0.15)
        assert wr.iterations_completed == 50
        assert wr.latencies_ms == []  # default empty

    def test_disabled_warmup(self):
        wr = WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=0,
            target_cv=0.05,
            max_prompts=50,
        )
        assert wr.converged is True
        assert wr.iterations_completed == 0


class TestEnvironmentMetadata:
    """Tests for EnvironmentMetadata model."""

    def test_construction(self):
        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920.0),
            cuda=CUDAEnvironment(version="12.4", driver_version="535.104"),
            thermal=ThermalEnvironment(temperature_c=42.0, power_limit_w=400.0),
            cpu=CPUEnvironment(governor="performance", platform="Linux"),
            container=ContainerEnvironment(detected=False),
            collected_at=datetime(2024, 6, 15, 10, 0, 0),
        )
        assert env.gpu.name == "NVIDIA A100-SXM4-80GB"
        assert env.cuda.version == "12.4"
        assert env.thermal.temperature_c == 42.0
        assert env.cpu.governor == "performance"
        assert env.container.detected is False

    def test_summary_line_format(self):
        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920.0),
            cuda=CUDAEnvironment(version="12.4", driver_version="535.104"),
            thermal=ThermalEnvironment(temperature_c=42.0),
            cpu=CPUEnvironment(platform="Linux"),
            container=ContainerEnvironment(detected=False),
            collected_at=datetime(2024, 6, 15, 10, 0, 0),
        )
        summary = env.summary_line
        assert "A100" in summary
        assert "CUDA 12.4" in summary
        assert "535.104" in summary
        assert "42C" in summary

    def test_summary_line_with_container(self):
        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920.0),
            cuda=CUDAEnvironment(version="12.4", driver_version="535.104"),
            cpu=CPUEnvironment(platform="Linux"),
            container=ContainerEnvironment(detected=True, runtime="docker"),
            collected_at=datetime(2024, 6, 15, 10, 0, 0),
        )
        assert "container" in env.summary_line

    def test_defaults_for_optional_fields(self):
        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="test", vram_total_mb=0.0),
            cuda=CUDAEnvironment(version="unknown", driver_version="unknown"),
            cpu=CPUEnvironment(platform="Linux"),
            collected_at=datetime(2024, 1, 1),
        )
        assert env.gpu.compute_capability is None
        assert env.thermal.temperature_c is None
        assert env.container.detected is False


class TestBackwardsCompatibility:
    """Tests that schema v3 fields are optional and v2-style results still work."""

    def _make_v2_result(self) -> RawProcessResult:
        """Create a RawProcessResult WITHOUT any v3 fields (v2 style)."""
        return RawProcessResult(
            experiment_id="exp_v2",
            process_index=0,
            gpu_id=0,
            config_name="test",
            model_name="gpt2",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 12, 0, 0),
                end=datetime(2024, 1, 1, 12, 0, 10),
                duration_sec=10.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=100,
                input_tokens=50,
                output_tokens=50,
                inference_time_sec=10.0,
                tokens_per_second=10.0,
                latency_per_token_ms=100.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=50.0,
                duration_sec=10.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=1e10,
            ),
        )

    def _make_v3_result(self) -> RawProcessResult:
        """Create a RawProcessResult WITH all v3 fields."""
        return RawProcessResult(
            experiment_id="exp_v3",
            process_index=0,
            gpu_id=0,
            config_name="test",
            model_name="gpt2",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 12, 0, 0),
                end=datetime(2024, 1, 1, 12, 0, 10),
                duration_sec=10.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=100,
                input_tokens=50,
                output_tokens=50,
                inference_time_sec=10.0,
                tokens_per_second=10.0,
                latency_per_token_ms=100.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=50.0,
                duration_sec=10.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=1e10,
            ),
            environment=EnvironmentMetadata(
                gpu=GPUEnvironment(name="A100", vram_total_mb=81920.0),
                cuda=CUDAEnvironment(version="12.4", driver_version="535"),
                cpu=CPUEnvironment(platform="Linux"),
                collected_at=datetime(2024, 1, 1),
            ),
            energy_breakdown=EnergyBreakdown(raw_j=50.0, adjusted_j=30.0, baseline_power_w=2.0),
            thermal_throttle=ThermalThrottleInfo(detected=False),
            warmup_result=WarmupResult(
                converged=True,
                final_cv=0.03,
                iterations_completed=10,
                target_cv=0.05,
                max_prompts=50,
            ),
        )

    def test_v2_result_without_v3_fields(self):
        """v2-style result (no v3 fields) must construct successfully."""
        result = self._make_v2_result()
        assert result.experiment_id == "exp_v2"
        assert result.environment is None
        assert result.energy_breakdown is None
        assert result.thermal_throttle is None
        assert result.warmup_result is None
        assert result.timeseries_path is None

    def test_v3_result_with_all_fields(self):
        """v3-style result with all new fields must construct successfully."""
        result = self._make_v3_result()
        assert result.experiment_id == "exp_v3"
        assert result.environment is not None
        assert result.environment.gpu.name == "A100"
        assert result.energy_breakdown is not None
        assert result.energy_breakdown.adjusted_j == 30.0
        assert result.thermal_throttle is not None
        assert result.thermal_throttle.detected is False
        assert result.warmup_result is not None
        assert result.warmup_result.converged is True

    def test_v2_and_v3_coexist(self):
        """Both v2 and v3 style results can exist simultaneously."""
        v2 = self._make_v2_result()
        v3 = self._make_v3_result()
        # Both are valid RawProcessResult instances
        assert v2.schema_version == v3.schema_version
        assert v2.inference_metrics.total_tokens == v3.inference_metrics.total_tokens
