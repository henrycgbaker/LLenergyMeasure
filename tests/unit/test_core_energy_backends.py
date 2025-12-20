"""Tests for energy measurement backends."""

from unittest.mock import MagicMock, patch

import pytest

from llm_bench.core.energy_backends.codecarbon import (
    CodeCarbonBackend,
    CodeCarbonData,
)
from llm_bench.domain.metrics import EnergyMetrics


class TestCodeCarbonData:
    """Tests for CodeCarbonData dataclass."""

    def test_creation(self) -> None:
        data = CodeCarbonData(
            cpu_power=50.0,
            gpu_power=150.0,
            ram_power=10.0,
            cpu_energy=0.1,
            gpu_energy=0.3,
            ram_energy=0.02,
            total_energy_kwh=0.001,
            emissions_kg=0.0005,
        )

        assert data.cpu_power == 50.0
        assert data.gpu_power == 150.0
        assert data.total_energy_kwh == 0.001

    def test_creation_with_nones(self) -> None:
        data = CodeCarbonData(
            cpu_power=None,
            gpu_power=None,
            ram_power=None,
            cpu_energy=None,
            gpu_energy=None,
            ram_energy=None,
            total_energy_kwh=0.0,
            emissions_kg=None,
        )

        assert data.cpu_power is None
        assert data.total_energy_kwh == 0.0


class TestCodeCarbonBackend:
    """Tests for CodeCarbonBackend."""

    def test_initialization(self) -> None:
        backend = CodeCarbonBackend(
            measure_power_secs=2,
            tracking_mode="machine",
        )

        assert backend._measure_power_secs == 2
        assert backend._tracking_mode == "machine"

    def test_name_property(self) -> None:
        backend = CodeCarbonBackend()
        assert backend.name == "codecarbon"

    def test_is_available_when_installed(self) -> None:
        with patch.dict("sys.modules", {"codecarbon": MagicMock()}):
            backend = CodeCarbonBackend()
            assert backend.is_available() is True

    def test_start_tracking_creates_and_returns_tracker(self) -> None:
        mock_tracker = MagicMock()

        with patch(
            "codecarbon.EmissionsTracker",
            return_value=mock_tracker,
        ):
            backend = CodeCarbonBackend()
            tracker = backend.start_tracking()

            mock_tracker.start.assert_called_once()
            assert tracker is mock_tracker

    def test_stop_tracking_without_tracker_returns_zeros(self) -> None:
        backend = CodeCarbonBackend()
        result = backend.stop_tracking(None)

        assert isinstance(result, EnergyMetrics)
        assert result.total_energy_j == 0.0
        assert result.gpu_power_w == 0.0

    def test_stop_tracking_extracts_and_converts_data(self) -> None:
        mock_tracker = MagicMock()
        mock_emissions = MagicMock()
        mock_emissions.energy_consumed = 0.001  # 1 Wh = 0.001 kWh
        mock_emissions.gpu_power = 100.0
        mock_tracker._prepare_emissions_data.return_value = mock_emissions

        backend = CodeCarbonBackend()
        result = backend.stop_tracking(mock_tracker)

        mock_tracker.stop.assert_called_once()
        # 0.001 kWh = 3.6 J
        assert result.total_energy_j == pytest.approx(3600.0)
        assert result.gpu_power_w == 100.0

    def test_convert_to_metrics(self) -> None:
        backend = CodeCarbonBackend()
        data = CodeCarbonData(
            cpu_power=50.0,
            gpu_power=150.0,
            ram_power=10.0,
            cpu_energy=0.05,
            gpu_energy=0.15,
            ram_energy=0.01,
            total_energy_kwh=0.01,  # 10 Wh = 36000 J
            emissions_kg=0.005,
        )

        result = backend._convert_to_metrics(data)

        assert isinstance(result, EnergyMetrics)
        assert result.total_energy_j == pytest.approx(36000.0)
        assert result.gpu_power_w == 150.0

    def test_get_raw_data_with_none_tracker(self) -> None:
        backend = CodeCarbonBackend()
        result = backend.get_raw_data(None)

        assert result is None

    def test_get_raw_data_with_tracker(self) -> None:
        mock_tracker = MagicMock()
        mock_emissions = MagicMock()
        mock_emissions.energy_consumed = 0.002
        mock_emissions.gpu_power = 200.0
        mock_tracker._prepare_emissions_data.return_value = mock_emissions

        backend = CodeCarbonBackend()
        result = backend.get_raw_data(mock_tracker)

        assert isinstance(result, CodeCarbonData)
        assert result.total_energy_kwh == 0.002
        assert result.gpu_power == 200.0
