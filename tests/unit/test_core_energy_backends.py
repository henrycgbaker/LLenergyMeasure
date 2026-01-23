"""Tests for energy measurement backends."""

from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.core.energy_backends import (
    CodeCarbonBackend,
    CodeCarbonData,
    EnergyBackend,
    clear_backends,
    get_backend,
    list_backends,
    register_backend,
)
from llenergymeasure.domain.metrics import EnergyMetrics
from llenergymeasure.exceptions import ConfigurationError


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


class TestPluginRegistry:
    """Tests for energy backend plugin registry."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Reset registry before each test, re-register defaults after."""
        # Clear and re-register defaults for clean state
        clear_backends()
        register_backend("codecarbon", CodeCarbonBackend)
        yield
        # Restore defaults after test
        clear_backends()
        register_backend("codecarbon", CodeCarbonBackend)

    def test_list_backends_includes_codecarbon(self) -> None:
        """Codecarbon should be registered by default."""
        backends = list_backends()
        assert "codecarbon" in backends

    def test_get_backend_returns_codecarbon(self) -> None:
        """get_backend should return a CodeCarbonBackend instance."""
        backend = get_backend("codecarbon")
        assert isinstance(backend, CodeCarbonBackend)
        assert backend.name == "codecarbon"

    def test_get_backend_with_kwargs(self) -> None:
        """get_backend should pass kwargs to constructor."""
        backend = get_backend("codecarbon", measure_power_secs=5)
        assert backend._measure_power_secs == 5

    def test_get_backend_unknown_raises_error(self) -> None:
        """get_backend should raise ConfigurationError for unknown backend."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_backend("nonexistent")
        assert "Unknown backend: 'nonexistent'" in str(exc_info.value)

    def test_register_custom_backend(self) -> None:
        """Should be able to register custom backends."""

        class MockBackend:
            """Mock backend for testing."""

            @property
            def name(self) -> str:
                return "mock"

            def start_tracking(self) -> None:
                pass

            def stop_tracking(self, tracker: None) -> EnergyMetrics:
                return EnergyMetrics(
                    total_energy_j=0.0,
                    gpu_energy_j=0.0,
                    cpu_energy_j=0.0,
                    ram_energy_j=0.0,
                    gpu_power_w=0.0,
                    cpu_power_w=0.0,
                    duration_sec=0.0,
                    emissions_kg_co2=0.0,
                    energy_per_token_j=0.0,
                )

            def is_available(self) -> bool:
                return True

        register_backend("mock", MockBackend)
        assert "mock" in list_backends()

        backend = get_backend("mock")
        assert backend.name == "mock"

    def test_register_duplicate_raises_error(self) -> None:
        """Registering same name twice should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            register_backend("codecarbon", CodeCarbonBackend)
        assert "already registered" in str(exc_info.value)

    def test_clear_backends(self) -> None:
        """clear_backends should remove all registered backends."""
        clear_backends()
        assert list_backends() == []

        # Re-register for other tests
        register_backend("codecarbon", CodeCarbonBackend)

    def test_codecarbon_implements_protocol(self) -> None:
        """CodeCarbonBackend should implement EnergyBackend protocol."""
        backend = CodeCarbonBackend()
        assert isinstance(backend, EnergyBackend)
