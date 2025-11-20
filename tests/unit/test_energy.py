"""
Unit tests for energy tracking.

Tests EnergyTracker wrapper around CodeCarbon.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llm_efficiency.metrics.energy import EnergyTracker


class TestEnergyTracker:
    """Tests for EnergyTracker class."""

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_initialization(self, mock_tracker_class):
        """Test tracker initialization."""
        tracker = EnergyTracker(
            experiment_id="test_001",
            output_dir=Path("test_output"),
        )
        
        assert tracker.experiment_id == "test_001"
        assert tracker.output_dir == Path("test_output")
        assert tracker._tracker is None

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_start_tracking(self, mock_tracker_class):
        """Test starting energy tracking."""
        mock_instance = Mock()
        mock_tracker_class.return_value = mock_instance
        
        tracker = EnergyTracker(experiment_id="test_001")
        tracker.start()
        
        mock_tracker_class.assert_called_once()
        mock_instance.start.assert_called_once()

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_stop_tracking(self, mock_tracker_class):
        """Test stopping energy tracking."""
        mock_instance = Mock()
        mock_instance.stop.return_value = 0.123  # Mock emissions
        mock_tracker_class.return_value = mock_instance
        
        tracker = EnergyTracker(experiment_id="test_001")
        tracker.start()
        emissions = tracker.stop()
        
        mock_instance.stop.assert_called_once()
        assert emissions == 0.123

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_context_manager(self, mock_tracker_class):
        """Test using tracker as context manager."""
        mock_instance = Mock()
        mock_instance.stop.return_value = 0.456
        mock_tracker_class.return_value = mock_instance
        
        with EnergyTracker(experiment_id="test_001") as tracker:
            assert tracker._tracker is not None
        
        mock_instance.start.assert_called_once()
        mock_instance.stop.assert_called_once()

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_get_results(self, mock_tracker_class):
        """Test getting energy results."""
        mock_instance = Mock()
        mock_instance.final_emissions = 0.5
        mock_instance.final_emissions_data = {
            "duration": 120.0,
            "emissions": 0.5,
            "energy_consumed": 0.1,
            "cpu_energy": 0.03,
            "gpu_energy": 0.05,
            "ram_energy": 0.02,
        }
        mock_tracker_class.return_value = mock_instance
        
        tracker = EnergyTracker(experiment_id="test_001")
        tracker.start()
        tracker.stop()
        
        results = tracker.get_results()
        
        assert "duration_seconds" in results
        assert "emissions_kg_co2" in results
        assert "energy_consumed_kwh" in results

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_stop_without_start(self, mock_tracker_class):
        """Test stopping without starting raises error."""
        tracker = EnergyTracker(experiment_id="test_001")
        
        with pytest.raises(RuntimeError, match="not started"):
            tracker.stop()

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_output_directory_created(self, mock_tracker_class, tmp_path):
        """Test output directory is created."""
        output_dir = tmp_path / "energy_results"
        
        tracker = EnergyTracker(
            experiment_id="test_001",
            output_dir=output_dir,
        )
        
        assert output_dir.exists()

    @patch('llm_efficiency.metrics.energy.EmissionsTracker')
    def test_multiple_start_calls(self, mock_tracker_class):
        """Test calling start multiple times."""
        mock_instance = Mock()
        mock_tracker_class.return_value = mock_instance
        
        tracker = EnergyTracker(experiment_id="test_001")
        tracker.start()
        
        # Second start should not create new tracker
        tracker.start()
        
        # Should only be called once
        assert mock_tracker_class.call_count == 1
