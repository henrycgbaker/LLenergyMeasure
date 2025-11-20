"""
Unit tests for distributed computing utilities.

Tests Accelerate integration and experiment ID generation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llm_efficiency.core.distributed import (
    setup_accelerator,
    generate_experiment_id,
    synchronize_processes,
    is_main_process,
    get_process_info,
    cleanup_distributed,
)


class TestSetupAccelerator:
    """Tests for setup_accelerator function."""

    @patch('llm_efficiency.core.distributed.Accelerator')
    def test_setup_default_config(self, mock_accelerator_class):
        """Test accelerator setup with default configuration."""
        mock_instance = Mock()
        mock_accelerator_class.return_value = mock_instance
        
        accelerator = setup_accelerator()
        
        assert accelerator == mock_instance
        mock_accelerator_class.assert_called_once_with(
            mixed_precision="no",
            gradient_accumulation_steps=1,
        )

    @patch('llm_efficiency.core.distributed.Accelerator')
    def test_setup_custom_config(self, mock_accelerator_class):
        """Test accelerator setup with custom configuration."""
        mock_instance = Mock()
        mock_accelerator_class.return_value = mock_instance
        
        accelerator = setup_accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=4,
        )
        
        assert accelerator == mock_instance
        mock_accelerator_class.assert_called_once_with(
            mixed_precision="fp16",
            gradient_accumulation_steps=4,
        )


class TestGenerateExperimentId:
    """Tests for experiment ID generation."""

    def test_new_id_file(self, tmp_path):
        """Test generating first experiment ID."""
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = True
        mock_accelerator.wait_for_everyone = Mock()
        
        id_file = tmp_path / "exp_id.txt"
        
        exp_id = generate_experiment_id(mock_accelerator, id_file)
        
        assert exp_id == "0001"
        assert id_file.exists()
        assert id_file.read_text() == "1"

    def test_increment_existing_id(self, tmp_path):
        """Test incrementing existing experiment ID."""
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = True
        mock_accelerator.wait_for_everyone = Mock()
        
        id_file = tmp_path / "exp_id.txt"
        id_file.write_text("5")
        
        exp_id = generate_experiment_id(mock_accelerator, id_file)
        
        assert exp_id == "0006"
        assert id_file.read_text() == "6"

    def test_non_main_process_waits(self, tmp_path):
        """Test non-main process waits for ID."""
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = False
        mock_accelerator.wait_for_everyone = Mock()
        
        id_file = tmp_path / "exp_id.txt"
        id_file.write_text("10")
        
        exp_id = generate_experiment_id(mock_accelerator, id_file)
        
        # Non-main process reads but doesn't write
        assert exp_id == "0010"
        assert id_file.read_text() == "10"  # Unchanged

    def test_zero_padding(self, tmp_path):
        """Test ID is zero-padded to 4 digits."""
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = True
        mock_accelerator.wait_for_everyone = Mock()
        
        id_file = tmp_path / "exp_id.txt"
        id_file.write_text("99")
        
        exp_id = generate_experiment_id(mock_accelerator, id_file)
        
        assert exp_id == "0100"
        assert len(exp_id) == 4


class TestSynchronizeProcesses:
    """Tests for process synchronization."""

    def test_synchronize_with_accelerator(self):
        """Test synchronization with accelerator."""
        mock_accelerator = Mock()
        mock_accelerator.wait_for_everyone = Mock()
        
        synchronize_processes(mock_accelerator)
        
        mock_accelerator.wait_for_everyone.assert_called_once()

    def test_synchronize_without_accelerator(self):
        """Test synchronization without accelerator does nothing."""
        # Should not raise
        synchronize_processes(None)


class TestIsMainProcess:
    """Tests for main process check."""

    def test_is_main_with_accelerator(self):
        """Test main process check with accelerator."""
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = True
        
        assert is_main_process(mock_accelerator) is True

    def test_is_not_main_with_accelerator(self):
        """Test non-main process check with accelerator."""
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = False
        
        assert is_main_process(mock_accelerator) is False

    def test_is_main_without_accelerator(self):
        """Test main process check without accelerator defaults to True."""
        assert is_main_process(None) is True


class TestGetProcessInfo:
    """Tests for getting process information."""

    def test_get_info_with_accelerator(self):
        """Test getting process info with accelerator."""
        mock_accelerator = Mock()
        mock_accelerator.process_index = 2
        mock_accelerator.num_processes = 4
        mock_accelerator.device = "cuda:2"
        
        info = get_process_info(mock_accelerator)
        
        assert info["process_index"] == 2
        assert info["num_processes"] == 4
        assert info["device"] == "cuda:2"
        assert info["is_main_process"] is False

    def test_get_info_without_accelerator(self):
        """Test getting process info without accelerator."""
        info = get_process_info(None)
        
        assert info["process_index"] == 0
        assert info["num_processes"] == 1
        assert info["is_main_process"] is True


class TestCleanupDistributed:
    """Tests for distributed cleanup."""

    @patch('llm_efficiency.core.distributed.torch.distributed.is_initialized')
    @patch('llm_efficiency.core.distributed.torch.distributed.destroy_process_group')
    def test_cleanup_when_initialized(self, mock_destroy, mock_is_init):
        """Test cleanup when distributed is initialized."""
        mock_is_init.return_value = True
        
        cleanup_distributed()
        
        mock_destroy.assert_called_once()

    @patch('llm_efficiency.core.distributed.torch.distributed.is_initialized')
    @patch('llm_efficiency.core.distributed.torch.distributed.destroy_process_group')
    def test_cleanup_when_not_initialized(self, mock_destroy, mock_is_init):
        """Test cleanup when distributed is not initialized."""
        mock_is_init.return_value = False
        
        cleanup_distributed()
        
        mock_destroy.assert_not_called()
